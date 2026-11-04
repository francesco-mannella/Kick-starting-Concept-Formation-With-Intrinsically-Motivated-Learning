import argparse
import contextlib
import io
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

from SMMain import Main  # noqa: F401  (pickled dumps look up Main in __main__)

GRID = 10
K = 5
METRICS = ["success", "touch", "match", "max_match", "hit2", "end_touch_d", "end_proprio_d"]


def episode_metrics(tr):
    touch_cols = [f"s{i}" for i in range(40)]
    rows = []
    for ep, g in tr.groupby("episode_id"):
        touch = g[touch_cols].values.sum(1) > 0
        dist = np.hypot(g.touch_x.values - g.prototype_x.values, g.touch_y.values - g.prototype_y.values)
        dist_p = np.hypot(g.proprio_x.values - g.prototype_x.values, g.proprio_y.values - g.prototype_y.values)
        ends = g.assign(dt=dist, dp=dist_p).groupby("tr_id").tail(5).groupby("tr_id")[["dt", "dp"]].mean()
        rows.append({"ep": int(ep), "success": float(g.tr_id.nunique() >= 2), "touch": touch.mean(), "match": g.match.mean(),
                     "max_match": g.match.max(), "hit2": (dist[touch] <= 2).mean() if touch.any() else 0.0,
                     "end_touch_d": float(ends.dt.median()), "end_proprio_d": float(ends.dp.median())})
    return pd.DataFrame(rows)


def segment_table(tr, episodes, eseed):
    seg = tr.drop_duplicates(["episode_id", "tr_id"]).copy()
    seg["node"] = (np.round(seg.prototype_x.values) * GRID + np.round(seg.prototype_y.values)).astype(int)
    ep = seg.episode_id.astype(int).values % len(episodes)
    out = pd.DataFrame({"group": [f"{eseed}_{e}" for e in seg.episode_id.astype(int)], "node": seg.node.values,
                        "gx": seg.prototype_x.values, "gy": seg.prototype_y.values,
                        "ctx": episodes.context.values[ep], "stretch": episodes.stretch.values[ep], "rot": episodes.rotation.values[ep]})
    out["shape"] = episodes["shape"].values[ep] if "shape" in episodes else "none"
    return out


def design(df, kind):
    cols = []
    if kind in ("ctx", "ctx_pos"):
        for c in sorted(df.ctx.unique()):
            cols.append((df.ctx == c).astype(float))
        for s in sorted(df["shape"].unique())[1:]:
            cols.append((df["shape"] == s).astype(float))
        cols.append((df.stretch - df.stretch.mean()).astype(float))
        for r in sorted(df.rot.unique())[1:]:
            cols.append((df.rot == r).astype(float))
    if kind in ("pos", "pos2", "ctx_pos"):
        x, y = (df.gx - 4.5) / 4.5, (df.gy - 4.5) / 4.5
        cols += [x, y]
        if kind in ("pos2", "ctx_pos"):
            cols += [x * x, y * y, x * y]
    return np.column_stack([np.ones(len(df))] + [np.asarray(c, float) for c in cols])


def cv_r2(X, Y, groups, folds=5, lookup_nodes=None, ridge=1e-3):
    pred = np.zeros_like(Y)
    for tr_i, te_i in GroupKFold(n_splits=folds).split(Y, Y, groups):
        if lookup_nodes is not None:
            glob = Y[tr_i].mean(0)
            for i in te_i:
                same = tr_i[lookup_nodes[tr_i] == lookup_nodes[i]]
                pred[i] = Y[same].mean(0) if len(same) else glob
        else:
            A = X[tr_i]
            beta = np.linalg.solve(A.T @ A + ridge * np.eye(A.shape[1]), A.T @ Y[tr_i])
            pred[te_i] = X[te_i] @ beta
    return 1 - ((Y - pred) ** 2).sum(0) / ((Y - Y.mean(0)) ** 2).sum(0)


def regression(seg, scores, rng, n_perm):
    groups = seg.group.values
    out = {}
    nodes = seg.node.values
    for name, kind in (("context", "ctx"), ("position", "pos"), ("position_quad", "pos2"), ("context_and_position", "ctx_pos")):
        X = design(seg, kind)
        out[name] = cv_r2(X, scores, groups)
    out["goal_node_lookup"] = cv_r2(None, scores, groups, lookup_nodes=nodes)
    ep_feat = seg.drop_duplicates("group")[["group", "ctx", "stretch", "rot", "shape"]].reset_index(drop=True)
    null = {"context": [], "position_quad": []}
    for _ in range(n_perm):
        perm = ep_feat.assign(**{c: ep_feat[c].values[rng.permutation(len(ep_feat))] for c in ("ctx", "stretch", "rot", "shape")})
        merged = seg[["group", "gx", "gy"]].merge(perm, on="group", how="left")
        null["context"].append(cv_r2(design(merged, "ctx"), scores, groups))
        shuffled = seg.assign(gx=seg.gx.values[(p := rng.permutation(len(seg)))], gy=seg.gy.values[p])
        null["position_quad"].append(cv_r2(design(shuffled, "pos2"), scores, groups))
    for k, v in null.items():
        out[f"{k}_null_p95"] = np.percentile(np.array(v), 95, axis=0)
    return out


def main():
    parser = argparse.ArgumentParser(description="Parametrized-skill analysis: regression of prototype parameters on context and graded-outcome interventions")
    parser.add_argument("--storage", required=True)
    parser.add_argument("--epoch", type=int, required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--eseeds", default="101,102,103,104,105,106")
    parser.add_argument("--iseeds", default="201,202,203", help="Evaluation seeds of the interventions")
    parser.add_argument("--n-episodes", type=int, default=36)
    parser.add_argument("--draws", type=int, default=2)
    parser.add_argument("--min-segments", type=int, default=2)
    parser.add_argument("--n-perm", type=int, default=30)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    rng = np.random.RandomState(args.seed)

    folder = Path(args.storage) / f"{args.epoch:06d}"
    main_obj = np.load(folder / "main.dump.npy", allow_pickle=True)[0]
    main_obj.params.use_wandb = False
    controller = main_obj.controller
    weights = controller.stm_a.get_weights().copy()
    protos = weights.T + getattr(controller, "policy_shared", 0)
    episodes = main_obj.episode_dataset

    def evaluate(w, eseed):
        controller.stm_a.set_weights(w)
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            _, tr = main_obj.evaluation_episodes(orig_controller=controller, epoch=0, n_episodes=args.n_episodes,
                                                 render=None, save_stats=False, e_seed=eseed)
        controller.stm_a.set_weights(weights)
        return tr

    segs = []
    for eseed in (int(s) for s in args.eseeds.split(",")):
        segs.append(segment_table(evaluate(weights, eseed), episodes, eseed))
    seg = pd.concat(segs, ignore_index=True)
    counts = seg.node.value_counts()
    used = np.array(sorted(counts[counts >= args.min_segments].index))
    seg = seg[seg.node.isin(used)].reset_index(drop=True)
    mean = protos[used].mean(0)
    _, sv, vt = np.linalg.svd(protos[used] - mean, full_matrices=False)
    comps, share = vt[:K], (sv ** 2 / (sv ** 2).sum())[:K]
    scores_all = (protos - mean) @ comps.T
    scores = scores_all[seg.node.values]
    result = {"used_nodes": int(len(used)), "segments": int(len(seg)), "episodes": int(seg.group.nunique()),
              "common_share_used": float((mean ** 2).sum() / (protos[used] ** 2).sum(1).mean()), "pc_variance_share": share.tolist()}
    reg = regression(seg, scores, rng, args.n_perm)
    result["regression"] = {k: np.round(v, 4).tolist() for k, v in reg.items()}

    def build(kind, k=None):
        p = protos.copy()
        s = scores_all[used]
        if kind == "flat":
            p[used] = protos[used] - np.outer(s[:, k], comps[k])
        elif kind == "shuffle":
            p[used] = protos[used] + np.outer(s[rng.permutation(len(used)), k] - s[:, k], comps[k])
        elif kind == "flat_top":
            p[used] = protos[used] - s @ comps
        elif kind == "keep_top":
            p[used] = mean + s @ comps
        elif kind == "flat_all":
            p[used] = mean
        return p.T

    conds = [("intact", None, None, 1)]
    conds += [(f"flat_pc{k + 1}", "flat", k, 1) for k in range(K)]
    conds += [(f"shuffle_pc{k + 1}", "shuffle", k, args.draws) for k in range(K)]
    conds += [("flat_top5", "flat_top", None, 1), ("keep_top5", "keep_top", None, 1), ("flat_all", "flat_all", None, 1)]
    rows = []
    for name, kind, k, draws in conds:
        for draw in range(draws):
            w = weights if kind is None else build(kind, k)
            for eseed in (int(s) for s in args.iseeds.split(",")):
                table = episode_metrics(evaluate(w, eseed))
                rows.append({"cond": name, "draw": draw, "eseed": eseed, **table[METRICS].mean().to_dict()})
    outcome = pd.DataFrame(rows).groupby("cond")[METRICS].mean().reindex([c[0] for c in conds])
    outcome.to_csv(f"{args.out}_outcome.csv")
    result["outcome"] = {c: {m: round(float(outcome.loc[c, m]), 4) for m in METRICS} for c in outcome.index}
    json.dump(result, open(f"{args.out}_skill.json", "w"), indent=1)
    pd.set_option("display.width", 200)
    print(f"used nodes {result['used_nodes']}, segments {result['segments']}, common share {result['common_share_used']:.3f}")
    print("PC variance share", np.round(share, 3))
    print(pd.DataFrame({k: v for k, v in result["regression"].items()}, index=[f"PC{i + 1}" for i in range(K)]).T.round(3).to_string())
    print(outcome.round(3).to_string())


if __name__ == "__main__":
    main()

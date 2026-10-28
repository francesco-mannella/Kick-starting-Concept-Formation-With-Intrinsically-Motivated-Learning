#!/usr/bin/env python
import argparse
import contextlib
import io
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import adjusted_mutual_info_score, mutual_info_score
from sklearn.model_selection import GroupKFold, cross_val_score

SRC = Path(__file__).resolve().parent
sys.path.insert(0, str(SRC))

from SMMain import Main  # noqa: E402  (pickled dumps look up Main in __main__)

_ = Main
GRID = 10


def parse_epochs(spec):
    epochs = []
    for part in spec.split(","):
        if ":" in part:
            start, stop, step = (int(x) for x in part.split(":"))
            epochs += list(range(start, stop, step))
        else:
            epochs.append(int(part))
    return epochs


def load(storage, epoch, data_name):
    folder = Path(storage) / f"{epoch:06d}"
    main = np.load(folder / "main.dump.npy", allow_pickle=True)[0]
    main.params.use_wandb = False
    return main, np.load(folder / data_name, allow_pickle=True)[0]


def segments(data, episodes):
    actions = data["a"]
    rows, policies = [], []
    for ep in range(actions.shape[0]):
        last, k = None, 0
        for t in range(actions.shape[1]):
            v = actions[ep, t]
            if not v.any() or (last is not None and np.allclose(v, last)):
                continue
            policies.append(v.copy())
            rows.append((ep, episodes.context[ep], episodes.stretch[ep], round(episodes.rotation[ep], 2), k,
                         int(data["v_r"][ep, t].argmax())))
            last, k = v, k + 1
    return np.array(policies), pd.DataFrame(rows, columns=["ep", "ctx", "stretch", "rot", "k", "node"])


def postures(data, episodes, epoch):
    active = data["a"].any(-1)
    touch = np.linalg.norm(data["ss"], axis=-1) > 1e-6
    rows = []
    for ep in range(active.shape[0]):
        idx = np.where(active[ep])[0]
        if len(idx) < 5:
            continue
        contact = idx[touch[ep, idx]]
        rows.append((f"{epoch}_{ep}", episodes.context[ep], data["p"][ep, idx].mean(0),
                     data["p"][ep, contact].mean(0) if len(contact) else None))
    return rows


def eta2(policies, labels):
    mu = policies.mean(0)
    total = ((policies - mu) ** 2).sum()
    between = sum((labels == lab).sum() * ((policies[labels == lab].mean(0) - mu) ** 2).sum()
                  for lab in np.unique(labels))
    return between / total


def participation(x):
    ev = np.linalg.eigvalsh(np.cov(x.T)).clip(0)
    return ev.sum() ** 2 / (ev**2).sum()


def permute_units(units, label, rng):
    # permute labels over whole (epoch, episode) units so repeated policies of one episode stay together
    first = pd.DataFrame({"unit": units, "label": label}).groupby("unit").label.first()
    mapping = dict(zip(first.index, rng.permutation(first.values)))
    return pd.Series(units).map(mapping).values


def unit_null(policies, units, label, rng, n_perm):
    values = [eta2(policies, permute_units(units, label, rng)) for _ in range(n_perm)]
    return float(np.mean(values)), float(np.percentile(values, 95))


def residual_node(policies, nodes):
    out = policies.copy()
    for n in np.unique(nodes):
        out[nodes == n] -= policies[nodes == n].mean(0)
    return out


def mi_time_weighted(data, main, policies, rng, n_shuffle):
    params = main.params
    actions = data["a"]
    mask = np.any(actions != 0, axis=2)
    mask[:, : params.drop_first_n_steps + params.policy_selection_steps] = False
    index = KMeans(10, n_init=5, random_state=0).fit(policies).predict(actions[mask])
    contexts = main.episode_dataset.context.values

    def mi(ctx, score=mutual_info_score):
        return score(np.repeat(ctx[:, None], actions.shape[1], 1)[mask], index)

    return (mi(contexts), float(np.mean([mi(rng.permutation(contexts)) for _ in range(n_shuffle)])),
            mi(contexts, adjusted_mutual_info_score))


def static_stats(args, rng):
    pooled, labels, protos, goal_protos = [], [], None, []
    mi_raw, mi_null, mi_adj = [], [], []
    posture_rows = []
    for epoch in parse_epochs(args.epochs):
        main, data = load(args.storage, epoch, args.data)
        policies, label = segments(data, main.episode_dataset)
        if len(policies) < 12:
            continue
        label["epoch"] = epoch
        posture_rows += postures(data, main.episode_dataset, epoch)
        pooled.append(policies)
        labels.append(label)
        protos = getattr(main, args.controller).stm_a.get_weights().T
        goal_protos.append(protos[label.node.values])
        raw, null, adjusted = mi_time_weighted(data, main, policies, rng, args.n_perm_mi)
        mi_raw.append(raw)
        mi_null.append(null)
        mi_adj.append(adjusted)
    policies, label = np.vstack(pooled), pd.concat(labels, ignore_index=True)
    units = (label.epoch.astype(str) + "_" + label.ep.astype(str)).values
    first = (label.k == 0).values
    out = {"checkpoints": len(pooled), "segments": len(policies), "episodes": len(np.unique(units))}

    for c in (1, 2, 3):
        per_ep = label[label.ctx == c].groupby(["epoch", "ep"]).size()
        out[f"policies_per_episode_ctx{c}"] = float(per_ep.mean())

    dist = np.sqrt(((policies[:, None] - protos[None]) ** 2).sum(-1))
    out["policy_norm"] = float(np.linalg.norm(policies, axis=1).mean())
    out["prototype_norm"] = float(np.linalg.norm(protos, axis=1).mean())
    out["dist_to_prototype_over_norm"] = float(dist.min(1).mean() / out["prototype_norm"])
    out["nodes_used"] = int(len(np.unique(dist.argmin(1))))
    out["participation_policies"] = float(participation(policies))
    out["participation_prototypes"] = float(participation(protos))
    unit = protos / np.linalg.norm(protos, axis=1, keepdims=True)
    cos = unit @ unit.T
    g = np.arange(GRID * GRID)
    grid_d = np.hypot((g // GRID)[:, None] - (g // GRID)[None], (g % GRID)[:, None] - (g % GRID)[None])
    iu = np.triu_indices(len(g), 1)
    out["cos_prototypes_neighbours"] = float(cos[iu][grid_d[iu] == 1].mean())
    out["cos_prototypes_far"] = float(cos[iu][grid_d[iu] > 6].mean())

    condition = (label.ctx.astype(str) + "_" + label.stretch.astype(str) + "_" + label.rot.astype(str)).values
    for name, lab in (("context", label.ctx.values), ("stretch", label.stretch.values),
                      ("rotation", label.rot.values), ("condition", condition)):
        for subset, sel in (("all", np.ones(len(label), bool)), ("first", first)):
            p, u, l = policies[sel], units[sel], lab[sel]
            observed = eta2(p, l)
            null_mean, null_p95 = unit_null(p, u, l, rng, args.n_perm)
            out[f"eta2_{name}_{subset}"] = float(observed)
            out[f"eta2_{name}_{subset}_null"] = null_mean
            out[f"eta2_{name}_{subset}_p95"] = null_p95

    fp, fc = policies[first], label.ctx.values[first]
    unit_fp = fp / np.linalg.norm(fp, axis=1, keepdims=True)
    cfp = unit_fp @ unit_fp.T
    iu = np.triu_indices(len(fp), 1)
    same = (fc[:, None] == fc[None])[iu]
    out["cos_first_same_context"] = float(cfp[iu][same].mean())
    out["cos_first_diff_context"] = float(cfp[iu][~same].mean())

    out["mi_context_policy"] = float(np.mean(mi_raw))
    out["mi_context_policy_shuffled"] = float(np.mean(mi_null))
    out["mi_context_policy_excess"] = out["mi_context_policy"] - out["mi_context_policy_shuffled"]
    out["ami_context_policy"] = float(np.mean(mi_adj))

    nodes, contexts = label.node.values, label.ctx.values
    goal_protos = np.vstack(goal_protos)
    out["mi_context_goal_node"] = float(mutual_info_score(contexts, nodes))
    out["mi_context_goal_node_shuffled"] = float(np.mean(
        [mutual_info_score(permute_units(units, contexts, rng), nodes) for _ in range(args.n_perm_mi)]))
    out["goal_nodes_used"] = int(len(np.unique(nodes)))
    out["eta2_goal_node"] = float(eta2(policies, nodes))
    out["eta2_goal_node_null"] = float(np.mean([eta2(policies, rng.permutation(nodes)) for _ in range(args.n_perm)]))
    residual = residual_node(policies, nodes)
    out["eta2_context_given_node"] = float(eta2(residual, contexts))
    out["eta2_context_given_node_null"], out["eta2_context_given_node_p95"] = unit_null(
        residual, units, contexts, rng, args.n_perm)
    total = ((policies - policies.mean(0)) ** 2).sum()
    out["noise_share"] = float(((policies - goal_protos) ** 2).sum() / total)
    out["eta2_context_goal_prototype"] = float(eta2(goal_protos, contexts))
    out["eta2_context_goal_prototype_null"], out["eta2_context_goal_prototype_p95"] = unit_null(
        goal_protos, units, contexts, rng, args.n_perm)

    for name, rows in (("posture", posture_rows), ("posture_contact", [r for r in posture_rows if r[3] is not None])):
        ids = np.array([r[0] for r in rows])
        ctx = np.array([r[1] for r in rows])
        x = np.array([r[2] if name == "posture" else r[3] for r in rows])
        out[f"eta2_{name}"] = float(eta2(x, ctx))
        out[f"eta2_{name}_null"], out[f"eta2_{name}_p95"] = unit_null(x, ids, ctx, rng, args.n_perm)
        out[f"{name}_episodes"] = len(rows)

    epochs_first = label.epoch.values[first]
    if len(np.unique(epochs_first)) >= 5:
        clf = LogisticRegression(max_iter=2000, C=0.1)
        cv = GroupKFold(n_splits=5)
        out["classifier_accuracy"] = float(cross_val_score(clf, fp, fc, groups=epochs_first, cv=cv).mean())
        out["classifier_accuracy_shuffled"] = float(np.mean([
            cross_val_score(clf, fp, rng.permutation(fc), groups=epochs_first, cv=cv).mean()
            for _ in range(args.n_perm_clf)]))
    return out


def episode_table(tr, episodes):
    touch_cols = [f"s{i}" for i in range(40)]
    rows = []
    for ep, g in tr.groupby("episode_id"):
        touch = g[touch_cols].values.sum(1) > 0
        goal_x, goal_y = g.prototype_x.values, g.prototype_y.values
        dist = np.hypot(g.touch_x.values - goal_x, g.touch_y.values - goal_y)
        rows.append({"ep": int(ep), "ctx": int(episodes.context[int(ep) % len(episodes)]),
                     "success": float(g.tr_id.nunique() >= 2), "match": g.match.mean(),
                     "max_match": g.match.max(), "touch": touch.mean(),
                     "hit2": (dist[touch] <= 2).mean() if touch.any() else 0.0})
    return pd.DataFrame(rows)


def node_groups(tr, episodes, n_nodes, min_segments):
    seg = tr.drop_duplicates(["episode_id", "tr_id"])
    node = (np.round(seg.prototype_x.values) * GRID + np.round(seg.prototype_y.values)).astype(int)
    ctx = episodes.context.values[seg.episode_id.astype(int).values % len(episodes)]
    count = np.zeros((n_nodes, 3))
    for n, c in zip(node, ctx):
        count[n, int(c) - 1] += 1
    group = np.where(count.sum(1) >= min_segments, count.argmax(1), -1)
    return group, count


def counterfactual(args, rng):
    main, _ = load(args.storage, parse_epochs(args.epochs)[-1], args.data)
    controller = getattr(main, args.controller)
    weights = controller.stm_a.get_weights().copy()
    protos = weights.T
    episodes = main.episode_dataset

    def evaluate(w, eseed):
        controller.stm_a.set_weights(w)
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            _, tr = main.evaluation_episodes(orig_controller=controller, epoch=0, n_episodes=args.n_episodes,
                                             render=None, save_stats=False, e_seed=eseed)
        controller.stm_a.set_weights(weights)
        return tr, float(re.search(r"eval success rate ([\d.]+)", buf.getvalue()).group(1))

    eseeds = [int(s) for s in args.eseeds.split(",")]
    tr0, _ = evaluate(weights, eseeds[0])
    group, count = node_groups(tr0, episodes, len(protos), args.min_segments)
    sizes = {c + 1: int((group == c).sum()) for c in range(3)}
    print(f"nodes by dominant context: {sizes}, unassigned {int((group < 0).sum())}")
    used = np.where(group >= 0)[0]

    def build(kind):
        p = protos.copy()
        if kind == "swap":
            for c in range(3):
                dst, src = used[group[used] == c], used[group[used] == (c + 1) % 3]
                if len(dst) and len(src):
                    p[dst] = protos[rng.choice(src, len(dst), replace=len(src) < len(dst))]
        elif kind == "shuffle_within_context":
            for c in range(3):
                idx = used[group[used] == c]
                p[idx] = protos[rng.permutation(idx)]
        elif kind == "shuffle_all_used":
            p[used] = protos[rng.permutation(used)]
        elif kind == "constant_mean":
            p[used] = protos[used].mean(0)
        return p.T

    rows = []
    for kind in ("intact", "swap", "shuffle_within_context", "shuffle_all_used", "constant_mean"):
        for draw in range(1 if kind in ("intact", "constant_mean") else args.draws):
            w = build(kind) if kind != "intact" else weights
            for eseed in eseeds:
                tr, success = evaluate(w, eseed)
                table = episode_table(tr, episodes)
                table["cond"], table["eseed"], table["draw"] = kind, eseed, draw
                rows.append(table)
    table = pd.concat(rows, ignore_index=True)
    return table, sizes


def parse_arguments():
    parser = argparse.ArgumentParser(description="Statistics of the policies of a finished run")
    parser.add_argument("--storage", required=True, help="Run storage folder holding the epoch folders")
    parser.add_argument("--epochs", required=True, help="Comma list or start:stop:step, e.g. 500:1000:50,999")
    parser.add_argument("--data", default="data.npy", help="data.npy or data_par.npy")
    parser.add_argument("--controller", default="controller", help="controller or controller_par")
    parser.add_argument("--out", default=None, help="Prefix of the output files")
    parser.add_argument("--counterfactual", action="store_true")
    parser.add_argument("--eseeds", default="101,102,103")
    parser.add_argument("--n-episodes", type=int, default=36)
    parser.add_argument("--draws", type=int, default=5)
    parser.add_argument("--min-segments", type=int, default=2)
    parser.add_argument("--n-perm", type=int, default=200)
    parser.add_argument("--n-perm-mi", type=int, default=50)
    parser.add_argument("--n-perm-clf", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    rng = np.random.default_rng(args.seed)
    stats = static_stats(args, rng)
    print(json.dumps(stats, indent=1))
    if args.out:
        Path(args.out + "_static.json").write_text(json.dumps(stats, indent=1))
    if args.counterfactual:
        table, sizes = counterfactual(args, rng)
        cols = ["success", "match", "max_match", "touch", "hit2"]
        overall = table.groupby("cond")[cols].mean().round(3)
        by_ctx = table.groupby(["cond", "ctx"])[cols].mean().round(3)
        print(overall.to_string())
        print(by_ctx.to_string())
        if args.out:
            table.to_csv(args.out + "_counterfactual_episodes.csv", index=False)
            by_ctx.to_csv(args.out + "_counterfactual_by_context.csv")

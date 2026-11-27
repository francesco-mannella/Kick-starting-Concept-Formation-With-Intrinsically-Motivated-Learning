import contextlib
import io

import numpy as np
import pandas as pd

GRID = 10
DIMS = [f"d{j}" for j in range(1, 6)]


def eta2(x, labels):
    mu = x.mean(0)
    total = ((x - mu) ** 2).sum()
    between = sum((labels == lab).sum() * ((x[labels == lab].mean(0) - mu) ** 2).sum() for lab in np.unique(labels))
    return between / total


def node_groups(tr, episodes, n_nodes, min_segments):
    seg = tr.drop_duplicates(["episode_id", "tr_id"])
    node = (np.round(seg.prototype_x.values) * GRID + np.round(seg.prototype_y.values)).astype(int)
    ctx = episodes.context.values[seg.episode_id.astype(int).values % len(episodes)]
    count = np.zeros((n_nodes, 3))
    for n, c in zip(node, ctx):
        count[n, int(c) - 1] += 1
    group = np.where(count.sum(1) >= min_segments, count.argmax(1), -1)
    return group, count


def node_postures(main, controller, steps, nodes=None):
    protos = getattr(main, controller).stm_a.get_weights().T if isinstance(controller, str) else controller.stm_a.get_weights().T
    nodes = np.arange(len(protos)) if nodes is None else np.asarray(nodes)
    out = np.zeros((len(nodes), len(steps), 7))
    for i, n in enumerate(nodes):
        env, state = main.episode_runner.prepare_environment(
            episode_index=0, seed=0, obj_params=main.get_obj_params_for_episode(0), context=4, render=None, plot_path=None)
        main.agent.reset()
        main.agent.updatePolicy(protos[n])
        sim, names = env.b2d_env.sim, env.b2d_env.joint_names
        for t in range(1, max(steps) + 1):
            if (t - 1) % main.params.action_steps == 0:
                action = main.agent.step(state)
            state = env.step(action)
            if t in steps:
                out[i, steps.index(t)] = [sim.joints[j].angle for j in names]
    return out


def nearest_context(row, pool):
    d = {c: np.linalg.norm(row[DIMS].values.astype(float) - pool[pool.ctx == c][DIMS].values.mean(0))
         for c in (1, 2, 3) if (pool.ctx == c).any()}
    return min(d, key=d.get)


def early_eta2(x, ctx, rot, rng, n_perm):
    rot = np.round(rot * 100).astype(int)

    def three(c, r):
        e = {"ctx": eta2(x, c), "rot": eta2(x, r), "cell": eta2(x, c * 1000 + r)}
        e["inter"] = e["cell"] - e["ctx"] - e["rot"]
        return e

    obs = three(ctx, rot)
    null = [three(ctx[q], rot[q]) for q in (rng.permutation(len(ctx)) for _ in range(n_perm))]
    return {k: (float(obs[k]), float(np.mean([n[k] for n in null])), float(np.percentile([n[k] for n in null], 95)))
            for k in obs}


def early_rows(tr, episodes, onset, steps):
    touch_cols = [f"s{i}" for i in range(40)]
    rows = []
    for ep, g in tr.groupby("episode_id"):
        touched = np.cumsum(g[touch_cols].values.sum(1) > 0) > 0
        node = int(np.round(g.prototype_x.iloc[0]) * GRID + np.round(g.prototype_y.iloc[0]))
        row = int(ep) % len(episodes)
        for k in steps:
            if onset + k not in g.index:
                continue
            rows.append({"ep": int(ep), "ctx": int(episodes.context[row]), "rot": float(episodes.rotation[row]), "step": k,
                         "node": node, "free": bool(not touched[g.index.get_loc(onset + k)]),
                         **{f"d{j}": float(g.loc[onset + k, f"d{j}"]) for j in range(1, 6)}})
    return pd.DataFrame(rows)


def paired_early(early):
    key = ["eseed", "ep", "step"]
    base = early[early.cond == "intact"].drop_duplicates(key).set_index(key)
    summary, per_ep = [], []
    for step, b in base.reset_index().groupby("step"):
        ok = b[b.free]
        scale = float(np.sqrt(((ok[DIMS].values - ok[DIMS].values.mean(0)) ** 2).sum(1).mean()))
        for cond, e in early[early.cond != "intact"].groupby("cond"):
            m = e.merge(base.reset_index(), on=key, suffixes=("", "_i"))
            m = m[m.step == step]
            same, free = m.node == m.node_i, m.free & m.free_i
            v = m[same & free].copy()
            v["dist"] = np.linalg.norm(v[DIMS].values - v[[d + "_i" for d in DIMS]].values, axis=1)
            v["nearest_ctx"] = [nearest_context(r, ok[ok.ep != r.ep]) for _, r in v.iterrows()]
            v["own"] = v.nearest_ctx == v.ctx
            per_ep.append(v.assign(step=step)[["cond", "step", "eseed", "ep", "draw", "ctx", "rot", "dist", "nearest_ctx", "own"]])
            summary.append({"cond": cond, "step": step, "rows": len(m), "same_node": float(same.mean()), "free_both": float(free.mean()),
                            "n": len(v), "dist": float(v.dist.mean()), "dist_over_spread": float(v.dist.mean() / scale),
                            "nearest_own": float(v.own.mean()), **{f"nearest_own_ctx{c}": float(v[v.ctx == c].own.mean()) for c in (1, 2, 3)}})
        intact = [nearest_context(r, ok[ok.ep != r.ep]) == r.ctx for _, r in ok.iterrows()]
        summary.append({"cond": "intact", "step": step, "rows": len(b), "same_node": 1.0, "free_both": float(b.free.mean()),
                        "n": len(ok), "dist": 0.0, "dist_over_spread": 0.0, "nearest_own": float(np.mean(intact))})
    return pd.DataFrame(summary).sort_values(["step", "cond"]), pd.concat(per_ep, ignore_index=True)


def build_weights(kind, protos, group, rng):
    used = np.where(group >= 0)[0]
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


def evaluate_matched(main, controller, weights, n_episodes, rng_state):
    original = controller.stm_a.get_weights().copy()
    controller.rng.set_state(rng_state)
    controller.stm_a.set_weights(weights)
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            _, tr = main.evaluation_episodes(orig_controller=controller, epoch=main.epoch, n_episodes=n_episodes,
                                             render=None, save_stats=False)
    finally:
        controller.stm_a.set_weights(original)
    return tr


COND_KEYS = {"intact": "intact", "swap": "swap", "shuffle_within_context": "within", "constant_mean": "constant"}


def early_measures(main, controller, steps, n_episodes, seed=0, n_perm=50):
    """Early-posture measures of the agent's policy map; evaluation-only, training state is restored."""
    episodes = main.episode_dataset
    onset = main.params.drop_first_n_steps + main.params.policy_selection_steps
    weights = controller.stm_a.get_weights().copy()
    protos = weights.T
    rng = np.random.default_rng(seed)
    state = controller.rng.get_state()
    try:
        runs = {"intact": evaluate_matched(main, controller, weights, n_episodes, state)}
        group, _ = node_groups(runs["intact"], episodes, len(protos), 2)
        for kind in ("swap", "shuffle_within_context", "constant_mean"):
            runs[kind] = evaluate_matched(main, controller, build_weights(kind, protos, group, rng), n_episodes, state)
    finally:
        controller.rng.set_state(state)

    frames = []
    for kind, tr in runs.items():
        rows = early_rows(tr, episodes, onset, steps)
        rows["cond"], rows["eseed"], rows["draw"] = kind, 0, 0
        frames.append(rows)
    early = pd.concat(frames, ignore_index=True)
    summary, _ = paired_early(early)

    out = {}
    for _, r in summary.iterrows():
        k, name = int(r.step), COND_KEYS[r.cond]
        out[f"early_nearest_own_{name}_s{k}"] = r.nearest_own
        if name != "intact":
            out[f"early_dist_{name}_s{k}"] = r.dist_over_spread
        else:
            out[f"early_n_s{k}"] = r.n
            out[f"early_free_s{k}"] = r.free_both
    for k in steps:
        out[f"early_swap_cost_s{k}"] = out[f"early_nearest_own_within_s{k}"] - out[f"early_nearest_own_swap_s{k}"]

    first = early[early.cond == "intact"]
    for k in steps:
        rows = first[first.step == k]
        if len(rows) < 6:
            continue
        nodes = rows.node.values
        uniq = np.unique(nodes)
        posture = node_postures(main, controller, steps, uniq)[:, steps.index(k)]
        x = posture[np.searchsorted(uniq, nodes)]
        for name, (obs, null, _) in early_eta2(x, rows.ctx.values.astype(int), rows.rot.values, rng, n_perm).items():
            if name in ("ctx", "rot", "inter"):
                out[f"early_eta2_{name}_s{k}"] = obs
                out[f"early_eta2_{name}_null_s{k}"] = null
                out[f"early_eta2_{name}_excess_s{k}"] = obs - null
    return out

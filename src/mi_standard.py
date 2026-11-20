import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import mutual_info_score

PAIRS = (("context", "policy"), ("context", "touch"), ("policy", "touch"), ("context", "gripper"), ("policy", "gripper"))
MIN_LAYOUTS, MIN_SEGMENTS = 3, 20


def segment_variables(ep, ended, ctx, policy_ind, ss, joints, seed=0):
    new = np.r_[True, ended[:-1] | (ep[1:] != ep[:-1])]
    first = np.flatnonzero(new)
    touch_groups = np.add.reduceat(ss.reshape(len(ss), 4, -1).sum(-1), first, axis=0)
    touch = touch_groups.argmax(1) + 1
    touch[touch_groups.sum(1) == 0] = 0
    mean_joints = np.add.reduceat(joints, first, axis=0) / np.diff(np.r_[first, len(ep)])[:, None]
    gripper = KMeans(min(10, len(first)), random_state=seed).fit_predict(mean_joints)
    return first, dict(context=ctx[first], policy=policy_ind[first], touch=touch, gripper=gripper)


def layout_draws(seg_ep, seg_layout, n_perm, rng):
    eps = np.unique(seg_ep)
    members = {e: np.flatnonzero(seg_ep == e) for e in eps}
    ep_layout = {e: seg_layout[members[e][0]] for e in eps}
    layouts = np.unique(seg_layout)
    of_layout = {lay: [e for e in eps if ep_layout[e] == lay] for lay in layouts}
    rank = {e: of_layout[ep_layout[e]].index(e) for e in eps}
    draws = []
    for _ in range(n_perm):
        source = dict(zip(layouts, rng.permutation(layouts)))
        idx = np.empty(len(seg_ep), int)
        for e in eps:
            reps = of_layout[source[ep_layout[e]]]
            src = members[reps[rank[e] % len(reps)]]
            idx[members[e]] = src[np.arange(len(members[e])) % len(src)]
        draws.append(idx)
    return draws


def layout_excess(ep, ended, ctx, layout, policy_ind, ss, joints, n_perm=100, seed=0):
    # the evaluation repeats each layout (object parameters x context) many times, so the layout, not the episode, is the independent unit
    keys = {}
    for a, b in PAIRS:
        keys.update({f"mi2_{a}_{b}": np.nan, f"mi2_{a}_{b}_raw": np.nan, f"mi2_{a}_{b}_null": np.nan})
    first, v = segment_variables(ep, ended, ctx, policy_ind, ss, joints, seed)
    keys["mi2_n_segments"], keys["mi2_n_layouts"] = float(len(first)), float(len(np.unique(layout[first])))
    if len(first) < MIN_SEGMENTS or keys["mi2_n_layouts"] < MIN_LAYOUTS:
        return keys
    draws = layout_draws(ep[first], layout[first], n_perm, np.random.default_rng(seed))
    for a, b in PAIRS:
        x, y = v[a], v[b]
        raw = mutual_info_score(x, y)
        null = np.mean([mutual_info_score(x, y[idx]) for idx in draws])
        keys[f"mi2_{a}_{b}"], keys[f"mi2_{a}_{b}_raw"], keys[f"mi2_{a}_{b}_null"] = float(raw - null), float(raw), float(null)
    return keys

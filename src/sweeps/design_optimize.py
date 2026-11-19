import itertools
import json
import pickle
import sys
from pathlib import Path

import numpy as np

out = Path(sys.argv[1])
DELTA = 0.25
GROUPS = [range(0, 5), range(5, 10), range(10, 15), range(15, 20), range(20, 25), range(25, 30), range(30, 35), range(35, 40)]
res = pickle.load(open(out / "design_sweep.pkl", "rb"))
metas, masks, contact, joints = [], [], [], []
for meta, d in res:
    metas.append(meta)
    frac = d["frac"].astype(float)
    m = np.zeros(len(frac), dtype=int)
    for g, idx in enumerate(GROUPS):
        m |= ((frac[:, list(idx)] >= 0.05).any(1)).astype(int) << g
    masks.append(m)
    contact.append(d["contact"] & (m > 0))
    joints.append(d["first_joints"][:, :3])
masks, contact, joints = np.array(masks), np.array(contact), np.array(joints)
C = len(metas)
rate = contact.mean(1)


def transfer(use_proprio):
    T = np.zeros((C, C))
    for a in range(C):
        if not contact[a].sum():
            continue
        same = (masks == masks[a]) & contact & contact[a][None, :]
        if use_proprio:
            same &= (np.linalg.norm(joints - joints[a][None], axis=2) <= DELTA)
        T[a] = same.sum(1) / contact[a].sum()
    S = (T + T.T) / 2
    np.fill_diagonal(S, 1.0)
    return S


S_goal, S_touch = transfer(True), transfer(False)
lib = [i for i, m in enumerate(metas) if m["mode"] == "lib"]
base = [i for i, m in enumerate(metas) if m["mode"] == "base"]
print(f"{len(lib)} library cells + {len(base)} baseline cells; contact rate range {rate.min():.2f}-{rate.max():.2f}; patterns seen {len(set(masks[contact]))}")


def label(i):
    m = metas[i]
    pos = "default place" if m["pos"] is None else f"({m['pos'][0]:.1f},{m['pos'][1]:.1f})"
    return f"{m['shape']:12s} stretch {m['stretch']} rot {m['rot']:2d} at {pos}"


def stats(ids, S):
    pairs = np.array([S[a, b] for a, b in itertools.combinations(ids, 2)])
    shared = []
    q = np.round(joints[ids] / 0.5).astype(int)
    for p in range(contact.shape[1]):
        act = [k for k, i in enumerate(ids) if contact[i][p]]
        if not act:
            continue
        keys = [(masks[ids[k]][p], tuple(q[k][p])) for k in act]
        shared.append(max(keys.count(x) for x in set(keys)))
    return dict(mean=float(pairs.mean()), max=float(pairs.max()), p90=float(np.quantile(pairs, 0.9)), distinct=float((pairs <= 0.05).mean()),
                best_posture_shares_goal_in=int(max(shared)), mean_share=float(np.mean(shared)), min_rate=float(rate[ids].min()))


def show(name, ids):
    g, t = stats(ids, S_goal), stats(ids, S_touch)
    print(f"  {name}")
    print(f"    goal transfer (touch pattern + arm posture): mean {g['mean']:.3f}  90th pct {g['p90']:.2f}  max {g['max']:.2f}  pairs <= 0.05: {g['distinct']:.2f}  best single posture gives the same goal in {g['best_posture_shares_goal_in']}/{len(ids)} contexts  (mean over contacted postures {g['mean_share']:.1f})")
    print(f"    touch-only transfer: mean {t['mean']:.3f}  max {t['max']:.2f}; min contact rate over the contexts {g['min_rate']:.2f}")


print("\nBASELINE: the current task layout (3 original kinds x stretch 1/2 x rotation 0/45/90, champion physics)")
show("current 18 contexts", base)

by = {}
for i in lib:
    m = metas[i]
    by.setdefault((m["shape"], m["pos"]), []).append(i)
groups = [g for g, ids in by.items() if len(ids) == 6 and np.mean(rate[ids]) >= 0.08 and rate[ids].min() >= 0.04]
print(f"\nSTRUCTURED design: one (shape, position) per kind, variants = stretch x rotation; {len(groups)} candidate (shape, position) groups usable")
G = len(groups)
W = np.zeros(G)
M = np.zeros((G, G))
Wmax = np.zeros(G)
Mmax = np.zeros((G, G))
for x, g in enumerate(groups):
    ids = by[g]
    pr = [S_goal[a, b] for a, b in itertools.combinations(ids, 2)]
    W[x], Wmax[x] = np.sum(pr), np.max(pr)
    for y, h in enumerate(groups):
        blk = S_goal[np.ix_(ids, by[h])]
        M[x, y], Mmax[x, y] = blk.sum(), blk.max()
scores = []
for a, b, c in itertools.combinations(range(G), 3):
    tot = W[a] + W[b] + W[c] + M[a, b] + M[a, c] + M[b, c]
    mx = max(Wmax[a], Wmax[b], Wmax[c], Mmax[a, b], Mmax[a, c], Mmax[b, c])
    scores.append((tot / 153.0, mx, (a, b, c)))
scores.sort()
arr = np.array([s[0] for s in scores])
print(f"  {len(scores)} designs evaluated; mean pairwise goal transfer: best {arr.min():.3f}, 5th pct {np.quantile(arr, .05):.3f}, median {np.median(arr):.3f}, worst {arr.max():.3f}")
best = []
for mean, mx, trio in scores[:5]:
    ids = [i for t in trio for i in by[groups[t]]]
    names = "; ".join(f"{groups[t][0]} at ({groups[t][1][0]:.1f},{groups[t][1][1]:.1f})" for t in trio)
    show(f"design: {names}", ids)
    best.append(dict(kinds=[dict(shape=groups[t][0], position=list(groups[t][1])) for t in trio], mean_goal_transfer=mean, max_pair=mx))
top_ids = [i for t in scores[0][2] for i in by[groups[t]]]
print("\n  the 18 contexts of the best structured design:")
for t in scores[0][2]:
    g = groups[t]
    print(f"    kind: {g[0]} at ({g[1][0]:.1f},{g[1][1]:.1f}): contact rates by variant " + ", ".join(f"s{metas[i]['stretch']}r{metas[i]['rot']}:{rate[i]:.2f}" for i in by[g]))

print("\nFREE design: any 18 library cells (upper bound; needs a per-context position and shape, not only per kind)")
ok = [i for i in lib if rate[i] >= 0.08]
rng = np.random.default_rng(0)
bestset, bestval = None, 1e9
for restart in range(40):
    cur = list(rng.choice(ok, 18, replace=False))
    val = np.mean([S_goal[a, b] for a, b in itertools.combinations(cur, 2)])
    improved = True
    while improved:
        improved = False
        for k in range(18):
            for cand in ok:
                if cand in cur:
                    continue
                trial = cur.copy()
                trial[k] = cand
                v = np.mean([S_goal[a, b] for a, b in itertools.combinations(trial, 2)])
                if v < val - 1e-9:
                    cur, val, improved = trial, v, True
    if val < bestval:
        bestset, bestval = cur, val
show("best free design of 18 contexts", bestset)
for i in sorted(bestset, key=lambda i: str(metas[i]["pos"])):
    print("    ", label(i), f" contact {rate[i]:.2f}")
json.dump(dict(structured=best, free=[dict(**{k: metas[i][k] for k in ("shape", "stretch", "rot", "pos")}) for i in bestset],
               baseline_mean_goal_transfer=stats(base, S_goal)["mean"]), open(out / "design_proposals.json", "w"), indent=1, default=str)

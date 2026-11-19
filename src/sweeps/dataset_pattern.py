import itertools
import pickle
import sys
from pathlib import Path

import numpy as np

out = Path(sys.argv[1])
K_LIST = [int(x) for x in sys.argv[2].split(",")] if len(sys.argv) > 2 else [3, 6, 9, 18]
res = pickle.load(open(out / "claw_sweeps.pkl", "rb"))
GROUPS = [range(0, 5), range(5, 10), range(10, 15), range(15, 20), range(20, 25), range(25, 30), range(30, 35), range(35, 40)]
keys, masks, contact = [], [], []
for meta, d in res:
    keys.append((meta["shape"], meta["scale"], meta["density"], meta["pos"], meta["rot"]))
    frac = d["frac"].astype(float)
    m = np.zeros(len(frac), dtype=int)
    for g, idx in enumerate(GROUPS):
        m |= ((frac[:, list(idx)] >= 0.05).any(1)).astype(int) << g
    masks.append(m)
    contact.append(d["contact"] & (m > 0))
masks, contact = np.array(masks), np.array(contact)
rate = contact.mean(1)
use = np.where(rate >= 0.08)[0]
n = len(keys)
T = np.zeros((n, n))
for a in range(n):
    if contact[a].sum() == 0:
        continue
    same = (masks == masks[a]) & contact[a][None, :]
    T[a] = same.sum(1) / contact[a].sum()
S = (T + T.T) / 2
np.fill_diagonal(S, 1.0)
print(f"{out.name}: {n} cells; {len(use)} usable (contact rate >= 0.08); distinct 8-region patterns seen {len(set(masks[contact]))}")
vals = S[np.ix_(use, use)][np.triu_indices(len(use), 1)]
print(f"pattern transfer between usable cells: mean {vals.mean():.2f}, 10th percentile {np.quantile(vals, .1):.2f}, minimum {vals.min():.3f}")


def label(i):
    sh, sc, de, pos, rot = keys[i]
    return f"{sh}x{sc} d{de} rot{rot:2d} " + ("base(11.2,11.3)" if pos is None else f"({pos[0]:.1f},{pos[1]:.1f})")


def grow(tau, restarts=4000, seed=0, K=None):
    rng = np.random.default_rng(seed)
    best = []
    for _ in range(restarts):
        chosen = []
        for i in rng.permutation(use):
            if all(S[i, j] <= tau for j in chosen):
                chosen.append(i)
                if K and len(chosen) == K:
                    break
        if len(chosen) > len(best):
            best = chosen
            if K and len(best) == K:
                break
    return best


def evaluate(sel):
    sub = contact[sel]
    per = sub.sum(0)
    pairs = [S[a, b] for a, b in itertools.combinations(sel, 2)]
    # best single posture: number of contexts in which the posture gives the same pattern as in its most common context
    best_share = 0
    for p in np.where(per >= 1)[0]:
        pats = [masks[i][p] for i in sel if contact[i][p]]
        share = np.bincount(pats).max() / len(sel)
        best_share = max(best_share, share)
    return max(pairs), np.mean(pairs), best_share


for tau in (0.30, 0.20, 0.10, 0.05, 0.02):
    sel = grow(tau)
    mx, mean, share = evaluate(sel) if len(sel) > 1 else (np.nan, np.nan, np.nan)
    print(f"\n  largest set with every pairwise pattern transfer <= {tau}: K = {len(sel)}  (max pair {mx:.3f}, mean {mean:.3f}, best single posture gives the same pattern in {share*len(sel):.0f}/{len(sel)} contexts)")
    if len(sel) <= 20:
        for i in sel:
            print(f"      {label(i):44s} contact {rate[i]:.2f}")
print("\n  smallest achievable maximum pairwise transfer for a given dataset size K (greedy, 4000 restarts):")
for K in K_LIST:
    lo, hi = 0.0, 1.0
    for _ in range(14):
        mid = (lo + hi) / 2
        sel = grow(mid, restarts=1500, K=K)
        if len(sel) >= K:
            hi = mid
        else:
            lo = mid
    sel = grow(hi, restarts=3000, K=K)
    if len(sel) >= K:
        mx, mean, share = evaluate(sel)
        print(f"    K={K:2d}: max pair transfer {mx:.3f}, mean {mean:.3f}; best single posture gives the same pattern in {share*K:.1f}/{K} contexts")
        pickle.dump([keys[i] for i in sel], open(out / f"dataset_K{K}.pkl", "wb"))
        if K <= 9:
            for i in sel:
                print(f"        {label(i):44s} contact {rate[i]:.2f}")
    else:
        print(f"    K={K}: not reachable")

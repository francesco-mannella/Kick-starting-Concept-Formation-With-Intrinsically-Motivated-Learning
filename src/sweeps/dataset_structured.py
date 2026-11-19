import itertools
import pickle
import sys
from pathlib import Path

import numpy as np

GROUPS = [range(0, 5), range(5, 10), range(10, 15), range(15, 20), range(20, 25), range(25, 30), range(30, 35), range(35, 40)]


def load(out):
    res = pickle.load(open(Path(out) / "claw_sweeps.pkl", "rb"))
    keys, masks, contact = [], [], []
    for meta, d in res:
        keys.append((meta["mode"], meta["shape"], meta["scale"], meta["pos"], meta["rot"]))
        frac = d["frac"].astype(float)
        m = np.zeros(len(frac), dtype=int)
        for g, idx in enumerate(GROUPS):
            m |= ((frac[:, list(idx)] >= 0.05).any(1)).astype(int) << g
        masks.append(m)
        contact.append(d["contact"] & (m > 0))
    return keys, np.array(masks), np.array(contact)


def transfer_matrix(masks, contact):
    n = len(masks)
    T = np.zeros((n, n))
    for a in range(n):
        if contact[a].sum():
            T[a] = ((masks == masks[a]) & contact[a][None, :]).sum(1) / contact[a].sum()
    S = (T + T.T) / 2
    np.fill_diagonal(S, 1.0)
    return S


for out in sys.argv[1:]:
    keys, masks, contact = load(out)
    S = transfer_matrix(masks, contact)
    rate = contact.mean(1)
    idx = {k: i for i, k in enumerate(keys)}
    cellsB = sorted({(k[1], k[2], k[3]) for k in keys if k[0] == "B"}, key=str)
    good = [c for c in cellsB if all(("B", c[0], c[1], c[2], r) in idx and rate[idx[("B", c[0], c[1], c[2], r)]] >= 0.08 for r in (0, 90))]
    print(f"\n== {Path(out).name}: {len(good)} (shape, position) cells reachable in both rotations (contact rate >= 0.08)")
    best = []
    for trio in itertools.combinations(good, 3):
        if len({t[2] for t in trio}) < 3:
            continue
        ids = [idx[("B", t[0], t[1], t[2], r)] for t in trio for r in (0, 90)]
        pairs = [S[a, b] for a, b in itertools.combinations(ids, 2)]
        between = [S[a, b] for a, b in itertools.combinations(ids, 2) if (ids.index(a) // 2) != (ids.index(b) // 2)]
        within = [S[a, b] for a, b in itertools.combinations(ids, 2) if (ids.index(a) // 2) == (ids.index(b) // 2)]
        best.append((max(pairs), np.mean(pairs), max(between), max(within), trio, ids))
    best.sort(key=lambda x: (x[0], x[1]))
    print("best 6-context datasets (3 kinds x rotations 0/90), ranked by the worst pairwise pattern transfer:")
    for mx, mean, bt, wi, trio, ids in best[:4]:
        names = "; ".join(f"{t[0]}x{t[1]} at ({t[2][0]:.1f},{t[2][1]:.1f})" for t in trio)
        print(f"  worst pair {mx:.2f} (between kinds {bt:.2f}, same kind rot 0 vs 90 {wi:.2f}), mean {mean:.2f}: {names}")
    mx, mean, bt, wi, trio, ids = best[0]
    for t in trio:
        print(f"      {t[0]:6s} x{t[1]} at ({t[2][0]:5.1f},{t[2][1]:5.1f})  contact rate rot0 {rate[idx[('B', t[0], t[1], t[2], 0)]]:.2f} rot90 {rate[idx[('B', t[0], t[1], t[2], 90)]]:.2f}")
    pickle.dump(dict(trio=trio, ids=ids), open(Path(out) / "dataset_structured.pkl", "wb"))
    single = []
    for c in good:
        a, b = idx[("B", c[0], c[1], c[2], 0)], idx[("B", c[0], c[1], c[2], 90)]
        single.append((S[a, b], c))
    single.sort()
    print("kinds whose own rotations 0 and 90 separate best (lowest pattern transfer):", ", ".join(f"{c[0]}@({c[2][0]:.1f},{c[2][1]:.1f}) {v:.2f}" for v, c in single[:5]))

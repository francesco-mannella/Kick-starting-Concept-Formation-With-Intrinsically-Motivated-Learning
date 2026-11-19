import itertools
import pickle
import sys
from pathlib import Path

import numpy as np

out = Path(sys.argv[1])
res = pickle.load(open(out / "claw_sweeps.pkl", "rb"))
keys, C, R = [], [], []
for meta, d in res:
    keys.append((meta["shape"], meta["scale"], meta["density"], meta["pos"], meta["rot"], meta["mode"]))
    C.append(d["contact"])
    R.append(d["maxs"] >= 4)
C, R = np.array(C), np.array(R)
rate = C.mean(1)
use = np.where(rate >= 0.08)[0]
print(f"{out.name}: {len(keys)} cells, {len(use)} with claw-contact rate >= 0.08 (rate range {rate[use].min():.2f}-{rate[use].max():.2f})")
n = len(keys)
inter = (C[:, None, :] & C[None, :, :]).sum(2)
union = (C[:, None, :] | C[None, :, :]).sum(2)
J = np.where(union > 0, inter / np.maximum(union, 1), 1.0)
overlap_min = inter / np.maximum(np.minimum(C.sum(1)[:, None], C.sum(1)[None, :]), 1)


def label(i):
    sh, sc, de, pos, rot, mode = keys[i]
    return f"{sh}x{sc} d{de} rot{rot} " + ("at base (11.2,11.3)" if pos is None else f"at ({pos[0]:.1f},{pos[1]:.1f})")


def search(tau, matrix, restarts=3000, seed=0):
    rng = np.random.default_rng(seed)
    best = []
    adj = (matrix <= tau)
    np.fill_diagonal(adj, False)
    for _ in range(restarts):
        order = rng.permutation(use)
        chosen = []
        for i in order:
            if all(adj[i, j] for j in chosen):
                chosen.append(i)
        if len(chosen) > len(best):
            best = chosen
    return best


def describe(S):
    sub = C[S]
    per_posture = sub.sum(0)
    K = len(S)
    cover = per_posture.max() / K
    single = sub[:, per_posture.argmax()].sum()
    pairs = [J[a, b] for a, b in itertools.combinations(S, 2)]
    print(f"    K={K}: max pairwise overlap {max(pairs):.2f} mean {np.mean(pairs):.2f}; best single posture works in {single}/{K} contexts; postures that work in >= 2 contexts: {(per_posture >= 2).sum()} of {int((per_posture >= 1).sum())} useful")
    for i in S:
        print(f"      {label(i):52s} contact {rate[i]:.2f} rich {R[i].mean():.2f}")


for tau in (0.30, 0.20, 0.10, 0.05):
    S = search(tau, J)
    print(f"\n  largest set found with every pairwise contact-set overlap <= {tau}:")
    describe(S)

print("\n  same search restricted to ONE shape (stick 0.5x3, scale 1, density 0.2) over positions and rotations, overlap <= 0.2:")
sel = [i for i in use if keys[i][0] == "stick" and keys[i][1] == 1.0 and keys[i][2] in (0.2, 1.0)]
use_save = use
use = np.array(sel)
S = search(0.2, J)
describe(S)
use = use_save
print("\n  three kinds, one context each (3 contexts), the best triple by smallest maximum overlap:")
best = None
for trio in itertools.combinations(use, 3):
    m = max(J[a, b] for a, b in itertools.combinations(trio, 2))
    if best is None or m < best[0]:
        best = (m, trio)
describe(list(best[1]))
pickle.dump(dict(keys=keys, J=J), open(out / "dataset_J.pkl", "wb"))

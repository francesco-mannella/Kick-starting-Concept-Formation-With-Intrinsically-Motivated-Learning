import itertools
import pickle
import sys
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score

out = Path(sys.argv[1])
res = pickle.load(open(out / "claw_sweeps.pkl", "rb"))
cells = {}
for meta, d in res:
    key = (meta["mode"], meta["shape"], meta["scale"], meta["density"], meta["pos"], meta["rot"])
    level = np.where(d["contact"], np.where(d["maxs"] >= 4, 2, 1), 0)
    cells[key] = dict(d, level=level, rich=d["maxs"] >= 4)
ELONG = {"stick", "bar", "plate"}


def jac(a, b):
    u = (a | b).sum()
    return (a & b).sum() / u if u else np.nan


def compare(a, b):
    A, B = cells[a], cells[b]
    both = A["contact"] & B["contact"]
    cos = np.nan
    if both.sum() >= 5:
        fa, fb = A["frac"][both].astype(float), B["frac"][both].astype(float)
        cos = float(np.mean((fa * fb).sum(1) / (np.linalg.norm(fa, axis=1) * np.linalg.norm(fb, axis=1) + 1e-9)))
    either = (A["level"] > 0) | (B["level"] > 0)
    d3 = float((A["level"][either] != B["level"][either]).mean()) if either.any() else np.nan
    return jac(A["contact"], B["contact"]), jac(A["rich"], B["rich"]), d3, cos


def usable(k):
    return cells[k]["contact"].mean() > 0.05


def report(label, pairs):
    rows = np.array([compare(a, b) for a, b in pairs if usable(a) and usable(b)], dtype=float)
    if not len(rows):
        print(f"  {label:48s} no usable pairs")
        return
    m = np.nanmean(rows, axis=0)
    print(f"  {label:48s} n={len(rows):4d}  contact-set overlap {m[0]:.2f} (min {np.nanmin(rows[:, 0]):.2f})  rich-set overlap {m[1]:.2f}  outcome disagreement {m[2]:.2f}  touch-pattern cosine {m[3]:.2f}")


def accuracy(x, y):
    classes, counts = np.unique(y, return_counts=True)
    if len(classes) < 2 or counts.min() < 6:
        return np.nan
    cv = StratifiedKFold(min(5, counts.min()), shuffle=True, random_state=0)
    return float(cross_val_score(RandomForestClassifier(80, random_state=0, n_jobs=4), x, y, cv=cv).mean())


def feats(k):
    d = cells[k]
    return np.hstack([d["frac"][d["contact"]].astype(float), d["maxs"][d["contact"], None]])


def decode(groups, label, chance):
    accs = []
    for members in groups:
        ks = [k for k in members if usable(k)]
        if len(ks) < 2:
            continue
        x = np.vstack([feats(k) for k in ks])
        y = np.concatenate([[str(i)] * int(cells[k]["contact"].sum()) for i, k in enumerate(ks)])
        a = accuracy(x, y)
        if not np.isnan(a):
            accs.append(a)
    print(f"  decode {label:42s} {np.mean(accs):.2f} (chance about {chance}; {len(accs)} groups)" if accs else f"  decode {label}: no groups")


A = [k for k in cells if k[0] == "A"]
B = [k for k in cells if k[0] == "B"]
print(f"== {out.name}: claw-contact measures; A = {len(A)} cells (9 shapes x 3 sizes x 2 densities x 3 rotations, one place), B = {len(B)} cells (6 shapes x 9 positions x 2 rotations)")
print("overlap and disagreement are computed on IDENTICAL posture lists; 'rich' = at least 4 sensors active at some time\n")
for dens in (0.2, 1.0):
    print(f"A. one place, density {dens}")
    grp = lambda f: [(a, b) for a, b in itertools.combinations(A, 2) if a[3] == dens and b[3] == dens and f(a, b)]
    report("rotation 0 vs 90, same shape/size (all shapes)", grp(lambda a, b: a[1:4] == b[1:4] and {a[5], b[5]} == {0, 90}))
    report("rotation 0 vs 90, elongated (stick, bar, plate)", grp(lambda a, b: a[1:4] == b[1:4] and a[1] in ELONG and {a[5], b[5]} == {0, 90}))
    report("rotation 0 vs 90, compact (cubes, disks, hex, wedge)", grp(lambda a, b: a[1:4] == b[1:4] and a[1] not in ELONG and {a[5], b[5]} == {0, 90}))
    report("different shapes, same size and rotation", grp(lambda a, b: a[1] != b[1] and a[2] == b[2] and a[5] == b[5]))
    report("stick vs cube2, same size and rotation", grp(lambda a, b: {a[1], b[1]} == {"stick", "cube2"} and a[2] == b[2] and a[5] == b[5]))
    report("same shape, different size, same rotation", grp(lambda a, b: a[1] == b[1] and a[2] != b[2] and a[5] == b[5]))
print("\nB. positions (density 1)")
gB = lambda f: [(a, b) for a, b in itertools.combinations(B, 2) if f(a, b)]
report("same shape and rotation, different positions", gB(lambda a, b: a[1] == b[1] and a[5] == b[5] and a[4] != b[4]))
report("same position and rotation, different shapes", gB(lambda a, b: a[1] != b[1] and a[4] == b[4] and a[5] == b[5]))
report("same shape and position, rotation 0 vs 90", gB(lambda a, b: a[1] == b[1] and a[4] == b[4] and a[5] != b[5]))
print("\nDecoding from the time-integrated touch pattern (40 sensors + max sensors), contacted postures only, random forest, 5-fold")
decode([[k for k in A if k[1] == sh and k[2] == sc and k[3] == 0.2] for sh in SHAPES for sc in (0.6, 1.0, 1.6)] if False else
       [[k for k in A if k[1:4] == (sh, sc, 0.2)] for sh in sorted(set(k[1] for k in A)) for sc in (0.6, 1.0, 1.6)], "ROTATION (0/45/90), same shape and size", "0.33")
decode([[k for k in A if k[2] == sc and k[3] == 0.2 and k[5] == rot] for sc in (0.6, 1.0, 1.6) for rot in (0, 45, 90)], "SHAPE (9 shapes), same size and rotation", "0.11")
decode([[k for k in B if k[1] == sh and k[5] == rot] for sh, _ in [(s, 0) for s in sorted(set(k[1] for k in B))] for rot in (0, 90)], "POSITION (9), same shape and rotation", "0.11")
decode([[k for k in B if k[1] == sh and k[4] == pos] for sh in sorted(set(k[1] for k in B)) for pos in sorted(set(k[4] for k in B))], "ROTATION (0/90), same shape and position", "0.50")

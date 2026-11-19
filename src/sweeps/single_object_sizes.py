import itertools
import os

import numpy as np
import pandas as pd

H = os.path.expanduser("~/tmp/kss/")


def J(a, b):
    u = (a | b).sum()
    return (a & b).sum() / u if u else np.nan


def block(sub, eff, label):
    print(f"== {label}")
    rows = []
    for sh in sorted(sub["shape"].unique()):
        for d in sorted(sub.density.unique()):
            c = sub[(sub["shape"] == sh) & (sub.density == d)].sort_values("scale")
            E = {r.scale: eff(i) for i, r in c.iterrows()}
            rate = {k: v.mean() for k, v in E.items()}
            ok = [k for k in sorted(E) if rate[k] > 0.10]
            pairs = {(a, b): J(E[a], E[b]) for a, b in itertools.combinations(ok, 2)}
            rows.append((sh, d, rate, pairs))
    print("same shape, different size: effect rate per size (0.6 / 1 / 1.6) and overlap of effective sets between sizes")
    for sh, d, rate, pairs in rows:
        if d in (0.2, 1.0):
            text = ", ".join(f"{a}-{b}: {v:.2f}" for (a, b), v in pairs.items()) or "n/a (too heavy to move)"
            print(f"  {sh:6s} density {d}: effect " + " / ".join(f"{rate[k]:.2f}" for k in sorted(rate)) + "   overlap " + text)
    allsize = [v for *_, p in rows for v in p.values()]
    print(f"  all same-shape size pairs: n={len(allsize)} mean {np.mean(allsize):.2f} min {np.min(allsize):.2f}")
    sh_pairs = []
    for sc in (0.6, 1.0, 1.6):
        for d in (0.2, 1.0):
            c = sub[(sub.scale == sc) & (sub.density == d)]
            E = {r["shape"]: eff(i) for i, r in c.iterrows() if eff(i).mean() > 0.10}
            sh_pairs += [J(E[a], E[b]) for a, b in itertools.combinations(E, 2)]
    print(f"  different shape, same size and density: n={len(sh_pairs)} mean {np.mean(sh_pairs):.2f} min {np.min(sh_pairs):.2f}")


s = pd.read_csv(H + "l2_shapes/single.csv").set_index("id")
cl = np.load(H + "l2_shapes/classes.npz")
sub = s[(s.friction == 2.0) & (s.lin == 20.0) & (s.ang == 20.0)]
block(sub, lambda i: np.concatenate([cl[f"{i}_{r}"] >= 2 for r in (0, 90)]), "5-output hand")

d = np.load(H + "dof8_screens/shapes.npz", allow_pickle=True)
df = pd.DataFrame([dict(id=int(d["id"][j]), shape=str(d["shape"][j]), scale=float(d["scale"][j]), density=float(d["density"][j]),
                        rot=int(d["rot"][j]), eff=d["codes"][j] >= 2) for j in range(len(d["id"]))])
effs, meta = {}, {}
for (i, sh, sc, de), g in df.groupby(["id", "shape", "scale", "density"]):
    effs[i] = np.concatenate(list(g.sort_values("rot").eff))
    meta[i] = dict(shape=sh, scale=sc, density=de)
block(pd.DataFrame(meta).T.astype({"scale": float, "density": float}), lambda i: effs[i], "8-DoF hand (wrist + independent fingers)")

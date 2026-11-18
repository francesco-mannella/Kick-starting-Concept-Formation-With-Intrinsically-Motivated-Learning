#!/usr/bin/env python
import argparse
import itertools
from pathlib import Path

import numpy as np
import pandas as pd

import explore_shapes_l2 as E
import tune_l2_objects as T

QUANTILES = np.linspace(0.05, 0.95, 19)


def features(post):
    arm = post[:, :3] / np.pi
    return np.column_stack([arm, post[:, 3] / (np.pi / 2), post[:, 4] / np.pi, post[:, :3].sum(1) / (3 * np.pi)])


FEATURES = ["arm1", "arm2", "arm3", "claw_prox", "claw_dist", "orientation"]


def parse_arguments():
    parser = argparse.ArgumentParser(description="Which objects need different postures (offline, from explore_shapes_l2 output)")
    parser.add_argument("--dir", default=str(Path.home() / "tmp/kss/l2_shapes"))
    parser.add_argument("--n", type=int, default=150)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--min-effect", type=int, default=12)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    out = Path(args.dir)
    classes = np.load(out / "classes.npz")
    single = pd.read_csv(out / "single.csv").set_index("id")
    feats = {r: features(T.postures(args.seed + r, args.n)) for r in E.ROTS}
    eff = {}
    for i in single.index:
        for r in E.ROTS:
            cls = classes[f"{i}_{r}"]
            eff[(i, r)] = feats[r][(cls >= 3) | (cls == 2) | (cls == 5)]
    usable = [i for i in single.index if all(len(eff[(i, r)]) >= args.min_effect for r in E.ROTS) and single.loc[i, "fling"] <= 0.10]
    print(f"{len(usable)} candidates with at least {args.min_effect} effective postures per rotation and fling <= 0.10")
    q = {k: np.quantile(v, QUANTILES, axis=0) for k, v in eff.items() if k[0] in usable}
    rng = np.random.default_rng(0)
    floor = []
    for i in rng.choice(usable, 40, replace=False):
        for r in E.ROTS:
            v = eff[(i, r)]
            idx = rng.permutation(len(v))
            a, b = v[idx[: len(v) // 2]], v[idx[len(v) // 2:]]
            floor.append(np.abs(np.quantile(a, QUANTILES, axis=0) - np.quantile(b, QUANTILES, axis=0)).mean(0))
    floor = np.mean(floor, axis=0)
    print("noise floor (split halves of one candidate), per feature:", dict(zip(FEATURES, floor.round(3))))
    rows = []
    for a, b in itertools.combinations(usable, 2):
        d = np.mean([np.abs(q[(a, r)] - q[(b, r)]).mean(0) for r in E.ROTS], axis=0)
        rows.append((a, b, *d))
    table = pd.DataFrame(rows, columns=["a", "b", *FEATURES])
    table["claw"] = table[["claw_prox", "claw_dist"]].mean(1)
    table["arm"] = table[["arm1", "arm2", "arm3", "orientation"]].mean(1)
    table["excess_claw"] = table.claw - floor[3:5].mean()
    table["excess_arm"] = table.arm - floor[[0, 1, 2, 5]].mean()
    table.to_csv(out / "posture_pairs.csv", index=False)
    info = single[["shape", "scale", "density", "lin", "ang", "friction", "effect", "fling"]]
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 40)
    for key in ("claw", "arm"):
        top = table.sort_values(key, ascending=False).head(12)
        show = top[["a", "b", key, "claw_prox", "claw_dist", "orientation"]].copy()
        for side in "ab":
            for col in ("shape", "scale", "density", "lin", "ang"):
                show[f"{side}_{col}"] = show[side].map(info[col])
            show[f"{side}_eff"] = show[side].map(info["effect"])
        print(f"\ntop pairs by {key} posture difference")
        print(show.round(3).to_string(index=False))
    print("\ndistribution of claw and arm differences over all pairs: quantiles 50/90/99/max")
    for key in ("claw", "arm"):
        print(key, np.quantile(table[key], [0.5, 0.9, 0.99, 1.0]).round(3))

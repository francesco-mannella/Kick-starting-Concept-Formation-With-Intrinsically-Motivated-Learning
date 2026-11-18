#!/usr/bin/env python
import argparse
import itertools
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

import explore_shapes_l2 as E
import tune_l2_objects as T

BASE = (10.73279, 9.31968)
XS, YS = (4.0, 10.7, 14.0), (6.0, 9.3, 12.0)
POSITIONS = list(itertools.product(XS, YS))
SHAPE_SET = [("cube2", 1.0), ("bar", 1.0), ("plate", 0.6), ("disk3", 1.0), ("wedge", 1.0), ("hex4", 0.6)]
QUANTILES = np.linspace(0.05, 0.95, 19)


def candidates():
    return [dict(id=i, shape=s, scale=sc, density=1.0, lin=20.0, ang=20.0, friction=2.0) for i, (s, sc) in enumerate(SHAPE_SET)]


def job(args):
    cand, pos, rot, n, steps, seed, path = args
    params = T.Parameters(obj_x=pos[1] - BASE[1], obj_y=pos[0] - BASE[0], obj_damping=None, obj_friction=None,
                          obj_stretch_conditions=[1], obj_rotation_conditions=[T.ROT[rot]])
    env = T.SMEnv(0, params)
    env.b2d_env.world_files[T.WORLD["bar"]] = Path(path)
    res = [E.classify(env, a, steps)[0] for a in T.postures(seed + rot, n)]
    return cand["id"], pos, rot, np.array(res)


def parse_arguments():
    parser = argparse.ArgumentParser(description="Friction plus object position: do positions or shapes change the needed postures?")
    parser.add_argument("--out", default=str(Path.home() / "tmp/kss/l2_positions"))
    parser.add_argument("--n", type=int, default=250)
    parser.add_argument("--steps", type=int, default=60)
    parser.add_argument("--workers", type=int, default=8)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    out = Path(args.out)
    (out / "models").mkdir(parents=True, exist_ok=True)
    cands = candidates()
    paths = {c["id"]: str(E.model_file(c, out / "models")) for c in cands}
    jobs = [(c, p, r, args.n, args.steps, 0, paths[c["id"]]) for c in cands for p in POSITIONS for r in E.ROTS]
    with ProcessPoolExecutor(args.workers) as pool:
        results = list(pool.map(job, jobs, chunksize=2))
    store = {(i, p, r): cls for i, p, r, cls in results}
    np.savez(out / "classes.npz", **{f"{i}_{p[0]}_{p[1]}_{r}": cls for (i, p, r), cls in store.items()})
    feats = {r: E.__dict__["np"].column_stack([T.postures(r, args.n)[:, :3] / np.pi, T.postures(r, args.n)[:, 3] / (np.pi / 2),
                                               T.postures(r, args.n)[:, 4] / np.pi, T.postures(r, args.n)[:, :3].sum(1) / (3 * np.pi)]) for r in E.ROTS}

    def eff(i, p, r):
        c = store[(i, p, r)]
        return feats[r][(c >= 2) & (c != 4) | (c == 4)]

    rows = []
    for c in cands:
        for p in POSITIONS:
            allc = np.concatenate([store[(c["id"], p, r)] for r in E.ROTS])
            rows.append({"shape": c["shape"], "x": p[0], "y": p[1], "dist": float(np.hypot(p[0] - 0.42, p[1])), "touch": float((allc >= 1).mean()),
                         "effect": float((allc >= 2).mean())})
    table = pd.DataFrame(rows)
    table.to_csv(out / "single.csv", index=False)
    pd.set_option("display.width", 220)
    print("effect rate by shape and position (x, y)")
    print(table.pivot_table(index="shape", columns=["x", "y"], values="effect").round(2).to_string())

    def dist(a, b):
        return float(np.mean([np.abs(np.quantile(a[r], QUANTILES, axis=0) - np.quantile(b[r], QUANTILES, axis=0)).mean(0) for r in E.ROTS]))

    ok = lambda i, p: all(len(eff(i, p, r)) >= 12 for r in E.ROTS)
    rng = np.random.default_rng(0)
    floor = []
    for key in store:
        i, p, r = key
        if r == 0 and ok(i, p):
            v = eff(i, p, 0)
            idx = rng.permutation(len(v))
            floor.append(np.abs(np.quantile(v[idx[: len(v) // 2]], QUANTILES, axis=0) - np.quantile(v[idx[len(v) // 2:]], QUANTILES, axis=0)).mean())
    floor = float(np.mean(floor))
    same_pos, same_shape, cross = [], [], []
    keys = [(c["id"], p) for c in cands for p in POSITIONS if ok(c["id"], p)]
    for (i, p), (j, q) in itertools.combinations(keys, 2):
        d = dist({r: eff(i, p, r) for r in E.ROTS}, {r: eff(j, q, r) for r in E.ROTS})
        if p == q and i != j:
            same_pos.append(d)
        elif i == j and p != q:
            same_shape.append(d)
        elif i != j and p != q:
            cross.append(d)
    pd.set_option("display.precision", 3)
    print(f"\n{len(keys)} (shape, position) cells with enough effective postures")
    print(f"noise floor (split halves of one cell): {floor:.3f}")
    for name, v in (("same position, different shapes", same_pos), ("same shape, different positions", same_shape), ("different shape AND position", cross)):
        v = np.array(v)
        if not len(v):
            print(name, "no pairs")
            continue
        print(f"{name:34s} n={len(v):4d} mean {v.mean():.3f}  median {np.median(v):.3f}  90th {np.quantile(v, .9):.3f}  max {v.max():.3f}  share above 2x floor {(v > 2 * floor).mean():.2f}")

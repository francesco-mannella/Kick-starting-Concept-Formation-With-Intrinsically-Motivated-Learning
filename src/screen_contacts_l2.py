#!/usr/bin/env python
import argparse
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

import explore_positions_l2 as P
import explore_shapes_l2 as E
import tune_l2_objects as T


def first_contact(env, action, steps):
    env.reset(world=T.WORLD["bar"])
    for t in range(steps):
        obs = env.step(action)
        touch = np.asarray(obs["TOUCH_SENSORS"], dtype=float)
        if touch.any():
            return t, touch, np.asarray(obs["JOINT_POSITIONS"][:5], dtype=float)
    return -1, np.zeros(40), np.zeros(5)


def job(args):
    cand, pos, rot, n, steps, seed, path = args
    params = T.Parameters(obj_x=pos[1] - P.BASE[1], obj_y=pos[0] - P.BASE[0], obj_damping=None, obj_friction=None,
                          obj_stretch_conditions=[1], obj_rotation_conditions=[T.ROT[rot]])
    env = T.SMEnv(0, params)
    env.b2d_env.world_files[T.WORLD["bar"]] = Path(path)
    rows = [first_contact(env, a, steps) for a in T.postures(seed + rot, n)]
    return (cand["shape"], pos, rot, np.array([r[0] for r in rows]), np.array([r[1] for r in rows]), np.array([r[2] for r in rows]))


def parse_arguments():
    parser = argparse.ArgumentParser(description="First-contact touch and proprioception by shape and position")
    parser.add_argument("--out", default=str(Path.home() / "tmp/kss/l2_contacts"))
    parser.add_argument("--n", type=int, default=250)
    parser.add_argument("--steps", type=int, default=60)
    parser.add_argument("--workers", type=int, default=8)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    out = Path(args.out)
    (out / "models").mkdir(parents=True, exist_ok=True)
    cands = P.candidates()
    paths = {c["id"]: str(E.model_file(c, out / "models")) for c in cands}
    jobs = [(c, p, r, args.n, args.steps, 0, paths[c["id"]]) for c in cands for p in P.POSITIONS for r in E.ROTS]
    with ProcessPoolExecutor(args.workers) as pool:
        results = list(pool.map(job, jobs, chunksize=2))
    shapes = [r[0] for r in results]
    np.savez(out / "contacts.npz", shape=np.array(shapes), x=np.array([r[1][0] for r in results]), y=np.array([r[1][1] for r in results]),
             rot=np.array([r[2] for r in results]), step=np.stack([r[3] for r in results]), touch=np.stack([r[4] for r in results]),
             proprio=np.stack([r[5] for r in results]))
    print("saved", len(results), "cells to", out / "contacts.npz")

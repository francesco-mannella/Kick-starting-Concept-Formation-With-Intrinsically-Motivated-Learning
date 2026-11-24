import argparse
import time
from concurrent.futures import ProcessPoolExecutor
from itertools import product
from pathlib import Path

import box2dsim  # noqa: F401
import numpy as np
import pandas as pd

from params import Parameters
from SMEnv import SMEnv

SIZES = {1: 1.5, 2: 3.0, 3: 5.0}
LEFT_INNER, RIGHT_INNER = slice(10, 20), slice(20, 30)
HALF_PI = np.pi / 2


def make_env(pos, rot=0.0):
    return SMEnv(0, Parameters(obj_set="grasp3", obj_pos_kind={k: list(pos) for k in ("still", "movable", "controllable")},
                               obj_stretch_conditions=[1], obj_rotation_conditions=[float(rot)],
                               visual_pixels=24, visual_binary=True, visual_size=576, fovea_range=8.0))


def to_action(u):
    arm = (np.asarray(u[:3]) * 2 - 1) * HALF_PI
    a3 = u[3] * HALF_PI
    a4 = u[4] * min(HALF_PI, 2 * a3)
    return np.array([*arm, a3, a4])


def run(env, world, action, steps):
    env.reset(world=world)
    both = inner = contact = 0
    run_len = best = 0
    for _ in range(steps):
        touch = np.array(env.step(action)["TOUCH_SENSORS"])
        left, right = touch[LEFT_INNER].sum() > 0, touch[RIGHT_INNER].sum() > 0
        contact += touch.sum() > 0
        inner += left or right
        run_len = run_len + 1 if (left and right) else 0
        both += left and right
        best = max(best, run_len)
    return {"contact": int(contact), "inner": int(inner), "both": int(both), "grasp_run": int(best)}


def cell_random(args):
    world, rot_deg, pos, n, steps, seed = args
    rng = np.random.default_rng(seed)
    env = make_env(pos, np.radians(rot_deg))
    rows = []
    for _ in range(n):
        u = rng.uniform(0, 1, 5)
        rows.append({"world": world, "rot": rot_deg, "x": pos[0], "y": pos[1], **{f"u{i}": u[i] for i in range(5)}, **run(env, world, to_action(u), steps)})
    return rows


def score(res):
    return res["grasp_run"] + 0.2 * res["both"] + 0.02 * res["inner"] + 0.002 * res["contact"]


def cell_cem(args):
    world, rot_deg, pos, pop, iters, steps, seed = args
    rng = np.random.default_rng(seed)
    env = make_env(pos, np.radians(rot_deg))
    mu, sd = np.full(5, 0.5), np.full(5, 0.3)
    best = (-1.0, None, None)
    for _ in range(iters):
        samples = np.clip(rng.normal(mu, sd, (pop, 5)), 0, 1)
        results = [run(env, world, to_action(u), steps) for u in samples]
        scores = np.array([score(r) for r in results])
        order = np.argsort(scores)[::-1]
        if scores[order[0]] > best[0]:
            best = (scores[order[0]], samples[order[0]].copy(), results[order[0]])
        elite = samples[order[: max(2, pop // 8)]]
        mu, sd = elite.mean(0), np.maximum(elite.std(0), 0.03)
    u = best[1]
    return {"world": world, "rot": rot_deg, "x": pos[0], "y": pos[1], "score": best[0], **{f"u{i}": u[i] for i in range(5)},
            **{f"a{i}": v for i, v in enumerate(to_action(u))}, **best[2]}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("timing", "random", "cem"), required=True)
    parser.add_argument("--out", default=str(Path.home() / "tmp/kss/grasp3_gateA"))
    parser.add_argument("--tag", default="")
    parser.add_argument("--xs", type=float, nargs="+", default=[7.0, 10.7, 14.0])
    parser.add_argument("--ys", type=float, nargs="+", default=[4.0, 6.0, 9.3, 12.0])
    parser.add_argument("--rots", type=float, nargs="+", default=[0.0], help="object rotations in degrees")
    parser.add_argument("--n", type=int, default=200)
    parser.add_argument("--pop", type=int, default=32)
    parser.add_argument("--iters", type=int, default=8)
    parser.add_argument("--steps", type=int, default=160)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    cells = list(product(SIZES, args.rots, args.xs, args.ys))

    if args.mode == "timing":
        env = make_env((10.7, 9.3))
        t = time.time()
        rng = np.random.default_rng(0)
        for _ in range(5):
            run(env, 2, to_action(rng.uniform(0, 1, 5)), args.steps)
        print(f"{(time.time() - t) / 5:.2f} s per posture trial of {args.steps} steps")
        return

    with ProcessPoolExecutor(args.workers) as pool:
        if args.mode == "random":
            jobs = [(w, r, (x, y), args.n, args.steps, 1000 * w + i) for i, (w, r, x, y) in enumerate(cells)]
            rows = [row for chunk in pool.map(cell_random, jobs) for row in chunk]
            pd.DataFrame(rows).to_csv(out / f"random{args.tag}.csv", index=False)
        else:
            jobs = [(w, r, (x, y), args.pop, args.iters, args.steps, 7 + i) for i, (w, r, x, y) in enumerate(cells)]
            pd.DataFrame(list(pool.map(cell_cem, jobs))).to_csv(out / f"cem{args.tag}.csv", index=False)
    print("done", args.mode)


if __name__ == "__main__":
    main()

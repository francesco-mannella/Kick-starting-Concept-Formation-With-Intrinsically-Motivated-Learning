import argparse
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import box2dsim  # noqa: F401
import numpy as np
import pandas as pd

from params import Parameters
from SMEnv import SMEnv

LEFT_INNER, RIGHT_INNER = slice(10, 20), slice(20, 30)


def make_env(pos, rot):
    return SMEnv(0, Parameters(obj_set="grasp3", obj_pos_kind={k: list(pos) for k in ("still", "movable", "controllable")},
                               obj_stretch_conditions=[1], obj_rotation_conditions=[float(rot)], touch_overload=1e12,
                               visual_pixels=24, visual_binary=True, visual_size=576, fovea_range=8.0))


def run_cell(args):
    world, rot_deg, pos, arm, openings, steps = args
    env = make_env(pos, np.radians(rot_deg))
    rows = []
    for a in openings:
        env.reset(world=world)
        action = np.array([*arm, a, a])
        for t in range(steps):
            obs = env.step(action)
            touch = np.array(obs["TOUCH_SENSORS"])
            left, right = touch[LEFT_INNER].sum() > 0, touch[RIGHT_INNER].sum() > 0
            rows.append(dict(world=world, rot=rot_deg, opening=float(a), t=t, contact=int(touch.sum() > 0), both=int(left and right),
                             n_active=int((touch > 0).sum()), claw=float(abs(obs["JOINT_POSITIONS"][3])), effort=float(obs["TOUCH_EFFORT"])))
    return rows


def parse_arguments():
    parser = argparse.ArgumentParser(description="Claw motor effort against commanded claw opening, at the CEM arm postures of the common position")
    parser.add_argument("--cem", default=str(Path.home() / "tmp/kss/grasp3_gateA/cem_rot.csv"))
    parser.add_argument("--pos", default="13,4")
    parser.add_argument("--steps", type=int, default=60)
    parser.add_argument("--n-open", type=int, default=16)
    parser.add_argument("--workers", type=int, default=5)
    parser.add_argument("--out", required=True)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    x, y = (float(v) for v in args.pos.split(","))
    cem = pd.read_csv(args.cem)
    cem = cem[(cem.x == x) & (cem.y == y) & (cem.grasp_run > 0)]
    openings = np.linspace(0.05, 1.5, args.n_open)
    jobs = [(int(r.world), float(r.rot), (x, y), [r.a0, r.a1, r.a2], openings, args.steps) for r in cem.itertuples()]
    with ProcessPoolExecutor(args.workers) as pool:
        data = pd.DataFrame([row for res in pool.map(run_cell, jobs) for row in res])
    data.to_csv(args.out, index=False)
    print(f"{len(data)} steps, {int(data.contact.sum())} contact steps")

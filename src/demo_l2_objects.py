#!/usr/bin/env python
import argparse
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import box2dsim  # noqa: F401  registers the gym environment
import numpy as np
import pandas as pd

from params import Parameters
from SMEnv import SMEnv

WORLDS = {"post": 1, "plate": 2, "bar": 3}
ORIG_NAMES = {"post": "blue-cube", "plate": "red-triangle", "bar": "green-cube"}
ROT = {0: 0.0, 45: 0.785398, 90: 1.570796}
HALF_PI = np.pi / 2


def make_env(object_set, rot):
    params = Parameters(
        obj_set=object_set,
        obj_damping=20.0,
        obj_friction=2.0,
        obj_stretch_conditions=[1],
        obj_rotation_conditions=[ROT[rot]],
    )
    return SMEnv(0, params)


def sample_policies(rng, n):
    arm = rng.uniform(-HALF_PI, HALF_PI, (n, 3))
    prox = rng.uniform(0, HALF_PI, n)
    dist = rng.uniform(0, 1, n) * 2 * prox
    return np.column_stack([arm, prox, dist])


def rollout(env, kind, action, steps, render_name=None):
    env.reset(world=WORLDS[kind], render="offline" if render_name else None)
    body = env.b2d_env.sim.bodies[env.b2d_env.object_names[0]]
    start, angle0 = np.array(body.worldCenter), body.angle
    touch, moved = [], 0.0
    for _ in range(steps):
        obs = env.step(action)
        touch.append(np.asarray(obs["TOUCH_SENSORS"]) > 0)
        moved = max(moved, float(np.linalg.norm(np.array(body.worldCenter) - start)))
    touch = np.array(touch)
    contact = touch.any(axis=1)
    if render_name:
        env.b2d_env.renderer.close(render_name)
    return {
        "contact": float(contact.mean()),
        "first_contact": int(np.argmax(contact)) if contact.any() else -1,
        "max_sensors": int(touch.sum(axis=1).max()),
        "displacement": moved,
        "end_displacement": float(np.linalg.norm(np.array(body.worldCenter) - start)),
        "rotation_deg": float(np.degrees(abs(body.angle - angle0))),
    }


def search(job):
    object_set, kind, rot, n, steps, seed = job
    env = make_env(object_set, rot)
    rng = np.random.default_rng(seed)
    policies = sample_policies(rng, n)
    rows = []
    for i, action in enumerate(policies):
        rows.append({"set": object_set, "kind": kind, "rot": rot, "policy": i, **rollout(env, kind, action, steps)})
        for j in range(5):
            rows[-1][f"u{j}"] = float(action[j])
    return rows


def score(row):
    return row["displacement"] + 0.5 * row["contact"]


def parse_arguments():
    parser = argparse.ArgumentParser(description="Demo of the L2 object set (no agent, no learning)")
    parser.add_argument("--out", default=str(Path.home() / "tmp/kss/l2_demo"))
    parser.add_argument("--n", type=int, default=300)
    parser.add_argument("--steps", type=int, default=60)
    parser.add_argument("--sets", default="l2,orig")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--no-gifs", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    sets = args.sets.split(",")
    jobs = [(s, k, r, args.n, args.steps, args.seed + 1000 * w + 10 * r)
            for s in sets for w, k in enumerate(WORLDS) for r in ROT]
    with ProcessPoolExecutor(args.workers) as pool:
        results = list(pool.map(search, jobs))
    table = pd.DataFrame([row for rows in results for row in rows])
    table.to_csv(out / "all_policies.csv", index=False)

    table["touches"] = table.contact > 0
    table["moves"] = table.displacement > 0.5
    summary = table.groupby(["set", "kind", "rot"]).agg(
        touch_rate=("touches", "mean"), move_rate=("moves", "mean"),
        mean_contact=("contact", "mean"), max_sensors=("max_sensors", "max"),
        max_displacement=("displacement", "max"), max_rotation=("rotation_deg", "max"))
    summary.round(3).to_csv(out / "summary.csv")
    print(summary.round(3).to_string())

    best = {}
    for key, group in table.groupby(["set", "kind", "rot"]):
        best[key] = group.loc[group.apply(score, axis=1).idxmax()]
    transfer = []
    for object_set in sets:
        for src in WORLDS:
            for dst in WORLDS:
                for rot in ROT:
                    action = np.array([best[(object_set, src, rot)][f"u{j}"] for j in range(5)])
                    env = make_env(object_set, rot)
                    transfer.append({"set": object_set, "policy_from": src, "applied_to": dst, "rot": rot,
                                     **rollout(env, dst, action, args.steps)})
    transfer = pd.DataFrame(transfer)
    transfer.round(3).to_csv(out / "transfer.csv", index=False)
    pivot = transfer.groupby(["set", "policy_from", "applied_to"]).displacement.mean().unstack().round(2)
    print("\nmean displacement of the best policy of the row object applied to the column object")
    print(pivot.to_string())

    if not args.no_gifs:
        os.chdir(out)
        gifs = out / "gifs"
        gifs.mkdir(exist_ok=True)
        os.chdir(gifs)
        for object_set in sets:
            for kind in WORLDS:
                for rot in ROT:
                    action = np.array([best[(object_set, kind, rot)][f"u{j}"] for j in range(5)])
                    env = make_env(object_set, rot)
                    rollout(env, kind, action, args.steps, f"{object_set}_{kind}_rot{rot}_own")
            for src, dst in (("bar", "plate"), ("plate", "bar")):
                action = np.array([best[(object_set, src, 0)][f"u{j}"] for j in range(5)])
                env = make_env(object_set, 0)
                rollout(env, dst, action, args.steps, f"{object_set}_{src}policy_on_{dst}_rot0")
    print(f"\noutput in {out}")

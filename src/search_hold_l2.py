#!/usr/bin/env python
import argparse
import json
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

import tune_l2_objects as T

HALF_PI = np.pi / 2


def to_action(z):
    arm = (z[:3] * 2 - 1) * HALF_PI
    prox = z[3] * HALF_PI
    dist = z[4] * 2 * prox
    return np.array([*arm, prox, dist])


def evaluate(env, kind, z, steps):
    action = to_action(z)
    env.reset(world=T.WORLD[kind])
    sim = env.b2d_env.sim
    body = sim.bodies[env.b2d_env.object_names[0]]
    c0 = np.array(body.worldCenter)
    touches, centre, hand = [], [], []
    for _ in range(steps):
        obs = env.step(action)
        touches.append(np.asarray(obs["TOUCH_SENSORS"]) > 0)
        centre.append(np.array(body.worldCenter))
        hand.append((np.array(sim.bodies["Claw11"].worldCenter) + np.array(sim.bodies["Claw21"].worldCenter)) / 2)
    touches, centre, hand = np.array(touches), np.array(centre), np.array(hand)
    gap = np.linalg.norm(centre - hand, axis=1)
    n = touches.sum(1)
    last = slice(steps - 10, steps)
    held = bool((n[last] > 0).sum() >= 8 and gap[last].std() < 0.2 and gap[last].mean() < 3.5
                and np.linalg.norm(centre[-1] - centre[-6]) < 0.1)
    score = 5.0 * held + 2.0 * (n[last] > 0).mean() + 0.02 * n[last].mean() - 0.1 * gap[-1]
    signature = touches[last].mean(0)
    return score, held, float(np.linalg.norm(centre[-1] - c0)), signature


def search(job):
    kind, cand, rot, gens, pop, steps, seed, folder = job
    env = T.make_env(kind, T.model_file(kind, cand, Path(folder)), rot)
    rng = np.random.default_rng(seed)
    mean, std = np.full(5, 0.5), np.full(5, 0.3)
    best = (-1e9, None, None)
    held_total, total = 0, 0
    for _ in range(gens):
        zs = np.clip(rng.normal(mean, std, (pop, 5)), 0, 1)
        res = [evaluate(env, kind, z, steps) for z in zs]
        scores = np.array([r[0] for r in res])
        held_total += sum(r[1] for r in res)
        total += pop
        elite = np.argsort(scores)[-max(4, pop // 5):]
        mean, std = zs[elite].mean(0), zs[elite].std(0) + 0.03
        if scores.max() > best[0]:
            i = int(scores.argmax())
            best = (float(scores[i]), zs[i].copy(), res[i])
    score, z, r = best
    return {"kind": kind, "id": cand["id"], "rot": rot, "best_score": score, "best_held": bool(r[1]),
            "best_end_disp": r[2], "held_rate": held_total / total, "z": z.tolist(),
            "signature": r[3].tolist(), "n_sensors": int((r[3] > 0.5).sum())}


def parse_arguments():
    parser = argparse.ArgumentParser(description="Directed search for holds and touch signatures of bar and plate candidates")
    parser.add_argument("--out", default=str(Path.home() / "tmp/kss/l2_tune"))
    parser.add_argument("--bars", default="30,6,14,3")
    parser.add_argument("--plates", default="9,33,37")
    parser.add_argument("--gens", type=int, default=8)
    parser.add_argument("--pop", type=int, default=64)
    parser.add_argument("--steps", type=int, default=60)
    parser.add_argument("--workers", type=int, default=8)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    out = Path(args.out)
    for kind, cands in T.CANDIDATES.items():
        for i, cand in enumerate(cands):
            cand["id"] = i
    jobs = []
    for kind, ids in (("bar", args.bars), ("plate", args.plates)):
        for i in [int(x) for x in ids.split(",")]:
            for rot in (0, 90):
                jobs.append((kind, T.CANDIDATES[kind][i], rot, args.gens, args.pop, args.steps, 7 + i, str(out / "models")))
    with ProcessPoolExecutor(args.workers) as pool:
        rows = list(pool.map(search, jobs))
    (out / "hold_search.json").write_text(json.dumps(rows))
    table = pd.DataFrame(rows).drop(columns=["z", "signature"])
    pd.set_option("display.width", 200)
    print(table.round(3).to_string(index=False))
    sig = {(r["kind"], r["id"], r["rot"]): np.array(r["signature"]) for r in rows}
    print("\ncosine similarity of the touch signature of the best posture, plate rows vs bar columns (rot 0 / rot 90)")
    for rot in (0, 90):
        keys_b = [k for k in sig if k[0] == "bar" and k[2] == rot]
        keys_p = [k for k in sig if k[0] == "plate" and k[2] == rot]
        mat = pd.DataFrame([[float(sig[p] @ sig[b] / (np.linalg.norm(sig[p]) * np.linalg.norm(sig[b]) + 1e-9)) for b in keys_b]
                            for p in keys_p], index=[f"plate{k[1]}" for k in keys_p], columns=[f"bar{k[1]}" for k in keys_b])
        print(f"rot {rot}"); print(mat.round(2).to_string())

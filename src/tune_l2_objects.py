#!/usr/bin/env python
import argparse
import itertools
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import box2dsim  # noqa: F401  registers the gym environment
import numpy as np
import pandas as pd

from params import Parameters
from SMEnv import SMEnv

MODELS = Path(box2dsim.__file__).resolve().parent / "models"
WORLD = {"plate": 2, "bar": 3}
SOURCE = {"plate": ("movable.json", "movable"), "bar": ("controllable.json", "controllable")}
ROT = {0: 0.0, 45: 0.785398, 90: 1.570796}
HALF_PI = np.pi / 2

CANDIDATES = {
    "bar": [
        dict(w=w, h=h, density=d, damping=c, friction=f)
        for (w, h), d, c, f in itertools.product([(0.5, 6.0), (0.8, 6.0), (0.8, 4.5)], [0.2, 1.0, 3.0], [2.0, 20.0], [0.5, 2.0])
    ],
    "plate": [
        dict(w=w, h=h, density=d, damping=c, friction=f)
        for (w, h), d, c, f in itertools.product([(6.5, 2.5), (5.0, 2.5), (8.0, 3.5)], [0.2, 1.0, 3.0, 6.0], [2.0, 20.0], [0.5, 2.0])
    ],
}


def model_file(kind, cand, folder):
    src, name = SOURCE[kind]
    world = json.loads((MODELS / src).read_text())
    body = next(b for b in world["body"] if b["name"] == name)
    body["fixture"][0]["polygon"]["vertices"] = {
        "x": [0.0, cand["w"], cand["w"], 0.0, 0.0],
        "y": [cand["h"], cand["h"], 0.0, 0.0, cand["h"]],
    }
    body["fixture"][0]["density"] = cand["density"]
    body["fixture"][0]["friction"] = cand["friction"]
    body["linearDamping"] = body["angularDamping"] = cand["damping"]
    path = folder / f"{kind}_{cand['id']}.json"
    path.write_text(json.dumps(world))
    return path


def make_env(kind, path, rot):
    params = Parameters(obj_damping=None, obj_friction=None, obj_stretch_conditions=[1], obj_rotation_conditions=[ROT[rot]])
    env = SMEnv(0, params)
    env.b2d_env.world_files[WORLD[kind]] = path
    return env


def postures(seed, n):
    rng = np.random.default_rng(seed)
    arm = rng.uniform(-HALF_PI, HALF_PI, (n, 3))
    prox = rng.uniform(0, HALF_PI, n)
    dist = rng.uniform(0, 1, n) * 2 * prox
    return np.column_stack([arm, prox, dist])


def outcome(env, kind, action, steps):
    env.reset(world=WORLD[kind])
    sim = env.b2d_env.sim
    body = sim.bodies[env.b2d_env.object_names[0]]
    c0, a0 = np.array(body.worldCenter), body.angle
    touch, centre, hand = [], [], []
    for _ in range(steps):
        obs = env.step(action)
        touch.append(int((np.asarray(obs["TOUCH_SENSORS"]) > 0).sum()))
        centre.append(np.array(body.worldCenter))
        hand.append((np.array(sim.bodies["Claw11"].worldCenter) + np.array(sim.bodies["Claw21"].worldCenter)) / 2)
    touch, centre, hand = np.array(touch), np.array(centre), np.array(hand)
    gap = np.linalg.norm(centre - hand, axis=1)
    end = float(np.linalg.norm(centre[-1] - c0))
    rot = float(np.degrees(abs(body.angle - a0)))
    last = slice(steps - 10, steps)
    hold = bool((touch[last] > 0).sum() >= 8 and gap[last].std() < 0.2 and gap[last].mean() < 3.5
                and np.linalg.norm(centre[-1] - centre[-6]) < 0.1)
    return {"touch": bool(touch.any()), "hold": hold, "moved": end >= 2.0, "fling": end >= 8.0,
            "spun": rot >= 60.0, "end": end, "rot": rot}


def job(args):
    kind, cand, rot, n, steps, seed, folder = args
    env = make_env(kind, model_file(kind, cand, Path(folder)), rot)
    rows = [outcome(env, kind, a, steps) for a in postures(seed + rot, n)]
    frame = pd.DataFrame(rows)
    frame["effect"] = frame.hold | frame.moved | frame.spun
    return kind, cand["id"], rot, frame


def parse_arguments():
    parser = argparse.ArgumentParser(description="Tune the bar and plate so they afford different actions (physics only)")
    parser.add_argument("--out", default=str(Path.home() / "tmp/kss/l2_tune"))
    parser.add_argument("--n", type=int, default=250)
    parser.add_argument("--steps", type=int, default=60)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--workers", type=int, default=8)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    out = Path(args.out)
    models = out / "models"
    models.mkdir(parents=True, exist_ok=True)
    for kind, cands in CANDIDATES.items():
        for i, cand in enumerate(cands):
            cand["id"] = i
    jobs = [(k, c, r, args.n, args.steps, args.seed, str(models)) for k, cs in CANDIDATES.items() for c in cs for r in ROT]
    with ProcessPoolExecutor(args.workers) as pool:
        results = list(pool.map(job, jobs, chunksize=4))
    store = {(k, i, r): f for k, i, r, f in results}
    rows = []
    for kind, cands in CANDIDATES.items():
        for cand in cands:
            frames = [store[(kind, cand["id"], r)] for r in ROT]
            rows.append({"kind": kind, **cand, **{m: float(np.mean([f[m].mean() for f in frames]))
                         for m in ("touch", "hold", "moved", "fling", "spun", "effect")}})
    single = pd.DataFrame(rows)
    single.to_csv(out / "single.csv", index=False)

    pairs = []
    for pc, bc in itertools.product(CANDIDATES["plate"], CANDIDATES["bar"]):
        jac, pe, be = [], [], []
        for r in ROT:
            ep, eb = store[("plate", pc["id"], r)].effect.values, store[("bar", bc["id"], r)].effect.values
            union = (ep | eb).sum()
            jac.append((ep & eb).sum() / union if union else 1.0)
        sp = single[(single.kind == "plate") & (single.id == pc["id"])].iloc[0]
        sb = single[(single.kind == "bar") & (single.id == bc["id"])].iloc[0]
        spec = 1 - float(np.mean(jac))
        reach = min(sp.effect, sb.effect, 0.2) / 0.2
        calm = 1 - min(1.0, sp.fling + sb.fling)
        contrast = float(np.clip(sb.hold - sp.hold, 0, 0.3) / 0.3)
        pairs.append({"plate": pc["id"], "bar": bc["id"], "score": spec * reach * calm * (0.5 + 0.5 * contrast),
                      "specificity": spec, "reach": reach, "calm": calm, "hold_bar_minus_plate": sb.hold - sp.hold,
                      "plate_moved": sp.moved, "bar_moved": sb.moved, "plate_hold": sp.hold, "bar_hold": sb.hold,
                      "plate_fling": sp.fling, "bar_fling": sb.fling, "plate_effect": sp.effect, "bar_effect": sb.effect})
    pairs = pd.DataFrame(pairs).sort_values("score", ascending=False)
    pairs.to_csv(out / "pairs.csv", index=False)
    pd.set_option("display.width", 250)
    print("single-object table (means over rotations)")
    print(single.round(3).to_string(index=False))
    print("\ntop 15 pairs")
    print(pairs.head(15).round(3).to_string(index=False))
    for kind in CANDIDATES:
        print(f"\n{kind} candidates:", {c["id"]: {k: v for k, v in c.items() if k != "id"} for c in CANDIDATES[kind]})

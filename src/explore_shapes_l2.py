#!/usr/bin/env python
import argparse
import itertools
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import box2dsim  # noqa: F401  registers the gym environment
import numpy as np
import pandas as pd

import tune_l2_objects as T

CLASSES = ["none", "touch", "hold", "push", "fling", "spin"]
ROTS = (0, 90)


def regular(n, radius):
    angles = 2 * np.pi * np.arange(n) / n + np.pi / n
    return radius * np.cos(angles), radius * np.sin(angles)


def box(w, h):
    return np.array([0, w, w, 0.0]), np.array([h, h, 0, 0.0])


SHAPES = {
    "cube2": box(1.9, 1.9),
    "cube1": box(1.0, 1.0),
    "bar": box(0.6, 6.0),
    "plate": box(8.0, 3.5),
    "disk3": regular(8, 1.5),
    "disk5": regular(8, 2.5),
    "hex4": regular(6, 2.0),
    "wedge": (np.array([0.0, 4.0, 0.0]), np.array([0.0, 0.0, 4.0])),
}
GRID = dict(scale=[0.6, 1.0, 1.6], density=[0.2, 1.0, 3.0], lin=[2.0, 20.0], ang=[2.0, 20.0], friction=[0.5, 2.0])
CANDIDATES = [
    dict(id=i, shape=s, scale=sc, density=d, lin=l, ang=a, friction=f)
    for i, (s, sc, d, l, a, f) in enumerate(itertools.product(SHAPES, *GRID.values()))
]


def model_file(cand, folder):
    world = json.loads((T.MODELS / "controllable.json").read_text())
    body = next(b for b in world["body"] if b["name"] == "controllable")
    x, y = SHAPES[cand["shape"]]
    x, y = [float(v) * cand["scale"] for v in x], [float(v) * cand["scale"] for v in y]
    x, y = x + [x[0]], y + [y[0]]
    body["fixture"][0]["polygon"]["vertices"] = {"x": x, "y": y}
    body["fixture"][0]["density"] = cand["density"]
    body["fixture"][0]["friction"] = cand["friction"]
    body["linearDamping"], body["angularDamping"] = cand["lin"], cand["ang"]
    path = folder / f"cand_{cand['id']}.json"
    path.write_text(json.dumps(world))
    return path


def classify(env, action, steps):
    env.reset(world=T.WORLD["bar"])
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
    held = (touch[last] > 0).sum() >= 8 and gap[last].std() < 0.2 and gap[last].mean() < 3.5 and np.linalg.norm(centre[-1] - centre[-6]) < 0.1
    if held:
        code = 2
    elif end >= 8.0:
        code = 4
    elif end >= 2.0:
        code = 3
    elif rot >= 60.0:
        code = 5
    elif touch.any():
        code = 1
    else:
        code = 0
    return code, int(touch.max())


def job(args):
    cand, rot, n, steps, seed, folder = args
    params = T.Parameters(obj_damping=None, obj_friction=None, obj_stretch_conditions=[1], obj_rotation_conditions=[T.ROT[rot]])
    env = T.SMEnv(0, params)
    env.b2d_env.world_files[T.WORLD["bar"]] = model_file(cand, Path(folder))
    res = [classify(env, a, steps) for a in T.postures(seed + rot, n)]
    return cand["id"], rot, np.array([r[0] for r in res]), np.array([r[1] for r in res])


def parse_arguments():
    parser = argparse.ArgumentParser(description="Explore shapes and physics for objects that afford different actions")
    parser.add_argument("--out", default=str(Path.home() / "tmp/kss/l2_shapes"))
    parser.add_argument("--n", type=int, default=150)
    parser.add_argument("--steps", type=int, default=60)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--limit", type=int, default=None)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    out = Path(args.out)
    (out / "models").mkdir(parents=True, exist_ok=True)
    cands = CANDIDATES[: args.limit] if args.limit else CANDIDATES
    jobs = [(c, r, args.n, args.steps, args.seed, str(out / "models")) for c in cands for r in ROTS]
    with ProcessPoolExecutor(args.workers) as pool:
        results = list(pool.map(job, jobs, chunksize=4))
    store = {(i, r): (cls, ns) for i, r, cls, ns in results}
    np.savez(out / "classes.npz", **{f"{i}_{r}": cls for (i, r), (cls, _) in store.items()})
    rows = []
    for c in cands:
        allc = np.concatenate([store[(c["id"], r)][0] for r in ROTS])
        row = {**c, **{k: float((allc == j).mean()) for j, k in enumerate(CLASSES)}}
        row["effect"] = float((allc >= 2).mean())
        row["sensors_when_touched"] = float(np.mean(np.concatenate([store[(c["id"], r)][1] for r in ROTS])[allc >= 1]))
        rows.append(row)
    single = pd.DataFrame(rows)
    single.to_csv(out / "single.csv", index=False)

    ids = [c["id"] for c in cands]
    arr = {i: np.concatenate([store[(i, r)][0] for r in ROTS]) for i in ids}
    pairs = []
    for a, b in itertools.combinations(ids, 2):
        ca, cb = arr[a], arr[b]
        ea, eb = ca >= 2, cb >= 2
        union = ea | eb
        if union.sum() == 0:
            continue
        disagree = float((ca[union] != cb[union]).mean())
        ra, rb = ea.mean(), eb.mean()
        fa, fb = (ca == 4).mean(), (cb == 4).mean()
        ma = np.bincount(ca[ea], minlength=6).argmax() if ea.any() else -1
        mb = np.bincount(cb[eb], minlength=6).argmax() if eb.any() else -1
        reach = min(ra, rb, 0.15) / 0.15
        calm = 1 - min(1.0, fa + fb)
        score = disagree * reach * calm * (1.5 if ma != mb else 1.0)
        pairs.append({"a": a, "b": b, "score": score, "disagree": disagree, "reach_a": ra, "reach_b": rb,
                      "fling_a": fa, "fling_b": fb, "mode_a": CLASSES[ma] if ma >= 0 else "-", "mode_b": CLASSES[mb] if mb >= 0 else "-"})
    pairs = pd.DataFrame(pairs, columns=["a", "b", "score", "disagree", "reach_a", "reach_b", "fling_a", "fling_b", "mode_a", "mode_b"])
    pairs = pairs.sort_values("score", ascending=False)
    pairs.to_csv(out / "pairs.csv", index=False)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 40)
    info = single.set_index("id")[["shape", "scale", "density", "lin", "ang", "friction"]]
    top = pairs.head(25).copy()
    for side in ("a", "b"):
        for col in ("shape", "scale", "density", "lin", "ang", "friction"):
            top[f"{side}_{col}"] = top[side].map(info[col])
    print("single-object class rates (top 20 by effect)")
    print(single.sort_values("effect", ascending=False).head(20).round(3).to_string(index=False))
    print("\ntop 25 pairs")
    print(top.round(3).to_string(index=False))

#!/usr/bin/env python
import argparse
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

SRC = Path(__file__).resolve().parent
sys.path.insert(0, str(SRC))
sys.path.insert(0, str(SRC.parent / "tools" / "box2dsim"))

SIGN = np.array([-1, -1, -1, 1, 1, 1, 1, 1])
COLOURS = {1: "blue", 2: "red", 3: "green"}
JOINTS = {1: "ratchet_ccw_blue", 2: "spring_red", 3: "ratchet_cw_green"}
SHAPES = ("square", "triangle")


def policies(n, seed):
    return np.random.RandomState(seed).uniform(-1, 1, (n, 8))


def features(trace):
    active = trace > 0
    frames = active.any(1)
    n = int(frames.sum())
    if n == 0:
        return dict(frames=0, first=-1, last=-1, runs=0, sensors=0, peak=0.0, centroid=-1.0, spread=0.0)
    idx = np.arange(trace.shape[1])
    weight = trace.sum(0)
    centroid = float((idx * weight).sum() / weight.sum())
    spread = float(np.sqrt(((idx - centroid) ** 2 * weight).sum() / weight.sum()))
    runs = int((np.diff(np.r_[0, frames.astype(int)]) == 1).sum())
    return dict(frames=n, first=int(np.argmax(frames)), last=int(len(frames) - 1 - np.argmax(frames[::-1])), runs=runs,
                sensors=int(active.any(0).sum()), peak=float(trace.max()), centroid=centroid, spread=spread)


def run_task(args):
    scenario, world, draw, n_policies, steps, seed = args
    import box2dsim  # noqa: F401
    from params import Parameters
    from SMEnv import SMEnv

    params = Parameters(scenario=scenario)
    base = np.hstack([params.policy_base_arm, params.policy_base_wrist, np.ones(4) * params.policy_base])
    rand = {"stretch_conditions": params.obj_stretch_conditions, "rotation_conditions": params.obj_rotation_conditions, "pos": [params.obj_x, params.obj_y]}
    env = SMEnv(seed + 1000 * draw + world, params, params.action_steps, rand_obj_params=rand)
    rng = np.random.RandomState(seed + 7 * draw + world)
    rows = []
    for k, u in enumerate(policies(n_policies, seed)):
        shape = SHAPES[rng.randint(2)] if scenario == "affordance" else None
        env.rand_obj_params = {**rand, **({} if shape is None else {"shape": shape})}
        obs = env.reset(world=world)
        sim = env.b2d_env.sim
        joint = sim.joints.get(JOINTS[world]) if scenario == "affordance" else None
        start = np.array(obs["OBJ_POSITION"], float).ravel()
        touch, response = [], []
        for t in range(steps):
            obs = env.step(u * SIGN + base)
            touch.append(obs["TOUCH_SENSORS"].copy())
            if joint is not None:
                response.append(joint.translation if world == 2 else joint.angle)
        touch = np.array(touch)
        final = np.array(obs["OBJ_POSITION"], float).ravel()
        row = dict(scenario=scenario, world=world, colour=COLOURS[world], draw=draw, policy=k, shape=shape or "original",
                   displacement=float(np.linalg.norm(final - start)), **features(touch))
        if response:
            response = np.array(response)
            row.update(response_final=float(response[-1]), response_extreme=float(response[np.abs(response).argmax()]),
                       response_end_abs=float(abs(response[-1])))
        rows.append(row)
    return rows


def classify(table, label, folds=5):
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import GroupKFold, cross_val_score

    d = table[table.frames > 0]
    cols = ["frames", "first", "last", "runs", "sensors", "peak", "centroid", "spread"]
    groups = d.policy.values
    y = d[label].astype(str)
    model = RandomForestClassifier(200, random_state=0, min_samples_leaf=3)
    return float(cross_val_score(model, d[cols].values, y, groups=groups, cv=GroupKFold(folds)).mean()), float(y.value_counts(normalize=True).max())


def summarise(table):
    rows = []
    for scenario, d in table.groupby("scenario"):
        by = d.groupby(["colour", "policy"]).apply(lambda g: pd.Series(dict(contact=(g.frames > 0).mean())), include_groups=False).unstack(0)["contact"]
        corr = by.corr(method="spearman")
        pairs = [corr.loc["blue", "red"], corr.loc["blue", "green"], corr.loc["red", "green"]]
        acc_colour, chance_colour = classify(d, "colour")
        row = dict(scenario=scenario, episodes=len(d), kind_rate_corr=float(np.mean(pairs)), touch_any=float((by > 0).any(axis=1).mean()),
                   colour_from_touch=acc_colour, colour_chance=chance_colour)
        if scenario == "affordance":
            acc_shape, chance_shape = classify(d, "shape")
            row.update(shape_from_touch=acc_shape, shape_chance=chance_shape)
            red = d[d.colour == "red"]
            blue = d[d.colour == "blue"]
            green = d[d.colour == "green"]
            row.update(contact_blue=float((blue.frames > 0).mean()), contact_red=float((red.frames > 0).mean()), contact_green=float((green.frames > 0).mean()),
                       red_pressed=float((red.response_extreme.abs() > 0.5).mean()), red_back_at_end=float((red[red.response_extreme.abs() > 0.5].response_end_abs < 0.2).mean()),
                       blue_turned=float((blue.response_final > 0.3).mean()), green_turned=float((green.response_final < -0.3).mean()),
                       blue_wrong_way=float((blue.response_final < -0.05).mean()), green_wrong_way=float((green.response_final > 0.05).mean()))
        else:
            for colour in ("blue", "red", "green"):
                sel = d[d.colour == colour]
                row[f"contact_{colour}"] = float((sel.frames > 0).mean())
                row[f"moved_{colour}"] = float((sel.displacement > 0.5).mean())
        rows.append(row)
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--policies", type=int, default=300)
    parser.add_argument("--draws", type=int, default=4)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    tasks = [(sc, w, d, args.policies, args.steps, args.seed) for sc in ("original", "affordance") for d in range(args.draws) for w in COLOURS]
    rows = []
    with ProcessPoolExecutor(args.workers) as pool:
        for i, r in enumerate(pool.map(run_task, tasks)):
            rows += r
            print(f"{i + 1}/{len(tasks)} tasks", flush=True)
    table = pd.DataFrame(rows)
    table.to_csv(out / "episodes.csv", index=False)
    summary = summarise(table)
    summary.to_csv(out / "affordance_screen.csv", index=False)
    pd.set_option("display.width", 220, "display.max_columns", 40)
    print(summary.round(3).T.to_string())


if __name__ == "__main__":
    main()

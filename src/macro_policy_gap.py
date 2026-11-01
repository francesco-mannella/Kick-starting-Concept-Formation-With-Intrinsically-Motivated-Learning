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
KINDS = {1: "still", 2: "movable", 3: "controllable"}
LEVELS = {
    "control": {},
    "A1": {"obj_pos_jitter": (2.0, 2.0)},
    "A2": {"obj_pos_jitter": (4.0, 4.0)},
    "A3": {"obj_pos_jitter": (5.0, 5.0)},
    "B1": {"obj_scale": (1.0, 0.25, 1.0)},
    "B3": {"obj_stretch_conditions": [1, 2, 3]},
    "B4": {"obj_density": (1.0, 10.0, 0.2), "obj_damping": (None, 4.0, None)},
}


def policies(n, seed):
    return np.random.RandomState(seed).uniform(-1, 1, (n, 8))


def run_task(args):
    level, draw, kind, n_policies, steps, seed = args
    import box2dsim  # noqa: F401
    from params import Parameters
    from SMEnv import SMEnv

    params = Parameters(**LEVELS[level])
    base = np.hstack([params.policy_base_arm, params.policy_base_wrist, np.ones(4) * params.policy_base])
    env = SMEnv(seed + 1000 * draw + kind, params, params.action_steps)
    rows = []
    for k, u in enumerate(policies(n_policies, seed)):
        obs = env.reset(world=kind)
        start = np.array(obs["OBJ_POSITION"], float)
        touched, first, frames = False, -1, 0
        for t in range(steps):
            obs = env.step(u * SIGN + base)
            contact = bool(np.any(obs["TOUCH_SENSORS"] > 0))
            if contact:
                frames += 1
                if first < 0:
                    first = t
        final = np.array(obs["OBJ_POSITION"], float)
        rows.append(dict(level=level, draw=draw, kind=kind, policy=k, contact=int(frames > 0), first_contact=first, frames=frames,
                         displacement=float(np.linalg.norm(final - start))))
    return rows


def summarise(table, n_policies):
    rows = []
    for level, d in table.groupby("level"):
        c = d.groupby(["draw", "policy", "kind"]).contact.mean().unstack()
        half = d.draw.max() // 2
        sel = d[d.draw <= half].groupby(["policy", "kind"]).contact.mean().unstack()
        ev = d[d.draw > half].groupby(["policy", "kind"]).contact.mean().unstack()
        single = sel.min(axis=1).idxmax()
        specialists = {k: sel[k].idxmax() for k in sel.columns}
        generalist = float(ev.loc[single].min())
        specialist = float(np.mean([ev.loc[specialists[k], k] for k in sel.columns]))
        rate = d.groupby(["policy", "kind"]).contact.mean().unstack()
        any_kind = (rate > 0).any(axis=1).mean()
        all_kinds = (rate > 0).all(axis=1).mean()
        pattern = (rate > 0.5).astype(int).astype(str).agg("".join, axis=1)
        share = pattern.value_counts(normalize=True)
        entropy = float(-(share * np.log2(share)).sum())
        moved = d[d.contact == 1].groupby("kind").displacement.mean()
        corr = rate.corr(method="spearman")
        pairs = [corr.loc[1, 2], corr.loc[1, 3], corr.loc[2, 3]]
        sets = [set(rate.index[rate[k] > 0]) for k in (1, 2, 3)]
        jacc = [len(sets[i] & sets[j]) / max(1, len(sets[i] | sets[j])) for i, j in ((0, 1), (0, 2), (1, 2))]
        row = dict(level=level, episodes=len(d), kind_rate_corr=float(np.mean(pairs)), touched_set_jaccard=float(np.mean(jacc)), touch_any=float(any_kind), touch_all_kinds=float(all_kinds),
                   rate_still=float(rate[1].mean()), rate_movable=float(rate[2].mean()), rate_controllable=float(rate[3].mean()),
                   pattern_entropy_bits=entropy, specialist_rate=specialist, single_policy_rate=generalist, macro_gap=specialist - generalist,
                   disp_still=float(moved.get(1, np.nan)), disp_movable=float(moved.get(2, np.nan)), disp_controllable=float(moved.get(3, np.nan)))
        rows.append(row)
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--levels", default=",".join(LEVELS))
    parser.add_argument("--policies", type=int, default=300)
    parser.add_argument("--draws", type=int, default=8)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    levels = args.levels.split(",")
    tasks = [(lv, d, k, args.policies, args.steps, args.seed) for lv in levels for d in range(args.draws) for k in KINDS]
    rows = []
    with ProcessPoolExecutor(args.workers) as pool:
        for i, r in enumerate(pool.map(run_task, tasks)):
            rows += r
            print(f"{i + 1}/{len(tasks)} tasks", flush=True)
    table = pd.DataFrame(rows)
    table.to_csv(out / "episodes.csv", index=False)
    summary = summarise(table, args.policies)
    summary.to_csv(out / "macro_policy_gap.csv", index=False)
    pd.set_option("display.width", 220, "display.max_columns", 30)
    print(summary.round(3).to_string(index=False))


if __name__ == "__main__":
    main()

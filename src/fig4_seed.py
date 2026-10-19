#!/usr/bin/env python
import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

SRC = Path(__file__).resolve().parent


def storage(root, seed):
    return Path(root) / f"seed_{seed}" / f"parasite_{seed:03d}" / "simulations" / f"parasite_{seed:03d}" / "storage"


def render_reps(store, reps, eval_seed):
    env = {**os.environ, "OMP_NUM_THREADS": "1", "PYTHONPATH": str(SRC), "MPLBACKEND": "agg"}
    procs = []
    for r in range(reps):
        folder = store / f"rep_{r:02d}"
        folder.mkdir(exist_ok=True)
        for name in ("main.dump.npy", "weights.npy"):
            shutil.copy(store / "001499" / name, folder / name)
        if not (folder / "trajectory_df.csv").exists():
            log = open(folder / "render.log", "w")
            procs.append(subprocess.Popen([sys.executable, str(SRC / "run_evaluation.py"), "-n", "1", "-s", str(eval_seed + r), "-p"],
                                          cwd=folder, env=env, stdout=log, stderr=subprocess.STDOUT))
    for p in procs:
        p.wait()


def merge_reps(store, reps):
    final = store / "001499" / "final"
    (final / "episodes").mkdir(parents=True, exist_ok=True)
    frames = []
    for r in range(reps):
        folder = store / f"rep_{r:02d}"
        frames.append(pd.read_csv(folder / "trajectory_df.csv"))
        for gif in (folder / "episodes").glob("episode_*_0.gif"):
            shutil.copy(gif, final / "episodes" / f"episode_{gif.name.split('_')[1]}_{r}.gif")
    merged = pd.concat(frames, ignore_index=True)
    merged["Unnamed: 0"] = np.arange(len(merged))
    merged.to_csv(final / "trajectory_df.csv", index=False)
    shutil.copy(store / "001499" / "weights.npy", final / "weights.npy")
    return final


def parse_arguments():
    parser = argparse.ArgumentParser(description="Figure 4 (goal posture grid) for the active agent of one sweep seed")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--root", default="experiments/gifs/seeds")
    parser.add_argument("--reps", type=int, default=20)
    parser.add_argument("--eval-seed", type=int, default=300)
    parser.add_argument("--out", default="experiments/fig567_improved/figures")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    store = storage(args.root, args.seed)
    render_reps(store, args.reps, args.eval_seed)
    final = merge_reps(store, args.reps)
    for folder in ("demos", "logs", "goal_frames"):
        (final / folder).mkdir(exist_ok=True)
    env = {**os.environ, "PATH": f"{SRC}:{os.environ['PATH']}", "PYTHONPATH": str(SRC), "MPLBACKEND": "agg", "OMP_NUM_THREADS": "1"}
    with open(final / "fig4.log", "w") as log:
        subprocess.run([sys.executable, str(SRC / "run_postures.py"), "-g", "-r", *map(str, range(args.reps))],
                       cwd=final, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    shutil.copy(final / "goalgrid.png", out / f"fig4_goalgrid_seed{args.seed}.png")
    shutil.copy(final / "goals.csv", out / f"fig4_goals_seed{args.seed}.csv")
    print(f"seed {args.seed}: {len(pd.read_csv(final / 'goals.csv'))} grid nodes filled")

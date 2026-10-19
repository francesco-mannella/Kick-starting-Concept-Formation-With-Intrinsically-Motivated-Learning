#!/usr/bin/env python
import argparse
import glob
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

SRC = Path(__file__).resolve().parent
sys.path.insert(0, str(SRC))

from policy_demos import ENV, checkpoint, wait  # noqa: E402


def select(work, seeds, top):
    frames = []
    for seed in seeds:
        seg = pd.read_csv(checkpoint(work, seed) / "final_ckpt" / "eval" / "segment_scores.csv")
        seg["seed"] = seed
        frames.append(seg)
    seg = pd.concat(frames)
    green = seg[(seg.episode % 3 == 2) & seg.good]
    ranked = green.sort_values("score", ascending=False).drop_duplicates(["seed", "episode", "node_x", "node_y"])
    return ranked.head(top).reset_index(drop=True), len(green)


def ensure_rendered(work, pairs, eval_seed):
    processes = []
    for seed, rep in pairs:
        store = checkpoint(work, seed)
        folder = store / f"rep_{rep:02d}"
        folder.mkdir(exist_ok=True)
        for f in ("main.dump.npy", "weights.npy"):
            shutil.copy(store / "final_ckpt" / f, folder / f)
        if not (folder / "trajectory_df.csv").exists():
            log = open(folder / "render.log", "w")
            processes.append(subprocess.Popen(
                [sys.executable, str(SRC / "run_evaluation.py"), "-n", "1", "-s", str(eval_seed + rep), "-p"],
                cwd=folder, env=ENV, stdout=log, stderr=subprocess.STDOUT))
    wait(processes)


def render(work, chosen):
    processes = []
    for _, r in chosen.iterrows():
        folder = checkpoint(work, int(r.seed)) / f"rep_{int(r.rep):02d}"
        for sub in ("demos", "logs", "goal_frames"):
            (folder / sub).mkdir(exist_ok=True)
        processes.append(subprocess.Popen(
            [sys.executable, str(SRC / "plot_postures.py"), "-e", str(int(r.episode)), "-r", "0", "-g", str(int(r.goal))],
            cwd=folder, env=ENV, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL))
    wait(processes)


def collect(work, chosen, out):
    rows = []
    for rank, r in chosen.iterrows():
        folder = checkpoint(work, int(r.seed)) / f"rep_{int(r.rep):02d}" / "demos"
        found = glob.glob(str(folder / f"e{int(r.episode):02d}_g{int(r.goal):02d}_r0_*.gif"))
        if len(found) != 1:
            print(f"rank {rank + 1}: render missing for seed {int(r.seed)} episode {int(r.episode)}", flush=True)
            continue
        name = f"green{rank + 1:02d}_seed{int(r.seed)}_ep{int(r.episode):02d}_goal{int(r.node_x)}-{int(r.node_y)}_rep{int(r.rep)}.gif"
        shutil.copy(found[0], out / name)
        rows.append(dict(file=name, rank=rank + 1, seed=int(r.seed), episode=int(r.episode), eval_rep=int(r.rep),
                         goal_node_x=int(r.node_x), goal_node_y=int(r.node_y), steps=int(r.steps),
                         touch_dist_start=round(r.touch_start, 1), touch_dist_end=round(r.touch_end, 1),
                         proprio_dist_start=round(r.proprio_start, 1), proprio_dist_end=round(r.proprio_end, 1),
                         score=round(r.score, 1)))
    index = pd.DataFrame(rows)
    index.to_csv(out / "index.csv", index=False)
    return index


def parse_arguments():
    parser = argparse.ArgumentParser(description="Best policy demos of the green-stick episodes across seeds")
    parser.add_argument("--work", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--seeds", default="0-9")
    parser.add_argument("--top", type=int, default=10)
    parser.add_argument("--eval-seed", type=int, default=300)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    lo, _, hi = args.seeds.partition("-")
    seeds = list(range(int(lo), int(hi or lo) + 1))
    chosen, available = select(args.work, seeds, args.top)
    print(f"{available} good green segments; using the best {len(chosen)} distinct ones", flush=True)
    ensure_rendered(args.work, sorted({(int(r.seed), int(r.rep)) for _, r in chosen.iterrows()}), args.eval_seed)
    render(args.work, chosen)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    index = collect(args.work, chosen, out)
    print(index[["rank", "seed", "episode", "eval_rep", "goal_node_x", "goal_node_y", "steps", "touch_dist_start",
                 "touch_dist_end", "proprio_dist_start", "proprio_dist_end", "score"]].to_string(index=False), flush=True)

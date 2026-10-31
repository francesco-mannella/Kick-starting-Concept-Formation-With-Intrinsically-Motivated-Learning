#!/usr/bin/env python
import argparse
import glob
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

SRC = Path(__file__).resolve().parent
sys.path.insert(0, str(SRC))

from best_goal_trajectories import score_segments  # noqa: E402

SIM = SRC.parent / "tools" / "box2dsim"   # the simulator of this checkout, not an installed copy of another one
ENV = {**os.environ, "OMP_NUM_THREADS": "1", "MPLBACKEND": "agg", "PYTHONPATH": os.pathsep.join([str(SRC), str(SIM)])}
OBJECTS = {1: "blue-cube", 2: "red-triangle", 3: "green-stick"}


def checkpoint(work, seed):
    name = f"parasite_{seed:03d}"
    return Path(work) / f"seed_{seed}" / name / "simulations" / name / "storage"


def wait(processes):
    for p in processes:
        p.wait()


def evaluate_all(sweep, work, seeds, reps, eval_seed, epoch):
    processes = []
    for seed in seeds:
        name = f"parasite_{seed:03d}"
        source = Path(sweep) / name / "simulations" / name / "storage" / f"{epoch:06d}"
        store = checkpoint(work, seed)
        target = store / "final_ckpt"
        target.mkdir(parents=True, exist_ok=True)
        for f in ("main.dump.npy", "weights.npy"):
            shutil.copy(source / f, target / f)
        if not (target / "eval" / "trajectory_df.csv").exists():
            log = open(target / "eval.log", "w")
            processes.append(subprocess.Popen(
                [sys.executable, str(SRC / "run_evaluation.py"), "-n", str(reps), "-s", str(eval_seed), "-o", "eval"],
                cwd=target, env=ENV, stdout=log, stderr=subprocess.STDOUT))
    wait(processes)


def rank(work, seeds, criteria):
    rows = []
    for seed in seeds:
        folder = checkpoint(work, seed) / "final_ckpt" / "eval"
        seg = score_segments(pd.read_csv(folder / "trajectory_df.csv"), **criteria)
        seg.round(3).to_csv(folder / "segment_scores.csv", index=False)
        good = seg[seg.good]
        best = good.sort_values("score").groupby("episode").tail(1)
        rows.append(dict(seed=seed, segments=len(seg), good_segments=len(good), episode_types=best.episode.nunique(),
                         distinct_nodes=good[["node_x", "node_y"]].drop_duplicates().shape[0],
                         distinct_pairs=good[["episode", "node_x", "node_y"]].drop_duplicates().shape[0],
                         mean_best_score=round(best.score.mean(), 2) if len(best) else 0.0))
    return pd.DataFrame(rows).sort_values(["episode_types", "distinct_nodes", "mean_best_score"], ascending=False)


def render_reps(store, reps, eval_seed):
    processes = []
    for r in range(reps):
        folder = store / f"rep_{r:02d}"
        folder.mkdir(exist_ok=True)
        for f in ("main.dump.npy", "weights.npy"):
            shutil.copy(store / "final_ckpt" / f, folder / f)
        if not (folder / "trajectory_df.csv").exists():
            log = open(folder / "render.log", "w")
            processes.append(subprocess.Popen(
                [sys.executable, str(SRC / "run_evaluation.py"), "-n", "1", "-s", str(eval_seed + r), "-p"],
                cwd=folder, env=ENV, stdout=log, stderr=subprocess.STDOUT))
    wait(processes)


def merge_reps(store, reps):
    final = store / "final_ckpt" / "final"
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
    shutil.copy(store / "final_ckpt" / "weights.npy", final / "weights.npy")
    for folder in ("demos", "logs", "goal_frames"):
        (final / folder).mkdir(exist_ok=True)
    return final


def render_gifs(final, jobs, parallel):
    running, failed = [], []
    for episode, rep, goal in jobs:
        while len(running) >= parallel:
            running = [(p, j) for p, j in running if p.poll() is None]
            if len(running) >= parallel:
                running[0][0].wait()
        running.append((subprocess.Popen(
            [sys.executable, str(SRC / "plot_postures.py"), "-e", str(episode), "-r", str(rep), "-g", str(goal)],
            cwd=final, env=ENV, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL), (episode, rep, goal)))
    for p, job in running:
        if p.wait() != 0:
            failed.append(job)
    return failed


def collect(final, good, out, seed):
    from params import Parameters
    from SMMain import build_episode_dataset

    dataset, _ = build_episode_dataset(Parameters())
    best = good.sort_values("score", ascending=False).groupby(["episode", "node_x", "node_y"]).head(1)
    best = best.sort_values(["episode", "score"], ascending=[True, False])
    rows = []
    for _, r in best.iterrows():
        found = glob.glob(str(final / "demos" / f"e{int(r.episode):02d}_g{int(r.goal):02d}_r{int(r.rep)}_*.gif"))
        if len(found) != 1:
            continue
        t = dataset[dataset["index"] == int(r.episode)].iloc[0]
        name = (f"ep{int(r.episode):02d}_{OBJECTS[int(t.context)]}_stretch{t.stretch}_rot{int(round(np.degrees(t.rotation)))}"
                f"_goal{int(r.node_x)}-{int(r.node_y)}_seed{seed}rep{int(r.rep)}.gif")
        shutil.copy(found[0], out / name)
        rows.append(dict(file=name, seed=seed, episode=int(r.episode), object=OBJECTS[int(t.context)], stretch=t.stretch,
                         rotation_deg=int(round(np.degrees(t.rotation))), eval_rep=int(r.rep), goal_node_x=int(r.node_x),
                         goal_node_y=int(r.node_y), steps=int(r.steps), touch_dist_start=round(r.touch_start, 1),
                         touch_dist_end=round(r.touch_end, 1), proprio_dist_start=round(r.proprio_start, 1),
                         proprio_dist_end=round(r.proprio_end, 1), score=round(r.score, 1)))
    index = pd.DataFrame(rows)
    index.to_csv(out / "index.csv", index=False)
    return index


def parse_arguments():
    parser = argparse.ArgumentParser(description="Policy demo GIFs of the active agent of a parasite sweep")
    parser.add_argument("--sweep", required=True, help="Sweep root with parasite_NNN run folders")
    parser.add_argument("--work", required=True, help="Working folder for evaluations and renders")
    parser.add_argument("--out", required=True, help="Folder for the final GIFs and index.csv")
    parser.add_argument("--seeds", default="0-9")
    parser.add_argument("--reps", type=int, default=20)
    parser.add_argument("--eval-seed", type=int, default=300)
    parser.add_argument("--epoch", type=int, default=1499)
    parser.add_argument("--seed", type=int, default=None, help="Skip the ranking and use this seed")
    parser.add_argument("--parallel", type=int, default=12)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    lo, _, hi = args.seeds.partition("-")
    seeds = list(range(int(lo), int(hi or lo) + 1))
    criteria = dict(min_steps=15, max_end=3.5, min_gain=1.5, min_directness=0.0, max_ratio=0.6)
    evaluate_all(args.sweep, args.work, seeds, args.reps, args.eval_seed, args.epoch)
    table = rank(args.work, seeds, criteria)
    print(table.to_string(index=False), flush=True)
    seed = args.seed if args.seed is not None else int(table.seed.iat[0])
    print(f"using seed {seed}", flush=True)

    store = checkpoint(args.work, seed)
    render_reps(store, args.reps, args.eval_seed)
    final = merge_reps(store, args.reps)
    segments = pd.read_csv(store / "final_ckpt" / "eval" / "segment_scores.csv")
    good = segments[segments.good]
    unique = good.sort_values("score", ascending=False).drop_duplicates(["episode", "node_x", "node_y"])
    jobs = [(int(r.episode), int(r.rep), int(r.goal)) for _, r in unique.iterrows()]
    failed = render_gifs(final, jobs, args.parallel)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    index = collect(final, good, out, seed)
    print(f"{len(index)} GIFs for seed {seed}: {index.episode.nunique()} episode types, "
          f"{index[['goal_node_x', 'goal_node_y']].drop_duplicates().shape[0]} goal nodes; failed renders: {failed}", flush=True)

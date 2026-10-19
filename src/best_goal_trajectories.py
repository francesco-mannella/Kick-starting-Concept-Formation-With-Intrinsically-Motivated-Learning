#!/usr/bin/env python
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

MODALITIES = {"touch": ("touch_x", "touch_y"), "proprio": ("proprio_x", "proprio_y")}


def segment_stats(segment, tail=5):
    goal = segment[["prototype_x", "prototype_y"]].iloc[0].to_numpy(float)
    out = {"steps": len(segment)}
    for name, cols in MODALITIES.items():
        points = segment[list(cols)].to_numpy(float)
        dist = np.linalg.norm(points - goal, axis=1)
        path = np.linalg.norm(np.diff(points, axis=0), axis=1).sum()
        out[f"{name}_start"] = dist[0]
        out[f"{name}_end"] = dist[-tail:].mean()
        out[f"{name}_gain"] = dist[0] - dist[-tail:].mean()
        out[f"{name}_directness"] = np.linalg.norm(points[-1] - points[0]) / path if path > 0 else 0.0
    return out


def score_segments(df, min_steps, max_end, min_gain, min_directness, max_ratio=1.0):
    df = df.copy()
    df["rep"] = df.e_seed.map({s: i for i, s in enumerate(sorted(df.e_seed.unique()))})
    rows = []
    for (rep, episode, goal), segment in df.groupby(["rep", "episode_id", "goal_id"]):
        if len(segment) < min_steps:
            continue
        stats = segment_stats(segment.sort_values("ts"))
        stats.update(rep=int(rep), episode=int(episode), goal=int(goal),
                     node_x=int(segment.prototype_x.iat[0]), node_y=int(segment.prototype_y.iat[0]))
        rows.append(stats)
    seg = pd.DataFrame(rows)
    good = np.ones(len(seg), bool)
    seg["score"] = 0.0
    for name in MODALITIES:
        good &= (seg[f"{name}_end"] <= max_end) & (seg[f"{name}_gain"] >= min_gain) & (seg[f"{name}_directness"] >= min_directness)
        good &= seg[f"{name}_end"] <= max_ratio * seg[f"{name}_start"]
        seg["score"] += seg[f"{name}_gain"] - seg[f"{name}_end"] + seg[f"{name}_directness"]
    seg["good"] = good
    return seg


def parse_arguments():
    parser = argparse.ArgumentParser(description="Rank seeds by good internal-space trajectories toward the goal")
    parser.add_argument("--root", default="experiments/gifs/seeds")
    parser.add_argument("--seeds", default="0-9")
    parser.add_argument("--min-steps", type=int, default=15)
    parser.add_argument("--max-end", type=float, default=3.5)
    parser.add_argument("--min-gain", type=float, default=1.5)
    parser.add_argument("--max-ratio", type=float, default=0.6, help="End distance at most this fraction of the start distance")
    parser.add_argument("--min-directness", type=float, default=0.0)
    parser.add_argument("--eval-dir", default="eval20", help="Folder with trajectory_df.csv inside the checkpoint folder")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    lo, _, hi = args.seeds.partition("-")
    summary = []
    for k in range(int(lo), int(hi or lo) + 1):
        path = Path(args.root) / f"seed_{k}" / f"parasite_{k:03d}" / "simulations" / f"parasite_{k:03d}" / "storage" / "001499"
        seg = score_segments(pd.read_csv(path / args.eval_dir / "trajectory_df.csv"), args.min_steps, args.max_end, args.min_gain, args.min_directness, args.max_ratio)
        seg.round(3).to_csv(path / args.eval_dir / "segment_scores.csv", index=False)
        good = seg[seg.good]
        best = good.sort_values("score").groupby("episode").tail(1)
        summary.append(dict(seed=k, segments=len(seg), good_segments=len(good), episode_types_with_good=best.episode.nunique(),
                            distinct_good_nodes=good[["node_x", "node_y"]].drop_duplicates().shape[0],
                            mean_best_score=round(best.score.mean(), 2) if len(best) else 0.0))
    table = pd.DataFrame(summary).sort_values(["episode_types_with_good", "distinct_good_nodes", "mean_best_score"], ascending=False)
    print(table.to_string(index=False))

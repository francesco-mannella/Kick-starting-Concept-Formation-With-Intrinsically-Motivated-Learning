#!/usr/bin/env python3
"""Schedule and execute parallel posture plotting tasks.

This script coordinates the generation of posture plots by distributing
work across multiple processes. It identifies pending tasks by checking
for existing output files and only processes missing combinations.
"""

import argparse
import glob
import os
import re
import subprocess
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import SMGraphs
from params import Parameters


script_folder = Path(__file__).resolve().parent
NUM_PROCESSES = 8

os.environ["COLUMNS"] = "999"


def extract_goal(params):
    """Execute a single goal posture plot subprocess.

    Args:
        params: Tuple containing (episode_id, goal_idx, rep) identifiers
            for the specific posture plot to generate.

    Returns:
        str: Path to the log file created during execution.
    """
    log_path = Path("logs")
    log_path.mkdir(parents=True, exist_ok=True)
    episode_id, goal_idx, rep = params
    name = f"lf_e{int(episode_id):02d}_g{goal_idx:02d}_r{rep}"
    log_file = f"{log_path / name}.log"
    err_file = f"{log_path / name}.err"
    with open(log_file, "w") as out, open(err_file, "w") as err:
        subprocess.run(
            [
                f"{script_folder}/plot_postures.py",
                "--episode_id",
                episode_id,
                "--goal_id",
                str(goal_idx),
                "--rep",
                str(rep),
                "-d",
            ],
            stdout=out,
            stderr=err,
        )
        print(name)
    return log_file


def run_plot(params):
    """Execute a single posture plot generation subprocess.

    Args:
        params: Tuple containing (episode_id, goal_idx, rep) identifiers
            for the specific posture plot to generate.

    Returns:
        str: Path to the log file created during execution.
    """
    log_path = Path("logs")
    log_path.mkdir(parents=True, exist_ok=True)
    episode_id, goal_idx, rep = params
    name = f"e{int(episode_id):02d}_g{goal_idx:02d}_r{rep}"
    log_file = f"{log_path / name}.log"
    err_file = f"{log_path / name}.err"
    if len(glob.glob(f"{name}*png")) == 0:
        with open(log_file, "w") as out, open(err_file, "w") as err:
            subprocess.run(
                [
                    f"{script_folder}/plot_postures.py",
                    "--episode_id",
                    episode_id,
                    "--goal_id",
                    str(goal_idx),
                    "--rep",
                    str(rep),
                ],
                stdout=out,
                stderr=err,
            )
        print("plot: ", name)
        return log_file


if __name__ == "__main__":
    """Parse arguments and dispatch posture plotting tasks in parallel.

    Queries available postures from plot_postures.py, filters based on
    user-specified repetition indices, skips already-generated plots,
    and executes remaining tasks using a process pool.
    """
    parser = argparse.ArgumentParser(description="Schedule postures.")
    parser.add_argument(
        "-r", "--rep", type=int, nargs="+", help="Rep index/indices to process."
    )
    parser.add_argument(
        "-g", "--goal_grid", action="store_true", help="Generate goal grid (requires -r)."
    )

    args = parser.parse_args()

    if args.goal_grid and not args.rep:
        parser.error("-g/--goal_grid requires -r/--rep")

    # Query plot_postures.py for available posture schedule
    result = subprocess.run(["plot_postures.py", "-l"], capture_output=True, text=True)
    posture_schedule = result.stdout.strip().split("\n")[-18:]
    print(result.stdout)

    frames = Path("goal_frames")

    # Build task list, skipping combinations with existing output files
    tasks = []
    for item in posture_schedule:
        tokens = item.split()
        episode_id, goals = tokens[1], [int(x) for x in tokens[2:-1]]
        for rep, goal_count in enumerate(goals):
            if args.rep is None or rep in args.rep:
                for goal_idx in range(goal_count):
                    pattern = (
                        frames / f"lf_e{int(episode_id):02d}_g{goal_idx:02d}_r{rep}_*.png"
                    )
                    pattern = str(pattern)
                    if not glob.glob(pattern):
                        tasks.append((episode_id, goal_idx, rep))

    # Execute tasks in parallel using process pool
    with ProcessPoolExecutor(max_workers=NUM_PROCESSES) as executor:
        results = list(executor.map(extract_goal if args.goal_grid else run_plot, tasks))
        print(f"{len(results)} processes executed")

    if args.goal_grid:

        g = SMGraphs.GraphManager(None, Parameters())
        policy_colors = g.policy_map(wfile="weights.npy", local=True, plot=False)
        policy_colors = np.array(
            [[i, j, *el] for i, d1 in enumerate(policy_colors) for j, el in enumerate(d1)]
        )
        policy_proto = np.hstack(
            [
                policy_colors[:, :2].copy(),
                np.zeros([policy_colors.shape[0], 1]),
            ]
        )

        #    Find all goal event image files matching pattern
        goalevent_files = glob.glob(str(frames / "lf*png"))

        # Parse filename components into structured data
        data = []
        pattern = re.compile(r".*lf_e(\d+)_g(\d+)_r(\d+)_(\d)(\d).png")
        for f in goalevent_files:
            match = pattern.match(f)
            if match:
                e, g, r, px, py = map(int, match.groups())
                data.append(
                    {
                        "episode_id": e,
                        "goal_id": g,
                        "rep": r,
                        "px": px,
                        "py": py,
                        "filename": f,
                    }
                )

        # Create DataFrame from goal data dictionary
        goal_df = pd.DataFrame(data)
        # Load trajectory data from CSV file
        traj_df = pd.read_csv("trajectory_df.csv")

        # Map unique seeds to sequential repetition indices
        unique_seeds = np.sort(traj_df["e_seed"].unique())
        seed_to_rep = dict(zip(unique_seeds, range(len(unique_seeds))))
        traj_df["rep"] = traj_df["e_seed"].map(seed_to_rep)

        # Propagate final match value (at max timestamp) to all rows in group
        group_cols = ["episode_id", "goal_id", "rep"]
        traj_df["match"] = traj_df.groupby(group_cols)["match"].transform(
            lambda g: g.loc[traj_df.loc[g.index, "ts"].idxmax()]
        )

        # Reduce to one row per group, keeping only relevant columns
        keep_cols = group_cols + ["prototype_x", "prototype_y", "match"]
        traj_df = traj_df.groupby(group_cols).first().reset_index()[keep_cols]

        # Merge match results into goal DataFrame
        merge_cols = group_cols + ["match"]
        goal_df = goal_df.merge(
            traj_df[merge_cols], on=group_cols, how="left"
        )

        # Get last (best) match event for each prototype position
        goalevents = (
            goal_df.sort_values(["px", "py", "match"])
            .groupby(["px", "py"])
            .last()
            .reset_index()
        )

        goalevents = goalevents.query("match>0.3")
        goalevents.match = goalevents.match.round(2)
        goalevents.to_csv("goals.csv")

        h, w, _ = plt.imread(goalevents.filename.iat[0]).shape

        # Grid and image dimensions
        border = 2
        b2 = border // 2
        grid_dims = (10, 10)
        im_dims = np.array((h + border, w + border))
        n_rows, n_cols = grid_dims
        im_h, im_w = im_dims

        # Initialize 4D array: (grid_row, grid_col, img_height, img_width, RGBA)
        global_array = np.zeros((n_rows, n_cols, im_h, im_w, 4))

        for idx, row in goalevents.iterrows():
            px, py = row[["px", "py"]].to_list()
            f = row["filename"]
            r = n_rows - px - 1
            c = py
            global_array[r, c, b2:-b2, b2:-b2, :] = plt.imread(f)
            policy_proto[(policy_proto[:, 0] == r) & (policy_proto[:, 1] == c), 2] = 1

        # Reshape grid of images into single composite image
        global_array = global_array.transpose(0, 2, 1, 3, 4).reshape(
            n_rows * im_h, n_cols * im_w, 4
        )

        # Create figure and display composite image
        fig, ax = plt.subplots(1, 1, figsize=(3, 3 * (h / w)))

        coords = ((policy_colors[:, :2] + [0.5, 0.5]) * im_dims).T[::-1]
        rect_size = im_dims[::-1]
        innrect_size = im_dims[::-1] - 60
        for i in range(policy_colors.shape[0]):
            rect = plt.Rectangle(
                (coords[0, i] - rect_size[0] / 2, coords[1, i] - rect_size[1] / 2),
                *rect_size,
                fill=True,
                edgecolor=None,  # policy_colors[i, 2:],
                facecolor=policy_colors[i, 2:],
                linewidth=0,
                alpha=1,
                zorder=0,
            )
            ax.add_patch(rect)
            if policy_proto[i, -1] > 0:
                wrect = patches.FancyBboxPatch(
                    (
                        coords[0, i] - innrect_size[0] / 2,
                        coords[1, i] - innrect_size[1] / 2,
                    ),
                    boxstyle="round,pad=10",
                    *innrect_size,
                    fill=True,
                    facecolor="#fff",
                    linewidth=0,
                    alpha=1,
                    zorder=5,
                )
                ax.add_patch(wrect)

        global_array[global_array[:, :, :3].mean(-1) > 0.99, 3] = 0

        ax.imshow(global_array, interpolation="none", zorder=10)

        # Configure grid lines at image boundaries (no labels)
        ax.set_xticks(range(0, im_w * (n_cols + 1), im_w))
        ax.set_yticks(range(0, im_h * (n_rows + 1), im_h))
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.grid(True, which="both")

        fig.tight_layout(pad=0)

        fig.savefig("goalgrid.png", dpi=400)

        # return global_array

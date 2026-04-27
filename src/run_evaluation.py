#!/usr/bin/env python

import argparse
import glob
import os
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from SMMain import Main
from storage import StorageManager


os.environ["OMP_NUM_THREADS"] = "1"

_ = Main


def run_evaluation(
    main_dump_path,
    full_render=False,
    render_path=None,
    plot=False,
    seed=0,
    suffix="",
):
    """
    Run a set of evaluation episodes

    Args:
        main_dump_path: Path to the main.dump.npy file
        seed: Additional offset to add to the seed for variation
    """
    if not Path(main_dump_path).is_file():
        raise FileNotFoundError(f"Could not find main dump file: {main_dump_path}")

    main = np.load(main_dump_path, allow_pickle=True)[0]
    orig_dir = (Path.cwd() / ".." / ".." / ".." / "..").resolve()
    exp_name = (Path.cwd() / ".." / "..").resolve().stem

    main.sm = StorageManager(exp_name, orig_dir)
    main.params.use_wandb = False
    if full_render:
        main.params.full_render = True

    goal_counts, trajectories = main.evaluation_episodes(
        epoch=main.epoch,
        suffix=f"_{suffix}",
        render_path=render_path,
        save_stats=False,
        n_episodes=main.params.tests,
        render="offline" if plot else None,
        e_seed=seed,
    )

    return trajectories


def parse_arguments():
    parser = argparse.ArgumentParser(description="Run a set of episodes for evaluation")

    parser.add_argument(
        "-s",
        "--seed",
        type=int,
        default=200,
        help="Seed for the evaluation set",
    )
    parser.add_argument(
        "-n",
        "--num_reps",
        type=int,
        default=1,
        help="Number of repetitions for episode",
    )
    parser.add_argument(
        "-o",
        "--output_dir",
        type=str,
        help="Output directory for rendered episode",
        default=".",
    )
    parser.add_argument(
        "-p",
        "--plot",
        action="store_true",
        help="Render animations",
    )
    parser.add_argument(
        "-g",
        "--gpu",
        action="store_true",
        help="Use GPU if available",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()

    # Setup device
    device = "cuda" if torch.cuda.is_available() and args.gpu else "cpu"
    torch.set_default_device(device)

    # Setup output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    trajectory_file = output_dir / "trajectory_df.csv"

    if not trajectory_file.exists():

        # render
        all_trajectories = []
        for i in range(args.num_reps):
            seed = i + args.seed
            trajectories = run_evaluation(
                "main.dump.npy",
                seed=seed,
                suffix=i,
                plot=args.plot,
                render_path=output_dir,
                full_render=False,
            )
            all_trajectories.append(trajectories)
        merged_trajectories = pd.concat(all_trajectories, ignore_index=True)

        merged_trajectories.to_csv(trajectory_file)

        episode_gifs = glob.glob("episode*gif")
        episodes = Path("episodes")
        episodes.mkdir(exist_ok=True)

        for episode_gif in episode_gifs:
            shutil.move(episode_gif, episodes)

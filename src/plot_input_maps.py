#!/usr/bin/env python
import argparse
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from params import Parameters
from plot_postures import plot_polyline
from SMGraphs import GraphManager

SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e5e4e0"
ACCENT = "#2a78d6"
ARM_LENGTHS = [1, 1, 1, 0.5, 0.5]


def map_path(side, data_dir):
    name = "StoredGripActuatorMap" if side == 5 else f"StoredGripActuatorMap_k{side}"
    return Path(data_dir) / f"{name}.npy"


def split_blocks(weights, side, norm=None):
    n = side * side
    if norm is not None:
        weights = weights * norm["scale"][:, None] + norm["mean"][:, None]
    return weights[:40, :n].T, np.degrees(weights[40:45, n : 2 * n].T), weights[45:47, 2 * n : 3 * n].T


def touch_tile(ax, values, sensors, vmax):
    ax.scatter(*sensors.T, s=2, c=GRID, linewidths=0)
    strength = np.clip(values / vmax, 0, 1)
    ax.scatter(*sensors.T, s=3 + 22 * strength, c=ACCENT, alpha=0.25 + 0.75 * strength, linewidths=0)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_axis_off()


def posture_tile(ax, angles):
    arm, grip = plot_polyline(angles.copy(), ARM_LENGTHS)
    ax.plot(*arm.T, "-", c=INK, lw=0.9)
    ax.plot(*grip.T, "-", c=ACCENT, lw=0.9)
    ax.set_xlim(-1.5, 3.5)
    ax.set_ylim(-1.5, 3.5)
    ax.set_aspect("equal")
    ax.set_axis_off()


def eye_lattice(ax, eye, side):
    grid = eye.reshape(side, side, 2)
    for i in range(side):
        ax.plot(grid[i, :, 0], grid[i, :, 1], "-", c=GRID, lw=1.0, zorder=1)
        ax.plot(grid[:, i, 0], grid[:, i, 1], "-", c=GRID, lw=1.0, zorder=1)
    ax.scatter(eye[:, 0], eye[:, 1], s=22 if side <= 5 else 10, c=ACCENT, zorder=2, linewidths=0)
    ax.set_facecolor(SURFACE)
    ax.set_aspect("equal")
    ax.tick_params(colors=INK2, labelsize=8)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)


def draw(side, weights, sensors, out, tag=None, norm=None):
    touch, posture, eye = split_blocks(weights, side, norm)
    n = side * side
    fig = plt.figure(figsize=(15, 5.6), facecolor=SURFACE)
    gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 1], wspace=0.08)
    vmax = np.percentile(touch, 99)
    for col, (title, items) in enumerate((("Touch prototypes (40 sensors)", touch), ("Posture prototypes (arm and gripper)", posture))):
        sub = gs[0, col].subgridspec(side, side, wspace=0.05, hspace=0.05)
        for idx in range(n):
            ax = fig.add_subplot(sub[idx // side, idx % side])
            ax.set_facecolor(SURFACE)
            touch_tile(ax, items[idx], sensors, vmax) if col == 0 else posture_tile(ax, items[idx])
        fig.text(0.02 + col * 0.335, 0.96, title, color=INK, fontsize=11, ha="left", va="top")
    ax = fig.add_subplot(gs[0, 2])
    eye_lattice(ax, eye, side)
    fig.text(0.68, 0.96, "Eye-position map (lattice in input space)", color=INK, fontsize=11, ha="left", va="top")
    label = f"Pre-learned input maps, {side}x{side} units per modality ({3 * n} hidden units)."
    if tag:
        label += f" Map: {tag}."
    fig.text(0.005, 0.005, label, color=INK2, fontsize=8)
    path = Path(out) / (f"input_maps_{tag}.png" if tag else f"input_maps_k{side}.png")
    fig.savefig(path, dpi=140, facecolor=SURFACE)
    plt.close(fig)
    return path


def parse_arguments():
    parser = argparse.ArgumentParser(description="Draw the pre-learned grip input maps")
    parser.add_argument("--data-dir", default="src/data")
    parser.add_argument("--out", default="test/demo_local/figures")
    parser.add_argument("--sides", default="3,5,8,10")
    parser.add_argument("--map", default=None, help="Draw this single map file instead of the side sweep")
    parser.add_argument("--norm", default=None, help="Normalisation stats (.npz) of a normalised map")
    parser.add_argument("--tag", default=None, help="Name used in the output file")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    Path(args.out).mkdir(parents=True, exist_ok=True)
    sensors = GraphManager(None, Parameters()).generate_sensor_points(40)
    if args.map:
        weights = np.load(args.map)
        norm = dict(np.load(args.norm)) if args.norm else None
        print(draw(int(round(np.sqrt(weights.shape[1] / 3))), weights, sensors, args.out, args.tag, norm))
    else:
        for side in [int(s) for s in args.sides.split(",")]:
            print(draw(side, np.load(map_path(side, args.data_dir)), sensors, args.out))

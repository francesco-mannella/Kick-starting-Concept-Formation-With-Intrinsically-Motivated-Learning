#!/usr/bin/env python
import argparse
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("agg")
import matplotlib.pyplot as plt  # noqa: E402

BLOCKS = ("touch", "proprioception", "eye")
OUTPUTS = ("joint 1", "joint 2", "joint 3", "claw 1", "claw 2")


def node_matrices(vector, blocks=3, units=25, outputs=5):
    """(outputs, blocks, 5, 5): the readout weights of one policy, one 5 x 5 matrix per output and input block."""
    weights = np.asarray(vector).reshape(blocks * units, outputs)
    side = int(round(np.sqrt(units)))
    return np.stack([[weights[b * units : (b + 1) * units, o].reshape(side, side) for b in range(blocks)] for o in range(outputs)])


def compose(policy, side=10, gap=1, node_gap=3, clip=99.0):
    """Big image of the policy map: node (row, column) at row-major index, row 0 at the bottom as in the policy-map figure."""
    first = node_matrices(policy[:, 0])
    outputs, blocks, m, _ = first.shape
    tile_h = outputs * m + (outputs - 1) * gap
    tile_w = blocks * m + (blocks - 1) * gap
    height = side * tile_h + (side - 1) * node_gap
    width = side * tile_w + (side - 1) * node_gap
    image = np.full((height, width), np.nan)
    for j in range(side * side):
        row, col = divmod(j, side)
        mats = node_matrices(policy[:, j])
        top = (side - 1 - row) * (tile_h + node_gap)
        left = col * (tile_w + node_gap)
        for o in range(outputs):
            for b in range(blocks):
                y, x = top + o * (m + gap), left + b * (m + gap)
                image[y : y + m, x : x + m] = mats[o, b]
    limit = np.percentile(np.abs(policy), clip)
    return image, limit, (tile_h, tile_w, node_gap)


def draw(policy, out, title=""):
    image, limit, (tile_h, tile_w, node_gap) = compose(policy)
    height, width = image.shape
    fig = plt.figure(figsize=(10.5, 0.62 * 10.5 * height / width / 0.90 + 0.6))
    ax = fig.add_axes([0.07, 0.03, 0.62, 0.90])
    cmap = plt.get_cmap("RdBu_r").copy()
    cmap.set_bad("white")
    shown = ax.imshow(image, cmap=cmap, vmin=-limit, vmax=limit, interpolation="nearest", aspect="equal")
    ax.set_axis_off()
    step_x, step_y = tile_w + node_gap, tile_h + node_gap
    for c in range(10):
        ax.text(c * step_x + tile_w / 2 - 0.5, -2.5, str(c), ha="center", va="bottom", fontsize=9, color="#444444")
    for r in range(10):
        ax.text(-2.5, (9 - r) * step_y + tile_h / 2 - 0.5, str(r), ha="right", va="center", fontsize=9, color="#444444")
    ax.text(-8.5, height / 2, "map row", rotation=90, ha="center", va="center", fontsize=9, color="#444444")
    ax.text(width / 2, -9.5, "map column", ha="center", va="bottom", fontsize=9, color="#444444")
    cax = fig.add_axes([0.71, 0.55, 0.018, 0.30])
    cbar = fig.colorbar(shown, cax=cax)
    cbar.set_label("policy weight", fontsize=9)
    key = fig.add_axes([0.745, 0.10, 0.235, 0.26])
    key.set_xlim(-1.9, 3.3)
    key.set_ylim(-0.4, 5.9)
    key.set_aspect("equal")
    key.set_axis_off()
    for o, name in enumerate(OUTPUTS):
        key.text(-0.15, 4 - o + 0.5, name, ha="right", va="center", fontsize=9)
        for b in range(3):
            key.add_patch(plt.Rectangle((b, 4 - o), 0.88, 0.88, fc="#cfd8dc", ec="#90a4ae", lw=0.8))
    for b, name in enumerate(("touch", "propr.", "eye")):
        key.text(b + 0.2, 5.1, name, ha="left", va="bottom", fontsize=8, rotation=40)
    key.text(1.44, -0.2, "one node: 5 outputs x 3 input blocks,\neach a 5 x 5 matrix (25 hidden units)", ha="center", va="top", fontsize=8, color="#555555")
    fig.text(0.5, 0.985, title or "Policy weights of the 10 x 10 map", ha="center", va="top", fontsize=11)
    fig.savefig(out, dpi=100, facecolor="white")
    plt.close(fig)


def parse_arguments():
    parser = argparse.ArgumentParser(description="Grid of the policy weights: per node 5 rows (outputs) of 3 (input blocks) 5 x 5 matrices")
    parser.add_argument("checkpoint", help="Folder with weights.npy (a storage/<epoch> folder of a run)")
    parser.add_argument("--out", required=True)
    parser.add_argument("--title", default="")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    policy = np.load(Path(args.checkpoint) / "weights.npy", allow_pickle=True)[0]["policy"]
    draw(policy, args.out, args.title)
    print(f"saved {args.out} (policy {policy.shape})")

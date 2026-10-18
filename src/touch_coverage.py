#!/usr/bin/env python
import argparse
import glob
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e5e4e0"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
GROUPS = [slice(10 * g, 10 * g + 10) for g in range(4)]


def touch_frames(x):
    return x[x.sum(1) > 0]


def coverage(frames, W, scale):
    z = frames / scale
    mass = np.abs(W).sum(1)
    d = (z ** 2).sum(1)[:, None] - 2 * z @ W + (W ** 2).sum(0)[None]
    counts = np.bincount(d.argmin(1), minlength=W.shape[1])
    p = counts / counts.sum()
    q, _ = np.linalg.qr(W)
    dominant = np.stack([z[:, g].sum(1) for g in GROUPS], 1).argmax(1)
    r2 = lambda a: 1 - ((a - a @ q @ q.T) ** 2).sum() / (a ** 2).sum()
    return {
        "sensors below 5% of mean prototype weight": int((mass / mass.mean() < 0.05).sum()),
        "prototype mass, sensors 0-9 / 10-19 / 20-29 / 30-39 (%)": " / ".join(f"{100 * np.abs(W[g]).sum() / np.abs(W).sum():.0f}" for g in GROUPS),
        "prototypes that win at least once (of 25)": int((counts > 0).sum()),
        "effective number of winning prototypes": round(float(1 / (p ** 2).sum()), 1),
        "R2 of touch frames in prototype span": round(float(r2(z)), 3),
        "R2 by dominant group 0-9 / 10-19 / 20-29 / 30-39": " / ".join(
            f"{r2(z[dominant == g]):.2f}" if (dominant == g).sum() > 20 else "n/a" for g in range(4)),
    }, mass / mass.mean()


def parse_arguments():
    parser = argparse.ArgumentParser(description="Touch map coverage: data, old map, new maps")
    parser.add_argument("--babble", required=True, help="glob of new babbling .npz files")
    parser.add_argument("--runtime-root", default="test/demo_local")
    parser.add_argument("--out", default="test/demo_local/figures")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    new = touch_frames(np.concatenate([np.load(f)["touch"] for f in sorted(glob.glob(args.babble))]))
    old = touch_frames(np.load("src/data/StoredGripGenerateData.npy", mmap_mode="r")[:, 300:340].copy())
    run = touch_frames(np.concatenate([
        np.load(f, allow_pickle=True)[0]["ss"].reshape(-1, 40)
        for f in sorted(glob.glob(f"{args.runtime_root}/parasite_*/simulations/*/storage/001499/data.npy"))]))
    maps = {"old map (current)": ("src/data/StoredGripActuatorMap.npy", None),
            "old data, normalised": ("src/data/StoredGripActuatorMap_oldnorm.npy", "src/data/StoredGripActuatorNorm_oldnorm.npz"),
            "new map v3 (balanced)": ("src/data/StoredGripActuatorMap_v3.npy", "src/data/StoredGripActuatorNorm_v3.npz"),
            "new map v3s (balanced + synthetic)": ("src/data/StoredGripActuatorMap_v3s.npy", "src/data/StoredGripActuatorNorm_v3s.npz")}
    rows, mass_by_map = [], {}
    for name, (mp, npz) in maps.items():
        W = np.load(mp)[:40, :25]
        scale = float(np.load(npz)["scale"][0]) if npz else 1.0
        for frames_name, frames in (("new babbling frames", new), ("runtime frames (10 agents)", run)):
            report, mass = coverage(frames, W, scale)
            rows.append({"map": name, "evaluated on": frames_name, **report})
            mass_by_map[name] = mass
    out = Path(args.out)
    pd.DataFrame(rows).to_csv(out / "touch_map_coverage.csv", index=False)

    fig, axes = plt.subplots(2, 1, figsize=(11, 7), facecolor=SURFACE, sharex=True)
    x = np.arange(40)
    for ax in axes:
        ax.set_facecolor(SURFACE)
        ax.grid(axis="y", color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(GRID)
        ax.tick_params(colors=INK2, labelsize=9)
        for g in (10, 20, 30):
            ax.axvline(g - 0.5, color=GRID, linewidth=1)
    for label, frames, color, dash in (("old babbling data", old, ORANGE, "-"), ("new babbling data", new, BLUE, "-"),
                                       ("runtime, final agents", run, AQUA, "--")):
        axes[0].plot(x, (frames > 0).mean(0), color=color, linewidth=1.8, linestyle=dash, label=label)
    axes[0].set_title("How often each sensor is active (fraction of touch frames)", color=INK, fontsize=11, loc="left")
    axes[0].legend(frameon=False, labelcolor=INK, fontsize=9)
    for label, color, dash in (("old map (current)", ORANGE, "-"), ("new map v3s", BLUE, "-")):
        key = "new map v3s (balanced + synthetic)" if label == "new map v3s" else label
        axes[1].plot(x, mass_by_map[key], color=color, linewidth=1.8, linestyle=dash, label=label)
    axes[1].set_title("Prototype weight each sensor receives (relative to the mean sensor)", color=INK, fontsize=11, loc="left")
    axes[1].axhline(0.05, color=INK2, linewidth=0.8, linestyle=":")
    axes[1].text(39.4, 0.07, "5% threshold", color=INK2, fontsize=8, ha="right")
    axes[1].set_xlabel("Touch sensor index (groups of 10 separated)", color=INK2, fontsize=9)
    axes[1].legend(frameon=False, labelcolor=INK, fontsize=9)
    fig.tight_layout()
    fig.savefig(out / "touch_coverage.png", dpi=170, facecolor=SURFACE)
    print(out / "touch_coverage.png", out / "touch_map_coverage.csv")

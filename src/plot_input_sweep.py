#!/usr/bin/env python
import argparse
import sqlite3
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e5e4e0"
RAMP = {3: "#8ab8f0", 5: "#4a8fdc", 8: "#1f5fb0", 10: "#0b3470"}
SIDES = (3, 5, 8, 10)
ENCODINGS = (("linear", "linear (current)"), ("wta_block", "winner-take-all per block"))
PANELS = [
    ("Internal alignment", "grid_comp_mean"),
    ("Success rate", "eval_episode_success_rate"),
    ("MI score context-policy", "mi_score_context_policy"),
    ("MI score context-touch", "mi_score_context_touch"),
]
BASELINE_RUNS = ("parasite_000", "parasite_001", "parasite_002")
EPOCHS = 500


def series(con, side, encoding, key):
    if side == 5 and encoding == "linear":
        where, params = f"r.name IN ({','.join('?' * len(BASELINE_RUNS))})", list(BASELINE_RUNS)
    else:
        where, params = "r.grp = ?", [f"k{side}_{encoding}"]
    df = pd.read_sql(
        "SELECT r.name, m.step, m.value FROM metrics m JOIN runs r ON r.id = m.run_id"
        f" WHERE {where} AND m.key = ? AND m.step < {EPOCHS}",
        con,
        params=params + [key],
    )
    if df.empty:
        return None
    return df.pivot(index="step", columns="name", values="value")


def style(ax):
    ax.set_facecolor(SURFACE)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=8)


def frames_per_epoch(root, side, encoding):
    out = []
    if side == 5 and encoding == "linear":
        dirs = [Path(root).parent / n for n in BASELINE_RUNS]
    else:
        dirs = sorted(Path(root).glob(f"agent_k{side}_{encoding}_*"))
    for d in dirs:
        f = d / "internal_trajectory_data.csv"
        if not f.is_file():
            continue
        df = pd.read_csv(f, usecols=["epoch", "sensory_change"], nrows=EPOCHS * 3600)
        per_epoch = df.groupby("epoch").sensory_change.sum()
        out.append((per_epoch.loc[0:99].mean(), per_epoch.loc[400:499].mean()))
    return np.array(out)


def parse_arguments():
    parser = argparse.ArgumentParser(description="Plot the input-map sweep (resolution x encoding)")
    parser.add_argument("--db", required=True)
    parser.add_argument("--root", required=True, help="Sweep folder with agent_k*_* run folders")
    parser.add_argument("--out", required=True, help="Folder for input_sweep_fig.png and input_sweep_table.csv")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    con = sqlite3.connect(args.db)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 4, figsize=(16, 7), facecolor=SURFACE, sharex=True)
    rows = []
    for r, (encoding, label) in enumerate(ENCODINGS):
        for c, (title, key) in enumerate(PANELS):
            ax = axes[r, c]
            style(ax)
            if r == 0:
                ax.set_title(title, color=INK, fontsize=11, loc="left")
            if c == 0:
                ax.set_ylabel(label, color=INK, fontsize=10)
            if r == 1:
                ax.set_xlabel("Epoch", color=INK2, fontsize=9)
            for side in SIDES:
                wide = series(con, side, encoding, key)
                if wide is None:
                    continue
                mean, sd = wide.mean(axis=1), wide.std(axis=1)
                ax.fill_between(mean.index, mean - sd, mean + sd, color=RAMP[side], alpha=0.14, linewidth=0)
                ax.plot(mean.index, mean, color=RAMP[side], linewidth=1.8, label=f"{side}x{side}")
                for window, lo, hi in (("epochs 0-99", 0, 100), ("epochs 400-499", 400, 500)):
                    per_seed = wide[(wide.index >= lo) & (wide.index < hi)].mean()
                    rows.append(dict(encoding=encoding, side=side, metric=key, window=window,
                                     mean=per_seed.mean(), sd=per_seed.std(), seeds=int(per_seed.notna().sum())))
            ax.set_xlim(0, EPOCHS)
            if key != "grid_comp_mean":
                ax.set_ylim(bottom=0)
    handles = [plt.Line2D([], [], color=RAMP[s], linewidth=2) for s in SIDES]
    fig.legend(handles, [f"{s}x{s} input maps" for s in SIDES], loc="upper right", ncol=4, frameon=False,
               labelcolor=INK, fontsize=10, bbox_to_anchor=(0.995, 1.0))
    fig.text(0.005, 0.005, "Mean and standard deviation over 3 seeds (baseline 5x5 linear: existing runs, seeds 0-2).",
             color=INK2, fontsize=8)
    fig.tight_layout(rect=(0, 0.02, 1, 0.97))
    fig.savefig(out / "input_sweep_fig.png", dpi=170, facecolor=SURFACE)
    table = pd.DataFrame(rows)
    frames = []
    for encoding, _ in ENCODINGS:
        for side in SIDES:
            f = frames_per_epoch(args.root, side, encoding)
            if len(f):
                for i, window in enumerate(("epochs 0-99", "epochs 400-499")):
                    frames.append(dict(encoding=encoding, side=side, metric="success_frames_per_epoch", window=window,
                                       mean=f[:, i].mean(), sd=f[:, i].std(ddof=1) if len(f) > 1 else np.nan, seeds=len(f)))
    table = pd.concat([table, pd.DataFrame(frames)], ignore_index=True)
    table.round(4).to_csv(out / "input_sweep_table.csv", index=False)
    print(table.pivot_table(index=["metric", "encoding", "side"], columns="window", values="mean").round(3).to_string())

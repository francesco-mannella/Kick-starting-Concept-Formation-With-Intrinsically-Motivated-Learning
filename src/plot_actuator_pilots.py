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
COLORS = ["#9ca3af", "#eb6834", "#2a78d6", "#1baf7a", "#9c36b5", "#c2255c", "#8a6d00"]
PANELS = [("Success rate (evaluation)", "eval_episode_success_rate"), ("Internal alignment", "grid_comp_mean"),
          ("Touch frequency", "touch_freq"), ("MI context-touch", "mi_score_context_touch")]


def series(con, names, key):
    out = []
    for name in names:
        df = pd.read_sql("SELECT m.step, m.value FROM metrics m JOIN runs r ON r.id = m.run_id"
                         " WHERE r.name = ? AND m.key = ? ORDER BY m.step", con, params=(name, key))
        out.append(df.set_index("step").value)
    return pd.concat(out, axis=1).sort_index()


def parse_arguments():
    parser = argparse.ArgumentParser(description="Learning curves of actuator pilot conditions vs baseline")
    parser.add_argument("--db", default="test/demo_local/runs.db")
    parser.add_argument("--prefix", default="", help="e.g. pilot2: run names agent_<prefix>_<cond>_<seed>")
    parser.add_argument("--conditions", default="A_oldmap,A,B,C,D")
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--baseline", default="parasite_{seed:03d}")
    parser.add_argument("--out", default="test/demo_local/figures")
    parser.add_argument("--label", default=None)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    con = sqlite3.connect(args.db)
    seeds = [int(s) for s in args.seeds.split(",")]
    groups = {"baseline (old actuator)": [args.baseline.format(seed=s) for s in seeds]}
    for c in args.conditions.split(","):
        groups[c] = ["_".join(filter(None, ["agent", args.prefix, c])) + f"_{s:03d}" for s in seeds]
    label = args.label or args.prefix or "pilot"
    fig, axes = plt.subplots(1, 4, figsize=(17, 4), facecolor=SURFACE)
    rows = []
    for ax, (title, key) in zip(axes, PANELS):
        ax.set_facecolor(SURFACE)
        ax.grid(axis="y", color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(GRID)
        ax.tick_params(colors=INK2, labelsize=9)
        ax.set_title(title, color=INK, fontsize=11, loc="left")
        ax.set_xlabel("Epoch", color=INK2, fontsize=9)
        for color, (name, runs) in zip(COLORS, groups.items()):
            df = series(con, runs, key)
            df = df[df.index < args.epochs]
            if key in ("grid_comp_mean", "touch_freq"):
                df = df.rolling(5, min_periods=1).mean()
            mean, sd = df.mean(axis=1), df.std(axis=1)
            ax.fill_between(mean.index, mean - sd, mean + sd, color=color, alpha=0.12, linewidth=0)
            ax.plot(mean.index, mean, color=color, linewidth=1.8, label=name)
            for lo, hi in ((0, 50), (50, 100), (100, args.epochs)):
                w = df[(df.index >= lo) & (df.index < hi)].mean(axis=0)
                rows.append({"condition": name, "metric": key, "window": f"epochs {lo}-{hi}", "mean": w.mean(),
                             "sd": w.std(), "seeds": int(w.notna().sum())})
        ax.set_xlim(0, args.epochs)
    axes[0].set_ylim(0, 1)
    fig.legend(*axes[0].get_legend_handles_labels(), loc="upper right", ncol=len(groups), frameon=False,
               labelcolor=INK, fontsize=9, bbox_to_anchor=(0.995, 1.0))
    fig.text(0.005, 0.01, f"Mean and sd over {len(seeds)} seeds; touch frequency and alignment smoothed over 5 epochs.",
             color=INK2, fontsize=8)
    fig.tight_layout(rect=(0, 0.04, 1, 0.92))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / f"actuator_{label}_fig.png", dpi=170, facecolor=SURFACE)
    table = pd.DataFrame(rows).round(4)
    table.to_csv(out / f"actuator_{label}_table.csv", index=False)
    print(table.pivot_table(index="condition", columns=["metric", "window"], values="mean", sort=False).round(3).to_string())

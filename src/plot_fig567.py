#!/usr/bin/env python
import argparse
import sqlite3
from pathlib import Path

import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e5e4e0"
COLORS = {"agent": "#2a78d6", "parasite": "#eb6834", "former parasite": "#1baf7a"}

PANELS = [
    ("Internal alignment", "grid_comp_mean", None),
    ("Success rate", "eval_episode_success_rate", (0, 1)),
    ("MI score context-policy", "mi_score_context_policy", None),
    ("MI score context-touch", "mi_score_context_touch", None),
]

FIGURES = {
    5: [("agent", "parasite", "")],
    6: [("agent", "parasite", ""), ("parasite", "parasite", "_par")],
    7: [("agent", "parasite", ""), ("former parasite", "former", "")],
    "6_noagent": [("parasite", "parasite", "_par")],
    "7_noagent": [("former parasite", "former", "")],
}


def series(con, group, key):
    df = pd.read_sql(
        "SELECT r.name, m.step, m.value FROM metrics m JOIN runs r ON r.id = m.run_id"
        " WHERE r.grp = ? AND m.key = ?",
        con,
        params=(group, key),
    )
    df["seed"] = df["name"].str[-3:].astype(int)
    wide = df.pivot(index="step", columns="seed", values="value")
    return wide.mean(axis=1), wide.std(axis=1), wide.shape[1]


def style(ax):
    ax.set_facecolor(SURFACE)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=9)


def panel_tops(con):
    tops = {}
    for _, key, _ in PANELS:
        for series_list in FIGURES.values():
            for _, group, suffix in series_list:
                mean, sd, _ = series(con, group, key + suffix)
                tops[key] = max(tops.get(key, 0), (mean + sd).max())
    return tops


def make_figure(con, number, out_dir, table, tops):
    fig, axes = plt.subplots(1, 4, figsize=(16, 3.8), facecolor=SURFACE)
    handles, nseeds = {}, set()
    for ax, (title, key, ylim) in zip(axes, PANELS):
        style(ax)
        ax.set_title(title, color=INK, fontsize=11, loc="left")
        ax.set_xlabel("Epoch", color=INK2, fontsize=9)
        for label, group, suffix in FIGURES[number]:
            name = "parasite" if suffix else label
            mean, sd, n = series(con, group, key + suffix)
            nseeds.add(n)
            color = COLORS[name]
            ax.fill_between(mean.index, mean - sd, mean + sd, color=color, alpha=0.16,
                            linewidth=0)
            (line,) = ax.plot(mean.index, mean, color=color, linewidth=1.8, label=name)
            handles[name] = line
            for step, m, s in zip(mean.index, mean, sd):
                table.append((number, title, name, step, m, s, n))
        ax.set_xlim(0, 1500)
        if ylim:
            ax.set_ylim(*ylim)
        else:
            ax.set_ylim(0, tops[key] * 1.02)
    fig.legend(handles.values(), handles.keys(), loc="upper right", ncol=len(handles),
               frameon=False, labelcolor=INK, fontsize=10, bbox_to_anchor=(0.995, 1.02))
    fig.text(0.005, 0.01, f"Mean and standard deviation over {max(nseeds)} replications.",
             color=INK2, fontsize=8)
    fig.tight_layout(rect=(0, 0.04, 1, 0.97))
    path = out_dir / f"fig{number}.png"
    fig.savefig(path, dpi=200, facecolor=SURFACE)
    plt.close(fig)
    return path


def parse_arguments():
    parser = argparse.ArgumentParser(description="Plot Figs 5-7 from the local database")
    parser.add_argument("--db", required=True)
    parser.add_argument("--out", default="figures")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(args.db)
    table = []
    tops = panel_tops(con)
    for number in FIGURES:
        print(make_figure(con, number, out_dir, table, tops))
    columns = ["figure", "panel", "series", "epoch", "mean", "sd", "replications"]
    pd.DataFrame(table, columns=columns).to_csv(out_dir / "fig567_table.csv", index=False)
    print(out_dir / "fig567_table.csv")

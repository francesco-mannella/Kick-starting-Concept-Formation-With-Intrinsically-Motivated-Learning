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

PANELS8 = [
    ("Own-size posture", "eval_early_nearest_own_intact_s{step}", (0, 1)),
    ("Context information (eta2 excess)", "eval_early_eta2_ctx_excess_s{step}", (-0.1, 0.8)),
    ("Episodes not yet touched", "eval_early_free_s{step}", (0, 1)),
    ("Swap cost (within minus swap)", "eval_early_swap_cost_s{step}", (-0.5, 1)),
]

EARLY_STEPS = (10, 20)

FIGURES = {
    5: [("agent", "parasite", "")],
    6: [("agent", "parasite", ""), ("parasite", "parasite", "_par")],
    7: [("agent", "parasite", ""), ("former parasite", "former", "")],
    8: [("agent", "parasite", ""), ("parasite", "parasite", "_par"), ("former parasite", "former", "")],
    "6_noagent": [("parasite", "parasite", "_par")],
    "7_all": [("agent", "parasite", ""), ("parasite", "parasite", "_par"),
              ("former parasite", "former", "")],
    "7_noagent": [("former parasite", "former", "")],
}


def series(con, group, key):
    df = pd.read_sql(
        "SELECT r.name, m.step, m.value FROM metrics m JOIN runs r ON r.id = m.run_id"
        " WHERE r.grp = ? AND m.key = ?",
        con,
        params=(group, key),
    )
    if df.empty:
        return pd.Series(dtype=float), pd.Series(dtype=float), 0
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


def panel_rows(number):
    if number == 8:
        return [[(f"{title}, step {k}", key.format(step=k), ylim) for title, key, ylim in PANELS8] for k in EARLY_STEPS]
    return [PANELS]


def has_data(con, number):
    keys = [key for row in panel_rows(number) for _, key, _ in row]
    count = con.execute(f"SELECT COUNT(*) FROM metrics WHERE key IN ({','.join('?' * len(keys))})", keys).fetchone()[0]
    return count > 0


def panel_tops(con):
    tops = {}
    for _, key, _ in PANELS:
        for number, series_list in FIGURES.items():
            if number == 8:
                continue
            for _, group, suffix in series_list:
                mean, sd, _ = series(con, group, key + suffix)
                tops[key] = max(tops.get(key, 0), (mean + sd).max())
    return tops


def make_figure(con, number, out_dir, table, tops):
    rows = panel_rows(number)
    fig, axes = plt.subplots(len(rows), 4, figsize=(16, 3.8 * len(rows)), facecolor=SURFACE, squeeze=False)
    handles, nseeds, xmax = {}, set(), [1]
    for ax, (title, key, ylim) in zip(axes.ravel(), [p for row in rows for p in row]):
        style(ax)
        ax.set_title(title, color=INK, fontsize=11, loc="left")
        ax.set_xlabel("Epoch", color=INK2, fontsize=9)
        for label, group, suffix in FIGURES[number]:
            name = "parasite" if suffix else label
            mean, sd, n = series(con, group, key + suffix)
            if mean.empty:
                ax.text(0.5, 0.5, "not logged in these runs", transform=ax.transAxes, ha="center", color=INK2, fontsize=9)
                continue
            nseeds.add(n)
            xmax[0] = max(xmax[0], mean.index.max())
            color = COLORS[name]
            ax.fill_between(mean.index, mean - sd, mean + sd, color=color, alpha=0.16,
                            linewidth=0)
            (line,) = ax.plot(mean.index, mean, color=color, linewidth=1.8, label=name)
            handles[name] = line
            for step, m, s in zip(mean.index, mean, sd):
                table.append((number, title, name, step, m, s, n))
        ax.set_xlim(0, xmax[0] if number == 8 else 1500)
        if ylim:
            ax.set_ylim(*ylim)
        else:
            ax.set_ylim(0, tops[key] * 1.02)
    fig.legend(handles.values(), handles.keys(), loc="upper right", ncol=len(handles),
               frameon=False, labelcolor=INK, fontsize=10, bbox_to_anchor=(0.995, 1 + 0.02 / len(rows)))
    fig.text(0.005, 0.01, f"Mean and standard deviation over {max(nseeds)} replications.",
             color=INK2, fontsize=8)
    fig.tight_layout(rect=(0, 0.04 / len(rows), 1, 1 - 0.03 / len(rows)))
    path = out_dir / f"fig{number}.png"
    fig.savefig(path, dpi=200, facecolor=SURFACE)
    plt.close(fig)
    return path


def parse_arguments():
    parser = argparse.ArgumentParser(description="Plot Figs 5-7 from the local database")
    parser.add_argument("--db", required=True)
    parser.add_argument("--out", default="figures")
    parser.add_argument("--early-steps", type=int, nargs="+", default=[10, 20], help="Steps after the first policy onset shown in Fig 8, one row each")
    parser.add_argument("--only", nargs="+", default=None, help="Figures to draw, e.g. --only 8 (default: all)")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(args.db)
    EARLY_STEPS = tuple(args.early_steps)
    table = []
    tops = panel_tops(con)
    groups = {g for (g,) in con.execute("SELECT DISTINCT grp FROM runs")}
    if "agent" in groups and not groups & {"parasite", "former"}:
        FIGURES[8] = [("agent", "agent", "")]
    for number in FIGURES:
        if args.only and str(number) not in args.only:
            continue
        if number == 8 and not has_data(con, number):
            print("fig8 skipped: no eval_early_* metrics in the database")
            continue
        print(make_figure(con, number, out_dir, table, tops))
    columns = ["figure", "panel", "series", "epoch", "mean", "sd", "replications"]
    pd.DataFrame(table, columns=columns).to_csv(out_dir / "fig567_table.csv", index=False)
    print(out_dir / "fig567_table.csv")

#!/usr/bin/env python
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.ndimage import maximum_filter1d

WINDOWS = [(0, 30), (30, 150), (150, 600), (600, 1500)]
BINS = ["exact (0)", "exactly +-1", "exactly +-2", "3 to 4", "5 or more, active succeeds in the episode",
        "active has no success in the episode"]
EPISODES, STEPS = 18, 200


def load(path):
    flags = pd.read_csv(path, usecols=["sensory_change"]).sensory_change.values.astype(bool)
    return flags.reshape(-1, EPISODES, STEPS)


def dilate(flags, k):
    if k == 0:
        return flags
    return maximum_filter1d(flags.astype(np.uint8), size=2 * k + 1, axis=2, mode="constant").astype(bool)


def count_bins(root):
    counts = {w: np.zeros(7) for w in WINDOWS}
    for run in sorted(Path(root).glob("parasite_*")):
        active = load(run / "internal_trajectory_data.csv")
        parasite = load(run / "internal_trajectory_data_parasite.csv")
        d0, d1, d2, d4 = (dilate(active, k) for k in (0, 1, 2, 4))
        for lo, hi in WINDOWS:
            p = parasite[lo:hi]
            a0, a1, a2, a4 = d0[lo:hi], d1[lo:hi], d2[lo:hi], d4[lo:hi]
            has = active[lo:hi].any(axis=2, keepdims=True)
            counts[(lo, hi)] += [(p & a0).sum(), (p & a1 & ~a0).sum(), (p & a2 & ~a1).sum(),
                                 (p & a4 & ~a2).sum(), (p & ~a4 & has).sum(), (p & ~has).sum(),
                                 p.sum()]
    return counts


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Percentage of parasite success frames by distance to the nearest"
        " active-agent success frame in the same episode"
    )
    parser.add_argument("--root", required=True, help="Experiment folder with parasite_NNN runs")
    parser.add_argument("--out", required=True, help="Folder for success_sync_bins.csv")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    counts = count_bins(args.root)
    counts["all epochs"] = sum(counts.values())
    table = {}
    for key, c in counts.items():
        name = key if isinstance(key, str) else f"epochs {key[0]}-{key[1]}"
        table[name] = list((100 * c[:6] / c[6]).round(1)) + [int(c[6])]
    table = pd.DataFrame(table, index=[f"% of parasite success frames: {b}" for b in BINS]
                         + ["parasite success frames (n)"])
    table = table.astype(object)
    table.iloc[-1] = table.iloc[-1].astype(int)
    table.index.name = "distance to nearest active success"
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    table.to_csv(out / "success_sync_bins.csv")
    print(table.to_string())

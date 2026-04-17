#!/usr/bin/env python

import argparse
import glob

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import regex as re
from PIL import Image

import subprocess

rep = 0

# Query plot_postures.py for available posture schedule
result = subprocess.run(["plot_postures.py", "-l"], capture_output=True, text=True)
posture_schedule = result.stdout.strip().split("\n")[-18:]
print(result.stdout)




# Constants for tile dimensions
TILE_WIDTH, TILE_HEIGHT = 350, 350

# Regex pattern to extract information from filenames
FILENAME_PATTERN = r"e(\d+)_g(\d+)_r(\d+)_(.)(.).png"

# Argument parsing for repetition selection
parser = argparse.ArgumentParser(description="Process image files.")
parser.add_argument(
    "-r", "--rep", type=int, default=0, help="Repetition number to process"
)
args = parser.parse_args()

# Get a sorted list of image files matching the pattern
image_files = sorted(glob.glob("e[0-9]*.png"))

# Parse information from filenames into a list of rows
parsed_rows = [
    [int(v) for v in re.findall(r"\d+", f)[:3]]
    + list(map(int, list(re.search(r"_(.)(.)\.png$", f).groups())))
    + [f]
    for f in image_files
]

# Create a DataFrame from parsed rows
df = pd.DataFrame(parsed_rows, columns=["episode", "goal", "rep", "x", "y", "file"])

# Filter DataFrame by the specified repetition
df = df[df["rep"] == args.rep]

# Group by 'rep' and 'episode', filter goals, and update DataFrame
group = df.groupby(["rep", "episode"])["goal"].apply(
    lambda x: x[(x > 1) & (x == x.max())]
)
mask = df.index.isin(group.index.get_level_values(-1))
df = df[~mask]
group = df.groupby(["rep", "episode"])["goal"].apply(
    lambda x: x[(x > 1) & (x == x.max())]
)
mask = df.index.isin(group.index.get_level_values(-1))
df = df[~mask]

# Create a dictionary of image frames
frames = {
    tuple(int(k) for k in idx): np.array(Image.open(grp.file.iloc[0]))[100:450, 150:500]
    for idx, grp in df.groupby(["episode", "goal", "rep"])
}

# Set up a 10x10 grid of subplots
fig, axes = plt.subplots(10, 10, figsize=(6, 6), sharex=True, sharey=True)

# Configure each subplot
for ax in axes.flatten():
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect("equal")

# Populate the grid with images based on DataFrame grouping
for idx, data in df.groupby(["rep", "x", "y"]):
    d = data[data.goal == data.goal.min()]
    episode, goal, rep, x, y = d.iloc[0, :-1]
    axes[9 - x, y].imshow(frames[(episode, goal, rep)], aspect="equal")

# Adjust layout and save the figure
fig.tight_layout(pad=0.01)
fig.savefig("goalgrid.png", dpi=400)

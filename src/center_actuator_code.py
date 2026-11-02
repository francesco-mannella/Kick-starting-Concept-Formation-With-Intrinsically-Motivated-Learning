#!/usr/bin/env python
import argparse
import sys
from pathlib import Path

import numpy as np

SRC = Path(__file__).resolve().parent
sys.path.insert(0, str(SRC))

from Actuator import Actuator  # noqa: E402
from grip_map_v2 import BLOCKS, N_INPUTS, load_data  # noqa: E402


def build_actuator(map_name, norm_name, side=5, touch_gain=3.0, n_out=8):
    weights = np.load(SRC / f"{map_name}.npy")
    return Actuator(None, weights, None, num_inputs=N_INPUTS, num_hidden=3 * side * side, num_outputs=n_out, encoding="rbf_block",
                    input_norm="block_rms", norm_name=norm_name, touch_gain=touch_gain, output="tanh",
                    input_blocks=(BLOCKS[0][1], BLOCKS[1][1] - BLOCKS[1][0], BLOCKS[2][1] - BLOCKS[2][0]))


def main():
    parser = argparse.ArgumentParser(description="Add the mean of the actuator code (centring vectors) to a copy of the norm file")
    parser.add_argument("--data", nargs="+", required=True, help=".npz babbling files")
    parser.add_argument("--map", default="data/StoredGripActuatorMap_v4")
    parser.add_argument("--norm", default="data/StoredGripActuatorNorm_v4")
    parser.add_argument("--out", required=True, help="output norm file (.npz) under src/data")
    parser.add_argument("--frames", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    actuator = build_actuator(args.map, args.norm)
    data = load_data(args.data)
    rng = np.random.RandomState(args.seed)
    states = np.hstack([data["touch"], data["posture"], data["eye"]])
    states = states[rng.choice(len(states), min(args.frames, len(states)), replace=False)]
    codes = np.array([actuator.hidden(s) for s in states])
    units = actuator.block_units
    zero = actuator.hidden(np.zeros(N_INPUTS))
    mean_all = codes.mean(0)
    mean_all[:units] = zero[:units]
    mean_touch = np.zeros_like(mean_all)
    mean_touch[:units] = zero[:units]
    stats = dict(np.load(SRC / f"{args.norm}.npz"))
    stats["hidden_mean_all"] = mean_all.astype(np.float32)
    stats["hidden_mean_touch"] = mean_touch.astype(np.float32)
    np.savez(SRC / f"{args.out}.npz", **stats)
    print("block sums of the code (mean over babbling):", [round(float(codes[:, i * units:(i + 1) * units].sum(1).mean()), 3) for i in range(3)])
    print("block sums of the centring vectors:", [round(float(mean_all[i * units:(i + 1) * units].sum()), 3) for i in range(3)])


if __name__ == "__main__":
    main()

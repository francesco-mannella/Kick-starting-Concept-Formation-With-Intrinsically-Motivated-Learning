#!/usr/bin/env python
import argparse
from pathlib import Path

import numpy as np

from Actuator import Actuator
from grip_map_v2 import balanced_touch, load_data

SRC = Path(__file__).resolve().parent


def build_hidden(args, rng):
    data = load_data(args.data)
    touch = data["touch"]
    states = np.hstack([touch, data["posture"], data["eye"]])
    uniform = states[rng.choice(len(states), args.n_samples, replace=False)]
    contact = balanced_touch(states[touch.sum(1) > 0], args.n_samples, rng)
    states = np.vstack([uniform, contact])
    actuator = Actuator(
        None, np.load((SRC / args.map).with_suffix(".npy")), None,
        num_inputs=50, num_hidden=75, num_outputs=8, encoding=args.encoding, input_norm="block_rms",
        output="tanh", norm_name=args.norm, touch_gain=args.touch_gain, input_blocks=(40, 8, 2),
    )
    return np.array([actuator.hidden(x) for x in states])


def parse_arguments():
    parser = argparse.ArgumentParser(description="Whitened PCA basis of the actuator hidden activity")
    parser.add_argument("--data", required=True, nargs="+", help=".npz babbling files from grip_map_v2.py collect")
    parser.add_argument("--map", default="data/StoredGripActuatorMap_v4")
    parser.add_argument("--norm", default="data/StoredGripActuatorNorm_v4")
    parser.add_argument("--encoding", default="rbf_block")
    parser.add_argument("--touch-gain", type=float, default=3.0)
    parser.add_argument("--n-samples", type=int, default=20000)
    parser.add_argument("--share", type=float, default=0.99, help="Fraction of hidden second moment to keep")
    parser.add_argument("--max-dim", type=int, default=32)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", required=True)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    hidden = build_hidden(args, np.random.RandomState(args.seed))
    vals, vecs = np.linalg.eigh(hidden.T @ hidden / len(hidden))
    vals, vecs = vals[::-1].clip(1e-12), vecs[:, ::-1]
    k = min(int(np.searchsorted(np.cumsum(vals) / vals.sum(), args.share) + 1), args.max_dim)
    np.savez(args.out, basis=vecs[:, :k], scale=np.sqrt(vals[:k]))
    print(f"kept {k} of 75 components, share {vals[:k].sum() / vals.sum():.4f}, smallest eigenvalue ratio {vals[k - 1] / vals[0]:.2e}")

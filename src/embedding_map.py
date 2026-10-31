#!/usr/bin/env python
import argparse

import numpy as np

from grip_map_v2 import BLOCKS, N_INPUTS, balanced_touch, block_stats, load_data


def build_embedding(data, rng, n_touch=5, extra_touch=None, balance=True):
    mean, scale = block_stats(data)
    frames = data["touch"][data["touch"].sum(1) > 0]
    if balance:
        frames = balanced_touch(frames, min(len(frames) * 2, 20000), rng)
    if extra_touch is not None:
        frames = np.vstack([frames, extra_touch])
    z = (frames - mean[: BLOCKS[0][1]]) / scale[: BLOCKS[0][1]]
    touch_mean = z.mean(0)
    _, sv, vt = np.linalg.svd(z - touch_mean, full_matrices=False)
    touch_scale = sv[:n_touch] / np.sqrt(len(z))
    return {"mean": mean, "scale": scale, "touch_mean": touch_mean.astype(np.float32),
            "touch_basis": vt[:n_touch].T.astype(np.float32), "touch_scale": touch_scale.astype(np.float32),
            "gain": np.ones(n_touch + N_INPUTS - BLOCKS[0][1], np.float32)}


def parse_arguments():
    parser = argparse.ArgumentParser(description="Embedding code: whitened touch PCA plus standardised joints and eye")
    parser.add_argument("--data", nargs="+", required=True, help=".npz babbling files")
    parser.add_argument("--visited", nargs="*", default=[], help=".npy arrays of visited 50-d states; their contact frames are pooled")
    parser.add_argument("--n-touch", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", required=True, help="output .npz")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    extra = None
    if args.visited:
        states = np.vstack([np.load(p) for p in args.visited])
        extra = states[states[:, :40].sum(1) > 0][:, :40]
    stats = build_embedding(load_data(args.data), np.random.RandomState(args.seed), args.n_touch, extra)
    np.savez(args.out, **stats)

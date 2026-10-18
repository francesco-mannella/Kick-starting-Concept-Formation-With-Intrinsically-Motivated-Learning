#!/usr/bin/env python
import argparse

import numpy as np
import torch

from GripMapping import PrototypeGenerator, params
from SMEnv import SMEnv

SENSOR_GROUPS = [slice(10 * g, 10 * g + 10) for g in range(4)]
BLOCKS = [(0, 40), (40, 45), (45, 47)]


def collect_babbling(env, trials, stime, rng, near_steps=10):
    """Babbles in the agent's own action space against randomised objects.

    Actuator outputs u in [-1, 1]^5 are mapped exactly as in SMAgent.step, so
    the arm only visits postures the controller can command. Returns touch
    (n, 40), posture (n, 5), eye position (n, 2; reversed order as at run
    time) and near (n,): frames within `near_steps` of a touch in the same trial.
    """
    touch, posture, eye, near = [], [], [], []
    sign = np.array([-1, -1, -1, 1, 1])
    base = np.hstack([params.policy_base_arm, np.ones(2) * params.policy_base])
    for _ in range(trials):
        env.b2d_env.rand_obj_params = {
            "stretch_conditions": [rng.choice([1, 1.5, 2])],
            "rotation_conditions": [rng.choice(np.pi * np.array([0, 0.25, 0.5]))],
            "pos": [params.obj_y + rng.uniform(-2, 2), params.obj_x + rng.uniform(-2, 3)],
        }
        env.reset(world=rng.choice([1, 2, 3]))
        u = np.zeros(5) if rng.rand() < 0.5 else rng.uniform(-0.7, 0.7, 5)
        first = len(touch)
        for t in range(stime):
            if t % 5 == 0:
                u = np.clip(u + rng.randn(5) * 0.35, -1, 1)
            state = env.step(u * sign + base)
            touch.append(state["TOUCH_SENSORS"].copy())
            posture.append(state["JOINT_POSITIONS"][:5].copy())
            eye.append(state["EYE_POS"][::-1] + rng.randn(2))
        touched = np.array([x.sum() > 0 for x in touch[first:]])
        near.extend(np.convolve(touched, np.ones(2 * near_steps + 1), "same") > 0)
    out = {k: np.array(v, np.float32) for k, v in (("touch", touch), ("posture", posture), ("eye", eye))}
    out["near"] = np.array(near, bool)
    return out


def balanced_touch(frames, n_samples, rng):
    """Resamples touch frames so each sensor group carries equal total mass."""
    mass = np.stack([frames[:, g].sum(1) for g in SENSOR_GROUPS], 1)
    weight = (mass / (mass.sum(0) + 1e-9)).sum(1)
    return frames[rng.choice(len(frames), n_samples, p=weight / weight.sum())]


def synthetic_touch(n_samples, rng, n_sensors=40):
    """Contiguous contact patches at random positions along the sensor order."""
    out = np.zeros((n_samples, n_sensors), np.float32)
    for i in range(n_samples):
        width = rng.randint(2, 7)
        start = rng.randint(0, n_sensors - width + 1)
        profile = np.exp(-0.5 * ((np.arange(width) - (width - 1) / 2) / (width / 4)) ** 2)
        out[i, start : start + width] = 1.5 * rng.uniform(0.7, 1.3) * profile
    return out


def block_stats(data):
    touch, posture, eye = data["touch"], data["posture"], data["eye"]
    mean = np.zeros(47, np.float32)
    mean[40:45], mean[45:] = posture.mean(0), eye.mean(0)
    scale = np.ones(47, np.float32)
    scale[:40] = np.sqrt((touch ** 2).mean())
    scale[40:45] = np.sqrt(((posture - mean[40:45]) ** 2).mean())
    scale[45:] = np.sqrt(((eye - mean[45:]) ** 2).mean())
    return mean, scale


def build_maps(data, side, rng, balance=True, synthetic_frac=0.0, max_frames=50000):
    """Trains normalised block maps; returns the 47 x (3 * side^2) map and stats."""
    units = side * side
    mean, scale = block_stats(data)
    frames = data["touch"][data["touch"].sum(1) > 0]
    if balance:
        frames = balanced_touch(frames, min(len(frames) * 2, 20000), rng)
    if synthetic_frac > 0:
        frames = np.vstack([frames, synthetic_touch(int(len(frames) * synthetic_frac), rng)])
    subset = lambda x: x[rng.choice(len(x), min(len(x), max_frames), replace=False)]
    posture = subset(data["posture"])
    if "near" in data and data["near"].any():
        near = data["posture"][data["near"]]
        posture = np.vstack([subset(near)[: max_frames // 2], posture[: max_frames // 2]])
    sources = [frames, posture, subset(data["eye"])]
    weights = np.zeros((47, 3 * units))
    for b, ((lo, hi), x) in enumerate(zip(BLOCKS, sources)):
        z = ((x - mean[lo:hi]) / scale[lo:hi]).astype(np.float32)
        weights[lo:hi, b * units : (b + 1) * units] = PrototypeGenerator(hi - lo, units, z)()

    everything = np.hstack([data["touch"], data["posture"], data["eye"]])
    z_all = ((everything - mean) / scale)[rng.choice(len(everything), min(len(everything), 20000), replace=False)]
    coords = np.stack(np.meshgrid(np.arange(side), np.arange(side))).reshape(2, -1).T
    bumps = np.exp(-0.5 * ((coords[:, None] - coords[None]) ** 2).sum(-1) / (side / 5) ** 2)
    stats = {"mean": mean, "scale": scale, "gain_linear": np.ones(3), "gain_wta_block": np.ones(3),
             "gain_rbf_block": np.ones(3), "rbf_sigma": np.ones(3)}
    for b, (lo, hi) in enumerate(BLOCKS):
        proto = weights[lo:hi, b * units : (b + 1) * units]
        z = z_all[:, lo:hi]
        stats["gain_linear"][b] = 1 / np.sqrt(((z @ proto) ** 2).mean())
        dist = ((z[:, :, None] - proto[None]) ** 2).sum(1)
        stats["gain_wta_block"][b] = 1 / np.sqrt((bumps[dist.argmin(1)] ** 2).mean())
        spacing = ((proto[:, :, None] - proto[:, None]) ** 2).sum(0)
        np.fill_diagonal(spacing, np.inf)
        stats["rbf_sigma"][b] = np.sqrt(np.median(spacing.min(1)))
        act = np.exp(-0.5 * dist / stats["rbf_sigma"][b] ** 2)
        act /= act.sum(1, keepdims=True) + 1e-12
        stats["gain_rbf_block"][b] = 1 / np.sqrt((act ** 2).mean())
    return weights, stats


def load_data(paths):
    parts = []
    for path in paths:
        if str(path).endswith(".npy"):
            raw = np.load(path)
            parts.append({"touch": raw[:, 300:340], "posture": raw[:, 340:345], "eye": raw[:, 345:347]})
        else:
            parts.append(dict(np.load(path)))
    keys = ("touch", "posture", "eye") + (("near",) if all("near" in p for p in parts) else ())
    return {k: np.concatenate([p[k] for p in parts]) for k in keys}


def parse_arguments():
    parser = argparse.ArgumentParser(description="Normalised, touch-balanced grip input maps")
    sub = parser.add_subparsers(dest="command", required=True)
    c = sub.add_parser("collect", help="babble the environment, store touch/posture/eye")
    c.add_argument("--trials", type=int, default=500)
    c.add_argument("--stime", type=int, default=100)
    c.add_argument("--seed", type=int, default=0)
    c.add_argument("--out", required=True)
    b = sub.add_parser("build", help="train the normalised block maps")
    b.add_argument("--data", nargs="+", required=True, help=".npz babbling files or the old .npy")
    b.add_argument("--side", type=int, default=5)
    b.add_argument("--seed", type=int, default=0)
    b.add_argument("--no-balance", action="store_true")
    b.add_argument("--synthetic-frac", type=float, default=0.0)
    b.add_argument("--out-map", required=True)
    b.add_argument("--out-norm", required=True)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    rng = np.random.RandomState(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.command == "collect":
        np.savez(args.out, **collect_babbling(SMEnv(args.seed, params), args.trials, args.stime, rng))
    else:
        weights, stats = build_maps(load_data(args.data), args.side, rng,
                                    balance=not args.no_balance, synthetic_frac=args.synthetic_frac)
        np.save(args.out_map, weights)
        np.savez(args.out_norm, **stats)

#!/usr/bin/env python
import argparse
import pickle
from pathlib import Path

import matplotlib
import numpy as np
import torch
from matplotlib.patches import Polygon, Rectangle
from sklearn.decomposition import PCA

matplotlib.use("agg")
import matplotlib.pyplot as plt  # noqa: E402

from SMGraphs import GraphManager  # noqa: E402
from SMMain import Main  # noqa: E402

_ = Main

OBJECTS = ("movable", "controllable", "still", "unreachable")


def rollout(main, policy, world, steps):
    env, state = main.episode_runner.prepare_environment(
        episode_index=0, seed=0, obj_params=main.get_obj_params_for_episode(0), context=world, render=None, plot_path=None
    )
    agent, action_steps = main.agent, main.params.action_steps
    agent.reset()
    agent.updatePolicy(policy)
    action, snapshots = None, {}
    for t in range(1, max(steps) + 1):
        if (t - 1) % action_steps == 0:
            action = agent.step(state)
        state = env.step(action)
        if t in steps:
            sim = env.b2d_env.sim
            snapshots[t] = dict(
                polygons={
                    name: np.vstack([body.GetWorldPoint(v) for v in body.fixtures[0].shape.vertices])
                    for name, body in sim.visible_bodies.items()
                    if name not in OBJECTS
                },
                joints=np.array([sim.joints[name].anchorA for name in env.b2d_env.joint_names]),
            )
    return snapshots


def draw(glyphs, layers, policy, side, path):
    n = len(glyphs)
    hue = PCA(3).fit_transform(policy.T[:n])
    hue = (hue - hue.min(0)) / np.ptp(hue, 0)
    points = np.vstack([q for node in glyphs for s in node.values() for q in s["polygons"].values()])
    lo, hi = points.min(0), points.max(0)
    span = (hi - lo).max()
    place = lambda q, c, i: (q - (lo + hi) / 2) / span * 0.95 + [c, i]
    widths, alphas = np.linspace(1.0, 2.2, len(layers)), np.linspace(0.55, 1.0, len(layers))
    fig, ax = plt.subplots(figsize=(6, 6))
    for j, node in enumerate(glyphs):
        i, c = divmod(j, side)
        ax.add_patch(Rectangle((c - 0.5, i - 0.5), 1, 1, fc=hue[j], alpha=0.55, lw=0))
        ax.add_patch(Rectangle((c - 0.5, i - 0.5), 1, 1, fc="none", ec="white", lw=1.2))
        tile = hue[j] * 0.55 + 0.45
        ghost = "#000000" if 0.299 * tile[0] + 0.587 * tile[1] + 0.114 * tile[2] > 0.6 else "#ffffff"
        for k, step in enumerate(layers):
            last = k == len(layers) - 1
            colour = "#111111" if last else ghost
            for name, q in node[step]["polygons"].items():
                claw = name.startswith("Claw")
                width = widths[k] * (1.25 if claw else 1)
                if last:
                    ax.add_patch(Polygon(place(q, c, i), closed=True, fc="white", ec="white", lw=width + 2.2, joinstyle="round", zorder=2 + 2 * k - 0.5))
                ax.add_patch(Polygon(place(q, c, i), closed=True, fc=colour, ec=colour, lw=width, alpha=alphas[k], joinstyle="round", zorder=2 + 2 * k + claw))
            if last:
                xy = place(node[step]["joints"], c, i)
                ax.scatter(xy[:, 0], xy[:, 1], s=7, fc="white", ec="#111111", lw=0.7, zorder=20)
    ax.set_xlim(-0.5, side - 0.5)
    ax.set_ylim(-0.5, side - 0.5)
    ax.set_aspect("equal")
    ax.set_axis_off()
    fig.savefig(path, dpi=300, facecolor="white", bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def parse_arguments():
    parser = argparse.ArgumentParser(description="Policy map: the arm and gripper after running each node's policy open loop, on tiles coloured by the policy weights")
    parser.add_argument("checkpoint", help="Folder with main.dump.npy and weights.npy (a storage/<epoch> folder of a run)")
    parser.add_argument("--out", default="policy_map")
    parser.add_argument("--layers", default="30,40,50", help="Simulation steps drawn as ghosts (earlier) and the solid glyph (last)")
    parser.add_argument("--world", type=int, default=4, help="Simulator world (4 = no object)")
    parser.add_argument("--all-maps", action="store_true", help="Also save the vision, touch and proprioception maps")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    checkpoint, out = Path(args.checkpoint), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    layers = [int(s) for s in args.layers.split(",")]
    torch.set_default_device("cpu")
    main = np.load(checkpoint / "main.dump.npy", allow_pickle=True)[0]
    main.params.use_wandb = False
    policy = np.load(checkpoint / "weights.npy", allow_pickle=True)[0]["policy"]
    side = int(np.sqrt(main.params.internal_size))
    glyphs = [rollout(main, policy[:, j], args.world, layers) for j in range(side * side)]
    with open(out / "policy_glyphs.pkl", "wb") as f:
        pickle.dump(dict(glyphs=glyphs, layers=layers), f)
    draw(glyphs, layers, policy, side, out / "policy_map.png")
    if args.all_maps:
        manager = GraphManager(None, main.params)
        for name, make in (("vision", manager.visual_map), ("touch", manager.somatosensory_map), ("proprio", manager.proprio_map)):
            fig, ax = plt.subplots(figsize=(6, 6))
            make(str(checkpoint / "weights.npy"), ax=ax)
            fig.savefig(out / f"{name}.png", dpi=300, facecolor="white", bbox_inches="tight", pad_inches=0.02)
            plt.close(fig)
    print(f"saved {out / 'policy_map.png'}")

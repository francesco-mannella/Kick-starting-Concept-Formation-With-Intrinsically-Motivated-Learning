"""Scripted demo of the 8-DoF arm: wrist, each finger on its own, an asymmetric grip.

Action layout: shoulder, elbow, third arm joint, wrist, then proximal and distal joint of finger 1 and of
finger 2. Writes a GIF with the simulator's own offline renderer.
"""
import argparse
from pathlib import Path

import box2dsim
import gymnasium as gym
import matplotlib

matplotlib.use("Agg")
import numpy as np

_ = box2dsim

PI = np.pi
REST = np.zeros(8)
REACH = np.array([0.2 * PI, -0.3 * PI, -0.4 * PI, 0, 0, 0, 0, 0])


def pose(**kargs):
    action = REACH.copy()
    for name, value in kargs.items():
        action[{"wrist": 3, "f1p": 4, "f1d": 5, "f2p": 6, "f2d": 7}[name]] = value
    return action


# (label, target action, steps to reach it)
SCRIPT = [
    ("rest", REST, 20),
    ("arm to the reaching pose", REACH, 60),
    ("wrist turns one way", pose(wrist=0.8), 40),
    ("wrist turns the other way", pose(wrist=-0.8), 60),
    ("wrist back to centre", pose(wrist=0.0), 40),
    ("finger 1 closes alone", pose(f1p=1.2, f1d=1.2), 40),
    ("finger 1 holds", pose(f1p=1.2, f1d=1.2), 15),
    ("finger 1 opens", pose(), 30),
    ("finger 2 closes alone", pose(f2p=1.2, f2d=1.2), 40),
    ("finger 2 holds", pose(f2p=1.2, f2d=1.2), 15),
    ("finger 2 opens", pose(), 30),
    ("different grips: finger 1 proximal, finger 2 distal", pose(f1p=1.3, f1d=0.2, f2p=0.2, f2d=1.3), 45),
    ("wrist turns with the grip closed", pose(wrist=0.6, f1p=1.3, f1d=0.2, f2p=0.2, f2d=1.3), 40),
    ("everything opens", REACH, 40),
]


def timeline():
    current = REST.copy()
    for label, target, steps in SCRIPT:
        for k in range(steps):
            yield label, current + (target - current) * (k + 1) / steps
        current = target


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="dof8_demo.gif")
    parser.add_argument("--world", type=int, default=3)
    parser.add_argument("--every", type=int, default=2, help="render one frame every N steps")
    parser.add_argument("--duration", type=int, default=40, help="frame duration, ms")
    args = parser.parse_args()

    out = Path(args.out).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)

    env = gym.make(
        "Box2DSimOneArmOneEye-v0",
        rand_obj_params={"stretch_conditions": [1], "rotation_conditions": [0.0], "pos": [2, 0.4]},
        render_params={"resolution_prop": 1, "duration": args.duration},
    ).unwrapped
    env.set_seed(0)
    env.set_renderer_figsize((5, 5))
    env.reset(args.world)
    env.render_init("offline")
    env.renderer.ax.set_xlim(-8, 14)
    env.renderer.ax.set_ylim(-2, 30)

    for t, (label, action) in enumerate(timeline()):
        env.step(action)
        if t % args.every == 0:
            env.renderer.ax.set_title(f"{label}\nwrist {action[3]:+.1f}   finger 1 ({action[4]:.1f}, {action[5]:.1f})   finger 2 ({action[6]:.1f}, {action[7]:.1f})", fontsize=8)
            env.render("offline")
    env.renderer.vm.mk_video(name=out.stem, dirname=str(out.parent))
    print(f"{len(env.renderer.vm.frames)} frames -> {out}")


if __name__ == "__main__":
    main()

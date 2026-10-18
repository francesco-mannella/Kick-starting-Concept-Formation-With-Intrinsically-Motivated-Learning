#!/usr/bin/env python
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

SRC = Path(__file__).resolve().parent
sys.path.insert(0, str(SRC))

from diagnose_input_layer import BLOCKS, ESEEDS, HIDDEN, evaluate, make_ablation  # noqa: E402
from SMMain import Main  # noqa: E402  (pickled dumps look up Main in __main__)

_ = Main
SCALES = (1, 0.3, 0.1, 0.03, 0.01, 0.003, 0.001, 0.0001)


def dump(root, cond, seed, epoch):
    name = f"agent_{cond}_{seed:03d}"
    return Path(root) / name / "simulations" / name / "storage" / f"{epoch:06d}" / "main.dump.npy"


def make_recorder(agent):
    grip = agent.grip_agent.grip
    original = agent.step
    records = []

    def out(raw):
        return np.tanh(raw) if grip.output == "tanh" else np.clip(raw, -1, 1)

    def step(state):
        gs = np.hstack([state["TOUCH_SENSORS"], state["JOINT_POSITIONS"][:5], state["EYE_POS"][::-1]]).astype(float)
        hidden = grip.hidden(gs)
        raw = hidden @ grip.params
        touch_part = hidden[HIDDEN["touch"]] @ grip.params[HIDDEN["touch"]]
        gs_off = gs.copy()
        gs_off[BLOCKS["touch"]] = 0
        raw_off = grip.hidden(gs_off) @ grip.params
        records.append(np.hstack([gs, raw, touch_part, out(raw), out(raw_off)]))
        return original(state)

    return step, records


def touch_stats(R, limit):
    gs, raw, tpart, a_on, a_off = R[:, :47], R[:, 47:52], R[:, 52:57], R[:, 57:62], R[:, 62:67]
    active = np.abs(raw).sum(1) > 0
    touch = (gs[:, :40].sum(1) > 0) & active
    stats = {"frames": int(active.sum()), "touch_frames": int(touch.sum()),
             "sat": float((np.abs(raw[active]) >= limit).mean()), "median_abs_raw": float(np.median(np.abs(raw[active])))}
    if touch.any():
        stats["flip_rate"] = float((np.sign(a_on[touch]) != np.sign(a_off[touch])).mean())
        stats["mean_abs_change"] = float(np.abs(a_on[touch] - a_off[touch]).mean())
        stats["median_touch_over_raw"] = float(np.median(np.abs(tpart[touch]) / (np.abs(raw[touch]) + 1e-9)))
    return stats


def worker(args):
    cond, seed = args.worker.rsplit(":", 1)
    seed = int(seed)
    main = np.load(dump(args.root, cond, seed, args.epoch), allow_pickle=True)[0]
    main.params.use_wandb = False
    agent = main.agent
    grip = agent.grip_agent.grip
    ctrl = main.controller
    w0 = ctrl.stm_a.get_weights().copy()
    limit = 1 if grip.output == "clip" else np.arctanh(0.95)
    original = agent.step
    rows = []
    for scale in SCALES:
        ctrl.stm_a.set_weights(w0 * scale)
        step, records = make_recorder(agent)
        agent.step = step
        intact = [evaluate(main, "intact", seed, es) for es in ESEEDS]
        agent.step = original
        R = np.array(records)
        row = dict(cond=cond, seed=seed, scale=scale, success=np.mean([r["success"] for r in intact]), **touch_stats(R, limit))
        gs = R[:, :47]
        fills = {"no_touch": ("touch", 0.0), "posture_frozen": ("posture", gs[:, BLOCKS["posture"]].mean(0)),
                 "eye_frozen": ("eye", gs[:, BLOCKS["eye"]].mean(0))}
        for label, (block, fill) in fills.items():
            agent.step = make_ablation(agent, block, fill)
            row[f"success_{label}"] = np.mean([evaluate(main, label, seed, es)["success"] for es in ESEEDS])
            agent.step = original
        rows.append(row)
    ctrl.stm_a.set_weights(w0)
    Path(args.tmp).mkdir(parents=True, exist_ok=True)
    (Path(args.tmp) / f"sat_{cond}_{seed}.json").write_text(json.dumps(rows))


def orchestrate(args):
    seeds = [int(s) for s in args.seeds.split(",")]
    running = []
    for cond in args.conds.split(","):
        for seed in seeds:
            while len(running) >= args.parallel:
                running = [p for p in running if p.poll() is None]
                time.sleep(2)
            cmd = [sys.executable, str(Path(__file__).resolve()), "--root", args.root, "--tmp", args.tmp,
                   "--epoch", str(args.epoch), "--worker", f"{cond}:{seed}"]
            running.append(subprocess.Popen(cmd, env={**os.environ, "OMP_NUM_THREADS": "1"}))
    for p in running:
        p.wait()
    df = pd.DataFrame([r for f in sorted(Path(args.tmp).glob("sat_*.json")) for r in json.loads(f.read_text())])
    df["drop_no_touch"] = df.success - df.success_no_touch
    df["drop_posture"] = df.success - df.success_posture_frozen
    df["drop_eye"] = df.success - df.success_eye_frozen
    cols = ["sat", "median_abs_raw", "success", "drop_no_touch", "drop_posture", "drop_eye", "flip_rate",
            "mean_abs_change", "median_touch_over_raw", "touch_frames"]
    summary = df.groupby(["cond", "scale"], sort=False)[cols].mean().round(3).reset_index()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    df.round(4).to_csv(out / f"{args.label}_per_seed.csv", index=False)
    summary.to_csv(out / f"{args.label}_summary.csv", index=False)
    print(summary.to_string(index=False))


def parse_arguments():
    parser = argparse.ArgumentParser(description="Output saturation versus touch ablation: rescale policy weights at evaluation")
    parser.add_argument("--root", default="pilot")
    parser.add_argument("--conds", default="base,D,D_tanh_tg3,A")
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--epoch", type=int, default=149)
    parser.add_argument("--out", default="pilot/figures")
    parser.add_argument("--tmp", default="pilot/sat_tmp")
    parser.add_argument("--label", default="sat_touch")
    parser.add_argument("--parallel", type=int, default=12)
    parser.add_argument("--worker", default=None)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    orchestrate(args) if args.worker is None else worker(args)

#!/usr/bin/env python
import argparse
import contextlib
import io
import json
import re
import sys
from pathlib import Path

import numpy as np

SRC = Path(__file__).resolve().parent
sys.path.insert(0, str(SRC))
sys.path.insert(0, str(SRC.parent / "tools" / "box2dsim"))

import policy_stats as ps  # noqa: E402
from SMMain import Main  # noqa: E402

_ = Main


def evaluate(main, hidden_mean, eseeds=(101, 102, 103)):
    grip = main.agent.grip_agent.grip
    original = grip.hidden_mean
    grip.hidden_mean = hidden_mean
    successes, touches = [], []
    try:
        for eseed in eseeds:
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                _, tr = main.evaluation_episodes(orig_controller=main.controller, epoch=0, n_episodes=36, render=None, save_stats=False, e_seed=eseed)
            successes.append(float(re.search(r"eval success rate ([\d.]+)", buf.getvalue()).group(1)))
            touches.append(float((tr[[f"s{i}" for i in range(40)]].values.sum(1) > 0).mean()))
    finally:
        grip.hidden_mean = original
    return float(np.mean(successes)), float(np.mean(touches))


def decompose(main, data):
    grip = main.agent.grip_agent.grip
    controller = main.controller
    W = controller.stm_a.get_weights().T.reshape(-1, grip.num_hidden, grip.num_outputs)
    eye = np.load(SRC / f"{main.params.grip_norm_name}.npz")["mean"][main.params.somatosensory_size + main.params.proprioception_size:]
    ss, p = data["ss"], data["p"]
    active = data["a"].any(-1)
    states = np.hstack([ss[active], p[active], np.tile(eye, (active.sum(), 1))])[::7]
    codes = np.array([grip.hidden(s) for s in states])
    mean = codes.mean(0)
    units = grip.block_units
    zero = grip.hidden(np.zeros(len(states[0])))
    const_all = np.einsum("h,nho->no", mean, W)
    var_part = np.einsum("th,nho->nto", codes - mean, W)
    const_touch = np.einsum("h,nho->no", np.r_[zero[:units], np.zeros(len(mean) - units)], W)
    e_const, e_var = float((const_all ** 2).mean()), float((var_part ** 2).mean())
    return dict(output_constant_share=e_const / (e_const + e_var), output_touch_zero_share=float((const_touch ** 2).mean()) / (e_const + e_var),
                prototype_common_share=float(((W.reshape(len(W), -1).mean(0)) ** 2).sum() / (W.reshape(len(W), -1) ** 2).sum(1).mean()))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--storage", required=True)
    parser.add_argument("--epoch", type=int, required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--norm", default="data/StoredGripActuatorNorm_v4c")
    args = parser.parse_args()
    main_obj, data = ps.load(args.storage, args.epoch, "data.npy")
    stats = np.load(SRC / f"{args.norm}.npz")
    row = dict(label=args.label)
    row.update(decompose(main_obj, data))
    row["success_intact"], row["touch_intact"] = evaluate(main_obj, None)
    for variant in ("touch", "all"):
        row[f"success_{variant}"], row[f"touch_{variant}"] = evaluate(main_obj, stats[f"hidden_mean_{variant}"])
    Path(args.out).mkdir(parents=True, exist_ok=True)
    (Path(args.out) / f"{args.label}.json").write_text(json.dumps(row))
    print(row)


if __name__ == "__main__":
    main()

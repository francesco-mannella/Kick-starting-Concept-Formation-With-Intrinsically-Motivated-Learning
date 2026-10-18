#!/usr/bin/env python
import argparse
import contextlib
import io
import json
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

SRC = Path(__file__).resolve().parent
sys.path.insert(0, str(SRC))

from SMMain import Main  # noqa: E402  (pickled dumps look up Main in __main__)

_ = Main

BLOCKS = {"touch": slice(0, 40), "posture": slice(40, 45), "eye": slice(45, 47)}
HIDDEN = {"touch": slice(0, 25), "posture": slice(25, 50), "eye": slice(50, 75)}
ESEEDS = (101, 102, 103)
RUN_PATTERN = "parasite_{seed:03d}"


def storage(root, seed, epoch):
    name = RUN_PATTERN.format(seed=seed)
    return Path(root) / name / "simulations" / name / "storage" / f"{epoch:06d}"


def evaluate(main, label, seed, eseed):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        _, tr = main.evaluation_episodes(orig_controller=main.controller, epoch=0, n_episodes=36,
                                         render=None, save_stats=False, e_seed=eseed)
    touch = tr[[f"s{i}" for i in range(40)]].values.sum(1) > 0
    success = float(re.search(r"eval success rate ([\d.]+)", buf.getvalue()).group(1))
    return dict(seed=seed, cond=label, eseed=eseed, success=success,
                touch_freq=float(touch.mean()), match=float(tr["match"].mean()))


def make_recorder(agent):
    grip = agent.grip_agent.grip
    original = agent.step
    records = []

    def step(state):
        gs = np.hstack([state["TOUCH_SENSORS"], state["JOINT_POSITIONS"][:5], state["EYE_POS"][::-1]]).astype(float)
        hidden = grip.hidden(gs)
        parts = [hidden[HIDDEN[b]] @ grip.params[HIDDEN[b]] for b in BLOCKS]
        records.append(np.hstack([gs, hidden @ grip.params] + parts))
        return original(state)

    return step, records


def make_ablation(agent, block, fill):
    params = agent.params
    grip = agent.grip_agent

    def step(state):
        gs = np.hstack([state["TOUCH_SENSORS"], state["JOINT_POSITIONS"][:5], state["EYE_POS"][::-1]]).astype(float)
        gs[BLOCKS[block]] = fill
        action = grip.step(gs)
        action *= [-1, -1, -1, 1, 1]
        action[:3] += params.policy_base_arm
        action[3:] += params.policy_base
        return action

    return step


def static_rank(M, policy_final, policy_init):
    out = {"map_rank_total": int(np.linalg.matrix_rank(M))}
    for b in BLOCKS:
        out[f"map_rank_{b}"] = int(np.linalg.matrix_rank(M[BLOCKS[b], HIDDEN[b]]))
    out["effective_policy_dim"] = out["map_rank_total"] * 5
    _, _, vt = np.linalg.svd(M)
    null = vt[out["map_rank_total"]:].T
    for name, w in (("final", policy_final), ("init", policy_init)):
        shares = []
        for j in range(w.shape[1]):
            p = w[:, j].reshape(75, 5)
            shares.append(np.sum((null.T @ p) ** 2) / np.sum(p ** 2))
        out[f"policy_null_share_{name}"] = float(np.mean(shares))
    out["null_share_random"] = float(null.shape[1] / 75)
    return out


def worker(args):
    seed = args.worker
    main = np.load(storage(args.root, seed, args.epoch) / "main.dump.npy", allow_pickle=True)[0]
    main.params.use_wandb = False
    agent = main.agent
    grip = agent.grip_agent.grip
    M = grip.map
    result = {"seed": seed}
    w_final = main.controller.stm_a.get_weights()
    w_init = np.load(storage(args.root, seed, 0) / "weights.npy", allow_pickle=True)[0]["policy"]
    result["static"] = static_rank(M, w_final, w_init)

    step, records = make_recorder(agent)
    original = agent.step
    agent.step = step
    intact = [evaluate(main, "intact", seed, es) for es in ESEEDS]
    agent.step = original
    R = np.array(records)
    gs, raw, parts = R[:, :47], R[:, 47:52], R[:, 52:].reshape(len(R), 3, 5)

    active = np.abs(raw).sum(1) > 0
    limit = 1 if grip.output == "clip" else np.arctanh(0.95)
    sat = {"frac_clipped_all": float((np.abs(raw) >= limit).mean()), "frac_steps_with_policy": float(active.mean()),
           "frac_clipped_with_policy": float((np.abs(raw[active]) >= limit).mean()),
           "frac_all_outputs_clipped_with_policy": float((np.abs(raw[active]) >= limit).all(1).mean()),
           "unique_clipped_sign_patterns": int(len(np.unique(np.sign(raw[active]), axis=0)))}
    for k in range(5):
        sat[f"frac_clipped_out{k}"] = float((np.abs(raw[:, k]) >= limit).mean())
    sat["median_abs_raw_output_with_policy"] = float(np.median(np.abs(raw[active])))
    var = parts.var(0)
    share = var / var.sum(0, keepdims=True)
    for i, b in enumerate(BLOCKS):
        sat[f"variance_share_{b}"] = float(share[i].mean())
    result["saturation"] = sat

    touch = gs[:, :40]
    tf = touch[touch.sum(1) > 0]
    cap = {"touch_frames": int(len(tf)), "touch_unique_patterns": int(len(np.unique(tf.round(3), axis=0))),
           "touch_data_rank": int(np.linalg.matrix_rank(tf)) if len(tf) else 0}
    if len(tf) > 5:
        s = np.linalg.svd(tf - tf.mean(0), compute_uv=False) ** 2
        cap["touch_pcs_95"] = int(np.searchsorted(np.cumsum(s) / s.sum(), 0.95) + 1)
        q, _ = np.linalg.qr(M[:40, :25])
        if grip.input_norm == "block_rms":
            tf = (tf - grip.input_mean[:40]) / grip.input_scale[:40]
        cap["touch_r2_in_prototype_span"] = float(1 - np.sum((tf - tf @ q @ q.T) ** 2) / np.sum(tf ** 2))
    for b in ("posture", "eye"):
        x = gs[:, BLOCKS[b]]
        cap[f"{b}_std"] = float(x.std(0).mean())
    result["touch_capacity"] = cap

    fills = {"no touch (zeros)": ("touch", 0.0), "posture frozen at mean": ("posture", gs[:, BLOCKS["posture"]].mean(0)),
             "eye frozen at mean": ("eye", gs[:, BLOCKS["eye"]].mean(0))}
    ablation = list(intact)
    for label, (block, fill) in fills.items():
        agent.step = make_ablation(agent, block, fill)
        ablation += [evaluate(main, label, seed, es) for es in ESEEDS]
        agent.step = original
    result["ablation"] = ablation
    out = Path(args.tmp) / f"diag_{seed}.json"
    out.write_text(json.dumps(result))


def orchestrate(args):
    tmp = Path(args.tmp)
    tmp.mkdir(parents=True, exist_ok=True)
    seeds = [int(s) for s in args.seeds.split(",")]
    running = []
    for seed in seeds:
        while len(running) >= args.parallel:
            running = [p for p in running if p.poll() is None]
            if len(running) >= args.parallel:
                import time
                time.sleep(2)
        cmd = [sys.executable, str(Path(__file__).resolve()), "--root", args.root, "--tmp", args.tmp, "--worker", str(seed),
               "--run-pattern", args.run_pattern, "--epoch", str(args.epoch), "--label", args.label]
        running.append(subprocess.Popen(cmd, env={**__import__("os").environ, "OMP_NUM_THREADS": "1"}))
    for p in running:
        p.wait()
    results = [json.loads((tmp / f"diag_{s}.json").read_text()) for s in seeds]
    rows = []
    for section in ("static", "saturation", "touch_capacity"):
        keys = results[0][section].keys()
        for k in keys:
            vals = np.array([r[section].get(k, np.nan) for r in results], float)
            rows.append(dict(section=section, metric=k, mean=np.nanmean(vals), sd=np.nanstd(vals, ddof=1), n=int(np.isfinite(vals).sum())))
    summary = pd.DataFrame(rows)
    ab = pd.DataFrame([x for r in results for x in r["ablation"]])
    per_seed = ab.groupby(["cond", "seed"])[["success", "touch_freq", "match"]].mean().reset_index()
    abl = per_seed.groupby("cond")[["success", "touch_freq", "match"]].agg(["mean", "std"])
    abl.columns = ["_".join(c) for c in abl.columns]
    order = ["intact", "no touch (zeros)", "posture frozen at mean", "eye frozen at mean"]
    abl = abl.reindex(order).round(3)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    summary.round(4).to_csv(out / f"{args.label}_summary.csv", index=False)
    abl.to_csv(out / f"{args.label}_ablation.csv")
    print(summary.round(3).to_string(index=False))
    print(abl.to_string())


def parse_arguments():
    parser = argparse.ArgumentParser(description="Diagnostics of the agent's input layer")
    parser.add_argument("--root", default="test/demo_local")
    parser.add_argument("--out", default="test/demo_local/figures")
    parser.add_argument("--tmp", default="input_diag_tmp")
    parser.add_argument("--parallel", type=int, default=5)
    parser.add_argument("--worker", type=int, default=None)
    parser.add_argument("--seeds", default="0,1,2,3,4,5,6,7,8,9")
    parser.add_argument("--run-pattern", default="parasite_{seed:03d}", help="Run folder name with a {seed} field")
    parser.add_argument("--epoch", type=int, default=1499, help="Epoch of the dump to analyse")
    parser.add_argument("--label", default="input_diag", help="Prefix of the output CSV files")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    RUN_PATTERN = args.run_pattern
    if args.worker is None:
        orchestrate(args)
    else:
        worker(args)

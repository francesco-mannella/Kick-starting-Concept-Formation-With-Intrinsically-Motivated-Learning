#!/usr/bin/env python
import argparse
import contextlib
import io
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

SRC = Path(__file__).resolve().parent
sys.path.insert(0, str(SRC))

from SMMain import Main  # noqa: E402  (pickled dumps look up Main in __main__)

_ = Main
ESEEDS = (101, 102, 103)
CONDITIONS = [(side, enc) for enc in ("linear", "wta_block") for side in (3, 5, 8, 10)]


def dump_path(sweep_root, baseline_root, side, enc, seed):
    if side == 5 and enc == "linear":
        name = f"parasite_{seed:03d}"
        return Path(baseline_root) / name / "simulations" / name / "storage" / "000500" / "main.dump.npy"
    name = f"agent_k{side}_{enc}_{seed:03d}"
    return Path(sweep_root) / name / "simulations" / name / "storage" / "000499" / "main.dump.npy"


def metrics(tr, ctrl, sigma):
    S = tr[[f"s{i}" for i in range(40)]].values
    P = tr[["d1", "d2", "d3", "d4", "d5"]].values
    T = S.sum(1) > 0
    out = dict(frames=len(tr), nT=int(T.sum()), touch_frames=int(T.sum()), match=float(tr["match"].mean()))
    if T.sum() == 0:
        return out
    gx, gy = tr.prototype_x.values[T], tr.prototype_y.values[T]
    gi = (np.round(gx) * 10 + np.round(gy)).astype(int)
    d = np.hypot(tr.touch_x.values[T] - gx, tr.touch_y.values[T] - gy)
    out.update(sum_d=float(d.sum()), hit2=int((d <= 2).sum()))
    for name, X, W in (("ss", S[T], ctrl.stm_ss.get_weights().T), ("p", P[T], ctrl.stm_p.get_weights().T)):
        D = ((X[:, None, :] - W[None]) ** 2).sum(-1)
        own = D[np.arange(len(D)), gi]
        out[f"sum_rank_{name}"] = float(((D < own[:, None]).sum(1) / 99).sum())
    return out


def worker(args):
    side, enc, seed = args.worker_side, args.worker_enc, args.worker_seed
    main = np.load(dump_path(args.sweep_root, args.baseline_root, side, enc, seed), allow_pickle=True)[0]
    main.params.use_wandb = False
    A = main.controller
    wa = A.stm_a.get_weights().copy()
    mean, std = wa.mean(1, keepdims=True), wa.std(1, keepdims=True)
    rows = []
    for es in ESEEDS:
        rng = np.random.default_rng(1000 * seed + es)
        conds = [("intact", wa), ("shuffled", wa[:, rng.permutation(wa.shape[1])]),
                 ("random same mean/std", mean + std * rng.normal(size=wa.shape)),
                 ("constant mean policy", np.repeat(mean, wa.shape[1], 1))]
        for label, w in conds:
            A.stm_a.set_weights(w)
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                _, tr = main.evaluation_episodes(orig_controller=A, epoch=0, n_episodes=36, render=None,
                                                 save_stats=False, e_seed=es)
            r = dict(side=side, enc=enc, seed=seed, eseed=es, cond=label,
                     success=float(re.search(r"eval success rate ([\d.]+)", buf.getvalue()).group(1)))
            r.update(metrics(tr, A, main.params.match_sigma))
            rows.append(r)
        A.stm_a.set_weights(wa)
    Path(args.tmp).mkdir(parents=True, exist_ok=True)
    (Path(args.tmp) / f"prec_{enc}_{side}_{seed}.json").write_text(json.dumps(rows))


def orchestrate(args):
    seeds = [int(s) for s in args.seeds.split(",")]
    jobs = [(s, e, seed) for s, e in CONDITIONS for seed in seeds
            if dump_path(args.sweep_root, args.baseline_root, s, e, seed).is_file()]
    running = []
    for side, enc, seed in jobs:
        while len(running) >= args.parallel:
            running = [p for p in running if p.poll() is None]
            time.sleep(2)
        cmd = [sys.executable, str(Path(__file__).resolve()), "--sweep-root", args.sweep_root,
               "--baseline-root", args.baseline_root, "--tmp", args.tmp, "--seeds", args.seeds,
               "--worker-side", str(side), "--worker-enc", enc, "--worker-seed", str(seed)]
        running.append(subprocess.Popen(cmd, env={**os.environ, "OMP_NUM_THREADS": "1"}))
    for p in running:
        p.wait()
    rows = [r for f in sorted(Path(args.tmp).glob("prec_*.json")) for r in json.loads(f.read_text())]
    df = pd.DataFrame(rows)
    per_job = df.groupby(["enc", "side", "cond", "seed"]).sum(numeric_only=True).reset_index()
    per_job["success"] = df.groupby(["enc", "side", "cond", "seed"]).success.mean().values
    per_job["touch_freq"] = per_job.touch_frames / per_job.frames
    per_job["mean_goal_distance"] = per_job.sum_d / per_job.nT
    per_job["hit_rate_2"] = per_job.hit2 / per_job.nT
    per_job["rank_touch_proto"] = per_job.sum_rank_ss / per_job.nT
    per_job["rank_proprio_proto"] = per_job.sum_rank_p / per_job.nT
    per_job["match"] = df.groupby(["enc", "side", "cond", "seed"]).match.mean().values
    cols = ["success", "touch_freq", "match", "mean_goal_distance", "hit_rate_2", "rank_touch_proto", "rank_proprio_proto"]
    summary = per_job.groupby(["enc", "side", "cond"])[cols].mean().round(3).reset_index()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    summary.to_csv(out / "input_sweep_precision.csv", index=False)
    diffs = []
    for (enc, side), g in per_job.groupby(["enc", "side"]):
        base = g[g.cond == "intact"].set_index("seed")
        for cond in ("shuffled", "random same mean/std", "constant mean policy"):
            other = g[g.cond == cond].set_index("seed")
            for col in ("success", "match", "hit_rate_2", "rank_touch_proto"):
                d = (base[col] - other[col]).dropna()
                diffs.append(dict(enc=enc, side=side, control=cond, metric=col, intact_minus_control=round(d.mean(), 3),
                                  seeds_intact_better=int((d > 0).sum() if col != "rank_touch_proto" else (d < 0).sum()),
                                  seeds=len(d)))
    pd.DataFrame(diffs).to_csv(out / "input_sweep_precision_paired.csv", index=False)
    print(summary.to_string(index=False))


def parse_arguments():
    parser = argparse.ArgumentParser(description="Precision and policy-shuffle controls on the sweep's final controllers")
    parser.add_argument("--sweep-root", default="test/demo_local/input_sweep")
    parser.add_argument("--baseline-root", default="test/demo_local")
    parser.add_argument("--out", default="test/demo_local/figures")
    parser.add_argument("--tmp", default="sweep_precision_tmp")
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--parallel", type=int, default=6)
    parser.add_argument("--worker-side", type=int, default=None)
    parser.add_argument("--worker-enc", default=None)
    parser.add_argument("--worker-seed", type=int, default=None)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    if args.worker_side is None:
        orchestrate(args)
    else:
        worker(args)

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
from sweep_precision import ESEEDS, metrics  # noqa: E402

_ = Main


def dump(root, pattern, seed, epoch):
    name = pattern.format(seed=seed)
    return Path(root) / name / "simulations" / name / "storage" / f"{epoch:06d}" / "main.dump.npy"


def worker(args):
    main = np.load(dump(args.root, args.pattern, args.worker_seed, args.epoch), allow_pickle=True)[0]
    main.params.use_wandb = False
    A = main.controller
    wa = A.stm_a.get_weights().copy()
    mean, std = wa.mean(1, keepdims=True), wa.std(1, keepdims=True)
    rows = []
    for es in ESEEDS:
        rng = np.random.default_rng(1000 * args.worker_seed + es)
        conds = [("intact", wa), ("shuffled", wa[:, rng.permutation(wa.shape[1])]),
                 ("random same mean/std", mean + std * rng.normal(size=wa.shape)),
                 ("constant mean policy", np.repeat(mean, wa.shape[1], 1))]
        for label, w in conds:
            A.stm_a.set_weights(w)
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                _, tr = main.evaluation_episodes(orig_controller=A, epoch=0, n_episodes=36, render=None,
                                                 save_stats=False, e_seed=es)
            r = dict(seed=args.worker_seed, eseed=es, cond=label,
                     success=float(re.search(r"eval success rate ([\d.]+)", buf.getvalue()).group(1)))
            r.update(metrics(tr, A, main.params.match_sigma))
            rows.append(r)
        A.stm_a.set_weights(wa)
    Path(args.tmp).mkdir(parents=True, exist_ok=True)
    (Path(args.tmp) / f"ctrl_{args.worker_seed}.json").write_text(json.dumps(rows))


def orchestrate(args):
    seeds = [int(s) for s in args.seeds.split(",")]
    running = []
    for seed in seeds:
        while len(running) >= args.parallel:
            running = [p for p in running if p.poll() is None]
            time.sleep(2)
        cmd = [sys.executable, str(Path(__file__).resolve())] + [
            x for k, v in vars(args).items() if k != "worker_seed" and v is not None
            for x in (f"--{k.replace('_', '-')}", str(v))] + ["--worker-seed", str(seed)]
        running.append(subprocess.Popen(cmd, env={**os.environ, "OMP_NUM_THREADS": "1"}))
    for p in running:
        p.wait()
    df = pd.DataFrame([r for f in sorted(Path(args.tmp).glob("ctrl_*.json")) for r in json.loads(f.read_text())])
    per = df.groupby(["cond", "seed"]).sum(numeric_only=True).reset_index()
    per["success"] = df.groupby(["cond", "seed"]).success.mean().values
    per["match"] = df.groupby(["cond", "seed"]).match.mean().values
    per["touch_freq"] = per.touch_frames / per.frames
    per["mean_goal_distance"] = per.sum_d / per.nT
    per["hit_rate_2"] = per.hit2 / per.nT
    per["rank_touch_proto"] = per.sum_rank_ss / per.nT
    per["rank_proprio_proto"] = per.sum_rank_p / per.nT
    cols = ["success", "touch_freq", "match", "mean_goal_distance", "hit_rate_2", "rank_touch_proto", "rank_proprio_proto"]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    per[["cond", "seed"] + cols].round(4).to_csv(out / f"{args.label}_controls_per_seed.csv", index=False)
    summary = per.groupby("cond")[cols].mean().round(3)
    print(args.label)
    print(summary.to_string())


def parse_arguments():
    parser = argparse.ArgumentParser(description="Policy shuffle / random / constant controls on final controllers")
    parser.add_argument("--root", required=True)
    parser.add_argument("--pattern", required=True, help="Run folder name with a {seed} field")
    parser.add_argument("--epoch", type=int, required=True)
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--label", required=True)
    parser.add_argument("--out", default="test/demo_local/actuator_sweep/diag")
    parser.add_argument("--tmp", default="policy_controls_tmp")
    parser.add_argument("--parallel", type=int, default=3)
    parser.add_argument("--worker-seed", type=int, default=None)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    orchestrate(args) if args.worker_seed is None else worker(args)

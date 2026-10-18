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

from diagnose_input_layer import ESEEDS, evaluate, make_recorder  # noqa: E402
from policy_topology import SIDE, storage, topology  # noqa: E402
from SMMain import Main  # noqa: E402  (pickled dumps look up Main in __main__)

_ = Main


def subspace(hidden, share=0.95):
    second = hidden.T @ hidden / len(hidden)
    vals, vecs = np.linalg.eigh(second)
    vals, vecs = vals[::-1].clip(0), vecs[:, ::-1]
    k = int(np.searchsorted(np.cumsum(vals) / vals.sum(), share) + 1)
    return vecs[:, :k], float(vals.sum() ** 2 / (vals ** 2).sum()), k


def nodes(w):
    return w.T.reshape(w.shape[1], -1, 5)


def worker(args):
    cond, seed = args.worker.rsplit(":", 1)
    seed = int(seed)
    main = np.load(storage(args.root, cond, seed, args.epochs[-1]) / "main.dump.npy", allow_pickle=True)[0]
    main.params.use_wandb = False
    agent = main.agent
    grip = agent.grip_agent.grip
    original = agent.step
    step, records = make_recorder(agent)
    agent.step = step
    evaluate(main, "intact", seed, ESEEDS[0])
    agent.step = original
    gs = np.array(records)[:, :47]
    hidden = np.array([grip.hidden(x) for x in gs])
    q, participation, k95 = subspace(hidden)
    proj = q @ q.T
    out = {"cond": cond, "seed": seed, "participation_ratio": participation, "components_95": k95,
           "random_share_outside": 1 - k95 / hidden.shape[1]}
    rows = None
    if grip.basis is None:
        m = grip.map
        rank = int(np.linalg.matrix_rank(m))
        rows = np.linalg.svd(m, full_matrices=False)[2][:rank].T
    rng = np.random.default_rng(seed)
    pairs = rng.integers(0, SIDE * SIDE, (500, 2))
    for epoch in args.epochs:
        w = np.load(storage(args.root, cond, seed, epoch) / "weights.npy", allow_pickle=True)[0]["policy"]
        p = nodes(w)
        inside = np.einsum("ij,njo->nio", proj, p)
        outside = p - inside
        out[f"null_share_data_{epoch}"] = float((outside ** 2).sum() / (p ** 2).sum())
        if rows is not None:
            out[f"null_share_map_{epoch}"] = float(((p - np.einsum("ij,njo->nio", rows @ rows.T, p)) ** 2).sum() / (p ** 2).sum())
        diff = p[pairs[:, 0]] - p[pairs[:, 1]]
        diff_out = diff - np.einsum("ij,njo->nio", proj, diff)
        out[f"inert_distance_share_{epoch}"] = float((diff_out ** 2).sum() / (diff ** 2).sum())
        for label, x in (("inside", inside), ("outside", outside)):
            t = topology(x.reshape(len(x), -1))
            out[f"ratio_{label}_{epoch}"], out[f"rho_{label}_{epoch}"] = t["ratio"], t["rho"]
    Path(args.tmp).mkdir(parents=True, exist_ok=True)
    (Path(args.tmp) / f"sub_{cond}_{seed}.json").write_text(json.dumps(out))


def orchestrate(args):
    seeds = [int(s) for s in args.seeds.split(",")]
    running = []
    for cond in args.conds.split(","):
        for seed in seeds:
            while len(running) >= args.parallel:
                running = [p for p in running if p.poll() is None]
                time.sleep(2)
            cmd = [sys.executable, str(Path(__file__).resolve()), "--root", args.root, "--tmp", args.tmp,
                   "--epochs", *map(str, args.epochs), "--worker", f"{cond}:{seed}"]
            running.append(subprocess.Popen(cmd, env={**os.environ, "OMP_NUM_THREADS": "1"}))
    for p in running:
        p.wait()
    df = pd.DataFrame([json.loads(f.read_text()) for f in sorted(Path(args.tmp).glob("sub_*.json"))])
    summary = df.drop(columns="seed").groupby("cond", sort=False).mean().round(3)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    df.round(4).to_csv(out / f"{args.label}_per_seed.csv", index=False)
    summary.to_csv(out / f"{args.label}_summary.csv")
    print(summary.T.to_string())


def parse_arguments():
    parser = argparse.ArgumentParser(description="Null space of the policy weights relative to the hidden-activity data subspace")
    parser.add_argument("--root", default="pilot")
    parser.add_argument("--conds", default="base,A_oldmap,A,B,C,D,C_tanh,D_tanh,C_tanh_tg3,D_tanh_tg3")
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--epochs", type=int, nargs="+", default=[0, 149])
    parser.add_argument("--out", default="pilot/figures")
    parser.add_argument("--tmp", default="pilot/sub_tmp")
    parser.add_argument("--label", default="policy_subspace")
    parser.add_argument("--parallel", type=int, default=10)
    parser.add_argument("--worker", default=None)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    orchestrate(args) if args.worker is None else worker(args)

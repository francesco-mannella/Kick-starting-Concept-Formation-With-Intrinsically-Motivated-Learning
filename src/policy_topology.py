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
from scipy.stats import spearmanr

SRC = Path(__file__).resolve().parent
sys.path.insert(0, str(SRC))

from diagnose_input_layer import ESEEDS, evaluate, make_recorder  # noqa: E402
from SMMain import Main  # noqa: E402  (pickled dumps look up Main in __main__)

_ = Main
SIDE = 10
GRID = np.stack(np.divmod(np.arange(SIDE * SIDE), SIDE), 1).astype(float)
GRID_DIST = np.sqrt(((GRID[:, None] - GRID[None]) ** 2).sum(-1))
UPPER = np.triu_indices(SIDE * SIDE, 1)
NEIGHBOURS = GRID_DIST[UPPER] == 1


def storage(root, cond, seed, epoch):
    name = f"agent_{cond}_{seed:03d}"
    return Path(root) / name / "simulations" / name / "storage" / f"{epoch:06d}"


def topology(features):
    d = np.sqrt(((features[:, None] - features[None]) ** 2).sum(-1))[UPPER]
    rho = spearmanr(GRID_DIST[UPPER], d)[0]
    return {"ratio": float(d[NEIGHBOURS].mean() / d.mean()), "rho": float(rho)}


def policy_features(w, probes, grip):
    policies = w.reshape(probes.shape[1], 5, -1)
    raw = np.einsum("nh,hok->nko", probes, policies)
    act = np.tanh(raw) if grip.output == "tanh" else np.clip(raw, -1, 1)
    per_node = lambda x: x.transpose(1, 0, 2).reshape(x.shape[1], -1)
    return {"policy_weights": w.T, "policy_actions": per_node(act), "policy_signs": per_node(np.sign(raw))}


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
    rng = np.random.default_rng(seed)
    probes = np.array([grip.hidden(x) for x in gs[rng.choice(len(gs), 300, replace=False)]])
    rows = []
    for epoch in args.epochs:
        w = np.load(storage(args.root, cond, seed, epoch) / "weights.npy", allow_pickle=True)[0]
        maps = {"visual": w["visual"].T, "touch": w["ssensory"].T, "proprio": w["proprio"].T}
        maps.update(policy_features(w["policy"], probes, grip))
        for name, feats in maps.items():
            rows.append(dict(cond=cond, seed=seed, epoch=epoch, map=name, **topology(feats)))
        shuffled = w["policy"][:, np.random.default_rng(seed).permutation(SIDE * SIDE)]
        rows.append(dict(cond=cond, seed=seed, epoch=epoch, map="policy_weights_shuffled", **topology(shuffled.T)))
    Path(args.tmp).mkdir(parents=True, exist_ok=True)
    (Path(args.tmp) / f"topo_{cond}_{seed}.json").write_text(json.dumps(rows))


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
    df = pd.DataFrame([r for f in sorted(Path(args.tmp).glob("topo_*.json")) for r in json.loads(f.read_text())])
    summary = df.groupby(["cond", "map", "epoch"], sort=False)[["ratio", "rho"]].mean().round(3).reset_index()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    df.round(4).to_csv(out / f"{args.label}_per_seed.csv", index=False)
    summary.to_csv(out / f"{args.label}_summary.csv", index=False)
    final = summary[summary.epoch == args.epochs[-1]]
    print(final.pivot_table(index="map", columns="cond", values="ratio", sort=False).round(2).to_string())
    print(final.pivot_table(index="map", columns="cond", values="rho", sort=False).round(2).to_string())


def parse_arguments():
    parser = argparse.ArgumentParser(description="Topology of the policy map: grid distance versus weight/action distance")
    parser.add_argument("--root", default="pilot")
    parser.add_argument("--conds", default="base,A,C,D,D_tanh_tg3")
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--epochs", type=int, nargs="+", default=[0, 50, 100, 149])
    parser.add_argument("--out", default="pilot/figures")
    parser.add_argument("--tmp", default="pilot/topo_tmp")
    parser.add_argument("--label", default="policy_topology")
    parser.add_argument("--parallel", type=int, default=15)
    parser.add_argument("--worker", default=None)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    orchestrate(args) if args.worker is None else worker(args)

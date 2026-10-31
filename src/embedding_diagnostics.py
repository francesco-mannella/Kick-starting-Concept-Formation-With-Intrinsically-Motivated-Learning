#!/usr/bin/env python
import argparse
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

BLOCKS = {"touch": slice(0, 40), "arm": slice(40, 43), "gripper": slice(43, 48), "eye": slice(48, 50)}
SOM_BLOCKS = {"touch": (slice(0, 40), slice(0, 25)), "posture": (slice(40, 48), slice(25, 50)), "eye": (slice(48, 50), slice(50, 75))}


def collect_worker(args):
    from diagnose_input_layer import ESEEDS, evaluate, make_recorder
    folder = Path(args.root) / f"agent_dof8_{args.worker:03d}" / "simulations" / f"agent_dof8_{args.worker:03d}" / "storage" / f"{args.epoch:06d}"
    main = np.load(folder / "main.dump.npy", allow_pickle=True)[0]
    main.params.use_wandb = False
    step, records = make_recorder(main.agent)
    main.agent.step = step
    for eseed in ESEEDS:
        evaluate(main, "intact", args.worker, eseed)
    states = np.array(records)[:, :50]
    np.save(Path(args.out) / f"visited_seed{args.worker}_epoch{args.epoch}.npy", states)


def collect(args):
    Path(args.out).mkdir(parents=True, exist_ok=True)
    procs = [subprocess.Popen([sys.executable, str(Path(__file__).resolve()), "collect", "--root", args.root, "--out", args.out,
                               "--epoch", str(args.epoch), "--worker", str(s)]) for s in range(args.seeds)]
    for p in procs:
        while p.poll() is None:
            time.sleep(2)


def load_babbling(paths, n=20000, seed=0):
    rng = np.random.RandomState(seed)
    d = [np.load(p) for p in paths]
    x = np.hstack([np.vstack([a["touch"] for a in d]), np.vstack([a["posture"] for a in d]), np.vstack([a["eye"] for a in d])])
    return x[rng.choice(len(x), min(n, len(x)), replace=False)], x


def som_codes(x, som_map, stats, units):
    z = (x - stats["mean"]) / stats["scale"]
    out = {}
    for name, (rows, cols) in SOM_BLOCKS.items():
        proto = som_map[rows, cols]
        dist = ((z[:, rows][:, :, None] - proto[None]) ** 2).sum(1)
        spacing = ((proto[:, :, None] - proto[:, None]) ** 2).sum(0)
        np.fill_diagonal(spacing, np.inf)
        sigma = np.sqrt(np.median(spacing.min(1)))
        act = np.exp(-0.5 * dist / sigma ** 2)
        out[name] = (act / (act.sum(1, keepdims=True) + 1e-12), dist.argmin(1))
    return out


def participation_ratio(c):
    ev = np.clip(np.linalg.eigvalsh(np.cov(c.T)), 0, None)
    return float(ev.sum() ** 2 / (ev ** 2).sum())


def readout_r2(c_fit, y_fit, c_eval, y_eval, ridge=1e-3):
    a = np.hstack([c_fit, np.ones((len(c_fit), 1))])
    w = np.linalg.solve(a.T @ a + ridge * np.eye(a.shape[1]), a.T @ y_fit)
    pred = np.hstack([c_eval, np.ones((len(c_eval), 1))]) @ w
    return float(1 - ((pred - y_eval) ** 2).sum() / ((y_eval - y_eval.mean(0)) ** 2).sum())


def fit_codes(block, z_fit, n_anchor=1500, seed=0):
    from sklearn.decomposition import PCA, KernelPCA

    rng = np.random.RandomState(seed)
    anchors = z_fit[rng.choice(len(z_fit), min(n_anchor, len(z_fit)), replace=False)]
    d = ((anchors[:200, None] - anchors[None, :200]) ** 2).sum(-1)
    gamma = 1 / (2 * np.median(d[d > 0]))
    codes = {}
    pca = PCA(n_components=min(5, z_fit.shape[1])).fit(z_fit)
    codes["linear_pca"] = pca.transform
    kpca = KernelPCA(n_components=5, kernel="rbf", gamma=gamma).fit(anchors)
    codes["kernel_pca"] = kpca.transform
    if block == "eye":
        x, y = z_fit[:, 0], z_fit[:, 1]
        feats = lambda z: np.stack([z[:, 0], z[:, 1], z[:, 0] ** 2, z[:, 0] * z[:, 1], z[:, 1] ** 2], 1)
        m, s = feats(z_fit).mean(0), feats(z_fit).std(0)
        codes["eye_poly"] = lambda z: (feats(z) - m) / s
    return codes


def summarise(name, code_fit, code_vis, y_fit, y_vis, k):
    sd_ratio = code_vis.std(0) / (code_fit.std(0) + 1e-12)
    return {"code": name, "dims": code_fit.shape[1],
            "pr_babble": participation_ratio(code_fit), "pr_visited": participation_ratio(code_vis),
            "comps_visited_sd_gt_20pct_babble": int((sd_ratio > 0.2).sum()),
            "r2_babble": readout_r2(code_fit, y_fit, code_fit, y_fit), "r2_visited": readout_r2(code_fit, y_fit, code_vis, y_vis),
            "sd_ratio_min": float(sd_ratio.min()), "sd_ratio_median": float(np.median(sd_ratio))}


def analyse(args):
    out = Path(args.out)
    stats = np.load(SRC / f"{args.norm}.npz")
    som_map = np.load(SRC / f"{args.map}.npy")
    fit_x, all_x = load_babbling([Path(p) for p in args.babbling], args.n)
    visited = {s: np.load(out / f"visited_seed{s}_epoch{args.epoch}.npy") for s in range(args.seeds)}
    vis_x = np.vstack(list(visited.values()))
    rows = []

    def contact(x):
        return x[x[:, :40].sum(1) > 0]

    som_fit, som_vis = som_codes(fit_x, som_map, stats, 25), som_codes(vis_x, som_map, stats, 25)
    for name in SOM_BLOCKS:
        rows_, cols_ = SOM_BLOCKS[name]
        win_b, win_v = som_fit[name][1], som_vis[name][1]
        if name == "touch":
            sel_b, sel_v = fit_x[:, :40].sum(1) > 0, vis_x[:, :40].sum(1) > 0
            win_b, win_v = win_b[sel_b], win_v[sel_v]
            ca_b, ca_v = som_fit[name][0][sel_b], som_vis[name][0][sel_v]
            y_b, y_v = ((fit_x - stats["mean"]) / stats["scale"])[sel_b][:, rows_], ((vis_x - stats["mean"]) / stats["scale"])[sel_v][:, rows_]
        else:
            ca_b, ca_v = som_fit[name][0], som_vis[name][0]
            y_b, y_v = ((fit_x - stats["mean"]) / stats["scale"])[:, rows_], ((vis_x - stats["mean"]) / stats["scale"])[:, rows_]
        usage = lambda w: np.bincount(w, minlength=25) / len(w)
        entropy = lambda w: float(-(usage(w)[usage(w) > 0] * np.log(usage(w)[usage(w) > 0])).sum() / np.log(25))
        row = summarise("som_rbf_25", ca_b, ca_v, y_b, y_v, 25)
        row.update(block=name, units_gt_1pct_babble=int((usage(win_b) > 0.01).sum()), units_gt_1pct_visited=int((usage(win_v) > 0.01).sum()),
                   winner_entropy_babble=entropy(win_b), winner_entropy_visited=entropy(win_v), n_visited=len(win_v))
        rows.append(row)

    for name, cols in BLOCKS.items():
        if name == "touch":
            fit, vis = contact(fit_x), contact(vis_x)
            fit = fit if len(fit) < 20000 else fit[:20000]
        else:
            fit, vis = fit_x, vis_x
        mean, scale = stats["mean"][cols], stats["scale"][cols]
        z_fit, z_vis = (fit[:, cols] - mean) / scale, (vis[:, cols] - mean) / scale
        for cname, fn in fit_codes(name, z_fit).items():
            row = summarise(cname, fn(z_fit), fn(z_vis), z_fit, z_vis, 5)
            row.update(block=name, n_visited=len(z_vis))
            rows.append(row)
    table = pd.DataFrame(rows)
    first = ["block", "code", "dims", "n_visited"]
    table = table[first + [c for c in table.columns if c not in first]]
    pd.set_option("display.width", 250, "display.max_columns", 30)
    print(table.round(3).to_string(index=False))
    table.to_csv(out / "embedding_diagnostics.csv", index=False)

    per_seed = []
    for s, x in visited.items():
        zs = (x - stats["mean"]) / stats["scale"]
        per_seed.append({"seed": s, "n": len(x), "touch_frac": float((x[:, :40].sum(1) > 0).mean()),
                         **{f"sd_{b}": float(zs[:, c].std(0).mean()) for b, c in BLOCKS.items()}})
    print(pd.DataFrame(per_seed).round(3).to_string(index=False))
    print("babbling sd:", {b: round(float(((fit_x - stats["mean"]) / stats["scale"])[:, c].std(0).mean()), 3) for b, c in BLOCKS.items()})


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)
    for name in ("collect", "analyse"):
        p = sub.add_parser(name)
        p.add_argument("--root", default="")
        p.add_argument("--out", required=True)
        p.add_argument("--epoch", type=int, default=150)
        p.add_argument("--seeds", type=int, default=5)
        p.add_argument("--worker", type=int, default=None)
        p.add_argument("--babbling", nargs="*", default=[])
        p.add_argument("--map", default="data/StoredGripActuatorMap_v4")
        p.add_argument("--norm", default="data/StoredGripActuatorNorm_v4")
        p.add_argument("--n", type=int, default=20000)
    args = parser.parse_args()
    if args.cmd == "collect":
        collect_worker(args) if args.worker is not None else collect(args)
    else:
        analyse(args)


if __name__ == "__main__":
    main()

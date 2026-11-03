import argparse
import glob
import os
import sqlite3

import numpy as np
import pandas as pd

SETS = {
    "orig1500": ("fig567_dof8", "former_*", "{name}"),
    "aff1500": ("affordance_probe/fig567", "former_*", "{name}"),
    "aff1000": ("affordance_probe/ladder1000", "agent_AFF1000_*", "{name}"),
    "fovea16": ("fovea_probe/long", "agent_F16long_*", "{name}"),
}
KEYS = ("mean_sigma", "mean_lr", "mean_comp")


def prototype_stats(W):
    mean = W.mean(0)
    energy = (W ** 2).sum(1).mean()
    R = W - mean
    s = np.linalg.svd(R, compute_uv=False) ** 2
    cos = W @ mean / (np.linalg.norm(W, axis=1) * np.linalg.norm(mean) + 1e-12)
    return dict(
        common_share=float((mean ** 2).sum() / energy),
        cos_median=float(np.median(cos)),
        pcs90=int(np.searchsorted(np.cumsum(s) / s.sum(), 0.9) + 1),
        participation=float(s.sum() ** 2 / (s ** 2).sum()),
        common_norm=float(np.linalg.norm(mean)),
        residual_norm=float(np.sqrt((R ** 2).sum(1).mean())),
    )


def metric_means(db, name, lo, hi):
    con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    out = {}
    for k in KEYS:
        out[k] = con.execute(
            "select avg(m.value) from metrics m join runs r on r.id=m.run_id where r.name=? and m.key=? and m.step>=? and m.step<?",
            (name, k, lo, hi)).fetchone()[0]
    return out


def collect(root, tag):
    base, pattern, _ = SETS[tag]
    rows = []
    for run in sorted(glob.glob(f"{root}/{base}/{pattern}")):
        name = os.path.basename(run)
        files = sorted(glob.glob(f"{run}/simulations/{name}/storage/*/weights.npy"))
        for f in files:
            epoch = int(os.path.basename(os.path.dirname(f)))
            w = np.load(f, allow_pickle=True)[0]
            W = w["policy"].T
            r = dict(set=tag, run=name, epoch=epoch, **prototype_stats(W))
            r.update(metric_means(f"{root}/{base}/runs.db", name, max(epoch - 50, 0), max(epoch, 1)))
            rows.append(r)
    return rows


def slope(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    return float(np.polyfit(x, y, 1)[0]) if len(x) > 2 else np.nan


def first_epoch(df, col, thr, above):
    d = df[df[col] >= thr] if above else df[df[col] <= thr]
    return d.epoch.min() if len(d) else np.nan


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", default=os.path.expanduser("~/tmp/kss"))
    p.add_argument("--out", default=os.path.expanduser("~/tmp/kss/context4_probe/check1"))
    a = p.parse_args()
    os.makedirs(a.out, exist_ok=True)
    t = pd.concat([pd.DataFrame(collect(a.root, tag)) for tag in SETS])
    t.to_csv(f"{a.out}/trajectory.csv", index=False)

    summ = []
    for (s, r), d in t.groupby(["set", "run"]):
        d = d.sort_values("epoch")
        late = d[d.epoch >= 300]
        summ.append(dict(
            set=s, run=r, share_150=d[d.epoch == 150].common_share.mean(), share_final=d.common_share.iloc[-1],
            slope_share_late=slope(late.epoch, late.common_share), slope_sigma_late=slope(late.epoch, late.mean_sigma),
            slope_resid_late=slope(late.epoch, late.residual_norm), slope_common_late=slope(late.epoch, late.common_norm),
            epoch_sigma_lt3=first_epoch(d, "mean_sigma", 3.0, False), epoch_share_gt08=first_epoch(d, "common_share", 0.8, True),
            sigma_final=d.mean_sigma.iloc[-1]))
    summ = pd.DataFrame(summ)
    summ["lag_share_after_sigma"] = summ.epoch_share_gt08 - summ.epoch_sigma_lt3
    summ.to_csv(f"{a.out}/summary.csv", index=False)
    pd.set_option("display.width", 250, "display.max_columns", 40)
    print(summ.round(4).to_string(index=False))
    print(summ.groupby("set").mean(numeric_only=True).round(4).T.to_string())

    # cross-run regression: common share on sigma with epoch as second regressor (checkpoints >= 100)
    x = t[(t.epoch >= 100)].dropna(subset=["mean_sigma", "common_share"])
    for cols in (["mean_sigma"], ["mean_sigma", "epoch"], ["epoch"]):
        X = np.column_stack([np.ones(len(x))] + [x[c].values if c != "epoch" else np.log(x[c].values) for c in cols])
        beta, res, *_ = np.linalg.lstsq(X, x.common_share.values, rcond=None)
        r2 = 1 - ((x.common_share.values - X @ beta) ** 2).sum() / ((x.common_share.values - x.common_share.mean()) ** 2).sum()
        print("regress common_share ~", cols, "(epoch as log)", np.round(beta, 4), "R2", round(r2, 3))

    import matplotlib
    matplotlib.use("agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 4, figsize=(18, 4))
    for s, d in t.groupby("set"):
        g = d.groupby("epoch")[["common_share", "residual_norm", "mean_sigma", "common_norm"]].mean()
        for i, c in enumerate(["common_share", "residual_norm", "mean_sigma", "common_norm"]):
            ax[i].plot(g.index, g[c], label=s)
            ax[i].set_title(c)
            ax[i].set_xlabel("epoch")
    ax[2].set_yscale("log")
    ax[0].legend()
    fig.tight_layout()
    fig.savefig(f"{a.out}/trajectory.png", dpi=110)


if __name__ == "__main__":
    main()

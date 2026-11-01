#!/usr/bin/env python
import argparse
import itertools
import sys
from pathlib import Path

import numpy as np
import pandas as pd

SRC = Path(__file__).resolve().parent
sys.path.insert(0, str(SRC))
sys.path.insert(0, str(SRC.parent / "tools" / "box2dsim"))

import box2dsim  # noqa: E402,F401
from params import Parameters  # noqa: E402
from SMEnv import SMEnv  # noqa: E402

SIGN = np.array([-1, -1, -1, 1, 1, 1, 1, 1])


def collect(fovea_range, reps, steps, seed):
    base_params = Parameters()
    base = np.hstack([base_params.policy_base_arm, base_params.policy_base_wrist, np.ones(4) * base_params.policy_base])
    rng = np.random.RandomState(seed)
    rows, retinas = [], []
    contexts = list(itertools.product((1, 2, 3), base_params.obj_stretch_conditions, base_params.obj_rotation_conditions))
    for rep in range(reps):
        for kind, stretch, rotation in contexts:
            params = Parameters(fovea_range=fovea_range)
            env = SMEnv(seed + rep, params, params.action_steps,
                        rand_obj_params={"stretch_conditions": [stretch], "rotation_conditions": [rotation],
                                         "pos": [params.obj_x, params.obj_y]})
            obs = env.reset(world=kind)
            color = np.array(env.get_b2d_state()["color"], float)
            u = rng.uniform(-0.7, 0.7, 8)
            group = rep * len(contexts) + contexts.index((kind, stretch, rotation))
            for t in range(steps):
                if t % 5 == 0:
                    u = np.clip(u + rng.randn(8) * 0.35, -1, 1)
                obs = env.step(u * SIGN + base)
                if t % params.action_steps == 0:
                    retina = obs["VISUAL_SENSORS"].copy()
                    obj_px = int((np.abs(retina - color).max(-1) < 0.05).sum())
                    other_px = int(((retina < 0.999).any(-1)).sum()) - obj_px
                    rows.append(dict(range=fovea_range, group=group, kind=kind, stretch=stretch, rotation=round(rotation, 2),
                                     object_px=obj_px, other_px=other_px))
                    retinas.append(retina.ravel())
    return pd.DataFrame(rows), np.array(retinas)


def decode(table, retinas, target, folds=5):
    from sklearn.decomposition import PCA
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GroupKFold, cross_val_score
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    y = table[target].astype(str) if target != "context" else table[["kind", "stretch", "rotation"]].astype(str).agg("_".join, axis=1)
    cv = GroupKFold(folds)
    out = {}
    models = {"logreg": make_pipeline(StandardScaler(), LogisticRegression(max_iter=300)),
              "knn": KNeighborsClassifier(5),
              "pca10_logreg": make_pipeline(PCA(10), StandardScaler(), LogisticRegression(max_iter=300))}
    for name, model in models.items():
        out[name] = float(cross_val_score(model, retinas, y, groups=table.group, cv=cv).mean())
    out["chance"] = float(y.value_counts(normalize=True).max())
    return out


def contact_sheet(frames, ranges, path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(len(ranges), 6, figsize=(9, 1.6 * len(ranges)))
    for i, r in enumerate(ranges):
        table, retinas = frames[r]
        shown = []
        for kind in (1, 2, 3):
            for stretch in (1, 2):
                sel = table[(table.kind == kind) & (table.stretch == stretch) & (table.rotation == 0)]
                shown.append(sel.object_px.idxmax() if len(sel) else None)
        for j, idx in enumerate(shown):
            ax = axes[i, j]
            if idx is not None:
                ax.imshow(retinas[table.index.get_loc(idx)].reshape(10, 10, 3))
            ax.set_xticks([])
            ax.set_yticks([])
            if j == 0:
                ax.set_ylabel(f"range {r}")
            if i == 0:
                ax.set_title(f"kind {j // 2 + 1} stretch {j % 2 + 1}", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=120)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--ranges", default="4,8,12,16")
    parser.add_argument("--reps", type=int, default=8)
    parser.add_argument("--steps", type=int, default=60)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    ranges = [int(r) for r in args.ranges.split(",")]
    frames, rows = {}, []
    for r in ranges:
        table, retinas = collect(r, args.reps, args.steps, args.seed)
        frames[r] = (table, retinas)
        np.save(out / f"retinas_{r}.npy", retinas)
        table.to_csv(out / f"retinas_{r}.csv", index=False)
        row = dict(range=r, n=len(table), pixel_size=r / 10, frac_with_object=float((table.object_px > 0).mean()),
                   mean_object_px=float(table.object_px.mean()), mean_object_px_when_seen=float(table.object_px[table.object_px > 0].mean()),
                   mean_other_px=float(table.other_px.mean()),
                   frac_object_and_other=float(((table.object_px > 0) & (table.other_px > 0)).mean()))
        for target in ("kind", "stretch", "rotation", "context"):
            for name, value in decode(table, retinas, target).items():
                row[f"{target}_{name}"] = value
        rows.append(row)
        print(pd.Series(row).round(3).to_string(), flush=True)
    summary = pd.DataFrame(rows)
    summary.to_csv(out / "fovea_diagnostics.csv", index=False)
    contact_sheet(frames, ranges, out / "fovea_contact_sheet.png")
    print(summary.round(3).T.to_string())


if __name__ == "__main__":
    main()

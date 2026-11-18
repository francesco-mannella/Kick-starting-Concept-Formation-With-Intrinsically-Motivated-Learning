#!/usr/bin/env python
import argparse
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score


def accuracy(x, y, seed=0):
    classes, counts = np.unique(y, return_counts=True)
    if len(classes) < 2 or counts.min() < 5:
        return np.nan
    cv = StratifiedKFold(n_splits=min(5, counts.min()), shuffle=True, random_state=seed)
    clf = RandomForestClassifier(n_estimators=100, random_state=seed, n_jobs=4)
    return float(cross_val_score(clf, x, y, cv=cv).mean())


def parse_arguments():
    parser = argparse.ArgumentParser(description="Can shape, position or rotation be decoded from the first-contact sensory outcome?")
    parser.add_argument("--dir", default=str(Path.home() / "tmp/kss/l2_contacts"))
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    d = np.load(Path(args.dir) / "contacts.npz")
    shapes, xs, ys, rots, step, touch, proprio = (d[k] for k in ("shape", "x", "y", "rot", "step", "touch", "proprio"))
    rows = []
    for c in range(len(shapes)):
        hit = step[c] >= 0
        for i in np.where(hit)[0]:
            rows.append((shapes[c], xs[c], ys[c], rots[c], i, touch[c][i], proprio[c][i]))
    shape = np.array([r[0] for r in rows])
    pos = np.array([f"{r[1]:.1f},{r[2]:.1f}" for r in rows])
    rot = np.array([r[3] for r in rows])
    pidx = np.array([r[4] for r in rows])
    touch_x = np.array([r[5] for r in rows])
    touch_b = (touch_x > 0).astype(float)
    n_sens = touch_b.sum(1, keepdims=True)
    left_right = np.column_stack([touch_b[:, :20].sum(1), touch_b[:, 20:].sum(1)])
    prop = np.array([r[6] for r in rows])
    feats = {"touch only": touch_b, "proprio only": prop, "touch+proprio": np.hstack([touch_b, prop]),
             "touch summary (count, left, right)": np.hstack([n_sens, left_right])}
    print(f"{len(rows)} first-contact events over {len(shapes)} (shape, position, rotation) cells")
    uniq_pos = sorted(set(pos))
    print("\nA. SHAPE decoded from the first contact, at the same position (chance 1/6 = 0.17); mean over positions and rotations")
    for name, x in feats.items():
        accs = []
        for p in uniq_pos:
            for r in (0, 90):
                m = (pos == p) & (rot == r)
                a = accuracy(x[m], shape[m])
                if not np.isnan(a):
                    accs.append(a)
        print(f"  {name:36s} {np.mean(accs):.2f}  (min {np.min(accs):.2f}, max {np.max(accs):.2f}, {len(accs)} groups)")
    print("\nB. POSITION decoded from the first contact, same shape (chance 1/9 = 0.11); mean over shapes and rotations")
    for name, x in feats.items():
        accs = []
        for s in sorted(set(shape)):
            for r in (0, 90):
                m = (shape == s) & (rot == r)
                a = accuracy(x[m], pos[m])
                if not np.isnan(a):
                    accs.append(a)
        print(f"  {name:36s} {np.mean(accs):.2f}  (min {np.min(accs):.2f}, max {np.max(accs):.2f}, {len(accs)} groups)")
    print("\nC. ROTATION (0 vs 90) decoded from the first contact, same shape and position (chance 0.50); mean")
    for name, x in feats.items():
        accs = []
        for s in sorted(set(shape)):
            for p in uniq_pos:
                m = (shape == s) & (pos == p)
                a = accuracy(x[m], rot[m])
                if not np.isnan(a):
                    accs.append(a)
        print(f"  {name:36s} {np.mean(accs):.2f}  (min {np.min(accs):.2f}, max {np.max(accs):.2f}, {len(accs)} groups)")
    print("\nD. mean touch footprint (sensors touched at first contact) by shape, and spread of mean joint posture by position")
    for s in sorted(set(shape)):
        m = shape == s
        print(f"  {s:6s} sensors {n_sens[m].mean():.2f}")
    for p in uniq_pos:
        m = (pos == p) & (shape == "cube2")
        if m.sum():
            print(f"  cube2 at {p:9s} n={m.sum():3d} mean joints (arm1 arm2 arm3 prox dist) {np.round(prop[m].mean(0), 2)}")

import itertools
import json
import pickle
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import box2dsim  # noqa: F401
import numpy as np

import design_sweep as D
import tune_l2_objects as T

SEED = 1
GROUPS = [range(0, 5), range(5, 10), range(10, 15), range(15, 20), range(20, 25), range(25, 30), range(30, 35), range(35, 40)]
DELTA = 0.25
out = Path(sys.argv[1])
prop = json.load(open(out / "design_proposals.json"))


def cell(args):
    meta, path, pos, rot, stretch, world = args
    orig = T.postures
    T.postures = lambda s, n: orig(SEED, n)
    try:
        return D.cell(args)
    finally:
        T.postures = orig


def shape_path(shape, scale):
    i = [s for s, _ in D.SHAPES_D].index(shape)
    cand = dict(id=100 + i, shape=shape, scale=scale, density=0.2, lin=20.0, ang=20.0, friction=2.0)
    return str(D.E.model_file(cand, out / "models"))


sets = {}
sets["current layout"] = [(dict(mode="base", shape=k, scale=1.0, stretch=s, rot=r, pos=None), None, None, r, s, w)
                          for k, w in D.KINDS.items() for s, r in itertools.product(D.STRETCH, D.ROTS)]
best = prop["structured"][0]
sets["best structured design"] = []
for kd in best["kinds"]:
    path = shape_path(kd["shape"], dict(D.SHAPES_D)[kd["shape"]])
    pos = tuple(kd["position"])
    for s, r in itertools.product(D.STRETCH, D.ROTS):
        sets["best structured design"].append((dict(mode="lib", shape=kd["shape"], scale=1.0, stretch=s, rot=r, pos=pos), path, pos, r, s, 3))
sets["best free design"] = []
for c in prop["free"]:
    sh = c["shape"]
    pos = tuple(eval(c["pos"]) if isinstance(c["pos"], str) else c["pos"])
    sets["best free design"].append((dict(mode="lib", shape=sh, scale=1.0, stretch=c["stretch"], rot=c["rot"], pos=pos), shape_path(sh, dict(D.SHAPES_D)[sh]), pos, c["rot"], c["stretch"], 3))
jobs = [j for v in sets.values() for j in v]
with ProcessPoolExecutor(8) as pool:
    res = list(pool.map(cell, jobs, chunksize=2))
k = 0
for name, v in sets.items():
    cells = res[k:k + len(v)]
    k += len(v)
    masks, contact, joints = [], [], []
    for _, d in cells:
        frac = d["frac"].astype(float)
        m = np.zeros(len(frac), dtype=int)
        for g, idx in enumerate(GROUPS):
            m |= ((frac[:, list(idx)] >= 0.05).any(1)).astype(int) << g
        masks.append(m)
        contact.append(d["contact"] & (m > 0))
        joints.append(d["first_joints"][:, :3])
    masks, contact, joints = np.array(masks), np.array(contact), np.array(joints)
    n = len(cells)
    S = np.zeros((n, n))
    for a in range(n):
        if contact[a].sum():
            same = (masks == masks[a]) & contact & contact[a][None, :] & (np.linalg.norm(joints - joints[a][None], axis=2) <= DELTA)
            S[a] = same.sum(1) / contact[a].sum()
    S = (S + S.T) / 2
    pairs = np.array([S[a, b] for a, b in itertools.combinations(range(n), 2)])
    shared = []
    q = np.round(joints / 0.5).astype(int)
    for p in range(contact.shape[1]):
        act = [i for i in range(n) if contact[i][p]]
        if act:
            keys = [(masks[i][p], tuple(q[i][p])) for i in act]
            shared.append(max(keys.count(x) for x in set(keys)))
    print(f"{name:26s} (fresh postures, seed {SEED}): mean goal transfer {pairs.mean():.3f}  90th pct {np.quantile(pairs, .9):.2f}  max {pairs.max():.2f}  pairs <= 0.05: {(pairs <= 0.05).mean():.2f}  best posture shares a goal in {max(shared)}/{n} contexts  min contact rate {contact.mean(1).min():.2f}")

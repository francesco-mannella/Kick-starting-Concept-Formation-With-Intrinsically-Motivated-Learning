import itertools
import sys
from pathlib import Path

import box2dsim  # noqa: F401
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score

import explore_shapes_l2 as E
import tune_l2_objects as T

HALF_PI = np.pi / 2
out = Path(sys.argv[1])
out.mkdir(parents=True, exist_ok=True)
N = 400
E.SHAPES["stick"] = E.box(0.5, 3.0)
E.SHAPES["stick_short"] = E.box(0.5, 1.5)
E.SHAPES["stick_thick"] = E.box(1.0, 3.0)


def sample(n, nj, seed=0):
    rng = np.random.default_rng(seed)
    if nj == 8:
        return np.hstack([rng.uniform(-HALF_PI, HALF_PI, (n, 4)), rng.uniform(0, HALF_PI, (n, 4))])
    return T.postures(seed, n)


def run(shape, rot, density):
    cand = dict(id=0, shape=shape, scale=1.0, density=density, lin=20.0, ang=20.0, friction=2.0)
    path = E.model_file(cand, out)
    params = T.Parameters(obj_stretch_conditions=[1], obj_rotation_conditions=[T.ROT[rot]])
    env = T.SMEnv(0, params)
    env.b2d_env.world_files[T.WORLD["bar"]] = Path(path)
    nj = env.b2d_env.num_joints
    rows = []
    for a in sample(N, nj):
        env.reset(world=T.WORLD["bar"])
        body = env.b2d_env.sim.bodies[env.b2d_env.object_names[0]]
        if not rows:
            v = np.array([body.transform * q for q in body.fixtures[0].shape.vertices])
            run.bbox = np.round(v.max(0) - v.min(0), 2)
        c0 = np.array(body.worldCenter)
        touches = []
        for _ in range(60):
            obs = env.step(a)
            touches.append(np.asarray(obs["TOUCH_SENSORS"], float) > 0)
        touches = np.array(touches)
        rows.append(dict(contact=touches.any(), maxs=int(touches.sum(1).max()), frac=touches.mean(0), moved=float(np.linalg.norm(np.array(body.worldCenter) - c0)) >= 2.0,
                         left=bool(touches[:, :20].any()), right=bool(touches[:, 20:].any())))
    return rows, run.bbox, nj


def jac(a, b):
    u = (a | b).sum()
    return (a & b).sum() / u if u else np.nan


for shape, density in (("stick", 0.2), ("stick", 1.0), ("stick_short", 0.2), ("stick_thick", 0.2)):
    res = {}
    for rot in (0, 90):
        res[rot] = run(shape, rot, density)
    nj = res[0][2]
    print(f"\n{nj}-joint hand, {shape} (bbox at rot 0: {res[0][1].tolist()}, at rot 90: {res[90][1].tolist()}), density {density}, {N} identical postures")
    for rot, (rows, bbox, _) in res.items():
        c = np.array([r["contact"] for r in rows]); m = np.array([r["maxs"] for r in rows]); mv = np.array([r["moved"] for r in rows])
        print(f"  rot {rot:2d} ({'vertical, along the fingers' if rot == 0 else 'horizontal, across the fingers'}): claw contact {c.mean():.3f}  moved {mv.mean():.3f}  max sensors | contact {m[c].mean():.2f}  contact with >=4 sensors {(m >= 4).mean():.3f}  both fingers touched {np.mean([r['left'] and r['right'] for r in rows]):.3f}")
    ca = np.array([r["contact"] for r in res[0][0]]); cb = np.array([r["contact"] for r in res[90][0]])
    ma = np.array([r["moved"] for r in res[0][0]]); mb = np.array([r["moved"] for r in res[90][0]])
    ra = np.array([r["maxs"] >= 4 for r in res[0][0]]); rb = np.array([r["maxs"] >= 4 for r in res[90][0]])
    print(f"  overlap vertical vs horizontal: claw-contact sets {jac(ca, cb):.2f}, moved sets {jac(ma, mb):.2f}, >=4-sensor sets {jac(ra, rb):.2f}")
    x = np.vstack([np.array([r["frac"] for r in res[0][0]])[ca], np.array([r["frac"] for r in res[90][0]])[cb]])
    y = np.array([0] * int(ca.sum()) + [1] * int(cb.sum()))
    acc = cross_val_score(RandomForestClassifier(100, random_state=0, n_jobs=4), x, y, cv=StratifiedKFold(5, shuffle=True, random_state=0)).mean()
    mx = np.vstack([[[r["maxs"]] for r in res[0][0]][i] for i in np.where(ca)[0]] + [[[r["maxs"]] for r in res[90][0]][i] for i in np.where(cb)[0]])
    acc2 = cross_val_score(RandomForestClassifier(100, random_state=0, n_jobs=4), mx, y, cv=StratifiedKFold(5, shuffle=True, random_state=0)).mean()
    print(f"  decode vertical vs horizontal from the time-integrated 40-sensor touch pattern: {acc:.2f}; from the number of sensors alone: {acc2:.2f} (chance 0.50, {len(y)} contacts)")

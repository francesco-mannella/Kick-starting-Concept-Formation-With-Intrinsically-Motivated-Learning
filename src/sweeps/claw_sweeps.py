import itertools
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import box2dsim  # noqa: F401
import numpy as np

import explore_positions_l2 as P
import explore_shapes_l2 as E
import tune_l2_objects as T

HALF_PI = np.pi / 2
E.SHAPES["stick"] = E.box(0.5, 3.0)
N, STEPS = 250, 60
ROTS_A, ROTS_B = (0, 45, 90), (0, 90)
SHAPES_A = ["cube1", "cube2", "stick", "bar", "plate", "disk3", "disk5", "hex4", "wedge"]
SHAPES_B = [("cube2", 1.0), ("stick", 1.0), ("bar", 1.0), ("plate", 0.6), ("disk3", 1.0), ("wedge", 1.0)]


def sample(nj):
    rng = np.random.default_rng(0)
    if nj == 8:
        return np.hstack([rng.uniform(-HALF_PI, HALF_PI, (N, 4)), rng.uniform(0, HALF_PI, (N, 4))])
    return T.postures(0, N)


def cell(args):
    path, pos, rot = args
    kw = {"obj_stretch_conditions": [1], "obj_rotation_conditions": [np.radians(rot)]}
    if pos is not None:
        kw.update(obj_x=pos[1] - P.BASE[1], obj_y=pos[0] - P.BASE[0])
    env = T.SMEnv(0, T.Parameters(**kw))
    env.b2d_env.world_files[T.WORLD["bar"]] = Path(path)
    nj = env.b2d_env.num_joints
    contact, maxs, frac, moved, t0s, ft, fj = [], [], [], [], [], [], []
    for a in sample(nj):
        env.reset(world=T.WORLD["bar"])
        body = env.b2d_env.sim.bodies[env.b2d_env.object_names[0]]
        c0 = np.array(body.worldCenter)
        touches, joints = [], []
        for _ in range(STEPS):
            obs = env.step(a)
            touches.append(np.asarray(obs["TOUCH_SENSORS"]) > 0)
            joints.append(np.asarray(obs["JOINT_POSITIONS"], float))
        touches, joints = np.array(touches), np.array(joints)
        anyt = touches.any(1)
        t0 = int(np.argmax(anyt)) if anyt.any() else -1
        contact.append(bool(anyt.any()))
        maxs.append(int(touches.sum(1).max()))
        frac.append(touches.mean(0).astype(np.float16))
        moved.append(float(np.linalg.norm(np.array(body.worldCenter) - c0)) >= 2.0)
        t0s.append(t0)
        ft.append(touches[t0] if t0 >= 0 else np.zeros(40, bool))
        fj.append(joints[t0] if t0 >= 0 else np.zeros(joints.shape[1]))
    return dict(contact=np.array(contact), maxs=np.array(maxs, dtype=np.int8), frac=np.stack(frac), moved=np.array(moved), t0=np.array(t0s),
                first_touch=np.stack(ft), first_joints=np.stack(fj))


def work(args):
    meta, a = args
    return meta, cell(a)


if __name__ == "__main__":
    out = Path(sys.argv[1])
    (out / "models").mkdir(parents=True, exist_ok=True)
    jobs = []
    i = 0
    for sh, sc, d in itertools.product(SHAPES_A, [0.6, 1.0, 1.6], [0.2, 1.0]):
        cand = dict(id=i, shape=sh, scale=sc, density=d, lin=20.0, ang=20.0, friction=2.0)
        path = str(E.model_file(cand, out / "models"))
        for rot in ROTS_A:
            jobs.append((dict(mode="A", shape=sh, scale=sc, density=d, pos=None, rot=rot), (path, None, rot)))
        i += 1
    for sh, sc in SHAPES_B:
        cand = dict(id=i, shape=sh, scale=sc, density=1.0, lin=20.0, ang=20.0, friction=2.0)
        path = str(E.model_file(cand, out / "models"))
        for pos in P.POSITIONS:
            for rot in ROTS_B:
                jobs.append((dict(mode="B", shape=sh, scale=sc, density=1.0, pos=pos, rot=rot), (path, pos, rot)))
        i += 1
    print(len(jobs), "cells,", len(jobs) * N, "rollouts", flush=True)
    with ProcessPoolExecutor(8) as pool:
        res = list(pool.map(work, jobs, chunksize=2))
    import pickle
    pickle.dump(res, open(out / "claw_sweeps.pkl", "wb"))
    print("saved", len(res), flush=True)

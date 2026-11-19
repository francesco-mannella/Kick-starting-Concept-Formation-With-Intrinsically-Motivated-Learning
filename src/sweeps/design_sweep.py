import itertools
import pickle
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import box2dsim  # noqa: F401
import numpy as np

import explore_positions_l2 as P
import explore_shapes_l2 as E
import tune_l2_objects as T

N, STEPS = 250, 60
E.SHAPES["stick"] = E.box(0.5, 3.0)
E.SHAPES["tri78"] = (np.array([0.0, 0.0, 7.74]), np.array([7.84, 0.0, 3.92]))
SHAPES_D = [("cube2", 1.0), ("stick", 1.0), ("plate", 0.6), ("disk3", 1.0), ("wedge", 1.0), ("tri78", 1.0)]
POS_D = list(itertools.product((7.0, 10.7, 14.0), (4.0, 6.0, 9.3, 12.0)))
STRETCH, ROTS = (1, 2), (0, 45, 90)
KINDS = {"still": 1, "movable": 2, "controllable": 3}


def cell(args):
    meta, path, pos, rot, stretch, world = args
    kw = {"obj_stretch_conditions": [stretch], "obj_rotation_conditions": [np.radians(rot)]}
    if meta["mode"] == "base":
        kw.update(obj_damping=20.0, obj_friction=2.0)
    if pos is not None:
        kw.update(obj_x=pos[1] - P.BASE[1], obj_y=pos[0] - P.BASE[0])
    env = T.SMEnv(0, T.Parameters(**kw))
    if path is not None:
        env.b2d_env.world_files[world] = Path(path)
    contact, maxs, frac, fj = [], [], [], []
    for a in T.postures(0, N):
        env.reset(world=world)
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
        fj.append(joints[t0] if t0 >= 0 else np.zeros(joints.shape[1]))
    return meta, dict(contact=np.array(contact), maxs=np.array(maxs, dtype=np.int8), frac=np.stack(frac), first_joints=np.stack(fj))


if __name__ == "__main__":
    out = Path(sys.argv[1])
    (out / "models").mkdir(parents=True, exist_ok=True)
    jobs = []
    for i, (sh, sc) in enumerate(SHAPES_D):
        cand = dict(id=i, shape=sh, scale=sc, density=0.2, lin=20.0, ang=20.0, friction=2.0)
        path = str(E.model_file(cand, out / "models"))
        for stretch, rot, pos in itertools.product(STRETCH, ROTS, POS_D):
            jobs.append((dict(mode="lib", shape=sh, scale=sc, stretch=stretch, rot=rot, pos=pos), path, pos, rot, stretch, 3))
    for kind, world in KINDS.items():
        for stretch, rot in itertools.product(STRETCH, ROTS):
            jobs.append((dict(mode="base", shape=kind, scale=1.0, stretch=stretch, rot=rot, pos=None), None, None, rot, stretch, world))
    print(len(jobs), "cells,", len(jobs) * N, "rollouts", flush=True)
    with ProcessPoolExecutor(8) as pool:
        res = list(pool.map(cell, jobs, chunksize=2))
    pickle.dump(res, open(out / "design_sweep.pkl", "wb"))
    print("saved", len(res), flush=True)

import itertools
import json
import sys
from pathlib import Path

import box2dsim  # noqa: F401
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score

import explore_shapes_l2 as E
import tune_l2_objects as T

HALF_PI = np.pi / 2
STRETCH, ROTS = (1, 2), (0, 45, 90)
LEFT, RIGHT = slice(0, 20), slice(20, 40)
out = Path(sys.argv[1])
out.mkdir(parents=True, exist_ok=True)
cand = dict(id=0, shape="cube2", scale=1.0, density=0.2, lin=20.0, ang=20.0, friction=2.0)
path = E.model_file(cand, out)


def sample(n, njoint, seed=0):
    rng = np.random.default_rng(seed)
    if njoint == 8:
        return np.hstack([rng.uniform(-HALF_PI, HALF_PI, (n, 4)), rng.uniform(0, HALF_PI, (n, 4))])
    return T.postures(seed, n)


def record(env, action, steps=60):
    env.reset(world=T.WORLD["bar"])
    sim = env.b2d_env.sim
    body = sim.bodies[env.b2d_env.object_names[0]]
    c0, a0 = np.array(body.worldCenter), body.angle
    touches, joints = [], []
    for _ in range(steps):
        obs = env.step(action)
        touches.append(np.asarray(obs["TOUCH_SENSORS"], float))
        joints.append(np.asarray(obs["JOINT_POSITIONS"], float))
    touches, joints = np.array(touches), np.array(joints)
    end = float(np.linalg.norm(np.array(body.worldCenter) - c0))
    rot = float(np.degrees(abs(body.angle - a0)))
    anyt = touches.sum(1) > 0
    t0 = int(np.argmax(anyt)) if anyt.any() else -1
    effect = end >= 2.0 or rot >= 60.0
    return effect, t0, (touches[t0] > 0) if t0 >= 0 else np.zeros(40, bool), joints[t0] if t0 >= 0 else np.zeros(joints.shape[1])


results = {}
for s, r in itertools.product(STRETCH, ROTS):
    params = T.Parameters(obj_stretch_conditions=[s], obj_rotation_conditions=[T.ROT.get(r, np.radians(r))])
    env = T.SMEnv(0, params)
    env.b2d_env.world_files[T.WORLD["bar"]] = Path(path)
    nj = env.b2d_env.num_joints
    rows = [record(env, a) for a in sample(300, nj)]
    results[(s, r)] = dict(effect=np.array([x[0] for x in rows]), t0=np.array([x[1] for x in rows]),
                           touch=np.stack([x[2] for x in rows]), joints=np.stack([x[3] for x in rows]))
print(f"hand with {nj} joints: single green cube (champion physics), stretch x rotation variants, 300 identical postures each")
for k, v in results.items():
    print(f"  stretch {k[0]} rotation {k[1]:2d}: effect rate {v['effect'].mean():.3f}   first-contact rate {(v['t0'] >= 0).mean():.3f}   mean sensors at first contact {v['touch'][v['t0'] >= 0].sum(1).mean():.2f}")
pairs = []
for a, b in itertools.combinations(results, 2):
    ea, eb = results[a]["effect"], results[b]["effect"]
    pairs.append(((a, b), (ea & eb).sum() / (ea | eb).sum()))
cpairs, rpairs = [], []
for a, b in itertools.combinations(results, 2):
    ca, cb = results[a]["t0"] >= 0, results[b]["t0"] >= 0
    cpairs.append(((a, b), (ca & cb).sum() / (ca | cb).sum()))
    ra, rb = results[a]["touch"].sum(1) >= 4, results[b]["touch"].sum(1) >= 4
    rpairs.append(((a, b), (ra & rb).sum() / max((ra | rb).sum(), 1)))
cv = np.array([p[1] for p in cpairs]); rv = np.array([p[1] for p in rpairs])
print(f"overlap of CLAW-CONTACT posture sets between variants: mean {cv.mean():.2f} min {cv.min():.2f} max {cv.max():.2f}")
print(f"  same rotation, different stretch: {np.mean([p[1] for p in cpairs if p[0][0][1]==p[0][1][1]]):.2f}; same stretch, different rotation: {np.mean([p[1] for p in cpairs if p[0][0][0]==p[0][1][0]]):.2f}")
print(f"overlap of postures with >=4 sensors at first contact: mean {rv.mean():.2f} min {rv.min():.2f}")
vals = np.array([p[1] for p in pairs])
print(f"overlap of effective postures between variants: mean {vals.mean():.2f} min {vals.min():.2f} max {vals.max():.2f}")
lo = min(pairs, key=lambda p: p[1])
print(f"  least overlap: {lo[0][0]} vs {lo[0][1]} = {lo[1]:.2f}")
same_rot = [p[1] for p in pairs if p[0][0][1] == p[0][1][1]]
same_st = [p[1] for p in pairs if p[0][0][0] == p[0][1][0]]
print(f"  same rotation, different stretch: mean {np.mean(same_rot):.2f}; same stretch, different rotation: mean {np.mean(same_st):.2f}")
for feat in ("touch", "joints", "touch+joints"):
    xs, ys = [], []
    for k, v in results.items():
        hit = v["t0"] >= 0
        f = {"touch": v["touch"][hit].astype(float), "joints": v["joints"][hit], "touch+joints": np.hstack([v["touch"][hit], v["joints"][hit]])}[feat]
        xs.append(f)
        ys += [f"{k}"] * int(hit.sum())
    x, y = np.vstack(xs), np.array(ys)
    cv = StratifiedKFold(5, shuffle=True, random_state=0)
    acc = cross_val_score(RandomForestClassifier(100, random_state=0, n_jobs=4), x, y, cv=cv).mean()
    print(f"  decode the variant (stretch, rotation) from the first contact, {feat}: {acc:.2f} (chance {1/6:.2f})")

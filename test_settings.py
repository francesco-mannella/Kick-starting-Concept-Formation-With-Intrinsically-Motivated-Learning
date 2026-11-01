import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "tools", "box2dsim"))
sys.path.insert(0, os.path.join(HERE, "src"))

import box2dsim  # noqa: E402,F401
from params import Parameters  # noqa: E402
from SMEnv import SMEnv  # noqa: E402

STILL, MOVABLE, CONTROLLABLE = 1, 2, 3


def make(seed=0, **options):
    params = Parameters(**options)
    return SMEnv(seed, params, params.action_steps)


def rollout(env, world, steps=30):
    obs = env.reset(world=world)
    out = [np.hstack([obs["TOUCH_SENSORS"], obs["JOINT_POSITIONS"], obs["VISUAL_SENSORS"].ravel()])]
    for _ in range(steps):
        obs = env.step(np.zeros(8))
        out.append(np.hstack([obs["TOUCH_SENSORS"], obs["JOINT_POSITIONS"], obs["VISUAL_SENSORS"].ravel()]))
    return np.array(out)


def body(env, name):
    return [b for b in env.b2d_env.world_dict["body"] if b["name"] == name][0]


for world in (STILL, MOVABLE, CONTROLLABLE):
    default = rollout(make(0), world)
    legacy_env = make(0)
    legacy_env.params = Parameters()
    legacy_env.b2d_env.rand_obj_params = dict(legacy_env.rand_obj_params)
    legacy_env.reset = lambda world=None, _e=legacy_env, **k: _e.b2d_env.reset(world, **k)
    legacy = rollout(legacy_env, world)
    assert np.array_equal(default, legacy), f"defaults must equal the legacy path (world {world})"

base_pos = make(0)
base_pos.reset(world=MOVABLE)
x_base, y_base = body(base_pos, "movable")["position"]["x"], body(base_pos, "movable")["position"]["y"]

a, b = make(3, obj_pos_jitter=(4.0, 4.0)), make(3, obj_pos_jitter=(4.0, 4.0))
positions_a, positions_b = [], []
for _ in range(6):
    a.reset(world=MOVABLE)
    b.reset(world=MOVABLE)
    positions_a.append((body(a, "movable")["position"]["x"], body(a, "movable")["position"]["y"]))
    positions_b.append((body(b, "movable")["position"]["x"], body(b, "movable")["position"]["y"]))
assert positions_a == positions_b, "jitter must be reproducible from the seed"
assert len(set(positions_a)) > 1, "jitter must differ between episodes"
base = body(a, "Base")["position"]
for x, y in positions_a:
    assert abs(x - x_base) <= 4.0 + 1e-9 and abs(y - y_base) <= 4.0 + 1e-9
    assert np.hypot(x - base["x"], y - base["y"]) <= 18.0 + 1e-9, "draws beyond the reach limit are rejected"

small = make(0, obj_scale=(1.0, 0.25, 1.0))
small.reset(world=MOVABLE)
normal = make(0)
normal.reset(world=MOVABLE)
ext = lambda e: np.ptp(np.array(body(e, "movable")["fixture"][0]["polygon"]["vertices"]["x"]))
assert np.isclose(ext(small), 0.25 * ext(normal)), (ext(small), ext(normal))
still_small = make(0, obj_scale=(1.0, 0.25, 1.0))
still_small.reset(world=STILL)
still_normal = make(0)
still_normal.reset(world=STILL)
assert np.isclose(ext_still := np.ptp(np.array(body(still_small, "still")["fixture"][0]["polygon"]["vertices"]["x"])),
                  np.ptp(np.array(body(still_normal, "still")["fixture"][0]["polygon"]["vertices"]["x"]))), "only the triangle is scaled"

heavy = make(0, obj_density=(1.0, 10.0, 0.2), obj_damping=(None, 4.0, None))
heavy.reset(world=MOVABLE)
plain = make(0)
plain.reset(world=MOVABLE)
assert np.isclose(body(heavy, "movable")["fixture"][0]["density"], 10 * body(plain, "movable")["fixture"][0]["density"])
assert body(heavy, "movable")["linearDamping"] == 4.0 and body(heavy, "movable")["angularDamping"] == 4.0
light = make(0, obj_density=(1.0, 10.0, 0.2), obj_damping=(None, 4.0, None))
light.reset(world=CONTROLLABLE)
plain_c = make(0)
plain_c.reset(world=CONTROLLABLE)
assert np.isclose(body(light, "controllable")["fixture"][0]["density"], 0.2 * body(plain_c, "controllable")["fixture"][0]["density"])
assert body(light, "controllable")["linearDamping"] == body(plain_c, "controllable")["linearDamping"]

env_re = make(0)
env_re.params = Parameters(obj_pos_jitter=(2.0, 2.0), obj_scale=(1.0, 0.25, 1.0))
env_re.reset(world=MOVABLE)
assert np.isclose(ext(env_re), 0.25 * ext(normal)), "settings must be applied at reset when params are set after construction"

print("ok")

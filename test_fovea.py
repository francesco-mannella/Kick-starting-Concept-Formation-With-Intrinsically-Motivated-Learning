import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "tools", "box2dsim"))
sys.path.insert(0, os.path.join(HERE, "src"))

import box2dsim  # noqa: E402,F401
from params import Parameters  # noqa: E402
from SMEnv import SMEnv  # noqa: E402


def rollout(fovea_range, steps=30, seed=0):
    params = Parameters() if fovea_range is None else Parameters(fovea_range=fovea_range)
    env = SMEnv(seed, params, params.action_steps)
    obs = env.reset(world=1)
    first = obs["VISUAL_SENSORS"].copy()
    retinas = [first]
    for _ in range(steps):
        obs = env.step(np.zeros(8))
        retinas.append(obs["VISUAL_SENSORS"].copy())
    return env, first, np.array(retinas)


default_env, first_default, default = rollout(None)
same_env, first_same, same = rollout(4)
assert default_env.b2d_env.fovea_width == 4
assert np.array_equal(default, same), "range 4 must equal the default"

footprint = {}
for r in (8, 12, 16):
    env, first, retinas = rollout(r)
    assert env.b2d_env.fovea_width == env.b2d_env.fovea_height == r
    assert retinas.shape == default.shape == (31, 10, 10, 3), retinas.shape
    assert np.isfinite(retinas).all()
    fovea = env.b2d_env.fovea
    assert np.allclose(fovea.scale, r / 10), fovea.scale
    covered = (retinas < 0.999).any(-1).mean()
    footprint[r] = covered

assert (first_same == first_default).all()
env12, first12, _ = rollout(12)
fresh = env12.b2d_env.fovea.step(env12.b2d_env.eye_pos)
assert fresh.shape == first12.shape

env_re = SMEnv(0, Parameters(), 5)
env_re.params = Parameters(fovea_range=12)
env_re.reset(world=1)
assert env_re.b2d_env.fovea_width == 12, "the range must be applied at reset when params are set after construction"

print("coverage of the retina by drawn bodies (default 4):", round(float((default < 0.999).any(-1).mean()), 3), {k: round(float(v), 3) for k, v in footprint.items()})
print("ok")

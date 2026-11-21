import box2dsim  # noqa: F401
import numpy as np

from params import Parameters
from SMEnv import SMEnv


def make(**kw):
    return SMEnv(0, Parameters(obj_stretch_conditions=[1], obj_rotation_conditions=[0.0], **kw))


def visual(env, world=2, steps=12):
    env.reset(world=world)
    out = []
    for _ in range(steps):
        obs = env.step(np.array([0.3, -0.4, 0.5, 0.6, 0.8]))
        out.append(np.array(obs["VISUAL_SENSORS"]).copy())
    return np.array(out)


def test_default_scale():
    assert Parameters().fovea_range == 4.0
    env = make()
    assert env.b2d_env.fovea_width == env.b2d_env.fovea_height == 4.0
    assert np.allclose(env.b2d_env.fovea.scale, [0.4, 0.4]) and env.b2d_env.fovea.shape == [10, 10]


def test_doubled_scale():
    env = make(fovea_range=8.0)
    assert np.allclose(env.b2d_env.fovea.scale, [0.8, 0.8]) and env.b2d_env.fovea.shape == [10, 10]
    assert env.reset(world=2)["VISUAL_SENSORS"].shape == (10, 10, 3)


def test_object_looks_smaller():
    a, b = visual(make()), visual(make(fovea_range=8.0))
    covered = lambda v: float((v.min(axis=-1) < 0.99).mean())
    assert not np.array_equal(a, b)
    assert covered(b) < covered(a), (covered(a), covered(b))


if __name__ == "__main__":
    test_default_scale()
    test_doubled_scale()
    test_object_looks_smaller()
    print("ok")

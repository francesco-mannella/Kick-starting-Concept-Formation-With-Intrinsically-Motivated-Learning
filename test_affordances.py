import itertools
import os
import pickle
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "tools", "box2dsim"))
sys.path.insert(0, os.path.join(HERE, "src"))

import box2dsim  # noqa: E402,F401
from params import Parameters  # noqa: E402
from SMEnv import SMEnv  # noqa: E402

BLUE, RED, GREEN = 1, 2, 3
NAMES = {BLUE: "blue_rotor_ccw", RED: "red_button", GREEN: "green_rotor_cw"}
JOINTS = {BLUE: "ratchet_ccw_blue", RED: "spring_red", GREEN: "ratchet_cw_green"}
FIXED = dict(stretch_conditions=[1], rotation_conditions=[0.0], pos=[2, 0.5])


def make(world, seed=0, scenario="affordance", **options):
    rand = options.pop("rand", {})
    params = Parameters(scenario=scenario, **options)
    env = SMEnv(seed, params, params.action_steps, rand_obj_params={**FIXED, **rand})
    env.reset(world=world)
    return env


def sim(env):
    return env.b2d_env.sim


def settle(env, steps, force=None, torque=None, world=RED):
    s = sim(env)
    body = s.bodies[NAMES[world]]
    trace = []
    for _ in range(steps):
        if force is not None:
            body.ApplyForceToCenter(tuple(force), True)
        if torque is not None:
            body.ApplyTorque(torque, True)
        s.step()
        trace.append(s.joints[JOINTS[world]].translation if world == RED else s.joints[JOINTS[world]].angle)
    return np.array(trace)


# original scenario is untouched by the new code
a, b = SMEnv(0, Parameters(), 5), SMEnv(0, Parameters(scenario="original"), 5)
for world in (1, 2, 3):
    oa, ob = a.reset(world=world), b.reset(world=world)
    assert np.array_equal(oa["VISUAL_SENSORS"], ob["VISUAL_SENSORS"])
    for _ in range(20):
        assert np.array_equal(a.step(np.zeros(8))["TOUCH_SENSORS"], b.step(np.zeros(8))["TOUCH_SENSORS"])
c = make(RED)
c.params = Parameters(scenario="original")
c.reset(world=RED)
assert [x for x in c.b2d_env.sim.bodies if x.startswith("red")] == [] and "movable" in c.b2d_env.sim.bodies, "scenario must switch back"

# red button: spring, travel, axis
for rot in (0.0, np.pi / 4, np.pi / 2):
    env = make(RED, rand={"rotation_conditions": [rot]})
    axis = np.array([np.cos(rot), np.sin(rot)])
    start = np.array(sim(env).bodies["red_button"].worldCenter)
    for sign in (+1, -1):
        trace = settle(env, 120, force=sign * 60 * axis)
        assert abs(trace[-1]) <= 2.0 + 0.05, f"travel limit exceeded {trace[-1]}"
        assert sign * trace[-1] > 1.0, f"button must move under force, got {trace[-1]}"
        moved = np.array(sim(env).bodies["red_button"].worldCenter) - start
        assert abs(moved @ np.array([-axis[1], axis[0]])) < 0.1, "no sideways motion"
        trace = settle(env, 240)
        assert abs(trace[-1]) < 0.1, f"button must return to rest, got {trace[-1]} (rot {rot})"
    transverse = settle(env, 120, force=60 * np.array([-axis[1], axis[0]]))
    assert abs(transverse[-1]) < 0.1, "force across the axis must not move the button along it"

# rotors: one-way ratchets, centre fixed
for world, direction in ((GREEN, -1.0), (BLUE, +1.0)):
    env = make(world)
    body = sim(env).bodies[NAMES[world]]
    centre = np.array(body.worldCenter)
    wrong = settle(env, 120, torque=-direction * 2.0, world=world)
    assert abs(wrong[-1] - wrong[0]) < 0.05, f"world {world} must not turn the wrong way: {wrong[-1] - wrong[0]}"
    right = settle(env, 200, torque=direction * 2.0, world=world)
    assert direction * (right[-1] - wrong[-1]) > 1.0, "rotor must turn in its direction"
    resting = settle(env, 200, world=world)
    assert abs(resting[-1] - resting[120]) < 0.05, "rotor stops and stays where it ends up"
    back = settle(env, 200, torque=-direction * 2.0, world=world)
    assert direction * (back[-1] - resting[-1]) < 0.05 and abs(back[-1] - resting[-1]) < 0.1, "ratchet must not return"
    assert np.allclose(body.worldCenter, centre, atol=1e-3), "centre must stay fixed"

# every colour builds in every shape, stretch and rotation
for world, shape, stretch, rot in itertools.product((BLUE, RED, GREEN), ("square", "triangle"), (1, 2), (0.0, np.pi / 4, np.pi / 2)):
    env = make(world, rand={"shape": shape, "stretch_conditions": [stretch], "rotation_conditions": [rot]})
    obj = sim(env).bodies[NAMES[world]]
    xs = np.array([v[0] for v in obj.fixtures[0].shape.vertices])
    ys = np.array([v[1] for v in obj.fixtures[0].shape.vertices])
    if world == RED:
        joint = [j for j in env.b2d_env.world_dict["joint"] if j["name"] == "spring_red"][0]
        assert np.allclose([joint["localAxisA"]["x"], joint["localAxisA"]["y"]], [np.cos(rot), np.sin(rot)])
    assert np.allclose(obj.position, sim(env).bodies[NAMES[world] + "_anchor"].position, atol=1e-6)
    if shape == "square" and stretch == 2 and rot == 0.0:
        assert np.ptp(xs) > 1.9 * np.ptp(ys), "stretch must make the x axis the long one"

# a reloaded checkpoint applies the scenario from its parameters
env = make(RED)
clone = pickle.loads(pickle.dumps(env))
clone.params = Parameters(scenario="affordance")
clone.reset(world=RED)
assert "red_button" in clone.b2d_env.sim.bodies

print("ok")

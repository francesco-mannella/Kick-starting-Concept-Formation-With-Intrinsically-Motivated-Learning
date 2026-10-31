import json
import os
import subprocess
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
NEW_PATH = os.path.join(HERE, "tools", "box2dsim")

PROBE = """
import json, sys, numpy as np, gymnasium as gym, box2dsim
env = gym.make("Box2DSimOneArmOneEye-v0").unwrapped
env.set_seed(0)
env.action_steps = 5
obs = env.reset(0)
n = int(sys.argv[1])
for _ in range(int(sys.argv[2])):
    obs = env.step(np.zeros(n))[0]
names = ["Arm3", "Claw11", "Claw12", "Claw21", "Claw22"]
print(json.dumps({k: list(map(float, env.sim.bodies[k].worldCenter)) for k in names}))
"""


def probe(path, n_actions, steps):
    env = dict(os.environ)
    if path:
        env["PYTHONPATH"] = path + os.pathsep + env.get("PYTHONPATH", "")
    out = subprocess.run([sys.executable, "-c", PROBE, str(n_actions), str(steps)], env=env, capture_output=True, text=True, cwd="/tmp")
    assert out.returncode == 0, out.stderr[-800:]
    return json.loads(out.stdout.strip().splitlines()[-1])


sys.path.insert(0, NEW_PATH)
import gymnasium as gym
import box2dsim

assert os.path.realpath(box2dsim.__file__).startswith(os.path.realpath(NEW_PATH)), box2dsim.__file__

env = gym.make("Box2DSimOneArmOneEye-v0").unwrapped
env.set_seed(0)
env.action_steps = 5
names = env.joint_names
assert len(names) == env.num_joints == 8 == env.num_joint_positions

obs = env.reset(0)
assert obs["JOINT_POSITIONS"].shape == (8,) and obs["TOUCH_SENSORS"].shape == (40,), (obs["JOINT_POSITIONS"].shape, obs["TOUCH_SENSORS"].shape)


def run(action, steps):
    env.reset(0)
    for _ in range(steps):
        o = env.step(np.array(action, dtype=float))[0]
    return o["JOINT_POSITIONS"]


def setpoints(action):
    env.reset(0)
    env.set_action(np.array(action, dtype=float))
    return np.array([env.sim.joint_pids[name].setpoint for name in names])


sign = np.array([1, 1, 1, 1] + [env.finger_signs[j] for j in names[4:]], dtype=float)

# 1. setpoints: each command goes to its own joint with the expected sign and range
for k in range(8):
    action = np.zeros(8)
    action[k] = 0.8
    sp = setpoints(action)
    expected = np.zeros(8)
    expected[k] = sign[k] * 0.8
    assert np.allclose(sp, expected), (names[k], sp)
for command, limit in ((5.0, np.pi / 2), (-5.0, -np.pi / 2)):
    sp = setpoints([command] * 8)
    assert np.allclose(sp[:4], limit), sp[:4]                     # arm and wrist clipped to +-pi/2, both directions
    assert np.allclose(sp[4:], sign[4:] * (np.pi / 2 if command > 0 else 0.0)), sp[4:]  # fingers close only
print("setpoints: ok")

# 2. early motion (before any finger contact): the commanded joint moves to the expected side, the rest stay put
for k in range(8):
    action = np.zeros(8)
    action[k] = 0.8
    steps = 80 if k < 3 else 10   # the heavy arm joints are slow
    angles = run(action, steps)
    others = np.delete(angles, k)
    print(f"{names[k]:18s} expected side {sign[k]:+.0f}  angle after {steps} steps {angles[k]:+.3f}  max |other joints| {np.abs(others).max():.3f}")
    assert np.sign(angles[k]) == sign[k] and abs(angles[k]) > 0.2
    if k >= 3:   # the wrist swings the fingers' inertia, so it gets a wider tolerance
        assert np.abs(others).max() < (0.3 if k == 3 else 0.15)

# 3. fingers are independent: closing one leaves the other open (short horizon, before they touch)
a = run([0, 0, 0, 0, 0.3, 0, 0, 0], 40)
b = run([0, 0, 0, 0, 0, 0, 0.3, 0], 40)
assert abs(a[6]) < 0.1 and abs(b[4]) < 0.1 and abs(a[4]) > 0.2 and abs(b[6]) > 0.2

# 4. wrist: the claws hang on the palm, the palm hangs on Arm3, and the palm turns with the wrist joint
for joint, a_name, b_name in (("Arm3_to_Wrist", "Arm3", "Wrist"), ("Wrist_to_Claw11", "Wrist", "Claw11"), ("Wrist_to_Claw21", "Wrist", "Claw21")):
    j = env.sim.joints[joint]
    assert (j.bodyA == env.sim.bodies[a_name]) and (j.bodyB == env.sim.bodies[b_name]), joint
wrist_angle = run([0, 0, 0, 0.6, 0, 0, 0, 0], 80)[3]
palm_turn = env.sim.bodies["Wrist"].angle - env.sim.bodies["Arm3"].angle
print(f"wrist joint angle {wrist_angle:+.3f}, palm angle relative to Arm3 {palm_turn:+.3f}")
assert abs(palm_turn - wrist_angle) < 0.02

# 4b. settling (regression test for a palm that was too light: fingers drifted to the wrong side and crossed):
# every wrist and finger joint reaches its own setpoint and the other wrist/finger joints stay put
for k in range(3, 8):
    value = 0.6 if k == 3 else 0.25   # a larger finger command swings the tip into the other finger (a real collision)
    action = np.zeros(8)
    action[k] = value
    angles = run(action, 160)
    others = np.delete(angles[3:], k - 3)
    print(f"settled {names[k]:18s} angle {angles[k]:+.3f} (setpoint {sign[k] * value:+.2f})  max |other wrist/finger joints| {np.abs(others).max():.3f}")
    assert abs(angles[k] - sign[k] * value) < 0.05, "the joint did not settle at its setpoint"
    assert np.abs(others).max() < 0.05, "a non-commanded wrist or finger joint moved"
combo = run([0, 0, 0, 0, 0.7, 0.7, 0.2, 0.9], 160)
assert np.allclose(combo[4:], sign[4:] * np.array([0.7, 0.7, 0.2, 0.9]), atol=0.06), combo[4:]

# 5. zero action reproduces the rest pose of the old 5-output environment (installed package = main checkout)
old = probe(None, 5, 20)
new = probe(NEW_PATH, 8, 20)
worst = max(np.abs(np.array(old[k]) - np.array(new[k])).max() for k in old)
print("rest pose, worst coordinate difference old vs new:", round(worst, 4))
assert worst < 0.05

print("ok")

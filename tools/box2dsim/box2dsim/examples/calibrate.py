import box2dsim
import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np


_ = box2dsim

rng = np.random.RandomState(62)
env = gym.make(
    "Box2DSimOneArmOneEye-v0",
    rand_obj_params={
        "stretch_conditions": [1, 1.5, 2],
        "rotation_conditions": np.pi * np.array([0, 0.25, 0.5]),
        "pos": [2, 0.4],
    },
).unwrapped

for world in range(1, 4):

    env.set_world(world)
    stime = 100
    init_action = np.pi * np.array([0.0, 0.0, 0.0, 0.5, 0])
    action = np.pi * np.array([0.0, 0.0, 0.0, 0, 0])

    env.reset()
    for t in range(stime):
        if t > stime * 0.6:

            action += 0.0 * rng.randn(5)
            action[:3] = np.pi * np.array([0.2, -0.4, -0.4])
            action[3:] = np.pi * np.array([0.2, 0.2])

            print(action)

            env.step(action)
            env.render()
            plt.pause(0.1)

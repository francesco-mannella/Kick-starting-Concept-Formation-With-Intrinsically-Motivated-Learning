import numpy as np
import gym
import box2dsim

rng = np.random.RandomState(62)
env = gym.make('Box2DSimOneArmOneEye-v0')
env.set_world(3)
env.render_init("human")
stime = 1000
action = [0, 0, 0, np.pi*0.3, np.pi*0.3]
for t in range(stime):
    env.render()
    if t < stime/2:
        action += 0.1*rng.randn(5)
        action[:3] = np.maximum(-np.pi*0.5, np.minimum(0, action[:3]))
        action[3:] = np.maximum(0, np.minimum(np.pi*0.5, action[3:]))
        print(action)
        env.step(action)

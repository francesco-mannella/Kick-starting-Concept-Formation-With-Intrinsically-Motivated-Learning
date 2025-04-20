import numpy as np
import gym
import box2dsim
import matplotlib.pyplot as plt

rng = np.random.RandomState(62)
env = gym.make('Box2DSimOneArmOneEye-v0')
stime = 20 

env.reset()
for t in range(stime):
    action = 10*rng.randn(5)
    env.step(action)
    env.render(None)

for t in range(stime):
    action = 10*rng.randn(5)
    env.step(action)
    env.render("human")


for t in range(stime):
    action = 10*rng.randn(5)
    env.step(action)
    env.render("offline")

env.renderer.close("demo")

import numpy as np
from scipy import interpolate
import gym
import box2dsim
import matplotlib.pyplot as plt
import shutil


env = gym.make('Box2DSimOneArmOneEye-v0')
env.set_seed(10)
stime = 100
trials = 3

random = False 

if random is True:
    actions = np.pi*env.rng.uniform(-0.5, 0.5, [10, 5])    
else:
    actions = np.pi*np.array([
        [0.00,   0.00,  0.00,  0.00,  0.00],
        [0.20,   -0.30, -0.20,  0.20 , 0.00],
        [0.20,   -0.40, -0.30,  0.50 , 0.50],
        [0.00,   -0.50, -0.30,  0.10 , 0.80],
        [0.00,   -0.50, -0.0,  0.0 , 0.80],
        [0.00,   -0.80, -0.0,  0.0 , 1.0],
        [0.00,   -0.80, -0.0,  0.0 , 1.2],
        ])

actions_interp = np.zeros([stime, 5])
for joint_idx, joint_timeline in enumerate(actions.T):
    x0 = np.linspace(0, 1, len(joint_timeline))
    f = interpolate.interp1d(x0, joint_timeline)
    x = np.linspace(0, 1, stime)
    joint_timeline_interp = f(x)
    actions_interp[:, joint_idx] = joint_timeline_interp

fig = plt.figure(figsize=(3,3))
ax = fig.add_subplot(121)
screen = ax.imshow(np.zeros([2,2]), vmin=-0.3, vmax=1.3, cmap=plt.cm.binary)
ax.set_axis_off()
ax.set_title("Saliency")
ax1 = fig.add_subplot(122)
fov = ax1.imshow(np.zeros([2, 2, 3]),vmin=0, vmax=1)
ax1.set_axis_off()
ax1.set_title("Fovea")


for q in range(4):
    for k in range(trials):
        env.set_world(q)
        env.reset()
        for t in range(stime):  
            env.render()
            
            env.renderer.ax.set_title("%s object" % 
                    env.world_object_names[env.world_id][0].capitalize())
    
            action = actions_interp[t]
            observation,*_ = env.step(action)
            touch = observation["TOUCH_SENSORS"]
            pos = observation["EYE_POS"]
            if sum(touch) > 0:
                print(''.join([f"{x}" for x in touch]), pos)
               

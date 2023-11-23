import numpy as np
from scipy import interpolate
import gym
import box2dsim
import matplotlib.pyplot as plt
import shutil
import os

plt.ion()

seed = np.frombuffer(os.urandom(4), dtype=np.uint32)[0]
env = gym.make('Box2DSimOneArmOneEye-v0')
env.set_seed(seed)
stime = 100
trials = 3


actions = np.pi*env.rng.uniform(-0.5, 0.5, [10, 5])    


fig = plt.figure(figsize=(15, 6))
ax = fig.add_subplot(121)
screen = ax.imshow(np.zeros([2,2]), vmin=-0.3, vmax=1.3, cmap=plt.cm.binary)
ax.set_axis_off()
ax.set_title("Saliency")
ax1 = fig.add_subplot(122)
fov = ax1.imshow(np.zeros([2, 2, 3]))
ax1.set_axis_off()
ax1.set_title("Fovea")


for q in [0, 3, 1, 2]:
    for k in range(trials):
        env.set_world(q)
        env.render_init("human")
        env.reset()
        for t in range(stime//5):  
            if t % (stime//5) == 0:
                action = 0.5*np.pi*env.rng.uniform(0, 1, [5])*[-1,-1,-1,1,1]    
            env.render()
            
            env.renderer.ax.set_title("%s object" % 
                    env.world_object_names[env.world_id][0].capitalize())
    
            observation,*_ = env.step(action)
            eye = observation["VISUAL_SENSORS"]
            sal = observation["VISUAL_SALIENCY"]

            fov.set_array(eye)
            screen.set_array(sal)
               

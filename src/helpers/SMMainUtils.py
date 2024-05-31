import glob
import os, sys
from pathlib import Path
import shutil
import matplotlib

matplotlib.use("Agg")

import numpy as np
import time

# Add parent directory to Python module path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import params
from SMMain import *
import figs

from SMGraphs import (
    remove_figs,
    blank_video,
    visual_map,
    comp_map,
    trajectories_map,
    representations_movements,
    log,
)

import matplotlib.pyplot as plt

class MainUtils(Main):
    
    def get_context_from_visual(self, visual):

        context_means = np.array(
            [
                [1, 1, 0],  # context 0
                [0, 0, 1],  # context 1
                [1, 0, 0],  # context 2
                [0, 1, 0],  # context 3
            ]
        )

        visual_means = visual.reshape(-1, 3)
        visual_means = (visual_means - visual_means.min()) / (
            visual_means.max() - visual_means.min()
        )
        visual_means = visual_means[np.any(visual_means > 0.9, axis=1)]
        visual_means = visual_means[visual_means.sum(1) < 2]
        visual_mean = visual_means.mean(0)
        diffs = [np.linalg.norm(visual_mean - x) for x in context_means]
        context = np.argmin(diffs)
        return context

    def collect_sensory_states(self):

        env = self.env
        agent = self.agent
        controller = self.controller

        trials = 1500

        data = []
        for context in range(1, 4):

            for trial in range(trials):

                print(f"{context} {trial}")

                # reset env and agent
                state = env.reset(
                    context,
                    plot=None,
                    render=None,
                )

                agent.reset()

                v = state["VISUAL_SENSORS"].ravel()
                ss = state["TOUCH_SENSORS"]
                p = state["JOINT_POSITIONS"][:5]
                a = np.zeros(agent.params_size)

                (
                    internal_representations,
                    internal_points,
                ) = controller.spread([[v], [ss], [p], [a]])

                # take only vision
                internal = internal_representations[0]
                state["internal"] = internal.reshape(-1)
                object_params = env.b2d_env.get_objects_params()
                data.append({"context": context, "observation": state, "object":object_params})
        np.save("objects_data.npy", [data])
        print("Collect sensory states: Done!!!")

    def demo_episode(self,  idx=0):
        
        side = int(np.sqrt(params.visual_size // 3))
        params.base_internal_sigma = 0.1

        env = self.env
        env.b2d_env.rendererType = TestPlotterVisualSalience
        agent = self.agent
        controller = self.controller
        
        
        batch_v = np.zeros([params.stime, params.visual_size])
        batch_ss = np.zeros([params.stime, params.somatosensory_size])
        batch_p = np.zeros([params.stime, params.proprioception_size])
        batch_a = np.zeros([params.stime, params.policy_size])

        v_r = np.zeros([params.stime, params.internal_size])
        ss_r = np.zeros([params.stime, params.internal_size])
        p_r = np.zeros([params.stime, params.internal_size])
        a_r = np.zeros([params.stime, params.internal_size])

        v_p = np.zeros([params.stime, 2])
        ss_p = np.zeros([params.stime, 2])
        p_p = np.zeros([params.stime, 2])
        a_p = np.zeros([params.stime, 2])
        
        # init figure
        fig1 = plt.figure()
        ax1 = fig1.add_subplot(111)
        img_vis = ax1.imshow(np.zeros([10, 10]))
        ax1.set_axis_off()

        fig2 = plt.figure()
        ax2 = fig2.add_subplot(111)
        ax2.set_axis_off()

        fig3 = plt.figure()
        ax3 = fig3.add_subplot(111)
        ax3.set_axis_off()

        regress_world = tf.keras.models.load_model("regress0_model.keras")
        regress_rot = tf.keras.models.load_model("regress1_model.keras")
        regress_color = tf.keras.models.load_model("regress2_model.keras")
        regress_rot_data = np.load("regress1_data.npy", allow_pickle=True)[0]

        # prototype
        i = idx
        goal_r = controller.stm_a.getRepresentation(
                controller.radial_grid[i],
                0.1)

        visual = controller.getVisualsFromRepresentations(goal_r.reshape(1, -1))
        touch = controller.getTouchesFromRepresentations(goal_r.reshape(1, -1))
        proprio = controller.getPropriosFromRepresentations(goal_r.reshape(1, -1))
        policy = controller.getPoliciesFromRepresentations(goal_r.reshape(1, -1)) 
        policy *= params.explore_sigma
        
        visual = visual / visual.max()
        img_vis.set_array(visual.reshape(side, side, 3))
        fig1.savefig(f"{site_dir}/demo_{i:04d}_goal.png")

        figs.ssensory(touch.ravel(), ax2)
        fig2.savefig(f"{site_dir}/demo_{i:04d}_touch.png")

        figs.proprio(proprio.ravel(), ax3)
        fig3.savefig(f"{site_dir}/demo_{i:04d}_proprio.png")

        # find context from visual
        context = self.get_context_from_visual(visual)
        

        w_rot = regress_rot.predict(visual.reshape(1, -1), verbose=0).ravel()
        w_color = regress_color.predict(visual.reshape(1, -1), verbose=0).ravel()
        
        colors = np.array([
            [1, 1, 0],
            [0, 0, 1],
            [1, 0, 0],
            [0, 1, 0],
            ])

        context = np.argmin(np.linalg.norm(colors - 
            w_color.reshape(1, -1), axis=1))

        print(context)
        mparams = { 
                "rot": w_rot,
                "rot_min": regress_rot_data["rot_min"],
                "rot_max": regress_rot_data["rot_max"],
                "color": np.maximum(0, np.minimum(1, w_color))
                }
        
        world_id = context
        world_dict = env.b2d_env.prepare_world(world_id=world_id)
        env.b2d_env.set_objects_params_rot(mparams)
            
        # reset env and agent
        state = env.reset(
            world = world_id,
            world_dict = world_dict,
            plot=f"{site_dir}/demo_%04d" % i,
            render="offline",
        )

        agent.reset()
        agent.updatePolicy(policy)
                
        batch_v[0, :] = state["VISUAL_SENSORS"].ravel()
        batch_ss[0, :] = state["TOUCH_SENSORS"]
        batch_p[0, :] = state["JOINT_POSITIONS"][:5]
        batch_a[0, :] = policy

        # iterate
        smcycle = SensoryMotorCicle()
        poses = np.zeros([params.stime, params.grip_output])
        for t in range(1, params.stime):
            state = smcycle.step(env, agent, state)

            poses[t] = state["JOINT_POSITIONS"][:5]
            batch_v[t, :] = state["VISUAL_SENSORS"].ravel()
            batch_ss[t, :] = state["TOUCH_SENSORS"]
            batch_p[t, :] = state["JOINT_POSITIONS"][:5]
            batch_a[t, :] = policy
        env.close()
        
        np.save(f"{site_dir}/demo_{i:04d}", poses)
        
        # get all representations
        Rs, Rp = controller.spread(
            [
                batch_v,
                batch_ss,
                batch_p,
                batch_a,
                batch_a,
            ]
        )
        v_r, ss_r, p_r, a_r, _ = Rs
        v_p, ss_p, p_p, a_p, _ = Rp
        mx = np.argmax(goal_r.ravel())
        mx = np.argmax(a_r.ravel())

        # get matches 
        match_value, match_increment = controller.computeMatch(
                np.stack([v_p, ss_p, p_p, a_p]), a_p) 
        
        visual_gens = controller.getVisualsFromRepresentations(v_r)

        data = {}
        data["match_value"] = match_value
        data["match_increment"] = match_increment
        data["v_g"] = visual_gens
        data["v_r"] = v_r
        data["ss_r"] = ss_r
        data["p_r"] = p_r
        data["a_r"] = a_r
        data["v_p"] = v_p
        data["ss_p"] = ss_p
        data["p_p"] = p_p
        data["a_p"] = a_p
        data["ss"] = batch_ss

        print(f"{site_dir}/demo_{i:04d}")

        return data

    def demo_episodes(self, n_episodes=params.internal_size):
       
        if n_episodes > params.internal_size:
            n_episodes = params.internal_size

        side = int(np.sqrt(params.visual_size // 3))
        params.base_internal_sigma = 0.1

        env = self.env
        agent = self.agent
        controller = self.controller
        
        batch_v = np.zeros([1, params.stime, params.visual_size])
        batch_ss = np.zeros([1, params.stime, params.somatosensory_size])
        batch_p = np.zeros([1, params.stime, params.proprioception_size])
        batch_a = np.zeros([1, params.stime, params.policy_size])
        batch_g = np.zeros([1, params.stime, params.internal_size])
        batch_c = np.ones([1, params.stime, 1])
        batch_log = np.ones([1, params.stime, 1])

        v_r = np.zeros([1, params.stime, params.internal_size])
        ss_r = np.zeros([1, params.stime, params.internal_size])
        p_r = np.zeros([1, params.stime, params.internal_size])
        a_r = np.zeros([1, params.stime, params.internal_size])

        v_p = np.zeros([1, params.stime, 2])
        ss_p = np.zeros([1, params.stime, 2])
        p_p = np.zeros([1, params.stime, 2])
        a_p = np.zeros([1, params.stime, 2])
        g_p = np.zeros([1, params.stime, 2])
        
        match_value = np.zeros([1, params.stime])
        match_value_per_mod = np.zeros([1, params.stime, 4])
        match_increment = np.zeros([1, params.stime])
        match_increment_per_mod = np.zeros([1, params.stime, 4])

        g_p_set = set()
        i = 0

        env = self.env
        while len(g_p_set) < n_episodes:
            context = (i % 3) + 1
            env.b2d_env.prepare_world(context)
            state = env.reset(context,
                              plot=f"{site_dir}/demo",
                              render="offline")
            batch_v[0, 0, :] = state["VISUAL_SENSORS"].ravel()
            batch_ss[0, 0, :] = state["TOUCH_SENSORS"]
            batch_p[0, 0, :] = state["JOINT_POSITIONS"][:5]
           
            # get Representations for initial states
            Rs, Rp = controller.spread(
                    [
                        batch_v[:, 0, :],
                        batch_ss[:, 0, :],
                        batch_p[:, 0, :],
                        batch_a[:, 0, :],
                        batch_g[:, 0, :],
                    ])
            v_r[:, 0, :], ss_r[:, 0, :], p_r[:, 0, :], a_r[:, 0, :], _ = Rs
            v_p[:, 0, :], ss_p[:, 0, :], p_p[:, 0, :], a_p[:, 0, :], g_p[:, 0, :] = Rp

            i += 1
            
            goal = tuple(g_p[0, 0, :])
            if goal in g_p_set:
                print(f"Skipping repeated prototype: {int(goal[0])}{int(goal[1])}")
                print(context)
                print(i)
                print(v_r[0, 0])
                continue

            print(f"{site_dir}/demo_00{int(goal[0])}{int(goal[1])}")
            print(goal)
            print(context)
            g_p_set.add(goal)
            
            matches, cum_match = self.run_episodes(
                batch_v, batch_ss, batch_p, batch_a, batch_g,
                batch_c, batch_log,
                v_r, ss_r, p_r, a_r,
                v_p, ss_p, p_p, a_p, g_p,
                match_value_per_mod,
                match_value,
                match_increment_per_mod,
                match_increment,
                agent, controller, [context],
                [env], [state], noise=False)
            
            env.render_info(match_value[0], matches[0])
            env.close()
            shutil.copyfile(f"{site_dir}/demo.gif", f"{site_dir}/demo_00{int(goal[0])}{int(goal[1])}.gif")

        
        print("demo episodes: Done!!!")        



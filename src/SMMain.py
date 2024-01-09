import glob
import os, sys
from pathlib import Path
import matplotlib
import torch

matplotlib.use("Agg")

import params
import numpy as np
import time
from SMController import SMController
from SMEnv import SMEnv
from SMAgent import SMAgent
from box2dsim.envs.Simulator import TestPlotterVisualSalience

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

np.set_printoptions(formatter={"float": "{:6.4f}".format})

storage_dir = "storage"
site_dir = "www"
os.makedirs(storage_dir, exist_ok=True)
os.makedirs(site_dir, exist_ok=True)

class TimeLimitsException(Exception):
    pass

class SensoryMotorCicle:
    def __init__(self):
        self.t = 0

    def step(self, env, agent, state, action_steps=5):
        if self.t % action_steps == 0:
            self.action = agent.step(state)
        state = env.step(self.action)
        self.t += 1
        return state

def modulate_param(base, limit, prop):
    return base + (limit - base) * prop

def softmax(x, t=0.01):
    e = np.exp(x / t)
    return e / (e.sum() + 1e-100)

class Main:
    def __init__(self, seed=None, plots=False):

        print("Main", flush=True)

        if seed is None:
            seed = np.frombuffer(os.urandom(4), dtype=np.uint32)[0]
        self.rng = np.random.RandomState(seed)
        torch.manual_seed(seed)
        self.seed = seed

        self.plots = plots
        self.start = time.perf_counter()
        if self.plots is True:
            remove_figs()
        self.env = SMEnv(seed)
        self.agent = SMAgent(self.env)
        self.controller = SMController(
            self.rng,
            load=params.load_weights,
            shuffle=params.shuffle_weights,
        )
        self.logs = np.zeros([params.epochs, 3])
        self.epoch = 0

    def __getstate__(self):
        return {
            "controller": self.controller.__getstate__(),
            "env": self.env.__getstate__(),
            "rng": self.rng.__getstate__(),
            "seed": self.seed,
            "plots": self.plots,
            "logs": self.logs,
            "epoch": self.epoch,
        }

    def __setstate__(self, state):

        self.plots = state["plots"]
        self.logs = state["logs"]
        self.epoch = state["epoch"]
        self.seed = state["seed"]
        torch.manual_seed(self.seed)
        self.rng = np.random.RandomState()
        self.rng.__setstate__(state["rng"])

        nlogs = len(self.logs)
        if params.epochs > nlogs:
            tmp = np.zeros([params.epochs, 3])
            tmp[:nlogs, : ] = self.logs.copy()
            self.logs = tmp

        self.env = SMEnv(self.seed)
        self.controller = SMController(
            self.rng,
            load=params.load_weights,
            shuffle=params.shuffle_weights,
        )

        self.controller.__setstate__(state["controller"])
        self.env.__setstate__(state["env"])

        self.agent = SMAgent(self.env)
        self.start = time.perf_counter()
        if self.plots is True:
            remove_figs(self.epoch)

    def train(self, time_limits):

        if self.epoch == 0:
            print("Training", flush=True)
        else:
            if self.epoch >= params.epochs - 1:
                raise TimeLimitsException

        env = self.env
        agent = self.agent
        controller = self.controller
        logs = self.logs
        batch_data_size = params.batch_size * (params.stime)
        epoch = self.epoch
        epoch_start = time.perf_counter()
        contexts = (np.arange(params.batch_size) % 3) + 1

        batch_v = np.zeros([batch_data_size, params.visual_size])
        batch_ss = np.zeros([batch_data_size, params.somatosensory_size])
        batch_p = np.zeros([batch_data_size, params.proprioception_size])
        batch_a = np.zeros([batch_data_size, params.policy_size])
        batch_c = np.ones([batch_data_size, 1])
        batch_log = np.ones([batch_data_size, 1])
        batch_g = np.zeros([batch_data_size, params.internal_size])
        v_r = np.zeros([batch_data_size, params.internal_size])
        ss_r = np.zeros([batch_data_size, params.internal_size])
        p_r = np.zeros([batch_data_size, params.internal_size])
        a_r = np.zeros([batch_data_size, params.internal_size])
        v_p = np.zeros([batch_data_size, 2])
        ss_p = np.zeros([batch_data_size, 2])
        p_p = np.zeros([batch_data_size, 2])
        a_p = np.zeros([batch_data_size, 2])
        g_p = np.zeros([batch_data_size, 2])

        while epoch < params.epochs:

            total_time_elapsed = time.perf_counter() - self.start
            if total_time_elapsed >= time_limits:
                if self.epoch > 0:
                    raise TimeLimitsException

            print(f"{epoch:6d}", end=" ", flush=True)

            controller.comp_grid = controller.getCompetenceGrid()
            comp = controller.comp_grid.mean()

            controller.match_sigma = modulate_param(
                params.base_match_sigma,
                params.match_sigma,
                1 - comp,
            )
            controller.curr_sigma = modulate_param(
                params.base_internal_sigma,
                params.internal_sigma,
                1 - comp,
            )
            controller.curr_lr = modulate_param(
                params.base_lr,
                params.stm_lr,
                1 - comp,
            )

            controller.explore_sigma = params.explore_sigma

            if epoch > params.pretest_epochs:
                controller.updateParams(
                    controller.curr_sigma, controller.curr_lr
                )

            # ----- episodes

            comps = batch_c
            world_dicts = []
            stime = params.stime
            for episode in range(params.batch_size):

                i = episode * stime

                env.b2d_env.prepare_world(contexts[episode])
                state = env.reset(contexts[episode])

                # ----- start episode

                smcycle = SensoryMotorCicle()
                for t in range(params.stime):

                    it = i + t

                    batch_v[it, :] = state["VISUAL_SENSORS"].ravel()
                    batch_ss[it, :] = state["TOUCH_SENSORS"]
                    batch_p[it, :] = state["JOINT_POSITIONS"][:5]

                    # ONLY FIRST TIMESTEP
                    if t == 0:

                        # get Representations at the first timestep
                        Rs, Rp = controller.spread(
                            [
                                batch_v[it : it + 1, :],
                                batch_ss[it : it + 1, :],
                                batch_p[it : it + 1, :],
                                batch_a[it : it + 1, :],
                                batch_g[it : it + 1, :],
                            ]
                        )
                        v_r[it, :], ss_r[it, :], p_r[it, :], a_r[it, :], _ = Rs 
                        v_p[it, :], ss_p[it, :], p_p[it, :], a_p[it, :], g_p[it, :] = Rp

                        # get policy at the first timestep
                        goal = v_r[it : it + 1, :]

                        (
                            policy,
                            competence,
                            rcompetence,
                        ) = controller.getPoliciesFromRepresentationsWithNoise(
                            goal,
                        )

                        # update policy at the first timestep
                        agent.updatePolicy(policy)

                    # ITERATION (EVERY TIMESTEP)
                    batch_a[it, :] = policy
                    batch_g[it, :] = goal
                    batch_c[it, :] = competence
                    batch_log[it, :] = rcompetence
                    state = smcycle.step(env, agent, state)

            # ---- end of episode: match_value and update

            # get all representations
            Rs, Rp = controller.spread(
                [
                    batch_v,
                    batch_ss,
                    batch_p,
                    batch_a,
                    batch_g,
                ]
            )
            v_r, ss_r, p_r, a_r, _ = Rs
            v_p, ss_p, p_p, a_p, g_p = Rp

            (
                match_value,
                match_increment,
            ) = controller.computeMatch(np.stack([v_p, ss_p, p_p, a_p]), g_p)

            pretest = epoch <= params.pretest_epochs
            (update_items, update_episodes,) = controller.update(
                batch_v,
                batch_ss,
                batch_p,
                batch_a,
                batch_g,
                match_value.reshape(-1, 1),
                match_increment.reshape(-1, 1),
                competences=batch_c,
                pretest=pretest,
            )

            # ---- print

            c = np.outer(contexts, np.ones(stime)).ravel()
            items = [np.sum(update_episodes[c == k]) for k in range(1, 4)]
            items = "".join(
                list(
                    map(
                        lambda x: "{: 6d} {}".format(*x),
                        zip(items, ["f", "m", "c"]),
                    )
                )
            )

            print(f"{update_items:#7d} {items}", end=" ", flush=True)
            print(f"{batch_ss.sum():#10.2f}", end=" ", flush=True)
            logs[epoch] = [
                batch_log.min(),
                batch_log.mean(),
                batch_log.max(),
            ]
            print(
                ("%8.7f " * 3)
                % (
                    batch_log.min(),
                    batch_log.mean(),
                    batch_log.max(),
                ),
                end="",
            )
            print(f" pretest={pretest}", flush=True)

            self.match_value = match_value
            self.match_increment = match_increment
            self.v_r = v_r
            self.ss_r = ss_r
            self.p_r = p_r
            self.a_r = a_r
            self.batch_v = batch_v
            self.batch_ss = batch_ss
            self.batch_p = batch_p
            self.batch_a = batch_a

            # diagnose
            if (epoch > 0 and epoch % params.epochs_to_test == 0) or epoch == (
                params.epochs - 1
            ):

                epoch_dir = f"storage/{epoch:06d}"
                os.makedirs(epoch_dir, exist_ok=True)
                np.save(f"{epoch_dir}/main.dump", [self], allow_pickle=True)
                self.diagnose()

                time_elapsed = time.perf_counter() - epoch_start
                print("---- TIME: %10.4f" % time_elapsed, flush=True)
                epoch_start = time.perf_counter()

            epoch += 1
            self.epoch = epoch
            sys.stdout.flush()
    
    def diagnose(self):

        np.save("main.dump", [self], allow_pickle=True)

        env = self.env
        agent = self.agent
        controller = self.controller
        logs = self.logs
        epoch = self.epoch

        data = {}
        data["match_value"] = self.match_value
        data["match_increment"] = self.match_increment
        data["v_r"] = self.v_r
        data["ss_r"] = self.ss_r
        data["p_r"] = self.p_r
        data["a_r"] = self.a_r
        data["v"] = self.batch_v
        data["ss"] = self.batch_ss
        data["p"] = self.batch_p
        data["a"] = self.batch_a

        epoch_dir = f"storage/{epoch:06d}"
        os.makedirs(epoch_dir, exist_ok=True)

        controller.save(epoch)
        np.save(f"{epoch_dir}/data", [data])
        np.save(f"{site_dir}/log", logs[: epoch + 1])
        np.save(f"{epoch_dir}/log", logs[: epoch + 1])

        if self.plots is False:
            return

        print("----> Graphs  ...", flush=True)
        remove_figs(epoch)
        visual_map()
        log()
        comp_map()

        if os.path.isfile("PLOT_SIMS"):
            print("----> Test Sims ...", end=" ", flush=True)

            for k in range(params.tests):

                context = (k // (params.tests//3)) + 1
                state = env.reset(
                    context,
                    plot=f"{site_dir}/episode%d" % k,
                    render="offline",
                )
                agent.reset()

                v = state["VISUAL_SENSORS"].ravel()
                ss = state["TOUCH_SENSORS"]
                p = state["JOINT_POSITIONS"][:5]
                a = np.zeros(agent.params_size)

                (
                    internal_representations,
                    internal_points,
                ) = controller.spread([np.array([v]), np.array([ss]), np.array([p]), np.array([a])])

                # take only vision
                internal_mean = internal_representations[0]
                policy, *_ = controller.getPoliciesFromRepresentationsWithNoise(
                    internal_mean
                )
                agent.updatePolicy(policy)

                batch_v = np.zeros([params.stime, params.visual_size])
                batch_ss = np.zeros(
                    [
                        params.stime,
                        params.somatosensory_size,
                    ]
                )
                batch_p = np.zeros(
                    [
                        params.stime,
                        params.proprioception_size,
                    ]
                )
                batch_a = np.zeros([params.stime, params.policy_size])
                batch_g = np.zeros([params.stime, params.internal_size])

                batch_v[0] = v
                batch_ss[0] = ss
                batch_p[0] = p
                batch_a[0] = policy.reshape(-1)
                batch_g[0] = internal_mean.reshape(-1)
                smcycle = SensoryMotorCicle()
                for t in range(params.stime):
                    state = smcycle.step(env, agent, state)

                    v = state["VISUAL_SENSORS"].ravel()
                    ss = state["TOUCH_SENSORS"]
                    p = state["JOINT_POSITIONS"][:5]

                    batch_v[t] = v
                    batch_ss[t] = ss
                    batch_p[t] = p
                    batch_a[t] = policy.reshape(-1)
                    batch_g[t] = internal_mean.reshape(-1)

                (
                    internal_representations,
                    internal_points,
                ) = controller.spread(
                    [batch_v, batch_ss, batch_p, batch_a, batch_g]
                )
                (match_value, match_increment,) = controller.computeMatch(
                    np.stack(internal_points[:4]),
                    internal_points[4],
                )

                env.render_info(
                    match_value,
                    (
                        (match_increment > params.match_incr_th)
                        & (match_value > params.match_th)
                    ),
                )

                env.close()
                if k % 2 == 0 or k == params.tests - 1:
                    print(
                        "{:d}% ".format(int(100 * (k / (params.tests - 1)))),
                        end=" ",
                        flush=True,
                    )
            print(flush=True)

        if os.path.isfile("COMPUTE_TRAJECTORIES"):
            print(
                "----> Compute Trajectories ...",
                end=" ",
                flush=True,
            )
            context = 4  # no object
            trj = np.zeros([params.internal_size, params.stime, 2])

            state = env.reset(context)
            agent.reset()
            for i, goal_r in enumerate(controller.goal_grid):
                policy = controller.getPoliciesFromRepresentations(np.array([goal_r]))
                agent.updatePolicy(policy)
                smcycle = SensoryMotorCicle()
                for t in range(params.stime):
                    state = smcycle.step(env, agent, state)
                    trj[i, t] = state["JOINT_POSITIONS"][-2:]
                if i % 10 == 0 or i == params.internal_size - 1:
                    print(
                        "{:d}% ".format(int(100 * (i / params.internal_size))),
                        end=" ",
                        flush=True,
                    )
            print(flush=True)
            np.save(f"{site_dir}/trajectories", trj)
            np.save(f"{epoch_dir}/trajectories", trj)
            trajectories_map()

    def collect_sensory_states():
        pass

    def demo_episode(idx):                  
        pass

    def demo_episodes():                  
        pass

    def get_context_from_visual():
        pass


if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-t",
        "--time",
        help="The maximum time for the simulation (seconds)",
        action="store",
        default=1e99,
    )
    parser.add_argument("-g", "--gpu", help="Use gpu", action="store_true")
    parser.add_argument(
        "-s",
        "--seed",
        help="Simulation seed",
        action="store",
        default=1,
    )
    parser.add_argument(
        "-x",
        "--plots",
        help="Plot graphs",
        action="store_true",
    )
    args = parser.parse_args()
    timing = float(args.time)
    gpu = bool(args.gpu)
    seed = int(args.seed)
    plots = bool(args.plots)

    if gpu:
        torch.set_default_device('cuda')

    if os.path.isfile("main.dump.npy"):
        main = np.load("main.dump.npy", allow_pickle="True")[0]

    else:
        main = Main(seed, plots)

    print(main.epoch)

    try:
        main.train(timing)
        print("Done!!", flush=True)
    except TimeLimitsException:
        print(f"Epoch {main.epoch}. end", flush=True)
        try:
            main.diagnose()
        except AttributeError:
            pass

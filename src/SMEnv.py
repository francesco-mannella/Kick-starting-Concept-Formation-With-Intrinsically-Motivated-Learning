import numpy as np
import params
import gymnasium as gym
import box2dsim


class SMEnv:
    def __init__(self, seed, action_steps=5):
        self.b2d_env = gym.make("Box2DSimOneArmOneEye-v0")
        self.b2d_env = self.b2d_env.unwrapped
        self.b2d_env.set_seed(seed)
        self.b2d_env.action_steps = action_steps

        self.b2d_env.set_taskspace(**params.task_space)
        self.render = None
        self.world = 0

    def __getstate__(self):
        return {"rng": self.b2d_env.rng}

    def __setstate__(self, state):
        self.__init__(0)
        self.b2d_env.rng = state["rng"]

    def step(self, action):
        observation, *_ = self.b2d_env.step(action)
        if self.render is not None:
            self.b2d_env.render(self.render)
        return observation

    def reset(
        self,
        world=None,
        world_dict=None,
        render=None,
        plot=None,
        plot_vision=None,
    ):
        self.render = render
        self.plot = plot
        if world is not None:
            self.world = world

        observation = self.b2d_env.reset(
            self.world, world_dict=world_dict
        )
        if self.render is not None:
            self.b2d_env.render_init(self.render)

        return observation

    def render_info(self, info, thresh):
        assert self.render is not None
        self.b2d_env.renderer.add_info_to_frames(
            info, thresh, params.cum_match_stop_th, params.match_incr_th,
            params.drop_first_n_steps
        )

    def close(self):
        if self.plot is not None:
            self.b2d_env.renderer.close(self.plot)

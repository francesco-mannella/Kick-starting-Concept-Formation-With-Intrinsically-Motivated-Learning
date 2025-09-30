import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from GripAgent import GripAgent
from params import Parameters
from SMEnv import SMEnv


class SMAgent:

    def __init__(self, env, *args, **kargs):
        self.env = env
        self.params = env.params
        self.grip_agent = GripAgent(
            env=self.env,
            num_inputs=self.params.grip_input,
            num_hidden=self.params.grip_hidden,
            num_outputs=self.params.grip_output,
            actuator_map_name="data/StoredGripActuatorMap",
            *args,
            **kargs
        )
        self.grip_param_shape = self.grip_agent.grip.params.shape
        self.params_size = np.prod(self.grip_param_shape)

    def step(self, state):
        pos = state["EYE_POS"][::-1]
        touch = state["TOUCH_SENSORS"]
        joints = state["JOINT_POSITIONS"][:5]
        grip_state = np.hstack([touch, joints, pos])
        grip_action = self.grip_agent.step(grip_state)

        grip_action *= [-1, -1, -1, 1, 1]

        grip_action[:3] += self.params.policy_base_arm
        grip_action[3:] += self.params.policy_base
        action = grip_action

        return action

    def reset(self):
        self.grip_agent.reset()

    def updatePolicy(self, policyParams):
        self.grip_agent.updatePolicy(
            self.params.policy_params_amplitude * policyParams
        )


if __name__ == "__main__":

    matplotlib.use("qtagg")

    plt.ion()

    params = Parameters()

    params.max_policy_noise = 100.0
    params.obj_x = 2.0
    params.obj_y = 0.5

    env = SMEnv(42, params)
    state = env.reset(3, render="human")
    agent = SMAgent(env)
    for t in range(100):
        action = agent.step(state)
        state = env.step(action)
        plt.pause(0.1)
    input()

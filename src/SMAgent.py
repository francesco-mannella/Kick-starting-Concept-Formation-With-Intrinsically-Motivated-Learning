import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from ArmAgent import ArmAgent
from GripAgent import GripAgent
from params import Parameters
from SMEnv import SMEnv


class SMAgent:

    def __init__(self, env, *args, **kargs):
        self.env = env
        self.params = env.params
        self.arm_agent = ArmAgent(
            env=None,
            num_inputs=self.params.arm_input,
            num_hidden=self.params.arm_hidden,
            num_outputs=self.params.arm_output,
            actuator_map_name="data/StoredArmActuatorMap",
            actuator_weights_name="data/StoredArmActuatorWeights",
            *args,
            **kargs
        )
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
        joints = state["JOINT_POSITIONS"][3:5]
        arm_state = state["EYE_POS"][::-1]
        grip_state = np.hstack([touch, joints, pos])
        arm_action = self.arm_agent.step(arm_state)
        grip_action = self.grip_agent.step(grip_state)

        grip_action[:3] -= self.params.policy_base_arm
        grip_action[3:] += self.params.policy_base
        arm_action[:3] *= 1 - self.params.reach_grip_prop
        grip_action[:3] *= self.params.reach_grip_prop
        action = np.hstack([arm_action + grip_action[:3], grip_action[3:]])
        return action

    def reset(self):
        self.arm_agent.reset()
        self.grip_agent.reset()

    def updatePolicy(self, policyParams):
        self.grip_agent.updatePolicy(
            self.params.policy_params_amplitude * policyParams
        )


if __name__ == "__main__":

    matplotlib.use("qtagg")

    plt.ion()

    params = Parameters()
    params.obj_x = 6
    params.obj_y = 6
    params.reach_grip_prop = 1.0
    params.policy_base_arm = np.pi * 0.01
    env = SMEnv(42, params)
    state = env.reset(3, render="human")
    agent = SMAgent(env)
    for t in range(100):
        action = agent.step(state)
        state = env.step(action)
    input()

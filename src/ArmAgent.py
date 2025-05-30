import pathlib
import numpy as np

from Actuator import Actuator
from ArmMapping import generate_arm_mapping

root_dir = pathlib.Path(__file__).parent


class ArmAgent:

    def __init__(
        self,
        env,
        actuator_map_name=None,
        actuator_weights_name=None,
        *args,
        **kargs
    ):
        if actuator_map_name is not None:
            try:
                actuator_map = np.load(
                    (root_dir / actuator_map_name)
                    .with_suffix(".npy")
                    .resolve()
                )
            except IOError:
                actuator_map = generate_arm_mapping(kargs["num_hidden"], env)
                np.save(actuator_map_name, actuator_map)
                print("Map Saved")
        if actuator_weights_name is not None:
            try:
                actuator_weights = np.load(
                    (root_dir / actuator_weights_name)
                    .with_suffix(".npy")
                    .resolve()
                )
                actuator_weights = actuator_weights.reshape(
                    kargs["num_hidden"], kargs["num_outputs"]
                )
            except IOError:
                print("Warning: {:} not found".format(actuator_weights_name))

        self.arm = Actuator(env, actuator_map, actuator_weights, **kargs)
        self.num_params = self.arm.num_hidden * self.arm.num_outputs

    def step(self, state):
        out = self.arm.step(state)
        out[:3] = 0.5 * (out[:3] * np.pi * np.array([1.8, 1, 1]) - np.pi)

        return out

    def reset(self):
        self.arm.reset()

    def updatePolicy(self, params):
        self.arm.params = np.reshape(
            params, [self.arm.num_hidden, self.arm.num_outputs]
        )


if __name__ == "__main__":

    # ArmActuator test
    import gym
    import box2dsim

    _ = box2dsim

    env = gym.make("Box2DSimOneArmOneEye-v0")

    agent = ArmAgent(
        env=None,
        num_inputs=2,
        num_hidden=100,
        num_outputs=3,
        actuator_map_name="data/StoredArmActuatorMap",
        actuator_weights_name="data/StoredArmActuatorWeights",
    )
    # agent.updatePolicy(1.2*np.ones([100, 3]))
    env.render_init("human")
    for t in range(100000):
        if t % 200 == 0:
            env.reset()
            action = agent.step([15, 15]) + 0.5 * np.random.randn(3)
        env.step(np.hstack([action, [0, 0]]))
        env.render("human")
    input()

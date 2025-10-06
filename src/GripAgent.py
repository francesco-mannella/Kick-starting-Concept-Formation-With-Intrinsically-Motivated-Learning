import pathlib

import numpy as np

from Actuator import Actuator
from GripMapping import generate_grip_mapping


root_dir = pathlib.Path(__file__).parent


class GripAgent:

    def __init__(
        self,
        env,
        actuator_map_name=None,
        actuator_weights_name=None,
        *args,
        **kargs
    ):
        actuator_map, actuator_weights = None, None
        if actuator_map_name is not None:
            try:
                actuator_map = np.load(
                    (root_dir / actuator_map_name)
                    .with_suffix(".npy")
                    .resolve()
                )
            except IOError:
                actuator_map = generate_grip_mapping(kargs["num_hidden"], env)
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

        self.env = env
        self.grip = Actuator(
            env, actuator_map, actuator_weights, *args, **kargs
        )
        self.num_params = self.grip.num_hidden * self.grip.num_outputs

    def step(self, state):
        out = self.grip.step(state)
        # out[:3] = out[:3] - 0.5
        # out[3:] = 0.5 * np.pi * (out[3:])
        
        return out

    def reset(self):
        self.grip.reset()

    def updatePolicy(self, params):
        self.grip.params = np.reshape(
            params, [self.grip.num_hidden, self.grip.num_outputs]
        )



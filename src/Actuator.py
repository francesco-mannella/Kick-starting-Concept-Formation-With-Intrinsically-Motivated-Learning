import numpy as np


def grid(side):
    x = np.arange(side)
    Z = np.stack(np.meshgrid(x, x)).reshape(2, -1).T
    return Z


class Actuator:
    def __init__(self, env, actuator_map, actuator_weights, **kargs):
        self.num_inputs = kargs["num_inputs"]
        self.num_hidden = kargs["num_hidden"]
        self.num_outputs = kargs["num_outputs"]
        self.use_esn = False
        self.grid = None
        self.side_hidden = int(np.sqrt(self.num_hidden))
        self.map = actuator_map
        self.params = np.zeros((self.num_hidden, self.num_outputs))
        if actuator_weights is not None:
            self.params = actuator_weights

        if "rng" in kargs:
            self.rng = kargs["rng"]
        else:
            self.rng = np.random.RandomState()

        self.hidden_func = lambda x: x

    def step(self, state):
        mapped_inp = np.dot(state, self.map)
        out = np.dot(mapped_inp, self.params)
        out = np.clip(out, -1, 1)
        return out

    def reset(self):
        if self.use_esn:
            self.echo.reset()

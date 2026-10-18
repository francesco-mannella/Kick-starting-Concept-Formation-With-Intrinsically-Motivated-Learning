from pathlib import Path

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

        self.encoding = kargs.get("encoding", "linear")
        self.input_norm = kargs.get("input_norm", "none")
        self.output = kargs.get("output", "clip")
        self.input_blocks = kargs.get("input_blocks", (40, 5, 2))
        self.block_units = self.num_hidden // len(self.input_blocks)
        self.gain = None
        if self.input_norm == "block_rms":
            stats = np.load(
                (Path(__file__).parent / kargs["norm_name"]).with_suffix(".npz")
            )
            self.input_mean = stats["mean"]
            self.input_scale = stats["scale"]
            self.gain = np.repeat(stats[f"gain_{self.encoding}"], self.block_units)
            self.rbf_sigma = stats["rbf_sigma"]
            self.gain[: self.block_units] *= kargs.get("touch_gain", 1.0)
        if self.encoding in ("wta_block", "rbf_block"):
            side = int(round(np.sqrt(self.block_units)))
            coords = grid(side)
            sq = ((coords[:, None, :] - coords[None, :, :]) ** 2).sum(-1)
            self.bumps = np.exp(-0.5 * sq / (side / 5) ** 2)

    def block_distances(self, state):
        start = 0
        for b, size in enumerate(self.input_blocks):
            units = slice(b * self.block_units, (b + 1) * self.block_units)
            prototypes = self.map[start : start + size, units]
            yield b, units, ((prototypes - state[start : start + size, None]) ** 2).sum(0)
            start += size

    def wta_hidden(self, state):
        hidden = np.zeros(self.num_hidden)
        for b, units, dist in self.block_distances(state):
            hidden[units] = self.bumps[np.argmin(dist)]
        return hidden

    def rbf_hidden(self, state):
        hidden = np.zeros(self.num_hidden)
        for b, units, dist in self.block_distances(state):
            act = np.exp(-0.5 * dist / self.rbf_sigma[b] ** 2)
            hidden[units] = act / (act.sum() + 1e-12)
        return hidden

    def hidden(self, state):
        if self.input_norm == "block_rms":
            state = (state - self.input_mean) / self.input_scale
        if self.encoding == "wta_block":
            hidden = self.wta_hidden(state)
        elif self.encoding == "rbf_block":
            hidden = self.rbf_hidden(state)
        else:
            hidden = np.dot(state, self.map)
        if self.gain is not None:
            hidden = hidden * self.gain
        return hidden

    def step(self, state):
        mapped_inp = self.hidden(state)
        out = np.dot(mapped_inp, self.params)
        if self.output == "tanh":
            return np.tanh(out)
        out = np.clip(out, -1, 1)
        return out

    def reset(self):
        if self.use_esn:
            self.echo.reset()

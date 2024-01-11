import numpy as np
import params
from esn import ESN


def grid(side):
    x = np.arange(side)
    Z = np.stack(np.meshgrid(x, x)).reshape(2, -1).T
    return Z


class Radial:

    def __init__(self, m):
        inp, out = m.shape
        self.l = int(np.sqrt(out))
        self.s = self.l/5
        self.Z = grid(self.l)

    def gauss(self, m, s):
        d = np.linalg.norm(self.Z - m, axis=1)
        return np.exp(-0.5*(s**-2)*d**2)

    def normalize(self, x):
        return x/((self.s**2)*2*np.pi)

    def normal(self, x):
        return self.normalize(self.gauss(x, self.s))

    def __call__(self, x, s=None):
        self.s = s if s is not None else self.s
        mx = np.argmax(x)
        m =  [mx//self.l, mx%self.l]
        return self.normal(m)


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
        self.radial = Radial(self.map)

        if "rng" in kargs:
            self.rng = kargs["rng"]
        else:
            self.rng = np.random.RandomState()

        self.hidden_func = lambda x: x
        if self.use_esn:
            self.echo = ESN(N=num_hidden,
                    stime=params.stime,
                    dt=1.0,
                    tau=params.esn_tau,
                    alpha=params.esn_alpha,
                    epsilon=params.esn_epsilon,
                    rng=self.rng)
            self.hidden_func = self.echo.step

    def step(self, state):
        mapped_inp = np.dot(state, self.map) 
        hidden = self.hidden_func(self.radial(mapped_inp))
        out = np.dot(hidden, self.params)
        out = np.maximum(-1, np.minimum(1, out))
        return out

    def reset(self):
        if self.use_esn:
            self.echo.reset()

    def interpolate(self, pos):
        if self.grid is None:
            self.grid = grid(self.self_hidden)
        ppos = self.grid[int(pos*(self.num_hidden-1))] 

        smooth = self.radial.normal(ppos)
        smooth = smooth/smooth.sum()
        inp = np.dot(self.map, smooth)
        return inp



import os
import sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))
from params import Parameters
from SMController import SMController

rng = np.random.RandomState(0)


def make(**kw):
    c = object.__new__(SMController)
    c.params = Parameters(**kw)
    c.policy_baseline = None
    return c


N = 40
goals = np.zeros((N, 100)); goals[np.arange(N), np.arange(N) % 5] = 1
match = rng.rand(N); local_lr = np.full((N,), 0.5)
sel = np.zeros(N, bool); sel[:30] = True

c = make(policy_update_baseline=0.05)
s, lr = c.baseline_select(sel, match, goals, local_lr)
print("hard: kept", s.sum(), "of", sel.sum(), "mass", lr[s].sum() / local_lr[sel].sum())

c = make(policy_update_baseline=0.05, policy_baseline_norm=True)
s, lr = c.baseline_select(sel, match, goals, local_lr)
print("hard+norm: kept", s.sum(), "mass ratio", round(lr[s].sum() / local_lr[sel].sum(), 3))
assert abs(lr[s].sum() / local_lr[sel].sum() - 1) < 1e-6

c = make(policy_update_baseline=0.05, policy_baseline_soft=0.1)
s, lr = c.baseline_select(sel, match, goals, local_lr)
print("soft: kept", s.sum(), "mass ratio", round(lr[s].sum() / local_lr[sel].sum(), 3), "min/max weight", round(lr[s].min() / 0.5, 3), round(lr[s].max() / 0.5, 3))
assert s.sum() == sel.sum() and abs(lr[s].sum() / local_lr[sel].sum() - 1) < 1e-6

# reach noise: variance profile and projection
c = make(policy_noise_reach_k=15, reach_var_profile="eig", policy_noise_rel=1.0)
Q = np.linalg.qr(rng.randn(75, 15))[0]
c.reach_basis = Q
c.reach_scale = np.sqrt(np.linspace(4, 1, 15) / np.linspace(4, 1, 15).mean())
noise = rng.randn(2000, 375).reshape(2000, 75, 5)
coef = np.einsum("hk,nho->nko", Q, noise) * c.reach_scale[None, :, None]
out = np.einsum("hk,nko->nho", Q, coef)
resid = out - np.einsum("hk,kj,njo->nho", Q, Q.T, out)
print("residual outside span:", float(np.abs(resid).max()), "variance by direction ratio first/last:", round(float(coef[:, 0].var() / coef[:, -1].var()), 2))
assert np.abs(resid).max() < 1e-8
print("ok")

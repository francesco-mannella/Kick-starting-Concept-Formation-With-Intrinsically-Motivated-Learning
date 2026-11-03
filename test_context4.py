import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "tools", "box2dsim"))
sys.path.insert(0, os.path.join(HERE, "src"))

from params import Parameters  # noqa: E402
from SMController import SMController  # noqa: E402
from update_decomposition import decompose_update  # noqa: E402

# one event at node 55 (5, 5): the update weight stays on its own node
phi = np.zeros((1, 100))
phi[0, 55] = 1.0
goals = np.zeros((1, 100))
goals[0, 55] = 1.0
out = decompose_update(phi, goals, np.ones((1, 600)))
assert out["upd_own_share"] == 1.0 and out["upd_weighted_distance"] == 0.0 and out["upd_spread_nodes"] == 1.0

# weight split between the goal node and a neighbour at distance 1 (node 56)
phi[0, 56] = 1.0
out = decompose_update(phi, goals, np.ones((1, 600)))
assert abs(out["upd_own_share"] - 0.5) < 1e-9 and abs(out["upd_weighted_distance"] - 0.5) < 1e-9 and out["upd_spread_nodes"] == 2.0

# two goal nodes with identical event policies: common share 1; orthogonal event policies: 1/2
phi2 = np.zeros((4, 100))
goals2 = np.zeros((4, 100))
for i, g in enumerate((10, 10, 90, 90)):
    phi2[i, g] = 1.0
    goals2[i, g] = 1.0
same = np.ones((4, 600))
assert abs(decompose_update(phi2, goals2, same)["upd_event_common_share"] - 1.0) < 1e-9
orth = np.zeros((4, 600))
orth[:2, 0] = 1.0
orth[2:, 1] = 1.0
out = decompose_update(phi2, goals2, orth)
assert abs(out["upd_event_common_share"] - 0.5) < 1e-9 and out["upd_goal_nodes"] == 2.0

# default path: no statistics are produced and the policy update is unchanged
assert Parameters().log_update_decomposition is False
controller = SMController(Parameters(policy_update="ema", policy_winner="anchored"), None, rng=np.random.RandomState(0))
g = np.random.RandomState(1).rand(5, 100)
x = np.random.RandomState(2).rand(5, 600)
before = controller.stm_a.get_weights().copy()
controller.update_policy_map(x, g, g, np.full((5, 1), 0.1))
after_default = controller.stm_a.get_weights().copy()
assert not hasattr(controller, "update_stats")

on = Parameters(log_update_decomposition=True, policy_update="ema", policy_winner="anchored")
controller2 = SMController(on, None, rng=np.random.RandomState(0))
controller2.stm_a.set_weights(before)
controller2.update_policy_map(x, g, g, np.full((5, 1), 0.1))
assert np.allclose(controller2.stm_a.get_weights(), after_default)
assert 0 < controller2.update_stats["upd_own_share"] <= 1

# policy-map-only sigma: identity by default, same endpoints (base, top) and linear in between
s = np.array([0.1, 5.05, 10.0])
assert (controller.policy_sigma(s) == s).all()
capped = SMController(Parameters(policy_internal_sigma=2.0), None, rng=np.random.RandomState(0))
assert np.allclose(capped.policy_sigma(s), [0.1, 1.05, 2.0])

print("test_context4 ok")

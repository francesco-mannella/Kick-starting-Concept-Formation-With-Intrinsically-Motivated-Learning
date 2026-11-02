import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "tools", "box2dsim"))
sys.path.insert(0, os.path.join(HERE, "src"))

from Actuator import Actuator  # noqa: E402
from center_actuator_code import build_actuator  # noqa: E402
from params import Parameters  # noqa: E402
from SMController import SMController  # noqa: E402

rng = np.random.RandomState(0)

# ---- centred actuator code
babble = np.load("/home/fmannella/tmp/kss/map_v4/babble_1.npz")
states = np.hstack([babble["touch"], babble["posture"], babble["eye"]])
states = states[rng.choice(len(states), 300, replace=False)]
plain = build_actuator("data/StoredGripActuatorMap_v4", "data/StoredGripActuatorNorm_v4c")
for variant in ("touch", "all"):
    act = Actuator(None, np.load(os.path.join(HERE, "src", "data", "StoredGripActuatorMap_v4.npy")), None, num_inputs=50, num_hidden=75, num_outputs=8,
                   encoding="rbf_block", input_norm="block_rms", norm_name="data/StoredGripActuatorNorm_v4c", touch_gain=3.0, output="tanh",
                   input_blocks=(40, 8, 2), center_code=variant)
    zero = act.hidden(np.zeros(50))
    assert np.allclose(zero[:25], 0.0, atol=1e-6), "no contact must give a zero touch-block input"
    codes = np.array([act.hidden(s) for s in states])
    base = np.array([plain.hidden(s) for s in states])
    touch_sums = codes[:, :25].sum(1)
    if variant == "touch":
        assert np.allclose(codes[:, 25:], base[:, 25:]), "touch variant must leave posture and eye blocks unchanged"
    else:
        for i in (1, 2):
            assert abs(np.median(codes[:, i * 25:(i + 1) * 25].sum(1))) < 0.05, "posture and eye blocks must sum to about 0 for a typical state"
assert np.allclose(plain.hidden(states[0]), build_actuator("data/StoredGripActuatorMap_v4", "data/StoredGripActuatorNorm_v4").hidden(states[0])), "norm v4c without centring equals v4"

# ---- shared base and residual policy map
params = Parameters()
controller = SMController(params, None, rng=np.random.RandomState(1))
assert not controller.policy_shared.any()
reps = controller.stm_a.getRepresentation(np.array([[2.0, 3.0]]), sigma=params.base_internal_sigma)
default = controller.getPoliciesFromRepresentations(reps)
assert np.array_equal(default, controller.stm_a.backward(reps)), "default path unchanged"

shared = Parameters(policy_shared_base=True, policy_update="ema", policy_winner="anchored")
a = SMController(shared, None, rng=np.random.RandomState(1))
b = SMController(shared, None, rng=np.random.RandomState(1))
base = rng.randn(shared.policy_size)
b.policy_shared = base.copy()
assert np.allclose(b.getPoliciesFromRepresentations(reps), b.stm_a.backward(reps) + base)
policy = rng.randn(1, shared.policy_size)
goals = np.zeros((1, shared.internal_size))
goals[0, 55] = 1.0
before_a, before_b = a.stm_a.get_weights().copy(), b.stm_a.get_weights().copy()
a.update_policy_map(policy, policy, goals, np.ones((1, 1)))
b.update_policy_map(policy, policy, goals, np.ones((1, 1)))
moved_a = a.stm_a.get_weights()[:, 55] - before_a[:, 55]
moved_b = b.stm_a.get_weights()[:, 55] - before_b[:, 55]
assert not np.allclose(moved_a, moved_b), "the residual map must update toward policy minus base"
assert np.dot(moved_a - moved_b, base) > 0, "with a base the node moves less toward the base direction (target is policy minus base)"

# persistence
state = b.__getstate__()
assert np.array_equal(state["policy_base"], base)
c = SMController(shared, None, rng=np.random.RandomState(1))
c.__setstate__(state)
assert np.array_equal(c.policy_shared, base)
legacy = {k: v for k, v in state.items() if k != "policy_base"}
c.__setstate__(legacy)
assert not c.policy_shared.any(), "a checkpoint without the base loads zeros"

print("ok")

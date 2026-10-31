import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "src"))

from Actuator import Actuator  # noqa: E402
from embedding_map import build_embedding  # noqa: E402

rng = np.random.RandomState(0)
n = 4000
touch = np.zeros((n, 40))
for i in rng.choice(n, 600, replace=False):
    start = rng.randint(0, 34)
    touch[i, start : start + 6] = rng.uniform(0.5, 1.5)
data = {"touch": touch, "posture": rng.randn(n, 8), "eye": rng.randn(n, 2) * 2}
stats = build_embedding(data, rng)
np.savez(os.path.join(HERE, "src", "data", "_test_embedding.npz"), **stats)

try:
    act = Actuator(None, None, None, num_inputs=50, num_hidden=15, num_outputs=8, encoding="embedding",
                   norm_name="data/_test_embedding", input_blocks=(40, 8, 2), output="tanh")
    state = np.hstack([data["touch"], data["posture"], data["eye"]])
    codes = np.array([act.hidden(s) for s in state])
    assert codes.shape == (n, 15) and np.isfinite(codes).all()
    assert act.params.shape == (15, 8)

    in_contact = touch.sum(1) > 0
    assert not codes[~in_contact, :5].any(), "no contact must give a zero touch code"
    assert np.allclose(codes[in_contact, :5].std(0), 1.0, atol=0.3), codes[in_contact, :5].std(0)
    assert np.allclose(codes[:, 5:].std(0), 1.0, atol=0.1), codes[:, 5:].std(0)
    assert np.allclose(codes[:, 5:].mean(0), 0.0, atol=0.1)

    act.params = rng.randn(15, 8) * 0.1
    out = act.step(state[0])
    assert out.shape == (8,) and np.abs(out).max() <= 1
finally:
    os.remove(os.path.join(HERE, "src", "data", "_test_embedding.npz"))

print("ok")

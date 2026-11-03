import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "tools", "box2dsim"))
sys.path.insert(0, os.path.join(HERE, "src"))

from params import Parameters  # noqa: E402
from policy_stats import episode_table  # noqa: E402
from SMController import SMController  # noqa: E402

# reach mask: contact required, both winners within the radius (Euclidean grid distance)
goal = np.array([[4.0, 5.0]] * 6)
touch = np.array([[4, 5], [4, 6], [5, 6], [4, 8], [4, 5], [4, 6]], float)
prop = np.array([[4, 5], [4, 5], [4, 5], [4, 5], [4, 7], [4, 6]], float)
contact = np.array([True, True, True, True, True, False])
for radius, expected in ((0, [1, 0, 0, 0, 0, 0]), (1, [1, 1, 0, 0, 0, 0]), (1.5, [1, 1, 1, 0, 0, 0]), (2.9, [1, 1, 1, 0, 1, 0]), (10, [1, 1, 1, 1, 1, 0])):
    got = SMController.reach_mask(touch, prop, goal, contact, radius)
    assert got.tolist() == [bool(x) for x in expected], (radius, got)

# default parameters leave the rule unchanged; schedule shrinks with competence
controller = SMController(Parameters(), None, rng=np.random.RandomState(0))
assert controller.event_radius is None and controller.scheduled_event_radius(0.5) is None
scheduled = SMController(Parameters(event_radius=3.0, event_radius_final=1.0), None, rng=np.random.RandomState(0))
assert scheduled.event_radius == 3.0
assert np.isclose(scheduled.scheduled_event_radius(0.0), 3.0) and np.isclose(scheduled.scheduled_event_radius(1.0), 1.0) and np.isclose(scheduled.scheduled_event_radius(0.5), 2.0)
warm = SMController(Parameters(event_radius=2.0, event_radius_warmup=50), None, rng=np.random.RandomState(0))
assert warm.event_radius is None and warm.scheduled_event_radius(0.5, 10) is None and warm.scheduled_event_radius(0.5, 50) == 2.0
fixed = SMController(Parameters(event_radius=2.0), None, rng=np.random.RandomState(0))
assert fixed.scheduled_event_radius(0.9) == 2.0
state = fixed.__getstate__()
assert state["event_radius"] == 2.0
clone = SMController(Parameters(), None, rng=np.random.RandomState(0))
clone.__setstate__(state)
assert clone.event_radius == 2.0

# reach columns of the episode table
steps = 20
rows = []
for tr_id, (tx, ty, px, py) in enumerate([(4, 5, 4, 5), (4, 7, 4, 5)]):
    for t in range(steps):
        row = {"episode_id": 0, "tr_id": tr_id, "prototype_x": 4.0, "prototype_y": 5.0, "touch_x": tx, "touch_y": ty, "proprio_x": px, "proprio_y": py, "match": 0.5}
        row.update({f"s{i}": (1.0 if (i == 3 and t >= 2) else 0.0) for i in range(40)})
        rows.append(row)
table = episode_table(pd.DataFrame(rows), pd.DataFrame({"context": [1, 2, 3]}))
row = table.iloc[0]
assert row.reach_any0 == 1.0 and row.reach_any2 == 1.0
assert np.isclose(row.reach_share0, (steps - 2) / (2 * steps)), row.reach_share0
assert np.isclose(row.reach_share1, row.reach_share0) and np.isclose(row.reach_share2, 2 * row.reach_share0) and np.isclose(row.reach_share3, row.reach_share2)
assert row.n_segments == 2 and np.isclose(row.end_touch_d, 1.0) and np.isclose(row.end_proprio_d, 0.0)

print("ok")

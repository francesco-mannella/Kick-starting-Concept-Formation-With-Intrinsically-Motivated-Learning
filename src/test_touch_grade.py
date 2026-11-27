from pathlib import Path

import types

import numpy as np

from params import Parameters
from touch_layout import SKETCH, inner_slices
from SMController import prune_policies
from SMMain import SensoryMotorCircle, gentle_bonus_value, held_onsets, near_record_match, smooth_effort, touch_grade, touch_input, touch_map_size, touch_overloaded


def touch(**active):
    v = np.zeros(40)
    for idx, val in active.items():
        v[int(idx[1:])] = val
    return v


def test_defaults_are_off():
    p = Parameters()
    assert not p.touch_normalize and not p.touch_grade
    v = touch(s11=2, s12=1, s22=3)
    assert np.array_equal(touch_input(p, v), v)


def test_normalised_input():
    p = Parameters(touch_normalize=True)
    v = touch(s11=2, s12=1, s22=3)
    x = touch_input(p, v)
    assert np.isclose(x.sum(), v.max()) and np.array_equal(x > 0, v > 0)
    assert np.allclose(x / x.sum(), v / v.sum())
    assert not touch_input(p, np.zeros(40)).any()
    batch = touch_input(p, np.stack([v, np.zeros(40)]))
    assert batch.shape == (2, 40) and batch[1].sum() == 0


def test_grade_values():
    p = Parameters(touch_grade=True, touch_grade_any=0.25, touch_grade_ref=2.0)
    assert touch_grade(p, np.zeros(40)) == 0.0
    assert np.isclose(touch_grade(p, touch(s11=1, s12=1)), 0.25)  # left inner only
    assert np.isclose(touch_grade(p, touch(s3=2, s35=1)), 0.25)  # outer faces only
    assert np.isclose(touch_grade(p, touch(s11=1, s22=1)), 0.25 + 0.75 * 0.5)  # one sensor on each inner face
    assert np.isclose(touch_grade(p, touch(s11=1, s12=1, s22=1, s23=1)), 1.0)  # two on each inner face
    assert np.isclose(touch_grade(p, touch(s11=1, s12=1, s13=1, s22=1, s23=1, s24=1)), 1.0)  # saturates
    batch = touch_grade(p, np.stack([touch(s11=1), touch(s11=1, s22=1)]))
    assert batch.shape == (2,) and batch[1] > batch[0]


def test_double_bonus():
    p = Parameters(touch_grade=True, touch_grade_double=4.0, touch_grade_ref=1.0)
    assert touch_grade(p, np.zeros(40)) == 0.0
    assert touch_grade(p, touch(s11=1, s12=1)) == 1.0
    assert touch_grade(p, touch(s3=2)) == 1.0
    assert touch_grade(p, touch(s11=1, s22=1)) == 4.0
    batch = touch_grade(p, np.stack([touch(s11=1), touch(s11=1, s22=1), np.zeros(40)]))
    assert list(batch) == [1.0, 4.0, 0.0]


def test_near_record_match():
    p = Parameters(match_abs_bonus=1.0, match_abs_frac=0.75)
    assert Parameters().match_abs_bonus == 0.0
    out = near_record_match(p, np.array([0.1, 0.5, 0.74, 0.76, 1.0]), 1.0)
    assert np.allclose(out, [0.0, 0.0, 0.0, 0.76, 1.0])
    assert np.allclose(near_record_match(p, np.array([0.31]), 0.4), [0.31])


def test_bilateral_input():
    p = Parameters(touch_bilateral_scale=4.0)
    assert touch_map_size(Parameters()) == 40 and touch_map_size(p) == 41
    one = touch_input(p, touch(s11=1, s12=2))
    two = touch_input(p, touch(s11=1, s22=1))
    assert one.shape == (41,) and one[40] == 0.0 and np.array_equal(one[:40], touch(s11=1, s12=2))
    assert two[40] == 4.0 and np.array_equal(two[:40], touch(s11=1, s22=1))
    assert touch_input(p, touch(s3=1, s35=1))[40] == 0.0
    batch = touch_input(p, np.stack([touch(s11=1), touch(s11=1, s22=1)]))
    assert batch.shape == (2, 41) and list(batch[:, 40]) == [0.0, 4.0]
    off = touch(s11=1, s22=1)
    assert np.array_equal(touch_input(Parameters(), off), off)


def test_bilateral_gain():
    p = Parameters(touch_bilateral_gain=16.0)
    assert touch_map_size(p) == 40
    two = touch_input(p, touch(s3=1, s11=1, s12=2, s22=1))
    assert np.allclose(two[[3, 11, 12, 22]], [1, 16, 32, 16]) and two.shape == (40,)
    one = touch(s3=1, s11=1, s12=2)
    assert np.array_equal(touch_input(p, one), one)
    batch = touch_input(p, np.stack([touch(s11=1, s22=1), touch(s11=1)]))
    assert batch[0, 11] == 16.0 and batch[1, 11] == 1.0
    off = touch(s11=1, s22=1)
    assert np.array_equal(touch_input(Parameters(), off), off)


def test_touch_overload():
    p = Parameters(touch_overload=0.3)
    v = touch(s11=1, s22=2)
    assert not touch_overloaded(Parameters(), v, 5.0)
    assert not touch_overloaded(p, v, None)
    assert not touch_overloaded(p, v, 0.2)
    assert touch_overloaded(p, v, 0.31)
    assert not touch_overloaded(p, np.zeros(40), 5.0)
    assert np.array_equal(touch_input(p, v, 0.2), v)
    zeroed = touch_input(p, v, 0.5)
    assert zeroed.shape == (40,) and not zeroed.any()
    pb = Parameters(touch_overload=0.3, touch_bilateral_scale=4.0)
    assert touch_input(pb, v, 0.2)[40] == 4.0
    out = touch_input(pb, v, 0.5)
    assert out.shape == (41,) and not out.any()
    assert np.array_equal(touch_input(Parameters(), v, 9.9), v)


def test_smooth_effort():
    assert smooth_effort(0.3, 2.0, np.nan) == 2.0
    assert smooth_effort(0.3, 2.0, None) == 2.0
    assert np.isclose(smooth_effort(0.3, 2.0, 1.0), 0.3 * 2.0 + 0.7 * 1.0)
    assert smooth_effort(1.0, 2.0, 1.0) == 2.0
    assert Parameters().touch_overload_alpha == 1.0
    p = Parameters(touch_overload=1.0, touch_overload_alpha=0.3)
    v = touch(s11=1)
    e1 = smooth_effort(p.touch_overload_alpha, 3.0, np.nan)
    e2 = smooth_effort(p.touch_overload_alpha, 0.0, e1)
    assert touch_overloaded(p, v, e1) and touch_overloaded(p, v, e2)
    e3 = smooth_effort(p.touch_overload_alpha, 0.0, e2)
    assert e3 < e1 and np.isclose(e3, 0.49 * 3.0)
    assert np.array_equal(touch_input(p, v, 0.5), v)


def test_held_onsets():
    T = 12
    onset = np.zeros((4, T), dtype=bool)
    ss = np.zeros((4, T, 40))
    changed = np.zeros((4, T), dtype=bool)
    onset[:, 2] = True
    ss[0, 2:7] = 1.0  # held
    ss[1, 2:4] = 1.0  # contact lost after 2 steps
    ss[2, 2:7] = 1.0  # held but the policy changes inside the window
    changed[2, 4] = True
    ss[3, 2:7] = 1.0
    ss[3, 4] = 0.0  # overload zeroes one step of the window
    assert list(held_onsets(onset, ss, changed, 5, 3)) == [True, False, False, False]
    assert not held_onsets(onset, ss, changed, 4, 3).any()
    assert not held_onsets(onset, ss, changed, 1, 3).any()
    assert Parameters().credit_hold_steps == 0


def test_sketch_layout_groups():
    left, right = inner_slices("grasp3s")
    assert (left, right) == SKETCH and left.stop - left.start == 14 and right.stop - right.start == 14
    assert inner_slices("grasp3") == (slice(10, 20), slice(20, 30)) and inner_slices("orig") == (slice(10, 20), slice(20, 30))
    p = Parameters(obj_set="grasp3s", touch_grade=True, touch_grade_double=4.0, touch_grade_ref=1.0)
    assert touch_grade(p, touch(s7=1, s21=1)) == 4.0  # both inner faces in the sketch layout
    assert touch_grade(p, touch(s7=1, s1=1)) == 1.0  # one inner and one outer sensor
    classic = Parameters(touch_grade=True, touch_grade_double=4.0, touch_grade_ref=1.0)
    assert touch_grade(classic, touch(s11=1, s22=1)) == 4.0
    pg = Parameters(obj_set="grasp3s", touch_bilateral_gain=16.0)
    out = touch_input(pg, touch(s7=1, s21=1, s2=1))
    assert out[7] == 16.0 and out[21] == 16.0 and out[2] == 1.0


def test_sketch_layout_plot_points():
    from SMGraphs import GraphManager
    from touch_layout import HAND_INNER, HAND_LINKS, HAND_POINTS, sensor_links

    pts = GraphManager(None, Parameters(obj_set="grasp3s")).generate_sensor_points(40)
    assert pts.shape == (40, 2) and np.allclose(pts, HAND_POINTS) and pts.min() >= 0.1 - 1e-9 and pts.max() <= 0.9 + 1e-9
    classic = GraphManager(None, Parameters(obj_set="grasp3")).generate_sensor_points(40)
    assert classic.shape == (40, 2) and not np.allclose(classic, pts)
    assert len(sensor_links("grasp3s")) == 4 and sensor_links("grasp3") == []
    left, right = inner_slices("grasp3s")
    assert HAND_INNER[left].all() and HAND_INNER[right].all() and HAND_INNER.sum() == 28
    assert all(r.shape == (4, 2) for r in HAND_LINKS)


def test_gentle_bonus_value():
    p = Parameters(gentle_bonus=0.5, gentle_limit=1.5)
    assert Parameters().gentle_bonus == 0.0 and Parameters().gentle_hold == 3
    mean, bonus = gentle_bonus_value(p, [0.2, 0.4, np.nan, 0.3], True)
    assert np.isclose(mean, 0.3) and np.isclose(bonus, 0.5 * (1 - 0.3 / 1.5))
    assert gentle_bonus_value(p, [0.2, 0.4], False) == (np.mean([0.2, 0.4]), 0.0)  # one finger only: no bonus
    assert gentle_bonus_value(p, [2.0, 2.2], True)[1] == 0.0  # hard squeeze: no bonus
    mean, bonus = gentle_bonus_value(p, [np.nan, np.nan], True)
    assert np.isnan(mean) and bonus == 0.0  # no contact in the window


def test_reset_restores_nan_arrays():
    import types

    from SMMain import Main

    ctrl = types.SimpleNamespace(model_data={"batch_effort_s": np.full((2, 3), 5.0), "batch_gentle": np.full((2, 3, 3), 5.0), "batch_ss": np.ones((2, 3, 4))})
    Main.reset_model_data(None, ctrl)
    assert np.isnan(ctrl.model_data["batch_effort_s"]).all() and np.isnan(ctrl.model_data["batch_gentle"]).all()
    assert (ctrl.model_data["batch_ss"] == 0).all()


def make_actuator():
    from Actuator import Actuator

    root = Path(__file__).parent
    return Actuator(None, np.load(root / "data/StoredGripActuatorMap_v3.npy"), None, num_inputs=47, num_hidden=75, num_outputs=5, encoding="rbf_block",
                    input_norm="block_rms", norm_name="data/StoredGripActuatorNorm_v3", touch_gain=3.0)


def test_actuator_records_last_activity():
    act = make_actuator()
    rng = np.random.default_rng(0)
    for _ in range(3):
        state = rng.uniform(-1, 1, 47)
        hidden = act.hidden(state)
        assert act.last_activity.shape == (75,) and np.allclose(act.last_activity * act.gain, hidden)


class FakeAgent:
    def __init__(self, activities):
        self.activities = iter(activities)
        self.grip_agent = types.SimpleNamespace(grip=types.SimpleNamespace(last_activity=None))

    def step(self, state):
        self.grip_agent.grip.last_activity = next(self.activities)
        return np.zeros(5)


class FakeEnv:
    def step(self, action):
        return {}


def test_circle_buffers_the_used_units():
    import types as _types

    main = _types.SimpleNamespace(params=Parameters(policy_prune=0.1), rng=np.random.default_rng(0))
    first, second, third = np.zeros(75), np.zeros(75), np.zeros(75)
    first[[3, 10]] = [0.5, 0.05]
    second[[3, 40]] = [0.2, 0.9]
    third[7] = 1.0
    agent = FakeAgent([first, second, third])
    circle = SensoryMotorCircle(main, action_steps=2)
    for _ in range(4):  # the actuator is queried at steps 0 and 2 only
        circle.step(FakeEnv(), agent, {})
    mask = circle.used_mask(0.1)
    assert mask.shape == (75,) and sorted(np.where(mask > 0)[0].tolist()) == [3, 40]  # unit 10 stayed under the threshold
    circle.reset_used()
    assert not circle.used_mask(0.1).any()
    circle.step(FakeEnv(), agent, {})
    circle.step(FakeEnv(), agent, {})
    assert np.where(circle.used_mask(0.1) > 0)[0].tolist() == [7]  # a new buffer starts after the reset
    off = _types.SimpleNamespace(params=Parameters(), rng=np.random.default_rng(0))
    circle = SensoryMotorCircle(off, action_steps=1)
    circle.step(FakeEnv(), FakeAgent([first]), {})
    assert circle.used is None  # default: nothing is recorded


def test_prune_policies():
    policies = np.ones((3, 375))
    used = np.zeros((3, 75))
    used[0, :10] = 1.0
    used[2, 5] = 1.0
    out = prune_policies(policies, used)
    assert out[0].reshape(75, 5)[:10].all() and not out[0].reshape(75, 5)[10:].any()  # unused rows zeroed
    assert (out[1] == 1).all()  # no mask for this row: kept
    assert out[2].reshape(75, 5)[5].all() and out[2].sum() == 5
    assert Parameters().policy_prune == 0.0


if __name__ == "__main__":
    test_defaults_are_off()
    test_normalised_input()
    test_grade_values()
    test_double_bonus()
    test_near_record_match()
    test_bilateral_input()
    test_bilateral_gain()
    test_touch_overload()
    test_smooth_effort()
    test_held_onsets()
    test_sketch_layout_groups()
    test_sketch_layout_plot_points()
    test_gentle_bonus_value()
    test_reset_restores_nan_arrays()
    test_actuator_records_last_activity()
    test_circle_buffers_the_used_units()
    test_prune_policies()
    print("ok")

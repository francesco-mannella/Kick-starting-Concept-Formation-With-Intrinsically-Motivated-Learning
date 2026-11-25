import numpy as np

from params import Parameters
from SMMain import near_record_match, touch_grade, touch_input, touch_map_size


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


if __name__ == "__main__":
    test_defaults_are_off()
    test_normalised_input()
    test_grade_values()
    test_double_bonus()
    test_near_record_match()
    test_bilateral_input()
    test_bilateral_gain()
    print("ok")

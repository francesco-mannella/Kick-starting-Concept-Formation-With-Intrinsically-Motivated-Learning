import numpy as np

import mi_standard


def simulate(rng, by_context, n_layouts=18, repeats=20):
    ctx_of_layout = np.arange(n_layouts) % 3 + 1
    policy_of_layout = rng.integers(0, 10, n_layouts)
    ep, ended, ctx, layout, policy = [], [], [], [], []
    for r in range(repeats):
        for lay in range(n_layouts):
            e = r * n_layouts + lay
            for _ in range(rng.integers(2, 5)):
                p = (ctx_of_layout[lay] * 3 + rng.integers(0, 2)) % 10 if by_context else policy_of_layout[lay]
                for t in range(5):
                    ep.append(e), ctx.append(ctx_of_layout[lay]), layout.append(lay), policy.append(p), ended.append(t == 4)
    n = len(ep)
    return (np.array(ep), np.array(ended), np.array(ctx), np.array(layout), np.array(policy),
            rng.random((n, 40)) * (rng.random((n, 1)) < 0.3), rng.random((n, 5)))


def test_no_dependence_at_layout_level():
    rng = np.random.default_rng(0)
    excess, raw = [], []
    for _ in range(10):
        res = mi_standard.layout_excess(*simulate(rng, by_context=False), n_perm=60)
        excess.append(res["mi2_context_policy"]), raw.append(res["mi2_context_policy_raw"])
    assert np.mean(raw) > 0.3, np.mean(raw)
    assert abs(np.mean(excess)) < 0.15, np.mean(excess)


def test_dependence_is_detected():
    res = mi_standard.layout_excess(*simulate(np.random.default_rng(1), by_context=True), n_perm=60)
    assert res["mi2_context_policy"] > 0.5, res["mi2_context_policy"]
    assert res["mi2_n_layouts"] == 18


def test_too_few_layouts_gives_nan():
    res = mi_standard.layout_excess(*simulate(np.random.default_rng(2), by_context=True, n_layouts=2, repeats=30), n_perm=10)
    assert np.isnan(res["mi2_context_policy"]) and res["mi2_n_layouts"] == 2


if __name__ == "__main__":
    test_no_dependence_at_layout_level()
    test_dependence_is_detected()
    test_too_few_layouts_gives_nan()
    print("ok")

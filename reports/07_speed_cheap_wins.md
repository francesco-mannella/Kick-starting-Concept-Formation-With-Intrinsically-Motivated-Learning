# Speed: low-risk modifications (branch `speed_cheap_wins`)

Not committed. The branch was created from `fix_parasite`, so the working tree also holds the uncommitted parasite and local-tracking changes; only the three files below belong to this report.

## Method
`cProfile` through `runpy` (profiling with `python -m cProfile` fails: `Main` cannot be pickled). Workload: `SMMain.py -s 3 -n prof -o epochs=12 -o epochs_to_test=100` on CPU (evaluation/diagnose runs at epoch 0 and epoch 11). Driver and profiles are in the job temp dir, not in the repo.

## Profile before changes (91.0 s total)
| Cost | Time | Where |
|---|---|---|
| STM squared distances (`torch.norm` on a B x D x K tensor) | ~14.4 s tottime (`SMController.spread` 14.9 s cum) | `stm.py` `get_norms_and_activation` |
| Env observation: touch-sensor loop (2 M `np.sum` calls on 1-element lists) | ~10 s cum in `numpy.sum` | `Box2DSim_env.get_observation` |
| Saliency `filter` (8 convolutions per call) | 10.0 s cum | `Box2DSim_env.filter` |
| Box2D `Simulator.step` (C world step + PID) | 9.7 s tottime | unchanged |
| `radial2d` + `get_activation` | ~7 s / 3.5 s cum | `stm.py` |
| `controller.update` | 2.5 s | unchanged cost, was dominated by the same norm |
| Env construction (`prepare_world`, `set_taskspace`) | ~1.1 s | negligible |
| KMeans, predictor, match computation | < 0.2 s | negligible |

This corrects `04_speed_improvement_plan.md`: rebuilding environments every epoch and the KMeans metrics are not worth optimising; the cost is in the STM distance computation and the env observation code.

## Changes (3 files, about 25 lines)
1. `src/stm.py`, `get_norms_and_activation`: squared distances as `|x|^2 - 2 x.W + |w|^2` (matrix product, clamped at 0) instead of building and norming the B x D x K difference tensor.
   - Micro-benchmark, batch 90, K=100: visual 4.36 to 0.12 ms, touch 1.33 to 0.06, proprio 1.03 to 0.05, policy 5.28 to 0.14.
   - Max abs difference 1e-6 to 1.5e-4 (float32 cancellation on the large values), winner index identical in all tested cases.
2. `src/stm.py`, `radial2d`: grid built once per size/device (`cached_grid`) and squared distance computed directly (no sqrt, no generic norm). Max difference 1.2e-7.
3. `tools/box2dsim/box2dsim/envs/Box2DSim_env.py`, `filter`: the 8 convolutions are linear and summed before `max(0, .)`, so one convolution with the summed kernel (`flts_sum`, built in `init_salience_filters`) gives the same result. Micro-benchmark 0.83 to 0.10 ms per call, max difference 3.9e-16.
4. Same file, `get_observation`: built-in `sum` instead of `np.sum` on the 1-element list of contact counts. Same integers, no numpy call overhead.

## Result (same 12-epoch workload)
| Version | Total | Change |
|---|---|---|
| Before | 91.0 s | |
| After 1-3 | 68.9 s | -24% |
| After 1-4 | 56.6 s | -38% |

Per function: `SMController.spread` 14.9 to 2.6 s, `controller.update` 2.5 to 0.1 s, `get_activation` 3.5 to 0.3 s, `filter` 10.0 to 5.5 s, `get_observation` 15.4 to 4.2 s. Each figure is a single run; run-to-run timing noise is a few percent.

## Seeded motor noise and equivalence check
Motor noise (`SensoryMotorCircle.noisy_step`) now uses the controller's seeded RNG instead of the global `np.random`, and the two `KMeans` calls use `random_state=0` (`src/SMMain.py`). Runs with the same `-s` seed are now reproducible, which allows a real before/after comparison.

Checks, 4 epochs, `-s 7`, `epochs_to_test=2`, CPU:
- Two plain runs with the same seed: weights at epochs 0, 2, 3 identical (max abs difference 0.0).
- Plain run vs the active agent of a `--parasite` run: identical (0.0). The parasite no longer perturbs the active agent's random streams.
- Original code (`stm.py` and `Box2DSim_env.py` from `HEAD`) vs this branch, same seed: weights identical (0.0) and `trajectories.csv` identical (2682 rows). The rounding-level differences did not change any winner in this run.
- Speed in that check: 7.1 / 10.1 / 7.1 s per test interval vs 10.6 / 15.4 / 10.5 s (about 33% faster).
This is a 4-epoch check; over thousands of epochs a tie flip could in principle make runs diverge, which would not indicate a bug.

Risks: changes 3 and 4 are exact up to rounding; changes 1 and 2 differ from the old code at float32 rounding level.

## Remaining hot spots (after all four changes, 56.6 s)
- `Simulator.step`: about 10 s tottime for 49.6 k env steps (Box2D world step plus a Python PID per joint). Changing solver iterations would alter the physics, so left alone.
- `numpy.array` and `numpy.sum` overhead: 1.3 M small `np.array` calls (3.1 s) in the observation code (joint angles, object positions); could be removed by reading Box2D attributes into preallocated arrays.
- `filter` still 5.5 s (image masks with `np.linalg.norm`, 3 per call); `fillPoly` 2 s (rendering the background for saliency each step).
- STM maps: `spread` 2.6 s and `get_point_and_representation` 1.4 s left; wrapping inference methods in `torch.no_grad()` and reusing the visual activation from `spread` would save a little more (a few percent).
- The larger structural options (parallel episodes, vectorised event loop) remain in `04_speed_improvement_plan.md`.

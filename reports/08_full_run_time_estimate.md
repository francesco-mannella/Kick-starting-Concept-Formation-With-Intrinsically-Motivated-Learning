# Time estimate: parasite experiments for Figs 5-7

Basis: measurements on this machine (i7-7700K, 4 cores / 8 threads, CPU only, `OMP_NUM_THREADS=1`), code on `speed_cheap_wins`, default parameters, `epochs_to_test=10`, no rendering (`-x` off). Numbers are estimates, not a run of the full experiment.

## Measured
| Quantity | Value |
|---|---|
| Epoch, plain run, solo | 3.0 s |
| Epoch, `--parasite` run, solo | 3.4 s |
| One evaluation (`diagnose`: 18 episodes + maps), solo | about 3.9 s (parasite mode runs two: about 7.8 s) |
| Slowdown with 4 parallel runs | none (3.05 s/epoch) |
| Slowdown with 8 parallel runs | 1.75x (5.35 s/epoch); throughput only about 14% above 4 parallel |

## Experiment design assumed (from the draft)
10 replications; 1000 training epochs (text of Result 1; the plot axes run to 1500). Fig 5 and Fig 6 (agent vs parasite) come from the same 10 `--parasite` runs, since the active agent in those runs is identical to a plain run. Fig 7 needs 10 further runs: the parasite weights saved at epoch 1000 (`weights-parasite.npy`) loaded into a normal run (`-l`) and trained as an active agent (assumed 1000 more epochs).

## Cost per run (solo)
| Run | 1000 epochs | 1500 epochs |
|---|---|---|
| Parasite-mode run: epochs + 100 (150) double evaluations | 3.4 ks + 0.8 ks = about 69 min | about 103 min |
| Former-parasite run (plain): epochs + 100 (150) evaluations | 3.0 ks + 0.4 ks = about 57 min | about 85 min |

## Totals
| | 1000 epochs | 1500 epochs |
|---|---|---|
| CPU time, all 20 runs | about 21 h | about 31 h |
| Wall clock, 4 runs in parallel | about 5.5-6 h | about 8-9 h |
| Wall clock, 8 runs in parallel (1.75x slower each) | about 5 h | about 7.5 h |

Fig 7 runs can start per seed as soon as that seed's parasite run finishes, so a job queue (for example `xargs -P`) wastes less than the wave-based `grid_search.py`, which waits for every process of a batch.

## Uncertainty (about +/-25%)
- Epoch cost may change as agents learn (more contacts, episodes ending early); the measurement covers only the first 32 epochs.
- The paper's exact hyperparameters are unknown; defaults and the recent `test.json` values differ (see `05`). Cost per epoch is largely independent of them, the learning outcome is not.
- `epochs_to_test=50` instead of 10 saves about 10-15%.

## Launching
- `src/launch_experiment.py --root <dir> -p 8` runs the 20 runs (parasite then former per seed, job pool, skips finished runs); `src/launch_run.py` runs a single one. `grid_search.py` also has `--parasite` now.
- Fixed on the way: `-l <weights>` crashed (`None/weights.npy`) because the storage folder was not set up before saving; the former-parasite runs depend on it.
- The former-parasite runs load `parasite_NNN/simulations/parasite_NNN/storage/<epochs-1>/weights-parasite.npy` of the same seed.
- Tested end to end with 3 seeds x 3 epochs (6 runs, 2 in parallel); the full 1000-epoch experiment has not been run.

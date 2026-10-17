# Plan: speed improvements

Based on reading the code; no profiling has been run yet, so estimates are hypotheses to confirm in step 0. Constraint: seeded runs should stay numerically identical unless a step says otherwise.

## 0. Measure first
- `python -m cProfile -o prof.out SMMain.py -s 1 -n prof -o epochs=5 -o epochs_to_test=5`, then `snakeviz`/`pstats`. Also `py-spy record` for a longer run.
- Epoch time is already printed (`---- TIME`); log per-phase timers: env stepping, `controller.spread`, match/event loop, `controller.update`, `diagnose`.
- Expected split (to verify): Box2D stepping + observation building dominates; then per-5-step STM `spread` calls on tiny CPU tensors; then `diagnose`.

## 1. Episode stepping (likely the largest cost)
Each epoch: 18 episodes x 200 steps = 3600 env steps, serial, plus env construction.
1. Parallelise episodes across processes (`multiprocessing` / `concurrent.futures`) with one env per worker kept alive; send actions, receive observations each 5-step block. Box2D is single-threaded and each episode is independent, so near-linear speedup up to 18 workers. Sync only at the `action_steps` boundary (already the policy-update cadence).
2. Stop rebuilding `SMEnv` (gym.make + Box2D world build + `set_taskspace`) for every episode of every epoch: create once, and `reset(world, seed)` per epoch. Verify that `reset` reproduces the same initial state as a fresh env for the same seed.
3. `SMEnv.step` appends `copy.deepcopy(observation)` only when `store_observations`; keep it off in training (already default).
4. In `Box2DSim_env.py` profile `get_observation` (40 touch sensors, 300-pixel vision) and `Simulator` loops for per-step Python overhead; vectorise sensor calculations with NumPy.

## 2. Remove per-step Python overhead in `run_episodes`
- Hoist invariants out of loops: `self.params.*` attribute chains, `np.s_` slices.
- `agent.updatePolicy` reshapes and writes parameters each step: only update when the policy changed (every `action_steps` at most).
- `SensoryMotorCircle.noisy_step` calls `np.random.normal` on the global RNG each step; draw noise for the whole noisy window up front using `self.rng` (also makes runs reproducible; this changes numbers, so do it as a separate flagged change).
- The inner `for i in range(t0, t)` loop of event detection runs 5 times per block with several boolean-mask assignments. Vectorise over the 5 steps with `np.diff` and `np.maximum.accumulate` on the block, or compile with `numba`.
- `cum_match`, `max_match`, masks are recomputed over `i:` tails (`max_match[mask, i:] = ...`): O(T^2) writes per episode. Store `max_match` per policy segment and expand once at the end.
- `model_data["batch_g"][success_mask, t:, :] = ...` fills the whole future tail for each policy change. Store goal/policy per segment index and gather when needed.

## 3. STM / PyTorch overhead
Called with batches of at most 18 x 5 = 90 vectors, so GPU transfer costs dominate on CUDA; CPU is likely faster for these sizes.
1. Compute `spread` for the four maps in one fused function: stack inputs, `torch.cdist` against kernels instead of `norm(x.unsqueeze(-1) - kernel.unsqueeze(0))` (avoids a B x D x 100 intermediate; for visual `D=300`).
2. `radial2d` rebuilds `make_grid(side)` on every call: cache the grid as a buffer in `STM.__init__`. Same for `getRepresentation` and `get_point_and_representation`.
3. Many `torch.tensor(...)` / `.cpu().numpy()` round trips per call. Keep numpy only at the module boundary and avoid converting arrays that are already float32 (`torch.from_numpy`). Use float32 end to end (buffers are float64 now).
4. `torch.set_default_device('cuda')` is set globally when `-g`: all small tensors, including the grid, go to the GPU. Either run training on CPU (`-g` off) or move only `update` to GPU; benchmark both.
5. `SMSTM.update` runs four separate Adam steps with `zero_grad`; fuse with `torch.optim.Adam(..., fused=True)` or batch the four maps if on GPU. Use `set_to_none=True`.
6. Use `torch.inference_mode()` for `spread`, `get_activation`, `backward` (no autograd needed).
7. `controller.update` builds `cond_ind` with a Python loop over `np.where(policy_changed == 1)`; vectorise with a cumulative-sum or `np.maximum.accumulate` trick, and note negative slice start (`i - 25 : i`) when `i < 25` wraps. Fix and speed together.

## 4. Matching and competence
- `computeMatchSimple` uses `np.stack` + `norm` on tiny arrays every block; switch to `np.hypot`/squared distance and avoid `moveaxis`.
- `getCompetenceGrid` recomputes `predictor.spread(goal_grid)` (100x100 matmul) on every call of `get_global_local_competence`, which is called per policy choice and per epoch. Cache and invalidate only when `predict.update` runs.
- `SMPredictKDE.update` with repeated indices: `competences[idx] += ...` does not accumulate duplicates; use `np.add.at` for correctness (changes results slightly) or deduplicate.

## 5. Per-epoch bookkeeping
- `internal_trajectory_data` builds Python lists of length 3600 per epoch (comprehensions) for 1000-3000 epochs, then concatenates DataFrames at the end. Use NumPy arrays and write every `epochs_to_test` epochs (append to CSV or a `.parquet`/`.npz`), which also fixes memory growth and loss of data on crash.
- wandb logging of many scalars per epoch: log in a single `wandb.log` (already) but compute the 6 `match_value_*` means once; the masked-mean expressions are re-evaluated repeatedly.
- `np.save(... "main.dump", [self])` is called by both `train` and `diagnose` on the same epoch: remove one. Pickling the whole `Main` including controllers is heavy; save only weights each test epoch and the full dump every N tests.

## 6. `diagnose` / evaluation
`diagnose` runs 18 evaluation episodes (serial, new `SMController` each time) plus four matplotlib renders every `epochs_to_test` epochs (10 in recent runs, i.e. every 10 epochs).
- Reuse one evaluation controller (copy weights in place instead of constructing `SMController` twice: once in `evaluation_episodes`, again as `controller_` in `_execute_and_collect_episode_results`, each building four SMSTMs, Adam optimisers and a 100x100 goal grid).
- Parallelise evaluation episodes the same way as step 1.
- `calc_mi_metrics` fits two `KMeans(n_clusters=10)` on up to 3600 x 375 points: set `n_init=1`/`'auto'`, use `MiniBatchKMeans`, or compute only when `use_wandb` is on / on the final epoch.
- Build `trajectories` DataFrames once per batch, not per episode with a `pd.merge`; use `np.concatenate` and one DataFrame.
- Graphs: draw maps every `k`-th test epoch and reuse figures (`fig.clf()` instead of new figure).
- Render gifs (`render="offline"`) only at the final epoch or a coarse schedule.

## 7. Post-processing
- `run_postures.py`: `NUM_PROCESSES = 8` fixed; set from `os.cpu_count()`. Each task launches a subprocess that re-imports torch and re-loads `main.dump.npy`. Load once per worker via `ProcessPoolExecutor(initializer=...)`.
- `postprocess_simulations.sh`: sequential over directories; run through `xargs -P`. Zip with `zip -1` or `tar --zstd`.
- `run_evaluation.py`: parameterised seeds (`-s 200 -n 30`) run serially; parallelise across seeds.

## 8. Grid search
`grid_search.py` waits for **all** processes of a batch before starting the next (`for process in processes: process.wait()`), leaving cores idle when run times differ. Use a semaphore/pool so a new run starts as soon as one finishes. Defaults to `-p 2`; set to the number of free cores minus the per-run worker count from step 1.

## Suggested order and expected gain (to confirm by profiling)
| Step | Effort | Expected gain |
|---|---|---|
| 0 profile | 1 h | decides everything else |
| 1.2 reuse envs, 3.2 cache grid, 3.6 inference_mode, 6 skip redundant controller/KMeans | 2-3 h | modest, safe, bit-identical |
| 5 trajectory/dump bookkeeping | 1-2 h | removes memory growth, small per-epoch gain |
| 2 vectorise event loop | half day | moderate |
| 1.1 parallel episodes | 1-2 days | largest, if env stepping dominates |
| 3.1/3.3 fused float32 spread | half day | moderate on CPU |
| 7, 8 | 2 h | wall-clock for batch workflows |

Validation for every step: seeded smoke run, compare `weights.npy` (bit-identical for pure refactors; statistically equivalent over 5 seeds for changes marked as altering numbers).

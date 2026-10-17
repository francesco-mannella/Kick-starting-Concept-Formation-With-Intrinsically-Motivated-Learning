# Project tree and state

Branch: `unsupervised_touch`. Paper: Mannella & Tummolini 2022, "Kick-starting concept formation with intrinsically motivated learning". Stack: Python, PyTorch (CPU or CUDA), NumPy, pandas, scikit-learn, gymnasium + Box2D, wandb (optional), matplotlib/seaborn.

## Top level

| Path | Content | State |
|---|---|---|
| `README.md`, `LICENSE.md` | Original paper README (CC BY 4.0) | Outdated: mentions `gym`, `-s/-t/-g/-x` only, not `-n -d -o -p --demo --render -w` |
| `docs/` | `demo.gif`, dependency tree (html/svg) | Dependency tree predates the new scripts |
| `src/` | All model, training, evaluation and plotting code | Active |
| `tools/box2dsim/` | Local pip package `box2dsim` (physics env `Box2DSimOneArmOneEye-v0`) | Stable; contains committed `.pyc` files |
| `combs.json`, `old.json`, `test.json` | Grid-search specs for `grid_search.py` (untracked) | Experiment scratch |
| `data/` (188M, untracked) | Output of `extract_data.sh` (one extracted simulation) | Generated |
| `src/data/` (133M, untracked) | Extracted `data.tar.gz`: grip actuator map/weights, episode PNGs | Generated; `StoredGrip*.npy` are required at runtime |
| `test` | Symlink to `/home/fmannella/tmp/kss` (logs `.log/.err`, `simulations/`, `wandb/`, `internal_trajectory_data.csv`) | Scratch output of a grid search |

## src/ by role (36 files, ~7.5k lines; `SMMain.py` alone is 2867)

### Model core
- `params.py` (161): `Parameters`, one flat constructor with about 85 hyperparameters; subclass of `ParameterManager`.
- `parameter_manager.py` (182): string/JSON/file (de)serialisation of parameters, hashing. Used for `-o KEY=VALUE` overrides.
- `stm.py` (304): `STM`/`SMSTM`, a torch self-organising map on a square grid (`internal_size=100`, so 10x10) with Gaussian neighbourhood, optional external "re-entrant" radials, Adam optimiser.
- `SMController.py` (499): holds four SMSTMs (visual, somatosensory, proprioceptive, policy) plus the competence predictor. Goal choice, noise modulation, match computation, learning update, save/load.
- `SMPredict.py` (134): `SMPredictKDE` (active, a per-prototype running mean of achieved match) and `SMPredict` (older linear/sigmoid predictor, only referenced by commented code).
- `SMAgent.py`, `GripAgent.py`, `Actuator.py` (68-38 lines each): linear policy `touch+joints+eye_pos -> map -> params -> 5 actions`. `GripMapping.py` (250) builds the fixed input map `StoredGripActuatorMap.npy` when missing.
- `SMEnv.py` (156): wrapper over the Box2D gym env; `SMEnvParasite` replays stored observations.
- `storage.py` (48): `StorageManager`, creates `simulations/<name>/{storage,www}` and per-epoch dirs.

### Orchestration
- `SMMain.py` (2867): `Main` class and CLI. Contains `SensoryMotorCircle`, `EpisodeRunner`, `build_episode_dataset`, `Main.run_episodes` (the simulation loop), `train`, `train_parasite` (about 570 lines, near-copy of `train`), `diagnose`, `evaluation_episodes`, `monte_carlo_episode_search`, `demo_episodes`, `demo_episodes_monte_carlo`, MI/KMeans metrics, argparse.
- `grid_search.py` (138): expands a JSON of parameter lists into subprocess runs of `SMMain.py`, N at a time. Hard-coded wandb project/entity and `-t 55000`.

### Post-processing (run on a finished simulation)
- `run_evaluation.py` (137): loads `main.dump.npy`, runs `evaluation_episodes`, writes goal grid, trajectories CSV, renderings.
- `run_postures.py` (297) schedules `plot_postures.py` (950) per goal/rep with a process pool; `plot_postures.py -l` lists the schedule.
- `extract_goal_frame.py` (89), `single_demo.py` (188), `plot_gripmapping.py` (183), `plot_match_values.py` (113), `run_policy_map.py` (4 lines).
- Shell: `postprocess_simulations.sh` (latest; analysis + zip), `extract_data.sh` (unzip one suffix), `run_eval_and_posts.sh` and `run_simulations.sh` (older, near-identical loops over `*000999*` dirs).

### Plot/visual helpers
- `SMGraphs.py` (400, `GraphManager`: visual, proprio, somatosensory, competence, policy maps), `tplot.py` (174, trajectory plots), `wplot.py` (187, weight plots; unreferenced), `mkvideo.py` (105, GIF/video; unreferenced, duplicate of `tools/box2dsim/.../envs/mkvideo.py` with a different default duration).

### Other
- `query.ia`: a prompt file for an AI assistant, not code.
- `policy_weights_*.npy`: initial policy weight experiments (loading code is commented out in `SMController`).
- `data.tar.gz`: packed `src/data` assets.

## tools/box2dsim
`Box2DSim_env.py` (697, gym env, rendering, saliency), `Simulator.py` (856), `JsonToPyBox2D.py` (611), `PID.py`, `svg2json.py`; models in `models/*.json` (arm, objects), `examples/` scripts. Installed with `pip install -e tools/box2dsim`.

## Runtime data flow

```
params.py -> SMMain.py (-o overrides, params.json)
  simulations/<name>/storage/<epoch>/{weights.npy, main.dump.npy, data.npy, trajectories.csv}
  simulations/<name>/www/{*_map.png, comp_grid.npy, log.npy, episode_*.gif}
  internal_trajectory_data.csv (cwd)
          |
          v  (post-processing, by directory name pattern)
run_evaluation.py -> run_postures.py -> plot_postures.py -> goalgrid, goals.csv, goal_frames, demos
```

## Known issues found while reading
1. `run_episodes`: `smcycles = [SensoryMotorCircle(...)] * batch_size` creates one shared object; its counter `t` advances once per episode per step, so the "choose a new action every `action_steps`" phase is shared across episodes. Probably unintended; verify before changing since published results depend on it.
2. `SMController.update` uses `np.bool` (works in NumPy 2.x, was removed in 1.24-1.x; breaks on older installs).
3. `monte_carlo_episode_search` calls `choose_policy_`, assigned to the controller only inside the same function; fragile monkeypatching.
4. `postprocess_simulations.sh` hard-codes `/home/fmannella/Projects/current/...` (the repo is under `Dropbox/Projects/current/`).
5. `diagnose` saves `main.dump` twice (also in `train`).
6. `train` rewrites `internal_trajectory_data.csv` only at the very end and keeps all epochs in RAM as Python lists.
7. `evaluation_episodes` has a comment that results without noise are "much worse"; `zero_noise=True` is the default anyway.
8. Dead or outdated code: `computeMatch`, `add_noise_to_vector_maintaining_norm`, `update_reentrant_connections`, `softmax`/`flt`, `SMPredict` (linear), `wplot.py`, `mkvideo.py`, `run_policy_map.py`, `run_simulations.sh`.
9. `.gitignore` is Python-only template; `.pyc` under `tools/` are tracked; generated dirs show as untracked.
10. No tests, no requirements/environment file.

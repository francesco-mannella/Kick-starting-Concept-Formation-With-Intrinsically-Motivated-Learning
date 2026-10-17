# Plan: rationalise scripts

Goal: fewer entry points, one place for paths and constants, no duplicated logic. Each step is independently verifiable; do them in order. Behaviour of training must not change (compare `weights.npy` after a seeded 20-epoch run before/after).

## 0. Safety net
- Add a seeded smoke run: `SMMain.py -s 1 -n smoke -o epochs=20 -o epochs_to_test=10` and save a checksum of `weights.npy` and `trajectories.csv`. Re-check after every step below.
- Add `requirements.txt` (or `environment.yml`) from the working env (numpy 2.4, torch 2.7, pandas, scikit-learn, seaborn, gymnasium, box2d, wandb, slugify, regex, scipy, pillow).

## 1. Repo hygiene (no code change)
- Extend `.gitignore`: `data/`, `src/data/`, `test`, `simulations/`, `wandb/`, `*.log`, `*.err`, `*.npy` dumps, `*.csv` outputs, `*.zip`.
- `git rm --cached` the `tools/box2dsim/**/__pycache__/*.pyc`.
- Move scratch specs `combs.json`, `old.json`, `test.json` into `experiments/` with dated names, or delete `old.json`/`test.json` if obsolete.
- Move `query.ia` out of the repo (it is a prompt, not code).

## 2. Remove duplicates and dead code
| Item | Action |
|---|---|
| `run_simulations.sh`, `run_eval_and_posts.sh` | Delete; `postprocess_simulations.sh -r` covers them |
| `src/mkvideo.py` vs `tools/box2dsim/.../mkvideo.py` | Keep one (the package copy); import it; drop unused `src/mkvideo.py` |
| `wplot.py`, `run_policy_map.py` | Unreferenced; delete or move to `scripts/legacy/` |
| `SMPredict` (linear), `computeMatch`, `add_noise_to_vector_maintaining_norm`, `softmax`, `flt`, `update_reentrant_connections` | Remove if confirmed unused by experiments you still run |
| `SMEnvParasite`, `train_parasite` | Decide: keep or drop. If kept, see step 4 |
| `single_demo.py` vs `Main.run_single_episode` / `demo_episodes*` | Keep `single_demo.py` as the CLI, and have it only call `run_single_episode` |

## 3. Directory layout
```
src/
  kss/                  # importable package
    params.py parameter_manager.py storage.py
    stm.py predict.py controller.py
    agent/ (actuator.py grip_agent.py grip_mapping.py sm_agent.py)
    env.py
    runner.py           # SensoryMotorCircle, EpisodeRunner, run_episodes
    metrics.py          # calc_mi_metrics, calc_match_inc_within_goal
    evaluation.py       # evaluation_episodes, trajectories assembly
    train.py            # Main.train / diagnose
    viz/ (graphs.py tplot.py plot_postures.py plot_gripmapping.py plot_match_values.py)
  cli/ (train.py evaluate.py postures.py demo.py grid_search.py)
scripts/ (postprocess_simulations.sh extract_data.sh)
```
Do this by moving code verbatim and re-exporting from the old module names for one release, so `main.dump.npy` pickles (which reference `SMMain.Main` etc.) still load. Pickles hold class paths: keep `SMMain.py` as a shim `from kss.train import *` until old dumps are retired.

## 4. Split `SMMain.py` (2867 lines)
In this order, each a pure move:
1. `parse_arguments`, `AppendParamsAction`, `__main__` block -> `cli/train.py`.
2. `build_episode_dataset`, `SensoryMotorCircle`, `EpisodeRunner` -> `runner.py`.
3. `calc_mi_metrics`, `calc_match_inc_within_goal` -> `metrics.py`.
4. `evaluation_episodes`, `_execute_and_collect_episode_results`, trajectory DataFrame building -> `evaluation.py`.
5. `demo_episodes`, `demo_episodes_monte_carlo`, `monte_carlo_episode_search` -> `demo.py`; replace the `choose_policy_` monkeypatching with a `policy_chooser` callable argument.
6. `train` and `train_parasite`: factor the shared epoch body into `_run_epoch(controller, ...)`; `train_parasite` becomes two calls of it (or is deleted). Removes about 500 duplicated lines.
7. Replace `initialize_model_data`'s repeated shape blocks with a dict of `{name: trailing_shape}`.

## 5. Parameters
- Replace the 85 assignments in `Parameters.__init__` with a `dataclass` (types come free, `param_types` machinery in `ParameterManager` can shrink to `asdict`/`replace`). Keep `update("k=v;..")`, `load/save` JSON and `__getstate__` compatibility.
- Move paths, wandb project/entity and the `-t 55000` literal from `grid_search.py` into CLI options or env vars.

## 6. Shell and workflow scripts
- One `postprocess_simulations.sh`; replace the hard-coded `PYTHONPATH=/home/fmannella/Projects/...` with `$(dirname "$0")`-relative resolution (the repo currently lives under `Dropbox/Projects`).
- `run_evaluation.py` and `run_postures.py` hard-code `Path.cwd()/../../../..` and the experiment name; take `--sim-dir` instead of relying on cwd.
- `extract_data.sh`: use `mktemp -d` instead of `/tmp` + `rm -fr /tmp/home`.
- Give scripts a common prefix (`kss-train`, `kss-eval`, `kss-postures`, `kss-grid`) via `pyproject.toml` entry points; drop the `#!/usr/bin/env python` + `PATH` hack.

## 7. Tests (small)
- `test_stm.py`: winner selection, radial shape, update lowers loss on toy data.
- `test_params.py`: `update("a=1;b=2")`, JSON roundtrip.
- `test_runner_smoke.py`: 2 episodes, 60 steps, shapes and no exceptions.
- `test_checkpoint.py`: save/load `main.dump` round trip.

## 8. Docs
Update `README.md` (CLI flags, install, new layout) and regenerate the dependency tree after the move.

## Order of effort
1 and 2 are minutes; 3-4 are the real work (about a day with the smoke checksum); 5-8 are optional follow-ups.

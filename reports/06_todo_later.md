# To do later (parked 2026-10-16)

Nothing here is started. Context: `05_knowledgebase_vs_code.md`.

## Decision needed first
Reference for the model: draft equations or current code behaviour. Open items 1-7 in `05` depend on it.

## Checks (no simulator needed)
1. Port `knowledgebase/stm_full_linear_ring.html` to a short Python test; compare C_tot curves for plain SGD vs `stm.SMSTM` (Adam).
2. Scale `local_lr` by 10x in `SMController.update` and check whether STM weights change (is the learning-rate modulation cancelled by Adam?).

## Then
3. Seeded smoke run + checksum of `weights.npy` (see `03`, step 0) before touching `SMMain.py`.
4. Rationalisation plan: `03_script_rationalization_plan.md`.
5. Speed plan: `04_speed_improvement_plan.md` (RNG and `np.add.at` changes alter numbers; do after the reference is decided).
6. Update the draft: environment text (18 episode types), match equation and width, predictor equation, Zubek citation year, evaluation-noise note, argmax vs blended readout.
7. Optional experiment: add contexts mid-training (draft limitation; link to Topographic Gating).

## Known bugs to confirm
- `run_episodes`: `[SensoryMotorCircle(...)] * batch_size` shares one counter across episodes.
- `SMController.update`: `cond_ind[i - 25 : i]` wraps for `i < 25`.
- `postprocess_simulations.sh`: hard-coded `/home/fmannella/Projects/...` path.

## Found on branch fix_parasite
- Motor noise was drawn from the global, unseeded `np.random` (runs not reproducible). Fixed on `speed_cheap_wins`: `SensoryMotorCircle` now draws from the controller's seeded RNG, and the two `KMeans` calls use `random_state=0`. Same-seed runs are now identical. Still unseeded but unused in normal runs: `GripMapping.py` (only runs when `StoredGripActuatorMap.npy` is missing).

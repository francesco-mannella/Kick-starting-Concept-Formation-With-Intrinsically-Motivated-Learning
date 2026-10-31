# Plan: improve on `both` (reachable-subspace noise + per-node baseline selection)

## 1. Where `both` stands

`both` = `policy_noise_reach_k=15` + `policy_update_baseline=0.05` on the champion (all-maps anchored + `touch_gate=True`). Epochs 800-999, 3 seeds, means (report 11, memory `reach-probe`):

| | Success | Swap | Constant policy | Weight eta2 | Posture eta2 | Prototype norm | Effective noise share |
|---|---|---|---|---|---|---|---|
| champion (gate) | 0.89 | 0.88 | 0.77 | 0.099 | 0.31 | 33-34 | 0.65-0.86 |
| both | 0.74 | 0.66 | 0.32 | 0.148 | 0.13 | 11 | 0.75-0.88 |

What worked: the policy content now matters (intact minus constant success 0.21 / 0.58 / 0.46 per seed, gate 0.10 / 0.05 / 0.22; swap cost 0.01-0.16, gate about 0). What did not: success is 0.15 lower, posture eta2 fell, the effective output noise did not drop, the prototypes halved in norm (the map is starved), and `reach15` seed 2 was degenerate (success 0.47, touch 0.09).

Goal: keep the content dependence of `both`, recover the gate's success (>= 0.85) and posture eta2 (>= 0.3), and lower the effective (5-d) noise share clearly below 0.65-0.88. Constraints: topological alignment stays (goal node is the shared anchor, no hindsight anchoring); no designed task pressure; all changes default off.

## 2. Hypotheses to test (each traceable to a measurement)

1. **Starvation by the baseline rule.** About half of the events pass and each is weighted by advantage / max, so the total update mass falls well below the unfiltered EMA. Prototypes then stay small (norm 11) and the map follows the noise less, but also learns less. Fix: keep the selection, normalise the mass.
2. **Hard elite cut is too noisy.** With few events per node, "advantage > 0" is a coin flip for nodes with a flat match. Fix: soft weights (softmax or sigmoid of the advantage with a temperature), with a floor.
3. **Competence saturates early (transfer from the arm lab).** In the arm lab the largest gain (hit rate 0.40 to 0.87-0.91) came from lowering `decay`/`local_decay` from 5 to 1.5/0.7, because `tanh(5c)` is already 0.96 at the match lattice value of 0.4 and the learning rate, neighbourhood and noise collapsed while the policies were still off. The main model uses `decay=5`, `local_decay=5` and the same lattice values (0.5 / 0.32 / 0.21). Unmeasured here: the actual trajectories of local incompetence, noise std, learning rate.
4. **Noise level is tied to the prototype scale.** Prototypes are an accumulation of noise, so their norm follows `max_policy_noise` and the signal-to-noise ratio is fixed by construction (the earlier cap probe changed nothing for that reason, and also because the output was saturated). Fix: bound the noise relative to the prototype's own output spread, not in absolute units.
5. **k = 15 and equal variance per direction may be the wrong subspace.** The basis comes from the uncentred second moment, so one direction (the constant offset, 83% of the energy) is in it and the state-dependent part is a small share. Test k and a variance profile proportional to the eigenvalues.

## 3. Steps

**Step 0: diagnostics (read-only, no runs).** From the existing `both`, gate and champion runs (`~/tmp/kss/reach_probe`, `gate_long`): trajectories of mean local incompetence, noise std and learning rate (`internal_trajectory_data*.csv.gz`; note the CSVs are being compressed, read with `.gz`); fraction of events passing the baseline per epoch; per-node event counts. Decides how much weight hypotheses 1 and 3 deserve.

**Step 1: screen in the arm lab (minutes per config).** `~/tmp/kss/arm_lab/` has the headless harness (`core_grid.js`, `sweep.js`) with the baseline option and noise parameters. Screen soft weights, mass normalisation and noise scaled to the prototype spread there first (hit rate, alignment, 8 seeds). The toy has 4-d policies with no unreachable dimensions, so it can test only hypotheses 1, 2 and 4, not 5.

**Step 2: implement in the main model (all default off, in `src/params.py`, `src/SMController.py`).**
- `policy_baseline_soft` (temperature; 0 = hard elite as now): replace `adv > 0` by `softmax(adv / T)` weights within `baseline_select`.
- `policy_baseline_norm` (bool): rescale the selected weights so their sum equals the unfiltered sum (`lr_pol` mass preserved).
- `policy_noise_rel` (float): cap the noise std at `rel x` the RMS of the prototype output spread in the reachable subspace (computed from the current `reach_basis` and map).
- `reach_var_profile` ('flat' | 'eig'): noise variance per reachable direction flat (now) or proportional to the eigenvalue.
- Reuse: `update_reach_basis`, `baseline_select`, `getPoliciesFromPointsWithNoise`, `policy_sel` mask. Competence decay needs no code (`decay`, `local_decay` already parameters).

**Step 3: screen runs (main model).** Gate base, seeds 0-2, 1000 epochs, `--epochs-to-test 50`, 5 parallel (about 70 min per batch), controls = gate and `both` runs in `~/tmp/kss/gate_long` and `reach_probe`. All with `both` options unless stated:
1. `both + decay=1.5, local_decay=0.7`
2. `both + policy_baseline_norm`
3. `both + policy_baseline_soft=T` (T chosen from step 0 / step 1)
4. `both + policy_baseline_soft + policy_baseline_norm`
5. `both + policy_noise_rel` (two values)
6. `both` with `policy_noise_reach_k` 5 and 25
7. `both + reach_var_profile='eig'`

About 8-9 conditions x 3 seeds = 24-27 runs, 5-6 batches (about 6-7 hours). Wait until the running champion sweep (`fig567_champion_gate`) is done, or the timings double.

**Step 4: combine and confirm.** Take the 1-2 winners, run the combination with 5 seeds (seed spread of the constant-policy cost is large: gate SD 0.05, control 0.31). Then score as before: `policy_stats.py --counterfactual --dims 3,10` for epochs 800-999, `effnoise.py` (`~/tmp/kss/signal_analysis`) for the 5-d noise share, and the table scripts (`reach_probe/table.py` as the template).

## 4. Decision rule

A variant counts as an improvement over `both` only if, with the gate as reference:
- success >= 0.85 (gate 0.89, `both` 0.74) in at least 2 of 3 seeds, none degenerate (touch >= 0.15);
- constant-policy cost and swap cost still above the gate's seed spread (cost >= 0.2 and swap cost >= 0.05 on average);
- posture eta2 >= 0.25;
- effective 5-d noise share below 0.65 on average, or a clear rise in the prototype norm (> 20) at the same noise.

Weight-space noise share falling alone does not count (it falls by construction when the noise is confined to the subspace). Report only differences above the seed spread; three seeds are a trend.

## 5. Risks and open points

- Three seeds, high variance: `reach15` seed 2 degenerated alone; a bad seed can mask a good change. Use 5 seeds for the final comparison.
- Lowering the competence decay lengthens the high-noise phase; it may reduce success early. Compare the success trajectory over epochs, not only epochs 800-999.
- The effective noise share depends on the evaluation of the reachable basis with the eye input at its mean; the eye variance is not measured.
- The arm lab result (decay) is a different system (4-d posture policy, 100-252 nodes): the transfer is a hypothesis, step 0 checks it before step 3 spends several CPU-hours.
- If none of the variants lowers the effective noise while keeping the success, the remaining directions are in the actuator (touch magnitude, compact parametrisation) and in checking the alignment of touch, proprioception and vision at the goal nodes; both need no change to the environment.

## 6. Files

Modified (step 2): `src/params.py`, `src/SMController.py`. Scripts: `~/tmp/kss/signal_analysis/effnoise*.py`, `~/tmp/kss/reach_probe/table.py`, `~/tmp/kss/arm_lab/{core_grid.js,sweep.js,run.js}`. Memory: `reach-probe`, `arm-lab-artifact`, `policy-signal-nonlinear-dependence`. Report: `reports/11_champion_configuration.md` (section 4).

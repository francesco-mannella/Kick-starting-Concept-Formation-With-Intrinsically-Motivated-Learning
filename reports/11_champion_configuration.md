# The champion configuration: algorithm, parameters and the evidence for it

Period: 2026-10-27 to 2026-10-31. Branches `speed_cheap_wins`, `noise_plan`, `dof8` and `embed_actuator`. Builds on reports 02 (main algorithm), 09 (actuator and policy learning), 10 (policy signal) and 12 (the actuator). All numbers are late checkpoints (epochs 800 to 999) of 1000-epoch runs, mean over seeds 0 to 2, unless stated. Sections 4 and 5 were measured with the former 5-output actuator (tag `champion-5dof`, commit `f72182f`); the champion has since moved to the 8-output actuator, which has only been checked for 150 epochs (section 4.3).

## 1. Summary

1. The best configuration found is the all-maps anchored set of report 09 section 9 plus the contact gate (`touch_gate=True`), run with an unsaturated actuator output (`policy_params_amplitude=0.07` instead of 8.0, section 4.1). It is the champion from now on. Against the previous champion (same set without the gate, amplitude 8) the gate gives steadier success (0.89 against 0.77, seed SD 0.04 against 0.11) and the clearest specificity gain (eta2 in the top-3 prototype components 0.51 against 0.33; posture eta2 0.31 against 0.25; constant-mean-policy success 0.77 against 0.56). Those gate numbers were measured at amplitude 8; the amplitude change leaves the general measures and the noise space of the policies unchanged within the seed spread, with one weak seed in five (success 0.81 against 0.90).
2. Twelve variants were tested on top of it (selection thresholds, stricter success rules, a tighter match width, noise confined to prototype or reachable directions, an output-space update, a per-node baseline, lower policy amplitude). None improves on the gate in success and specificity together. The nearest alternative, `policy_update_baseline` with `policy_noise_reach_k` (called `both`), is the only one in which the policy content matters for success, at a cost of 0.15 in success. It is kept as an experimental option, not as the champion.
3. No configuration produces policies that depend on the shape or the rotation of the object (stretch and rotation eta2 at null; swapping policies between episodes costs 0 to 0.1). Whether this is a limit of the learning rule or a property of the environment is open (section 7).

4. The actuator of the champion is now the 8-output one (three arm joints, a wrist, four independent finger commands; map v4, 50 inputs, 600 policy weights; section 2 point 4, report 12). In a 5-seed, 150-epoch check its learning curves matched those of the 5-output champion over the same epochs, and in the full Figs 5-7 sweep (10 seeds, 1500 epochs, section 4.3) alignment and touch frequency equal those of the 5-output sweep, success is 0.04 lower (inside the seed spread), the active agent's context-policy MI is lower (0.90 against 0.97) and 2 of 10 former parasites stay below 0.3 success against 1. The policy statistics of section 5 have not been repeated with it. A 15-number embedding input layer was tested in its place and rejected (section 4.3).

"Optimal" here means best of the tested set under the metrics of section 4, with three seeds per group. It is not a proven optimum.

## 2. Algorithm

The loop is that of report 02 (four 10 x 10 maps for vision, touch, proprioception and policy, a predictor of competence, goals chosen on the visual map, a match computed in the internal space, one event ends a policy). The champion differs in five places.

1. **Match on touch and proprioception only** (`modalities_weights=(0,1,1,0)`), arithmetic mean of the per-modality matches `exp(-d^2/sigma^2)` (`match_mode='arithmetic'`).
2. **Contact gate** (`touch_gate=True`, `SMController.computeMatchSimple`, `ss_contact` argument): the touch match is set to 0 when the touch input is zero. Without it the touch map's near-zero prototype (zero input, lowest-norm node) wins on non-contact steps and, when it lies near the goal, counts as a match (58% of non-contact steps in control seed 2).
3. **Anchored winners** for all maps (`policy_winner='anchored'`, `map_winner='anchored'`): the node updated is the winner among prototypes weighted by proximity to the goal node, so the visual, touch, proprioception and policy prototypes of one goal node are updated together (topological alignment). The policy map uses an EMA toward the executed policy (`policy_update='ema'`, `policy_ema_eta=0.05`); the sensory maps keep Adam.
4. **Actuator v4, 8 DoF** (report 09 for the encoding): RBF block encoding of touch (40), all 8 joints and eye position (50 inputs, blocks (40, 8, 2)), touch gain 3, 75 hidden units, `tanh` output, 75 x 8 = 600 policy weights per prototype. Outputs: three arm joints, a wrist joint (new palm link, +-pi/2), and four independent finger commands (close-only, `[0, pi/2]`; `policy_base_wrist=0`, `policy_base=0.8` on the fingers). Map `data/StoredGripActuatorMap_v4.npy`. Executed policy = prototype + Gaussian noise with std `base_policy_noise + (max_policy_noise - base_policy_noise) * local_incompetence`. The sections below were measured with the former 5-output actuator (v3, 375 weights; tag `champion-5dof`).
5. **Unsaturated output** (`policy_params_amplitude=0.07`): the policy is multiplied by this factor before the actuator. With the former value 8 the pre-tanh output had a median of about 200 (99% of the outputs above 3), so `tanh` always returned +-1 and only the sign pattern of the five outputs reached the arm. At 0.07 the median is about 1.9.

Event rule (unchanged): the touch match must rise against the previous step and the global match must exceed the running maximum of the policy by `match_incr_th` (0.001); one event ends a policy (`cum_match_stop_th` 1.0); success is at least two policy changes.

## 3. Parameters

Options on top of the defaults (`src/params.py`), as `KEY=VALUE` strings for `launch_run.py`, `launch_fig567.py` or a `launch_conditions.py` JSON:

```
grip_map_name='data/StoredGripActuatorMap_v4'
grip_norm_name='data/StoredGripActuatorNorm_v4'
grip_input_norm='block_rms'
grip_encoding='rbf_block'
grip_out_fn='tanh'
grip_touch_gain=3.0
policy_winner='anchored'
policy_update='ema'
map_winner='anchored'
touch_gate=True
```

The map and norm names above (v4, for the 8-output actuator) and the amplitude (`policy_params_amplitude=0.07`) are now the defaults in `params.py` and need no option; the 5-output champion needs the v3 pair, `grip_input=47`, `grip_output=5`, `proprioception_size=5` and the code of tag `champion-5dof`. Runs of the amplitude-8 champion need `-o policy_params_amplitude=8.0`. Runs and pickled checkpoints made earlier keep the value they were started with (8.0).

Defaults that matter and were not changed: `internal_size=100`, `batch_size=18`, `stime=200` (steps per episode), `action_steps=5`, `epochs=1000`, `match_sigma=1.5`, `internal_sigma=10`, `base_internal_sigma=0.1`, `base_lr=0.005`, `max_lr=1.0`, `base_policy_noise=0.01`, `max_policy_noise=20`, `match_incr_th=0.001`, `cum_match_stop_th=1.0`, `map_update='adam'`, `policy_ema_eta=0.05`.

`touch_gate` is still `False` in the defaults of `params.py`, so the gate is defined by the option above. All other probe parameters (`touch_incr_th`, `policy_update_min_match`, `policy_update_min_mod_match`, `policy_noise_subspace_k`, `policy_noise_floor`, `policy_output_space`, `policy_noise_reach_k`, `policy_update_baseline`, `policy_baseline_soft`, `policy_baseline_norm`, `policy_noise_rel`, `reach_var_profile`) default to off.

## 4. Evidence

Metrics: weight eta2 and top-3 eta2 are the share of executed-policy variance explained by context (object kind), in the 375-d weights and in the top three prototype components; posture eta2 uses the episode-mean joint posture; MI is the adjusted mutual information between context and policy node; success is the share of evaluation episodes with at least two policy changes; swap and constant are the success when policies are exchanged between episodes or replaced by the mean policy (a cost against intact means that policy content matters).

| Group | Weight eta2 | Top-3 eta2 | Posture eta2 | MI | Success | Swap | Constant |
|---|---|---|---|---|---|---|---|
| previous champion (control) | 0.072 | 0.33 | 0.25 | 0.51 | 0.77 | 0.79 | 0.56 |
| champion (gate) | 0.099 | 0.51 | 0.31 | 0.59 | 0.89 | 0.88 | 0.77 |
| stricter success (`match_incr_th` 0.1 / `touch_incr_th` 0.1) | | | | | 0.58 / 0.69 | | |
| selection 0.6 (`policy_update_min_match`) | 0.010 | 0.16 | 0.06 | 0.22 | 0.78 | 0.66 | 0.66 |
| mod0.5 (both modalities >= 0.5 for an update) | 0.020 | 0.15 | 0.06 | 0.37 | 0.39 | 0.43 | 0.33 |
| sigma0.8 (`base_match_sigma` 0.8) | 0.026 | 0.24 | 0.22 | 0.24 | 0.85 | 0.85 | 0.84 |
| subnoise10 (noise in prototype PCs) | 0.082 | 0.18* | 0.25 | 0.51 | 0.81 | 0.78 | 0.74 |
| outspace (output-space update) | 0.027 | 0.38 | 0.32 | 0.34 | 0.78 | 0.74 | 0.75 |
| amp0.1 (`policy_params_amplitude`) | 0.122 | 0.49 | 0.34 | 0.59 | 0.82 | 0.83 | 0.77 |
| amp0.03 | 0.081 | 0.48 | 0.35 | 0.56 | 0.89 | 0.87 | 0.87 |
| reach15 (`policy_noise_reach_k` 15) | 0.203 | 0.53 | 0.29 | 0.53 | 0.65 | 0.64 | 0.43 |
| base (`policy_update_baseline` 0.05) | 0.120 | 0.45 | 0.30 | 0.55 | 0.86 | 0.79 | 0.51 |
| both (reach15 and base) | 0.148 | 0.48 | 0.13 | 0.62 | 0.74 | 0.66 | 0.32 |

*Biased by construction (noise lives in the subspace being measured).

Reading:

- The gate is the only change that improves success and specificity together. Weight eta2 (0.099 against 0.072) is inside the seed spread; top-3 eta2, success and constant-policy success are outside it.
- Stricter rules, global update thresholds and a tighter match width lower success or specificity (the policy map is starved: prototype norm 8 to 18 against 33 to 34, noise share 1.0).
- Lowering the amplitude (the real pre-tanh output with amplitude 8 is about 200, saturated) leaves the gate's numbers unchanged within the seed spread (3-seed probe above, 5-seed confirmation in section 4.1); it was therefore adopted for the champion as a removal of a source of noise, not as a gain.
- `both` is the only group with a consistent policy-content effect: intact minus constant success 0.21, 0.58 and 0.46 across seeds (gate 0.10, 0.05, 0.22), swap cost 0.01 to 0.16. It comes with success 0.74 against 0.89, a posture eta2 of 0.13 and no drop in the 5-d effective noise share (0.75 to 0.88). It is a partial hint.

### 4.1 Removing the saturation (amplitude 8 against 0.07 and 0.03)

Goal: remove the saturation of the output as a source of noise; not to change the swap cost. Gate base, 1000 epochs, 5 seeds per column, epochs 800 to 999; intervals are 95% bootstrap on the difference of the means against amplitude 8.

| | amplitude 8 (gate) | amplitude 0.07 | amplitude 0.03 |
|---|---|---|---|
| Median \|pre-tanh\| output | 211 | **1.9** | **1.0** |
| Share of outputs above 3 | 99% | 33% | 8% |
| Prototype RMS / norm | 1.7-1.8 / 34 | 1.73 / 33 | 1.8 / 36 |
| Success | 0.902 | 0.813 (-0.089 [-0.24, +0.02]) | 0.822 (-0.080 [-0.20, 0.00]) |
| Touch frequency | 0.274 | 0.258 | 0.252 |
| Alignment | 0.618 | 0.605 | 0.537 (-0.081 [-0.12, -0.04]) |
| Weight eta2 / posture eta2 | 0.093 / 0.280 | 0.093 / 0.288 | 0.075 / 0.398 |
| MI context / policy | 0.555 | 0.473 | 0.519 |
| Weight noise share | 0.892 | 0.873 | 0.931 |
| Effective 5-d noise share | 0.752 | 0.786 (+0.03 [-0.09, +0.17]) | 0.853 (+0.10 [-0.01, +0.23]) |
| Prototype / executed output SD | 0.537 | 0.512 | 0.520 |
| Swap / constant-policy cost (information only) | 0.007 / 0.106 | 0.025 / 0.111 | 0.009 / 0.039 |
| Seeds with success below 0.6 | 0 of 5 | 1 (0.519) | 1 (0.593) |

- The weights keep the champion's scale and the arm works: no collapse. The unsaturated output carries no hidden noise: the noise space of the policies (effective noise share, prototype against executed spread, weight noise share) is the same as at amplitude 8 within the spread.
- Amplitude 0.07 keeps alignment and the noise-space measures at the gate's level and brings the median to 1.9; the share above 3 (33%) is just above the 30% target, so a value near 0.05 would sit inside it. Amplitude 0.03 removes the saturation more fully but loses alignment significantly.
- Success is lower on average in both (0.81 and 0.82 against 0.90, intervals including or touching 0) because of one weak seed each, with low touch (about 0.20). Pooled, 8 of 10 unsaturated seeds are 0.82 to 0.94, against 0.86 to 0.94 for the five amplitude-8 seeds. A weak-seed rate of about one in five may be real; five seeds cannot decide.
- Rotation eta2 is higher at 0.07 (0.035 against 0.014, [+0.007, +0.036]); I do not count it as an effect yet.
- Choice: 0.07 over 0.03 (alignment and noise space).

### 4.2 Full sweep at amplitude 8 (Figs 5-7, 10 seeds, 1500 epochs)

The champion (all-maps anchored + gate, amplitude 8) against the previous champion (all-maps anchored, `experiments/fig567_all_maps_anchored`), mean over seeds, epochs 1000 to 1500:

| | champion | previous |
|---|---|---|
| Agent: alignment / success | 0.645 / 0.776 | 0.640 / 0.784 |
| Former parasite: success | **0.659** (SD 0.27) | 0.510 (SD 0.36) |
| Former parasites with success below 0.3 | **1 of 10** | 4 of 10 |
| Former parasite: touch frequency | 0.129 | 0.088 |
| Former parasite: MI context / touch | 0.144 | 0.095 |

The agent is unchanged; the gate helps the recovery of former parasites (paired difference +0.15, higher in 7 of 10 seeds). Policy statistics at epoch 1499 are alike in both sweeps (eta2 of context 0.099 for the champion agent against 0.111, below or near the permutation p95; stretch and rotation eta2 below p95; noise share 0.92). Scripts: `~/tmp/kss/score_fig567.py`, data `~/tmp/kss/fig567_champion_gate`. This sweep was run at amplitude 8; a repeat at 0.07 has not been done.

### 4.3 The 8-output actuator and the embedding alternative

8-output actuator (branch `dof8`, `~/tmp/kss/dof8_probe`, 5 seeds, 150 epochs, champion options with map v4):
- Learning curves and success: no NaNs; success 0.68, touch frequency 0.21, mean competence 0.58 over the evaluation points of epochs 101 to 150. The epoch-150 counterfactual (`~/tmp/kss/embed_probe/cf`, 36 episodes x 3 evaluation seeds): intact success 0.70 (seed SD 0.11), swap cost 0.02, constant-policy cost 0.19 (0.59 and 0.29 in seeds 1 and 3, about 0 in the others), within-context shuffle cost 0.07.
- Saturation census at epoch 150: median pre-tanh 3.8 and 55% of outputs above 3, higher than the median of about 1.9 of the 5-output champion at epoch 999; this is an early-training value and was not followed to 1000 epochs.
- Fingers: the two fingers are used independently. In seed 0 finger 1 is bent at the distal joint and finger 2 straight in 27% of the steps, in a fixed way for all objects; in the other four seeds both fingers close and that pattern appears in at most 1% of the steps. The simulator is symmetric between the fingers (equal commands give mirrored angles; random babbling shows no touch advantage for either arrangement), so the pattern comes from learning (report 12 section 7).
- Phase 0 input-layer diagnostics at epoch 150 are in report 12 section 6: the agent visits 18 of the 25 posture cells and 20 to 22 of the 25 touch and eye cells.

Figs 5-7 sweep with the 8-output actuator (`~/tmp/kss/fig567_dof8`, 10 parasite and 10 former runs, 1500 epochs, former parasites started from the parasite weights at epoch 1000, champion options with the v4 defaults, amplitude 0.07, about 9.5 hours). The reference is the 5-output sweep of section 4.2 (amplitude 8), so the comparison changes the actuator and the amplitude together. Mean over seeds (SD), epochs 1000 to 1500, paired difference to the 5-output sweep and the number of seeds in which the 8-output run is higher:

| Series, measure | 8-output | 5-output | difference (higher in) |
|---|---|---|---|
| Agent: alignment | 0.652 (0.093) | 0.645 (0.024) | +0.007 (6/10) |
| Agent: success | 0.734 (0.129) | 0.776 (0.139) | -0.042 (3/10) |
| Agent: MI context / policy | 0.902 (0.091) | 0.965 (0.067) | -0.062 (2/10) |
| Agent: touch frequency | 0.175 (0.066) | 0.163 (0.048) | +0.012 (7/10) |
| Parasite: alignment | 0.656 (0.071) | 0.644 (0.049) | +0.011 (6/10) |
| Parasite: MI context / policy | 0.996 (0.050) | 0.932 (0.150) | +0.064 (6/10) |
| Former parasite: alignment | 0.628 (0.067) | 0.627 (0.036) | +0.001 (6/10) |
| Former parasite: success | 0.623 (0.322) | 0.659 (0.270) | -0.036 (4/10) |
| Former parasite: touch frequency | 0.136 (0.075) | 0.129 (0.063) | +0.006 (6/10) |
| Former parasite: MI context / touch | 0.156 (0.095) | 0.144 (0.079) | +0.012 (6/10) |
| Former parasites with success below 0.3 | 2 of 10 | 1 of 10 | |

Per seed, epochs 1000 to 1500, success of the agent: 0.77 0.77 0.83 0.70 0.78 0.56 0.90 0.81 0.74 0.47 (none below 0.3); of the former parasite: 0.31 0.25 0.93 0.97 0.85 0.10 0.55 0.89 0.86 0.51 (seeds 1 and 5 below 0.3; in the 5-output sweep seed 2 was the one at 0.04, here it recovers to 0.93). All differences are inside the seed spread; the two stuck former parasites against one is not a difference at ten seeds. The measures that show a loss are the agent's success (3 of 10 seeds higher) and its context-policy MI (2 of 10); the recovery of former parasites by the gate holds in the sense that 8 of 10 reach 0.3 or more. Policy statistics (eta2, noise share, counterfactuals) were not computed for this sweep. Figures: `~/tmp/kss/fig567_dof8/figures`; scoring: `~/tmp/kss/score_fig567.py fig567_dof8/runs.db fig567_champion_gate/runs.db`. Demo GIFs of the active agent inside the parasite runs at epoch 1499 (15 GIFs of seed 6, 9 episode types, default selection criteria; seeds 7 to 9 have 0 to 2 qualifying segments and give no demos): `~/tmp/kss/fig567_dof8/demos/gifs`, also in the viewer's figures folder as `demo_*.gif`.

Embedding input layer (branch `embed_actuator`, report 12 section 8): the three SOM blocks replaced by a 15-number code (5 whitened touch components, standardised joints and eye), 120 policy weights, 5 seeds, 150 epochs. At amplitude 0.07 it fails (success 0.06 to 0.07); at 0.3 success is 0.48 to 0.50, touch frequency 0.08 to 0.10 and mean competence 0.48 to 0.51, against 0.68, 0.21 and 0.58 for the SOM-RBF input. The swap cost is 0.135 in one variant (three seeds, interval of the difference to the SOM-RBF actuator [0.004, 0.216]) and 0.004 in the other; the constant-policy cost does not differ from the SOM-RBF actuator. The champion keeps the SOM-RBF encoding.

## 5. What does and does not carry the result

- Noise: executed policies are 88% to 99% noise around their node prototype (`noise_share`), also in the 5-d effective output (0.65 to 0.94). The relevant part of the 375 dims (5-output actuator; the 8-output actuator has 600 and was not measured) is about 15 x 5 = 75 (95% of the centred variance of the visited hidden vectors in 15 dims).
- The match depends on the effective output weakly and nonlinearly (random forest R2 0.02 to 0.19, linear about 0, sign pattern not the form). Report 10 has the details.
- Success depends on the sign pattern and posture of the arm far more than on the policy's content; the champion's policies are close to a 5-d offset plus a small state-dependent part.
- The saturation at amplitude 8 was real (always +-1) but it did not add noise to the policy statistics: removing it leaves the noise space unchanged (section 4.1).

## 6. Recommendation

1. Use the champion of section 3 (with the new default amplitude 0.07) as the baseline of every new sweep. The amplitude-8 champion is the historical reference: `~/tmp/kss/gate_long` (gate and control) and the 5-seed set in `~/tmp/kss/noiseplan2_probe/confirm`; the unsaturated runs are in `~/tmp/kss/satplan_probe`.
2. The baseline of new sweeps is the 8-output actuator with map v4 (the defaults of `params.py`); the 5-output reference is tag `champion-5dof`. `policy_params_amplitude` now defaults to 0.07 in `params.py`; `touch_gate=True` is still a launch option and its default is not changed.
3. Keep `policy_noise_reach_k=15` with `policy_update_baseline=0.05` (`both`) as an experimental variant for studies of content dependence; do not use it where success matters.
4. Report a change as a gain only if the swap or constant-policy cost exceeds the seed spread, stretch or rotation eta2 exceed their permutation p95, and success is not below 0.89 minus its seed spread.

## 7. Limitations and open questions

- Three seeds per group; seed SDs of eta2 and of the constant-policy success reach 0.04 and 0.31. Differences between the non-gate variants are mostly inside the seed spread.
- The probes were run on the gate base only, one setting each; combinations and other values (noise cap, `k`, baseline rate) are untested.
- Shape and rotation dependence is absent everywhere. By the design of the study the environment is not changed to create it (no coupling between affordance and position). Open: whether the aligned representations (touch, proprioception, visual prototypes) separate by shape and rotation at the goal nodes (not measured), whether a morphological change (freeing the two gripper fingers, with a matching rule that rewards the contact pattern) would produce it, and whether the actuator should be redesigned (touch magnitude lost by the sum-to-one normalisation; compact parametrisation of offset plus joint feedback).
- The mean of three seeds hides run-level failures (reach15 seed 2: success 0.47; mod0.5 seed 1: 0.16).
- The 8-output actuator was checked for 150 epochs on five seeds only: no 1000-epoch run, no noise-space measures (effective output dimension, `noise_share`, eta2), and no counterfactual or policy statistics on the 1500-epoch runs; the Figs 5-7 sweep and the former-parasite recovery were run (section 4.3). Matching the 5-output champion on the noise-space measures is assumed, not shown. The amplitude was not retuned for the 8 outputs (median pre-tanh 3.8 at epoch 150).
- Amplitude 0.07 was confirmed on five seeds, with one weak seed (0.519); the Figs 5-7 sweep was repeated at 0.07 only together with the actuator change (section 4.3), so the effect of the amplitude alone on the sweep is not separated.
- The noise-reduction variants screened after `both` (`policy_baseline_soft`, `policy_baseline_norm`, `policy_noise_rel`, `reach_var_profile`, subspace size, competence decay; 27 runs of `~/tmp/kss/noiseplan_probe` and 13 of `noiseplan2_probe`) gave no improvement over the gate: none reached success 0.85 together with a lower policy-space noise share. `both_norm` and `both_soft0.1` raised the prototype output spread (22 against 14) and lowered the executed-prototype distance in weights, with the effective noise share unchanged (0.68 and 0.72 against 0.75). Details are in `noise.md`.

## 8. Reproduction

Conditions JSON (`{name: [KEY=VALUE, ...]}`) as in `~/tmp/kss/gate_long/conditions.json`; launch and score:

```
python src/launch_conditions.py --conditions <conditions.json> --root <dir> --seeds 0,1,2 -p 5 --epochs 1000 --epochs-to-test 50
python src/policy_stats.py --storage <run>/simulations/<run>/storage --epochs 800,850,900,950,999 --counterfactual --out <stats>
```

Probe data and tables: `~/tmp/kss/dof8_probe` (8-output pilot, `sanity.py`, `figures/`), `~/tmp/kss/embed_probe` (embedding ladder, `score_curves.py`, `cf/`, `demos_B15/`), `~/tmp/kss/{gate_long, strict_probe, select_probe, align_probe, outspace_probe, sat_probe, reach_probe, noiseplan_probe, noiseplan2_probe, satplan_probe}` (`stats/`, `table.py`; the unsaturation runs and tables are in `satplan_probe`: `table3.out`, `noise_space_table.py`). Analysis scripts: `~/tmp/kss/signal_analysis/` (`sat_census.py`, `effnoise_all.py`). Demo GIFs: `gate_long/demos/gifs`, `reach_probe/demos/{gifs, gifs_both, green_reach15, green_both}`. Runs and probe data are not in the repository. Code state: the probe parameters of section 3 are committed on `speed_cheap_wins` (`1459676`); the viewer update (`1813fc7`), the noise options (`3782806`) and the new amplitude default are on branch `noise_plan`.

# 14. Two fixes for the single-policy problem: shared base with residuals, centred actuator code

Branch `context2` (on top of `affordance`). Plan: memory `context2`. Files: `src/SMController.py` (shared base), `src/Actuator.py`, `src/center_actuator_code.py`, `src/params.py`, `src/SMAgent.py`, `src/policy_stats.py`, `src/offset_diagnostics.py`, `test_context2.py`. Data: `~/tmp/kss/context2_probe`.

## 1. Problem and hypotheses

Prototypes of the policy map are 87% to 90% one common vector (share of prototype energy in the mean prototype; median cosine of a node to the mean 0.95 to 0.96; 90% of the variation in 4 to 5 of 600 dimensions; random start 0.14) in the original scenario and in the affordance scenario. Swap and constant-policy costs stay small. Hypotheses: (a) the touch block of the RBF code is a near-constant bias input, so the common posture is cheap to store; (b) every event updates the policy map with equal weight through a neighbourhood. This report tests the first two fixes derived from them.

## 2. What was built (all default-off)

- **Shared base plus residual** (`policy_shared_base=True`, `policy_base_eta=0.02`): the controller keeps one vector `b`; the policy map stores residuals; the executed mean policy is `b + residual` (+ noise). `b` moves toward the mean executed policy of the selected events (all events that pass the thresholds); the residual map is updated toward `policy - b`; the matching of executed policies against the map (`spread`) uses `policy - b`; `b` is saved in the controller state and in `weights.npy` (`policy_base`; missing key = zeros). `S1n`: residuals updated by every selected event; `S1`: residuals only from events that beat the node's running baseline (`policy_update_baseline=0.05`, weighted by advantage), `b` still from all events.
- **Centred actuator code** (`grip_center_code='touch'` or `'all'` with `grip_norm_name='data/StoredGripActuatorNorm_v4c'`): the code, whose blocks each sum to a constant, minus a mean vector; touch block centred on its exact zero-touch activation (no contact gives a zero input), posture and eye blocks on the babbling mean (`center_actuator_code.py`). `K1`: touch block only; `K2`: all blocks. `S1K1`: S1 with K1.
- Tests (`test_context2.py`): defaults unchanged; the base shifts the policy-map update; persistence; zero touch gives zero touch-block input; posture and eye block sums about 0 for a typical state.

## 3. Phase 0: how much of a trained output is constant (no training)

`src/offset_diagnostics.py` on the late checkpoints (visited states of the stored evaluation data; eye at its mean). Constant part = the mean code times the weights; energy shares of the pre-tanh output. Then the trained policy is evaluated again with the centring subtracted at run time (36 episodes x 3 evaluation seeds), without retraining.

| Checkpoint | Constant share of output energy | No-contact touch part (share) | Success intact | Success, touch block centred | Success, all blocks centred |
|---|---|---|---|---|---|
| affordance 1000 ep, seeds 0 / 1 / 2 | 0.81 / 0.83 / 0.62 | 0.83 / 0.81 / 0.93 | 0.57 / 0.86 / 0.86 | 0.22 / 0.21 / 0.74 | 0.23 / 0.29 / 0.28 |
| original sweep agent, 1500 ep, seeds 0 / 1 / 2 | 0.89 / 0.97 / 0.73 | 1.04 / 0.77 / 0.76 | 0.83 / 0.73 / 0.77 | 0.39 / 0.51 / 0.75 | 0.13 / 0.19 / 0.34 |
| original fovea-16 long run, 1000 ep | 0.87 | 1.03 | 0.75 | 0.81 | 0.03 |

Most of the output is a constant (62% to 97%), mostly the zero-touch activation of the touch block, and the trained policies rely on it (success falls in 6 of 7 checkpoints when it is removed without retraining). The gate of the plan (a constant under 30% of the output means option 2 is unlikely to matter) is not met, so all conditions were run.

## 4. Phase 1: original scenario (5 seeds, 150 epochs, champion options plus the change)

Runs end at epoch 149; control: the `dof8` pilot (epoch 150). Counterfactual: 36 episodes x 3 evaluation seeds x 5 draws. Mean over seeds:

| Measure | control | S1n | S1 | K1 | K2 | S1K1 |
|---|---|---|---|---|---|---|
| success (epochs 101-150) | 0.68 | 0.60 | 0.58 | 0.78 | 0.36 | 0.78 |
| touch frequency | 0.21 | 0.12 | 0.20 | 0.14 | 0.04 | 0.14 |
| intact success (counterfactual run) | 0.70 | 0.61 | 0.64 | 0.75 | 0.42 | 0.73 |
| swap cost | 0.020 | 0.013 | 0.065 | 0.024 | -0.044 | 0.027 |
| constant-policy cost | 0.19 | 0.02 | 0.08 | 0.03 | -0.02 | 0.02 |
| eta2 of context over its p95 | 1.02 | 0.95 | 1.04 | 0.88 | 0.89 | 0.82 |
| eta2 of stretch / rotation over p95 | 0.82 / 0.81 | 0.86 / 0.81 | 0.76 / 0.74 | 0.80 / 0.86 | 0.89 / 0.96 | 0.82 / 0.88 |
| nodes used | 32 | 29 | 31 | 27 | 22 | 31 |
| common share of the total prototypes | 0.65 | 0.96 | 0.94 | 0.61 | 0.62 | 0.94 |
| common share of the residual prototypes | 0.65 | 0.63 | 0.48 | 0.61 | 0.62 | 0.50 |
| components for 90% of the prototype variation | 4.8 | 4.8 | 4.8 | 4.4 | 5.0 | 5.0 |
| norm of the base over the residual rms norm | 0 | 0.09 | 0.09 | 0 | 0 | 0.09 |

Bootstrap 95% intervals of the difference to the control (5 seeds each): swap cost S1n -0.007 [-0.061, +0.046], S1 +0.045 [-0.041, +0.130], K1 +0.004 [-0.040, +0.050], K2 -0.064 [-0.169, +0.040], S1K1 +0.007 [-0.044, +0.056]; constant-policy cost S1n -0.17 [-0.39, +0.00], S1 -0.12 [-0.34, +0.09], K1 -0.16 [-0.38, +0.02], K2 -0.21 [-0.44, -0.01], S1K1 -0.17 [-0.39, +0.01]; eta2 of context over p95 S1n -0.06, S1 +0.03, K1 -0.14, K2 -0.13, S1K1 -0.20 (all intervals include 0); intact success K2 -0.28 [-0.46, -0.11], touch frequency K2 -0.16 [-0.23, -0.10], the others include 0.

Reading (original scenario): no condition separates the policies by the criterion of the plan (swap or constant-policy cost above the control beyond its interval, or eta2 above p95, without collapse). K2 collapses (success 0.36, touch 0.04). K1 and S1K1 keep success (0.78 and 0.75, 0.73 intact). The constant-policy cost falls to 0.02 to 0.03 in every new condition (control 0.19, driven by two seeds): a constant mean policy is as good as the learned one in all of them. The shared base absorbs the common vector as intended (common share of the residual prototypes 0.48 to 0.63 against 0.65), but the total prototypes are as common as before (0.94 to 0.96, base included) and the number of components carrying the variation does not change (4.4 to 5.0). Centring the code (K1, K2) does not reduce the common share (0.61 to 0.62 against 0.65): the policy map still converges to one dominant posture when the constant bias input is removed, so the constant input is not the cause.

## 5. Phase 1: affordance scenario

Same design (5 seeds, 150 epochs, runs end at epoch 149, `scenario='affordance'`, `obj_shapes=['square','triangle']`, `batch_size=36`, `tests=36`; control = the Phase 1 affordance runs of report 13). Mean over seeds:

| Measure | control | S1n | S1 | K1 | K2 | S1K1 |
|---|---|---|---|---|---|---|
| success (epochs 101-150) | 0.67 | 0.80 | 0.68 | 0.83 | 0.57 | 0.74 |
| touch frequency | 0.23 | 0.29 | 0.26 | 0.26 | 0.10 | 0.24 |
| intact success (counterfactual run) | 0.73 | 0.80 | 0.72 | 0.81 | 0.58 | 0.74 |
| swap cost | 0.019 | 0.077 | 0.014 | 0.050 | 0.015 | -0.033 |
| constant-policy cost | 0.10 | 0.13 | 0.03 | 0.11 | 0.02 | -0.02 |
| eta2 of colour over its p95 | 1.88 | 1.49 | 1.71 | 1.64 | 1.39 | 2.01 |
| eta2 of stretch / rotation over p95 | 0.69 / 0.69 | 0.79 / 0.84 | 0.75 / 0.81 | 0.80 / 0.79 | 0.80 / 0.86 | 0.67 / 0.75 |
| nodes used | 45 | 50 | 53 | 42 | 36 | 46 |
| common share of the total prototypes | 0.66 | 0.96 | 0.93 | 0.67 | 0.62 | 0.92 |
| common share of the residual prototypes | 0.66 | 0.63 | 0.51 | 0.67 | 0.62 | 0.47 |
| components for 90% of the prototype variation | 4.8 | 4.6 | 5.0 | 5.0 | 4.8 | 5.0 |

Bootstrap 95% intervals of the difference to the control (5 seeds each): swap cost S1n +0.059 [-0.027, +0.167], S1 -0.005 [-0.041, +0.031], K1 **+0.031 [+0.001, +0.064]**, K2 -0.004 [-0.044, +0.037], S1K1 -0.051 [-0.086, -0.016]; constant-policy cost S1n +0.030 [-0.087, +0.167], S1 -0.070 [-0.146, +0.013], K1 +0.007 [-0.059, +0.080], K2 -0.080 [-0.161, +0.009], S1K1 -0.119 [-0.189, -0.044]; eta2 of colour over p95 S1n -0.40 [-0.74, +0.02], S1 -0.17, K1 -0.24, K2 -0.50 [-0.89, -0.01], S1K1 +0.13 [-0.42, +0.72]; touch frequency K2 -0.13 [-0.21, -0.05], the others include 0; intact success all intervals include 0.

Reading (affordance scenario): the picture is the same as in the original scenario. K1 is the only condition whose swap cost lies above the control with an interval that excludes 0, but by a tiny margin (0.05 against 0.019, lower bound +0.001) and with five conditions and two cost measures compared, one such value is expected by chance; its constant-policy cost and eta2 of colour do not move. S1K1 has a lower swap and constant-policy cost than the control. K2 lowers the colour eta2 and collapses touch (0.10). The shared base moves the common vector into `b` (residual common share 0.47 to 0.63) but the total prototypes stay as common (0.92 to 0.96) and the number of components that carry the variation stays 4.6 to 5.0. The colour structure that the affordance scenario produced (eta2 1.9 times its p95 in the control) is not increased by either fix.

## 5.1 Verdict

Neither fix separates the policies by the criterion of the plan, in either scenario: no condition raises the swap or constant-policy cost beyond the control with a meaningful margin, and eta2 does not rise. The centred code is safe in its touch-only form (K1 keeps success, 0.75 original and 0.81 affordance intact) and unsafe when all blocks are centred (K2 collapses in the original scenario and lowers touch to 0.10 in the affordance one). The shared base absorbs the common vector as designed but changes nothing for the policies. Together with the observation that centring does not reduce the common share (0.61 to 0.67 against 0.65 to 0.66), this says that the common posture is not an artefact of the constant bias input or of the shared update of the map: the map keeps converging to one dominant posture with the bias removed or moved out of the nodes. What remains untested from the plan's hypotheses is the success rule (one touch-match rise is enough, so any contact posture is rewarded), which the earlier settings screen only changed by `cum_match_stop_th` (collapse) and which would need an intrinsic criterion that rewards a specific touch outcome.

## 6. Phase 2

Not started: no condition passed (section 5.1).

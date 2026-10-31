# Main algorithm

Intrinsically motivated learning of a sensorimotor map by a 3-DoF arm + 2-DoF gripper that interacts with one object (stretch x rotation x colour context). Current branch variant ("unsupervised_touch"): the match is computed on touch and proprioception only (`modalities_weights=(0,1,1,0)`), goals are picked from the visual map.

## Components

- Four SMSTMs on a 10x10 grid (`internal_size=100`): visual (300 inputs), somatosensory (40), proprioceptive (8 joints since the 8-DoF actuator, 5 before), policy (`grip_hidden*grip_output = 75*8 = 600`, 75*5 = 375 before). Each prototype of the policy map is a full parameter vector of the linear grip controller.
- Predictor `SMPredictKDE`: a vector of 100 expected competences (running mean of achieved max match per goal prototype).
- Agent: `action = tanh-clipped( [touch, joints[:5], eye_pos] @ fixed_map @ policy_params )`, with fixed offsets (`policy_base_arm`, `policy_base`); the policy chosen for an episode segment is written into the actuator every `action_steps` (5) steps.
- Env: Box2D, 200 steps/episode (`stime`), batch of 18 episodes per epoch (`batch_size`), 3 colour contexts x 6 object configurations.

## Per-epoch loop (`Main.train`)

1. Build `batch_size` envs (seed = base seed + episode + epoch), contexts cycle 1,2,3, object configs cycle in groups of 3.
2. `run_episodes` (below) for 200 steps, vectorised over episodes by Python loops.
3. Derive `policy_ended` (last step of each policy segment) from `policy_changed`.
4. Competence: `comp_grid = tanh(decay * predictor.spread(goal_grid))`, global competence = its mean. Local competence = `tanh(local_decay * predicted competence of the chosen goal)`.
   - `global_incompetence = 1 - global`; `local_incompetence = global_incompetence * (1 - local)`.
5. Modulate learning with local incompetence `x`: `sigma = base_internal_sigma + (internal_sigma - base_internal_sigma) * x`, `lr = base_lr + (max_lr - base_lr) * x`. High incompetence leads to wide neighbourhood and strong learning.
6. `controller.update(...)` (below).
7. Log (wandb optional), append rows to `internal_trajectory_data`.
8. Every `epochs_to_test` epochs: `diagnose` = save weights and `main.dump`, 18 evaluation episodes with zero policy noise, maps (visual, competence, somatosensory, proprio), optional gif demos.
9. After the last epoch write `internal_trajectory_data.csv`. A `TimeLimitsException` stops on wall-clock; rerun resumes from `main.dump.npy`.

## Inside one batch (`Main.run_episodes`)

Time loop `t = 1..stime`:

1. For each live episode: `agent.updatePolicy(batch_a[ep, t])`; step the env (`action_steps` constant action; motor noise `motor_noise` added during the first `action_steps` steps after `drop_first_n_steps=25`); stop the episode if the object leaves the task space.
2. Store visual/touch/joint observations in the batch buffers.
3. Every `action_steps` steps: compute representations of the last 5 steps through each SMSTM (`spread`): winner prototype `p` (point on the grid) and a Gaussian representation `r` with width `base_internal_sigma`. The goal `g` is the policy-map point currently pursued.
4. After the warm-up (25 steps) compute match between each modality and the goal point: `match_mod = exp(-(d/match_sigma)^2)`, `d` = grid distance of the modality's winner to the goal's winner; `match = weighted mean` over modalities (only touch and proprio here).
5. Event detection per step (after warm-up + `policy_selection_steps`=25):
   - `mmask` = the touch-match increased vs the previous step AND the global match exceeds the policy's running `max_match` by at least `match_incr_th`. `max_match` is updated on these events.
   - `cum_match` counts events within the current policy; reset on policy change.
6. Policy selection: when `cum_match >= cum_match_stop_th` (or at the first selection time) the episode picks a new goal and policy:
   - Take the visual representation of the last 25 steps, weight by predicted competence, project to the policy map to get goal point `g_p`.
   - Policy = prototype of the policy map at `g_p` (sigma = `base_internal_sigma` to get an exact prototype), plus Gaussian noise with std `base_policy_noise + (max_policy_noise - base_policy_noise) * local_incompetence`.
   - Write policy, goal, competence into the buffers from step `t` on; mark `policy_changed[t]`.

## Controller update (`SMController.update`)

- Selected samples: steps flagged by `mmask` ("effect" steps, touch onset). For these, update touch, proprio and policy SMs with weights `goals * local_lr` and per-sample `local_sigma`. The visual SM is updated on the 25 steps that preceded each policy change (the "condition" window).
- SMSTM step: pick winner by squared distance, build radial (Gaussian) neighbourhood around the winner, loss = mean(radial * dist^2 * modulation), one Adam step.
- Re-entrant coupling (`external_radial_prop`): each map's activation is mixed with an external radial; mechanism exists but `update_reentrant_connections` is not called in the current code.
- Predictor update at the end of each policy segment with `max_match > 0`: `competence[goal] += predict_lr * (max_match - competence[goal])`.

## Evaluation (`evaluation_episodes`)

Deterministic episodes from a fixed set of 18 types (6 objects x 3 contexts), zero policy noise, new copy of the controller. Outputs: goal counts, per-timestep trajectories (posture d1..d5, touch s0..s39, winners, match, goal id), success rate (episodes with at least 2 policies), touch frequency, correlation of match increase within goal, mutual information between context, policy cluster, touch region and gripper cluster (KMeans k=10).

## Key parameters

`internal_sigma` (max neighbourhood), `base_internal_sigma`, `max_lr`/`base_lr`, `decay`/`local_decay` (competence sharpness), `match_sigma`, `match_incr_th`, `cum_match_stop_th`, `max_policy_noise`, `policy_selection_steps`, `action_steps`, `epochs`, `epochs_to_test`.

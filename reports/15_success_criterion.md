# 15. The success criterion: goal-reaching diagnostic and the radius test

Branch `context3` (on top of `context2`). Plan: memory `context3`. Files: `src/params.py` (`event_radius`, `event_radius_final`, `event_radius_warmup`), `src/SMController.py` (`reach_mask`, `scheduled_event_radius`), `src/SMMain.py` (event mask), `src/policy_stats.py` (reach columns, `--old-rule`, `--kinds`), `test_context3.py`. Data: `~/tmp/kss/context3_probe` (Part A in `phaseA`, Part B runs and counterfactuals in `pod/ctx3`, `ladder_affordance`, `cf_affordance`). Part B ran on the RunPod pod (wide batch) and on the local machine (R3to1, affordance), see memory `runpod-usage`.

## 1. The question

The prototypes of the policy map are 87% to 90% one common vector and a constant mean policy is almost as good as the learned one (reports 11 to 14). The event rule, which defines success, is weak: an event is a rise of the touch match over the previous step and of the global match over the running maximum by 0.001 (`SMMain.run_episodes`); one event ends the policy; episode success is at least two policy changes. The match is `exp(-d^2/sigma^2)` of the grid distance between the winning node of the state and the goal node, sigma 1.5, so a rise of 0.001 is reached from about 3.9 cells away. Hypothesis: the rule rewards any contact near the goal region, so one posture serves many goals.

## 2. Part A: offline goal-reaching diagnostic (no training)

`policy_stats --counterfactual` extended with, per episode, `reach_anyR` (at least one step in contact with both the touch and the proprio winners within R cells of the goal node), `reach_shareR`, number of policy segments and the touch and proprio distances at the end of each segment. Mean over runs, learned policy against the constant mean policy (36 episodes x 3 evaluation seeds per run):

| Checkpoints | Learned reach r=1 | Constant mean r=1 | Gap r=1 | Gap r=2 | Segments per episode | Touch / proprio distance at segment end |
|---|---|---|---|---|---|---|
| original, 150 epochs (5 runs) | 0.22 | 0.15 | +0.07 | +0.11 | 4.1 | 5.1 / 5.2 |
| original, 1500 epochs (3 runs) | 0.48 | 0.38 | +0.10 | +0.11 | 4.9 | 4.9 / 3.9 |
| affordance, 150 epochs (5 runs) | 0.26 | 0.18 | +0.08 | +0.08 | 4.7 | 5.3 / 5.0 |
| affordance, 1000 epochs (3 runs) | 0.42 | 0.27 | +0.15 | +0.16 | 5.1 | 5.4 / 4.9 |
| fovea 16, 1000 epochs (1 run) | 0.39 | 0.24 | +0.15 | +0.06 | 4.2 | 4.3 / 3.5 |

The learned policies reach the goal region more often than the constant mean policy, by 6 to 16 points, and the policies end on average 4 to 5 cells from the goal node. Even the constant policy reaches radius 1 in 15% to 38% of the episodes. So the current rule selects goal-specific policies only a little, as hypothesised.

## 3. Part B: the radius test

Rule (`event_radius=R`): an event additionally requires contact and both the touch and the proprio winners within R grid cells of the goal node. `R3to1`: R shrinks from 3 to 1 as the global competence goes from 0 to 1. A radius rule from epoch 0 starves learning (no event, no map update: R2 and R1 had no event at all in 40 epochs, R3 only at epoch 39), so all conditions use `event_radius_warmup=50`: the original rule for the first 50 epochs, the radius rule for the remaining 100. This is a deviation from the plan. 5 seeds, 150 epochs, champion options; original scenario (batch 18) and affordance scenario (`scenario='affordance'`, `obj_shapes=['square','triangle']`, `batch_size=36`, `tests=36`). Controls: `dof8_probe` and the affordance Phase 1 runs of report 13.

Because the radius changes the meaning of success, every agent is also re-evaluated with the original rule (`--old-rule`): `intact_old_rule`, swap and constant-policy costs under the old rule, and the reach metrics. Mean over 5 seeds:

| Original scenario | control | R3 | R2 | R1 | R3to1 |
|---|---|---|---|---|---|
| success under the training rule (epochs 101-150) | 0.68 | 0.23 | 0.19 | 0.02 | 0.11 |
| success under the old rule (re-evaluated) | 0.70 | 0.62 | 0.58 | 0.58 | 0.51 |
| touch frequency | 0.21 | 0.14 | 0.13 | 0.18 | 0.16 |
| policy segments per episode (events + 1) | 4.1 | 1.9 | 1.4 | 1.1 | 1.4 |
| nodes used | 32 | 9 | 8 | 9 | 9 |
| swap cost, old rule | 0.020 | 0.084 | 0.111 | 0.018 | 0.026 |
| constant-policy cost, old rule | 0.18 | 0.17 | 0.18 | 0.19 | 0.07 |
| reach r=1, learned | 0.22 | 0.18 | 0.13 | 0.02 | 0.10 |
| reach r=1, gap to constant policy | 0.070 | 0.017 | 0.074 | 0.002 | 0.015 |
| reach r=2, gap to constant policy | 0.109 | 0.052 | 0.128 | 0.015 | 0.028 |
| common share of the prototypes | 0.65 | 0.49 | 0.47 | 0.46 | 0.48 |

| Affordance scenario | control | R3 | R2 | R1 | R3to1 |
|---|---|---|---|---|---|
| success under the training rule | 0.67 | 0.36 | 0.13 | 0.05 | 0.24 |
| success under the old rule | 0.73 | 0.73 | 0.64 | 0.68 | 0.76 |
| touch frequency | 0.23 | 0.25 | 0.25 | 0.21 | 0.29 |
| policy segments per episode | 4.7 | 1.7 | 1.2 | 1.1 | 1.5 |
| nodes used | 45 | 22 | 24 | 21 | 22 |
| swap cost, old rule | 0.019 | 0.020 | -0.013 | 0.057 | 0.037 |
| constant-policy cost, old rule | 0.08 | 0.17 | 0.08 | 0.24 | 0.14 |
| reach r=1, learned | 0.26 | 0.18 | 0.08 | 0.07 | 0.15 |
| reach r=1, gap to constant policy | 0.083 | 0.085 | 0.050 | 0.043 | 0.096 |
| reach r=2, gap to constant policy | 0.078 | 0.130 | 0.081 | 0.069 | 0.150 |
| common share of the prototypes | 0.66 | 0.55 | 0.52 | 0.50 | 0.56 |
| eta2 of colour over its p95 | 1.88 | 1.43 | 1.04 | 0.99 | 1.23 |

Bootstrap 95% intervals of the difference to the control (5 seeds each), main measures: success under the old rule: all intervals include 0 except `R3to1` in the original scenario (-0.19 [-0.39, -0.01]); segments per episode: all lower by 2.2 to 3.6 with intervals far from 0; reach r=1 of the learned policy: lower in R1 and R2 of the affordance scenario (about -0.19 [-0.31, -0.06]) and R1 of the original scenario (-0.20 [-0.34, -0.07]); reach gap to the constant policy: no condition above the control (R1 original -0.07 [-0.12, -0.01]); swap cost under the old rule: all include 0 (R2 original +0.09 [-0.03, +0.22]); constant-policy cost under the old rule: affordance R1 +0.16 [+0.01, +0.33] and R3 +0.09 [0.00, +0.18], the others include 0.

## 4. Reading

- The radius rule works as a filter and does what it says (events become rare: 1.1 to 1.9 policy segments per episode against 4.1 to 4.7) but the learned policies do not become more goal-specific. The gap in goal-reaching between the learned and the constant mean policy does not rise in any condition (original: 0.00 to 0.07 against 0.07; affordance: 0.04 to 0.10 against 0.08), the reach rate of the learned policy falls, and the nodes used fall from 32 (45) to about 9 (22): most goals receive no updates, so the policy map is trained on a small number of goals. The common share of the prototypes is lower (0.46 to 0.56), but this is mostly an effect of the unused nodes, which stay random.
- Success under the old rule, the rule-independent measure, is similar to or a little lower than the control (0.51 to 0.62 against 0.70 in the original scenario; 0.64 to 0.76 against 0.73 in the affordance one), so there is no collapse of behaviour, but also no gain; the success under the training rule is low (0.02 to 0.36) because events are rare.
- Swap costs stay small (0.02 to 0.11) and the constant-policy costs under the old rule are unchanged except in two affordance conditions (R1, R3) where the constant policy does worse by 0.09 to 0.16 with intervals that barely exclude 0; with 4 conditions x about 9 measures x 2 scenarios compared, a few such values are expected by chance, and the reach gap, which measures the same thing more directly, does not move.
- Part A and Part B together: under the current rule goals are reached only slightly better by learned than by constant policies, and making the rule stricter by a radius, with a warm-up, does not turn that into goal-specific policies in 150 epochs; it only removes updates. Verdict: negative in the form tested.

## 5. Not tested and caveats

- Only 100 epochs under the radius rule (50 of warm-up with the old rule); effects that need longer training are not excluded. A longer run of R3 or R3to1 would be a few long runs (local machine by the pod rule).
- Radius 0 (exact node) was left out (starvation).
- A contrastive match (distance in input space or relative fit across goals) is a change of the match itself, not tested.
- The unused-node effect makes the prototype common share hard to compare between conditions; node usage is reported next to it.

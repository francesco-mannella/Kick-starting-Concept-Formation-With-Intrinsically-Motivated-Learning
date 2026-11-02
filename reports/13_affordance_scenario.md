# 13. The affordance scenario: colour defines the physical response

Branch `affordance` (on top of `settings`, `fovea`, `embed_actuator`). Files: `tools/box2dsim/box2dsim/models/add_affordances.py` (world generator), `Simulator.py` (ratchets, prismatic spring), `Box2DSim_env.py` (`use_scenario`, shape library), `src/SMEnv.py`, `src/SMMain.py`, `src/params.py`, `test_affordances.py`, `src/affordance_screen.py`. Data: `~/tmp/kss/affordance_probe`.

## 1. Motivation

In the original world one policy serves all three objects (swap cost 0.02, constant-policy cost about 0.19 but 0 in three of five seeds, eta2 of context at the permutation null; reports 11, 12 and section 4.4 of report 11). The three kinds differ only in shape, size and static against dynamic. The scenario of this report makes the colour define a different physical response and lets every colour take every shape.

## 2. The scenario

- **red**: fixed in place, slides along its main axis like a button (travel +-2 units, either direction) and a spring returns it to the rest position when no force is applied.
- **green**: fixed in place, rotates around its centre in the clockwise direction only (one-way ratchet, no limit, no return).
- **blue**: the same, anticlockwise only.
- shapes (independent of colour): square 1.9 and triangle 7.8 x 7.7, stretch {1, 2}, rotation {0, 45, 90} as before: 3 colours x 2 shapes x 2 stretches x 3 rotations = 36 episode types. The main axis of the button is the long axis of the shape (the rotated x axis).
- selected with `scenario='affordance'` (default `'original'`, bit-identical to the old behaviour: tested), shapes with `obj_shapes=['square','triangle']`; the 36 types need `batch_size=36` and `tests=36`.

Affordance is tied to colour only; shape and position stay independent and the success rule is unchanged (touch-match events), so no engineered task pressure in the sense of memory `no-engineered-task-pressure`.

## 3. Physics (how it is built)

- Worlds `affordance_{blue,red,green}.json` (from `noobject.json`): an object body plus a static anchor body that collides with nothing (white, invisible); the anchor is moved to the object position at every reset (`randomize_objects`).
- Rotors: a revolute joint at the object centre (the bounding-box centre of the shape), motor off. `Simulator.step` makes it a ratchet: for a clockwise rotor the upper angle limit follows the minimum angle reached (`min(limit, angle)` every step), for an anticlockwise one the lower limit follows the maximum. The rotor can therefore only keep turning in its direction, jams in the other and stays where it was left.
- Button: a prismatic joint along the main axis with limits +-2 and a force-capped motor driven by the simulator's PID towards 0 (a constant-force spring of 40); the PID now reads `translation` for prismatic joints. Return to rest takes about 1 to 2 s (the test requires it within 240 steps).
- `SMEnv.track_object_response` records, for diagnostics only, the button travel or rotor angle of each episode (column `obj_response` of the trajectories); it is never used by the agent or the success rule.

Tests (`test_affordances.py`): original scenario unchanged; the button moves under a force along its axis in both directions, never beyond +-2, not sideways, returns to rest, ignores a force across its axis; rotors turn only their way, do not turn the wrong way, keep the angle once released, do not return, centre fixed; every colour builds in every shape, stretch and rotation, with the anchor at the object position and the button axis along the long axis; a reloaded checkpoint applies the scenario from its parameters.

## 4. Phase 0: random policies (no training)

`src/affordance_screen.py`: 300 random constant policies x 6 draws x the three colours, random shape, stretch and rotation, 100 steps; 5400 episodes per scenario (`~/tmp/kss/affordance_probe/phase0`).

| | original | affordance |
|---|---|---|
| contact rate blue / red / green | 0.28 / 0.43 / 0.28 | 0.37 / 0.36 / 0.38 |
| per-policy contact correlation between colours | 0.73 | 0.85 |
| colour readable from the touch trace (random forest, cross-validated by policy; chance in brackets) | 0.76 (0.44), but colour and shape are confounded | 0.46 (0.35) |
| shape readable from the touch trace | n/a | 0.78 (0.60) |

Responses under random policies: red pressed beyond 0.5 units in 44% of episodes (and back at rest within the episode in 9%, because the arm keeps pressing), blue turned in 34% and green in 14%; no rotor turned the wrong way. The gate of the plan (every colour responds) is met. The lower response of green is geometric: the arm pushes mostly from the side that turns a pivot anticlockwise.

## 5. Phase 1: training screen (5 seeds, 150 epochs, champion options, `scenario='affordance'`, `batch_size=36`)

Runs end at epoch 149. Control: the `dof8` pilot of the original scenario (epoch 150), so the comparison is indirect (different task). Counterfactual: 36 episodes x 3 evaluation seeds x 5 draws.

| Measure (mean over 5 seeds, SD) | control | affordance |
|---|---|---|
| success (epochs 101-150) | 0.68 (0.18) | 0.67 (0.21) |
| touch frequency | 0.21 (0.09) | 0.23 (0.10) |
| mean competence | 0.58 | 0.72 |
| intact success (counterfactual run) | 0.70 | 0.73 (0.20) |
| swap cost | 0.020 | 0.019 |
| constant-policy cost | 0.19 | 0.10 |
| eta2 of context (colour) over its permutation p95 | 1.02 (0.23) | 1.88 (0.47) |
| eta2 of shape over p95 | n/a | 1.00 (0.19) |
| eta2 of stretch / rotation over p95 | 0.82 / 0.81 | 0.69 / 0.69 |
| nodes used | 32 | 45 |
| share of evaluation episodes in which the object responded (red pressed > 0.5, blue turned > 0.3, green turned > 0.3) | n/a | 0.98 / 0.96 / 0.77 |
| intact success by colour blue / red / green | 0.87 / 0.59 / 0.63 | 0.72 / 0.72 / 0.76 |
| touch frequency by colour blue / red / green | 0.52 / 0.07 / 0.19 | 0.31 / 0.30 / 0.35 |
| intact success by shape square / triangle | n/a | 0.84 / 0.63 |

Bootstrap 95% intervals of the difference to the control (5 seeds each): intact success +0.03 [-0.16, +0.19], touch frequency +0.03 [-0.08, +0.13], swap cost -0.001 [-0.044, +0.041], constant-policy cost -0.09 [-0.32, +0.10], eta2 of context over p95 **+0.87 [+0.43, +1.23]**, stretch -0.13 [-0.28, +0.02], rotation -0.12 [-0.26, +0.02].

Per seed (affordance): eta2 of colour over p95 is 2.96, 1.21, 2.88, 3.04 and 2.08; intact success 0.87, 0.39, 0.82, 0.74 and 0.83 (seed 1 is the weak seed, touch 0.11). The objects are operated in nearly every evaluation episode by the trained agents (98% of red, 96% of blue, 77% of green), against 44%, 34% and 14% under random policies.

Reading: by the criterion of the plan the scenario separates affordances, through the eta2 criterion: the policies of the three colours differ by more than a permutation null (eta2 of colour 2.4 times its p95, all five seeds above 1.2) and significantly more than in the original world (+0.87), without collapse (success 0.73, touch 0.23). It does not separate them through the swap and constant-policy costs: swapping policies between episodes still costs 0.02 and the constant mean policy costs 0.10 (less than in the original world, whose value was dominated by two seeds), so the colour-specific part of the policies does not yet matter for the success rule, which counts one touch-match rise. The agents do operate the objects (the response diagnostic), the success rule just does not need it to be specific. The shape of the object does not shape the policies (eta2 of shape at its p95), nor do stretch and rotation.

## 6. Phase 2: 1000 epochs, demos and the Figs 5-7 sweep

### 6.1 1000-epoch runs (seeds 0 to 2, `~/tmp/kss/affordance_probe/ladder1000`)

Late window = epochs 800 to 999; counterfactual and statistics at epochs 800, 900 and 999 (`stats1000`). The reference is the original-scenario Figs 5-7 sweep (8-output champion, `fig567_dof8`, parasite-mode agent of seeds 0 to 2, epochs 1000 to 1500, statistics at epoch 1499), so the tasks and run types differ.

| Measure (mean of 3 seeds) | affordance, 1000 epochs | original, 1500 epochs |
|---|---|---|
| success (late window) | 0.70 (seeds 0.47, 0.76, 0.87) | 0.79 |
| touch frequency | 0.25 | 0.22 |
| alignment (`grid_comp_mean`) | 0.64 | 0.70 |
| MI context / policy | 0.80 | 0.96 |
| MI context / touch | 0.04 | 0.24 |
| intact success (counterfactual run) | 0.76 | 0.77 |
| swap cost | 0.031 (seeds -0.09, +0.06, +0.12) | -0.084 |
| constant-policy cost | 0.099 (0.15, 0.03, 0.12) | 0.019 |
| eta2 of colour over its p95 | **6.3** (4.5, 5.7, 8.5) | 1.85 (1.2, 2.2, 2.2) |
| eta2 of stretch / rotation over p95 | 0.58 / 0.80 | 0.54 / 0.50 |
| nodes used | 57 | 26 |
| share of evaluation episodes in which the object responded (red / blue / green) | 1.00 / 0.93 / 0.71 | n/a |
| intact success by colour blue / red / green | 0.71 / 0.78 / 0.78 | 0.86 / 0.79 / 0.65 |
| touch frequency by colour blue / red / green | 0.30 / 0.30 / 0.40 | 0.66 / 0.08 / 0.15 |

Reading: after 1000 epochs the colour-specific structure of the policies is much larger than in the original world (eta2 of colour 6.3 times its permutation p95 against 1.9; 57 nodes used against 26; all three seeds above 4.5). The agent touches all three colours about equally often (0.30 to 0.40) where in the original world one kind took most of the touches (0.66 against 0.08 and 0.15), and it operates the objects in nearly every episode. The swap cost is 0.03 on average and varies in sign between seeds, the constant-policy cost is 0.10 (0.03 to 0.15): the policies differ by colour, but replacing them by the mean policy or exchanging them costs only a little success, so the success rule (touch-match events) still does not require the colour-specific response. The aligned representations carry less colour information in touch (MI context / touch 0.04 against 0.24) and in policy (0.80 against 0.96) than in the original world; with three seeds and a different run type for the reference these two values are indications only. Success and alignment are slightly lower than the reference (0.70 against 0.79, 0.64 against 0.70), inside the seed spread (success per seed from 0.47 to 0.87).

### 6.2 Demo GIFs

15 GIFs of the best-ranked seed (seed 1) at epoch 999, 12 episode types and 12 goal nodes, default selection criteria (`~/tmp/kss/affordance_probe/demos1000/gifs`, also in the viewer figures folder of the Phase 1 database as `demo_epoch999_*.gif`; names give colour (blue-rotor, red-button, green-rotor), shape, stretch and rotation).

### 6.3 Figs 5-7 sweep

Launched because Phase 1 meets the eta2 criterion: 5 seeds (10 runs) instead of 10 to fit the time, 1500 epochs, former parasites from the parasite weights at epoch 1000, batch size 36, 2026-11-02 00:01 to 09:33 (`~/tmp/kss/affordance_probe/fig567`; figures in `fig567/figures`; scoring `fig567_score.out`, `~/tmp/kss/score_fig567.py`). The reference is the original-scenario sweep with the same actuator (`fig567_dof8`, 10 seeds, section 4.4 of report 11); the paired difference uses the 5 seeds present in both. Mean over seeds (SD), epochs 1000 to 1500:

| Series, measure | affordance | original | paired difference (affordance higher in) |
|---|---|---|---|
| Agent: alignment | 0.640 (0.023) | 0.652 (0.093) | -0.068 (2/5) |
| Agent: success | 0.784 (0.129) | 0.734 (0.129) | +0.014 (3/5) |
| Agent: MI context / policy | 0.845 (0.087) | 0.902 (0.091) | -0.078 (1/5) |
| Agent: MI context / touch | 0.048 (0.013) | 0.178 (0.065) | -0.169 (0/5) |
| Agent: touch frequency | 0.273 (0.048) | 0.175 (0.066) | +0.061 (4/5) |
| Parasite: alignment | 0.650 (0.018) | 0.656 (0.071) | -0.056 (0/5) |
| Parasite: success | 0.317 (0.259) | 0.368 (0.247) | -0.063 (2/5) |
| Former parasite: alignment | 0.662 (0.019) | 0.628 (0.067) | +0.014 (3/5) |
| Former parasite: success | **0.820 (0.051)** | 0.623 (0.322) | **+0.157 (3/5)** |
| Former parasite: touch frequency | **0.283 (0.065)** | 0.136 (0.075) | **+0.153 (5/5)** |
| Former parasite: MI context / policy | 0.871 (0.069) | 0.891 (0.063) | -0.022 (1/5) |
| Former parasite: MI context / touch | 0.040 (0.018) | 0.156 (0.095) | -0.107 (1/5) |
| Former parasites with success below 0.3 | 0 of 5 | 2 of 10 | |

Per seed, success of the former parasite, epochs 1000 to 1500: 0.81, 0.76, 0.82, 0.81 and 0.90 (original: 0.31, 0.25, 0.93, 0.97, 0.85, 0.10, 0.55, 0.89, 0.86, 0.51); of the agent: 0.56, 0.81, 0.83, 0.89 and 0.84 (none below 0.3).

Reading: the recovery of former parasites is better and steadier in the affordance scenario (success 0.82 with a seed SD of 0.05 against 0.62 with 0.32, no stuck seed against two, touch frequency doubled, higher in all five paired seeds), while the agent and the alignment are as in the original world and the agent's success is the same (0.78 against 0.73). The context information in touch is lower (MI context / touch about 0.04 against 0.16 to 0.18) and the information in the policy slightly lower (0.85 to 0.87 against 0.89 to 0.90): in this world the touch map separates the colours less, although the policies are more colour specific (section 6.1). With five seeds against ten, and the success of the former parasite higher in only 3 of 5 paired seeds (the original has large seed spread), the improvement of recovery is suggestive and not yet established; the touch frequency of the former parasite is the clearest difference (5 of 5). Demo GIFs of the active agent in the parasite run of seed 4 at epoch 1499 (15 GIFs, 13 episode types) are in the viewer figures folder as `demo_agent_epoch1499_*.gif`.

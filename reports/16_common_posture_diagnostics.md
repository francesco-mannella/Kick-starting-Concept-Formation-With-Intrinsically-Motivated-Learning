# 16. Common posture: time course against sigma and decomposition of the policy update

Branch `context4` (on top of `context3`). Plan: memory `context4`. Files: `src/common_share_trajectory.py` (Check 1), `src/update_decomposition.py`, `src/SMController.py` / `src/stm.py` / `src/SMMain.py` (default-off `log_update_decomposition`, metrics `upd_*`), `test_context4.py`. Data: `~/tmp/kss/context4_probe` (`check1` trajectory and summary csv, figure; `check2` 33 resumed runs, `score_check2.csv`).

## 1. The question

Late in training the policy-map prototypes are 87% to 94% one common vector. Either the neighbourhood update pulls all nodes toward the same recent policies (spillover), or each node converges on the same posture because the executed policies already share it (independent convergence).

## 2. Check 1: common share against sigma (no training)

Common share, node cosine to the mean, components for 90% of the variance and the norms of the common and residual parts at every saved checkpoint (every 50 epochs), joined with the mean sigma of the preceding 50 epochs. Sets: affordance 1000 epochs (3 runs, from scratch), fovea 16 (1 run, from scratch), original and affordance Figs 5-7 sweep formers (10 and 5 runs, continuations of earlier training: their epoch 0 is already at a common share of 0.85 to 0.89, so they only show late behaviour).

From scratch (affordance 1000, 3 runs; fovea 16 is the same):

| epoch | 0 | 50 | 150 | 300 | 500 | 700 | 999 |
|---|---|---|---|---|---|---|---|
| common share | 0.14 | 0.52 | 0.68 | 0.76 | 0.82 | 0.86 | 0.90 |
| mean sigma | 10.0 | 7.9 | 3.2 | 1.9 | 1.5 | 1.5 | 1.3 |
| residual norm | 3.3 | 17.9 | 16.2 | 14.9 | 13.3 | 11.9 | 10.5 |

Continuations (original, 10 runs): common share 0.87 (epoch 300) to 0.92 (1499), sigma 1.8 to 1.4, residual norm 13.5 to 10.7.

- The share is built in the first 50 epochs, while sigma is still 8 to 10 (neighbourhood covers most of the 10 x 10 map). After that it keeps rising (0.52 to 0.90) after sigma has reached its plateau of about 1.5 (epoch 400 on), and in the continuations it rises 0.87 to 0.92 with sigma flat. The late slope of the common share is positive in all runs and does not follow sigma (slope of sigma about -0.0004 per epoch, slope of the share +0.0001 to +0.0002).
- Cross-run regression of the common share (checkpoints from epoch 100): sigma alone R2 0.35, log epoch alone R2 0.34, both 0.48. Sigma and epoch are confounded; sigma does not explain the late rise.
- Reading: the first rapid build-up is compatible with spillover at large sigma, the slow late rise is not tied to sigma.

## 3. Check 2: where the update weight of a node comes from

A node receives from each selected event the weight `radial(winner, node; sigma) * goal activation(node)` times the learning rate (`stm.ema_step`, `phi`). Logged per epoch over the selected events (`decompose_update`): `upd_own_share` (share of the total weight received at the event's own goal node), `upd_weighted_distance` (weight-averaged grid distance of receiving nodes from the goal node), `upd_spread_nodes` (nodes with at least 10% of the event's maximum weight), and the common share of the executed event policies (mean per goal node, nodes with at least two events; and raw). Resumed 3 epochs from the saved dumps (`main.dump.npy`) of 3 affordance 1000-epoch runs (epochs 50, 150, 300, 600, 999) and 3 original sweep formers (50 to 1499), 33 resumes; mean of the 3 epochs and the runs (original former 1 had only 1 to 3 goal nodes at epochs 150-600, a stuck run, excluded from those means):

| set | epoch | own share | weighted distance (cells) | nodes spread | goal nodes | event policies common (per goal) | prototypes common |
|---|---|---|---|---|---|---|---|
| affordance | 50 | 0.07 | 2.10 | 29.7 | 21 | 0.11 | 0.52 |
| affordance | 150 | 0.10 | 1.86 | 10.9 | 26 | 0.13 | 0.68 |
| affordance | 300 | 0.15 | 1.66 | 6.3 | 27 | 0.25 | 0.76 |
| affordance | 600 | 0.24 | 1.38 | 4.0 | 28 | 0.46 | 0.84 |
| affordance | 999 | 0.24 | 1.27 | 3.7 | 28 | 0.60 | 0.90 |
| original | 150 | 0.23 | 1.16 | 8.6 | 10 | 0.54 | 0.81 |
| original | 600 | 0.17 | 1.41 | 8.4 | 15 | 0.49 | 0.87 |
| original | 1000 | 0.25 | 1.21 | 8.9 | 10 | 0.63 | 0.89 |
| original | 1499 | 0.21 | 1.21 | 6.8 | 19 | 0.58 | 0.92 |

- Late in training only 21% to 25% of the update weight of an event lands on its own goal node, the rest on neighbours (weighted distance 1.2 to 1.3 cells, 4 to 9 nodes receive at least 10% of the maximum): the update is dominated by neighbour weight, the below-0.3 criterion of the plan.
- The executed event policies, averaged per goal node, are also already common (0.58 to 0.63 late; raw 0.37 to 0.47), but less than the prototypes (0.89 to 0.92). The prototypes are more common than the per-goal event means by about 0.3 in share, but this gap is confounded: means over 2 to 3 events still contain exploration noise, which lowers their common share, while the prototypes average hundreds of events. The gap cannot be attributed to neighbour spreading.
- The event policies and the prototypes rise together (affordance: 0.11 to 0.60 and 0.52 to 0.90), and executed policies are the prototypes plus noise, so the events cannot be treated as independent of the prototypes: there is a feedback loop (prototype -> executed policy -> event -> prototype).

## 4. Reading

Both mechanisms are present and the checks do not isolate them: a fast early build-up under a wide neighbourhood, a slow late rise not tied to sigma, a neighbour-dominated update (which is structural: policies are learned topologically, so a low own share at sigma about 1.4 is expected and not informative), and events that are themselves 0.6 common through the feedback loop. Check 2 is therefore weak evidence; its useful content is that the late rise does not follow sigma and that distant nodes (zero update weight at sigma 1.4) share the same vector, which points to the early wide-neighbourhood imprint (section 6).

## 5. Caveats

Check 1 is correlational. Check 2 uses 3 epochs per resume and 3 runs per set. `mean_sigma` is a mean over steps of a competence-modulated sigma. The own share also depends on the number of goal nodes used and on event density (original: 0.55 at 7 goal nodes, 0.2 at 19). The resume prints a harmless `diagnose` error after the 3 epochs (the diagnostics object is not stored in the dump).

## 6. Test: narrow starting neighbourhood (common parent hypothesis)

Hypothesis (user): with `internal_sigma=10` on a 10 x 10 map the early neighbourhood covers the whole map, so all prototypes descend from one common parent. New default-off `policy_internal_sigma` caps the starting sigma of the policy map only (same competence schedule, `SMController.policy_sigma`); `internal_sigma` changes all four maps. Champion options, amplitude 0.07, 5 seeds, 150 epochs, pod; controls `dof8_probe` and `affordance_probe/ladder`. Conditions: `isig3o`, `isig1o`, `psig3o`, `psig1o` (original), `psig3a`, `psig1a` (affordance). Scoring: `~/tmp/kss/context4_probe/score_c4.py`; decisions in `decisions.md` (two runs failed at startup with a sqlite lock race and were rerun).

| | control | isig3o | isig1o | psig3o | psig1o | control aff | psig3a | psig1a |
|---|---|---|---|---|---|---|---|---|
| common share | 0.65 | 0.40 | 0.11 | 0.39 | 0.10 | 0.66 | 0.38 | 0.11 |
| participation ratio | 3.5 | 4.3 | 11.0 | 5.0 | 16.3 | 3.6 | 5.9 | 18.4 |
| swap cost | 0.02 | -0.05 | 0.04 | 0.00 | 0.08 | 0.02 | 0.07 | 0.03 |
| constant-policy cost | 0.18 | 0.06 | 0.12 | 0.18 | -0.03 | 0.08 | 0.18 | 0.20 |
| eta2 context over p95 | 1.02 | 0.80 | 1.18 | 0.93 | 0.93 | 1.88 | 1.78 | 1.18 |
| success, intact | 0.70 | 0.71 | 0.89 | 0.76 | 0.78 | 0.73 | 0.80 | 0.76 |
| nodes used | 32 | 17 | 18 | 29 | 18 | 45 | 39 | 30 |

- The common baseline is made by the wide starting neighbourhood: the common share falls with the starting sigma (0.65 to 0.38-0.40 at 3, to 0.10-0.11 at 1) and the cap on the policy map alone is as effective as the cap on all maps.
- The prototypes become diverse but not functional. Swap cost stays inside the seed spread (intervals include 0 except `isig3o`, -0.07 [-0.12, -0.01]); in the original scenario eta2 of context stays at the null; the constant-policy cost is unchanged or lower (`psig1o` -0.22 [-0.45, -0.02]).
- Affordance: constant-policy cost rises (+0.11 [0.00, +0.22] and +0.12 [0.00, +0.24], borderline) but swap cost does not, and colour eta2 falls with the narrow start (`psig1a` 1.18 against 1.88, -0.71 [-1.05, -0.29]): the colour specificity of the control is lost, not gained.
- Success: only `isig1o` is clearly higher (+0.19 [+0.10, +0.29]), a change of all maps, not of the policy map.
- Reading: the common parent explains the common baseline, but removing it does not make policies depend on the goal. Together with report 15 (success rule) and report 14 (shared base), the single-policy problem is not caused by the starting neighbourhood, the update or the rule; the open question is why diverse prototypes bring no functional gain (what the executed policy does at each goal).

## 7. Parametrized-skill analysis: prototype parameters, context and graded outcomes

Reading: the common vector is the skill template and the next few prototype principal components are its parameters, set by the goal node. `src/skill_analysis.py` (default arguments; output `~/tmp/kss/context4_probe/skill`, pooled by `skill/pool.py`): the prototypes of the used nodes (at least 2 segments in 6 evaluation seeds x 36 episodes) are centred and the first 5 principal components are the parameters; per segment the parameters of its goal node are regressed (5-fold cross-validated by episode, ridge) on (a) context features (colour, shape, stretch, rotation), (b) the goal position on the map (linear and quadratic), (c) both, (d) the per-node mean (ceiling); permutation null p95 about 0. Graded outcomes: with 3 other evaluation seeds the parameter k is set to the mean of the used nodes (`flat`) or shuffled across nodes (`shuffle`), all five flattened (`flat_top5`), only the five kept (`keep_top5`), or the whole prototype replaced by the mean (`flat_all`). Sets (3 runs each): affordance 1000 epochs (8-DoF), original 8-DoF sweep (parasite, epoch 1499), 5-output sweeps with sigma 3 and sigma 1 (tag `champion-5dof`, parasite, epoch 1499).

Mean over the 3 runs of each set (top-5 parameters hold 91% / 94% / 87% / 54% of the variance between used nodes):

| cross-validated R2, mean of PC1-PC5 | affordance | original 8-DoF | 5-out sigma 3 | 5-out sigma 1 |
|---|---|---|---|---|
| context features alone | 0.34 | 0.51 | 0.37 | 0.36 |
| goal position, linear | 0.47 | 0.45 | 0.39 | 0.32 |
| goal position, quadratic | 0.87 | 0.95 | 0.85 | 0.72 |
| context and position | 0.88 | 0.96 | 0.89 | 0.80 |
| per-node mean (ceiling) | 0.99 | 0.99 | 0.99 | 1.00 |

- The parameters are a smooth function of where the goal node lies on the map: linear in the position for PC1 and PC2 (R2 0.8 to 0.97), quadratic for PC3 to PC5. Context features explain 0.1 to 0.7 per component (null 0), but add only +0.01 to +0.15 beyond the position, so context acts on the parameters mainly through which part of the map its goals occupy. This is partly built in: the maps are topographic.
- With sigma 1 the position explains less (0.72) and the variance is spread over many more components (top five 54%).

Graded outcomes (cost = intact minus condition; mean over 3 runs; standard deviation over runs in brackets where cited):
- No single parameter has a cost clearly beyond the run-to-run spread in success, match, `max_match`, `hit2` or the end distance of the touch winner. The only consistent signal is the end distance of the proprioception winner to the goal node, which grows when PC1 is flattened or shuffled (+0.24 to +0.52 cells, in all four sets; run spread 0.1 to 0.5) and to a lesser degree PC2.
- The five parameters together carry what is learned: `flat_top5` costs about as much as `flat_all` in the 8-DoF sets (`max_match` 0.106 and 0.121 affordance, 0.073 and 0.077 original; `flat_all` in match 0.037 and 0.033, success 0.08 and 0.07, run spread 0.02 to 0.08) and `keep_top5` costs about nothing. The gain is a gain in match and not in the end distance of the touch winner (cost -0.03 to +0.22 cells).
- With sigma 1, `flat_all` costs most (success 0.28 [0.25], `max_match` 0.21, `hit2` 0.20), but `flat_top5` does not (0.05, 0.00, 0.06): the function sits in many components of low variance, not in the top five.

Reading: the learned policy is a template plus a few parameters that vary smoothly with the goal position and so with the object properties that place the goal. The parameters have a small joint benefit in match (and for the proprio end point), no benefit that shows individually, and none in how near the touch ends to the goal. Caveats: 3 runs per set and 108 evaluation episodes per condition, so differences below about 0.05 in success are not resolved; high R2 on position is expected from topography and does not by itself show that the parameters matter for the task; context and position are confounded in the affordance task because the visual map places objects by colour and shape.

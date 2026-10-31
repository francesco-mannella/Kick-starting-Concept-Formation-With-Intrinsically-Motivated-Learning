# Knowledgebase vs current code

Sources (in `knowledgebase/`):
1. `goal-formation.pdf` (22 p): Zubek, Mannella, Tummolini, "Learning flexible action control with self-supervised (or intrinsic?) ideomotor alignment". Ongoing draft. This is the paper the code should reproduce.
2. `topological_alignment_formalization.html`: math formalisation of the draft (STM energy, block coordinate descent, alignment algorithm, affordance space).
3. `topographic_gating.pdf` (42 p, dated 2026-09-24): Mannella, deep-decoder STM for continual learning (Split-MNIST/FashionMNIST). Imports the maturity modulation and the shared-anchor idea from goal-formation.
4. `stm_full_linear_ring.html`: interactive JS toy of STM alignment (two 8x8 maps, linear manifold s = pi = lambda, ring anchors, SGD M-step, annealing).

## 1. What each source says (short)

- **Draft**: goals are interactions, not states. Four topological maps (vision = condition, policy = action, touch + proprioception = effects) are trained with a shared anchor so one grid point encodes condition-action-effect (topological alignment). Intrinsic signal = internal match increase between touch/proprio winners and the goal point. Learning rate, update sigma and policy noise are modulated by predicted competence. Results: (1) alignment improves control, (2) a yoked "parasite" aligns the maps but does not gain control, (3) pretraining as parasite does not speed learning.
- **Formalisation**: STM energy `E = 1/2 sum phi_ij |x_i - w_j|^2` with `phi = neighbourhood(BMU, sigma_r) * anchor(t_i, sigma_t)`. Batch M-step = phi-weighted centroid (monotone within an epoch at fixed sigma). Alignment = same anchor for all modalities. Modulation eqs 14-16. Inverse mapping `W^T r(z)` (weighted).
- **Topographic Gating**: same STM gate, deep decoder readout, continual learning, leakage bound that decays with grid distance from the anchor. Explicitly states the shared-anchor cross-modal mechanism "has never been tested beyond a single modality" (Sec. 5.3, 5.5, Conclusion). Uses only the sigma_r part of the modulation.
- **Toy HTML**: a minimal executable form of the formalisation; useful as a unit test target (see section 5).

## 2. Draft vs code: what matches

| Draft | Code |
|---|---|
| 4 maps on a 10x10 grid | `internal_size=100`, `stm_v/ss/p/a` |
| Vision 10x10x3, touch 40, proprio 8 (5 before the 8-DoF actuator), policy 75x8 (75x5) | `visual_size=300`, 40, 8, `policy_size=600` (375) |
| Match uses touch and proprio only, vision excluded | `modalities_weights=(0,1,1,0)` |
| Success step: touch match rises and global match exceeds running max | `computeMatchSimple` + `mmask` in `run_episodes` |
| Max match re-initialised at policy selection | `max_match[policy_changed, i:] = match` |
| Goal chosen from visual representations weighted by predicted match | `choose_policy` (`v_rt` weighted by `predict.spread`) |
| Noise on policy, scaled by competence | `getPoliciesFromPointsWithNoise` |
| Predictor = KDE grid of expected match | `SMPredictKDE` |
| Arm controller: 3 pretrained 5x5 SOMs, 75x5 weights, 1000 trials x 50 steps | `GripMapping.py`, `StoredGripActuatorMap.npy` |
| Maps updated once per epoch, visual map on the window that chose the goal | `controller.update` (`cond_ind`) |
| Parasite (yoked) agent, former parasite | `train_parasite`, `SMEnvParasite`, `--parasite` |
| Success rate = at least 2 policies in an episode; MI context-touch / context-policy with KMeans k=10; touch in 5 categories | `episode_success_rate`, `calc_mi_metrics` |

## 3. Draft vs code: discrepancies

Ordered by how much they can affect results or the paper. Items marked (verify) are hypotheses from reading, not tested.

1. **Match width.** Draft: `exp(-(1/25) |x_s - x_goal|^2)` (width 5 grid units). Code: `exp(-d^2 / match_sigma^2)` with `match_sigma = 1.5` default and `0.8` in the recent `test.json` runs. The draft equation is probably stale; a width of 0.8 on a 10x10 grid makes almost any miss a zero match. The formalisation uses `2 sigma_mu^2`, code has no factor 2, so the three documents use three conventions.
2. **Learning-rate modulation probably cancelled by Adam (verify).** Draft eq: `w += eta_u a_u (d - w)` (SGD, eta locally modulated). Code: `SMSTM.update` multiplies the loss by `goals * local_lr` and then takes one **Adam** step. Adam normalises gradient scale, so a lower `local_lr` in a well-learned region reduces the update only relative to other samples in the same batch, not in absolute terms across epochs. `self.modulate` is never actually set (`updateParams` passes `lr=controller.curr_lr = None`). If confirmed, the "protect competent regions / lifelong learning" mechanism of the draft acts through sigma and noise only. Test: scale `local_lr` by 10x and check whether the weights change.
3. **Anchor mechanism differs from the written model.** Formalisation Def. 2 uses a fixed `sigma_t`, Def. 6 uses `sigma^u_j` for both gates. Code: the anchor is the goal grid representation with fixed `representation_sigma = 2.0`, passed as the `extrinsic` factor of the loss; neighbourhood sigma is the local `sigma_u`. This matches Def. 2, not Def. 6. The re-entrant mixing (`external_radial_prop`, `set_radials`) is dead code since `update_reentrant_connections` is never called. The toy HTML also anneals `sigma_t` (3.0 to 0.8); the code keeps it fixed.
4. **Inverse mapping.** Draft and formalisation eq 8: `W^T r(z)` (Gaussian-weighted blend). Code `STM.backward` uses the single argmax prototype ("CHANGE: Use single prototype instead of weighted average"). Topographic Gating builds its whole argument on a smooth, continuous readout. The goal-formation code deliberately does the opposite. The paper should say which one, and why.
5. **M-step.** Formalisation proves monotone energy decrease for the batch closed-form centroid (eq 6) or online SGD (eq 7). Code does one Adam step on the epoch's selected samples (momentum carried across epochs). The convergence statements do not transfer directly.
6. **Competence aggregation.** Draft: `C_tot = mean(c)`, gate `1 - tanh(3 C_tot)`, local gate `1 - tanh(c)`. Code: `global_competence = mean(tanh(decay * c))` with `decay = 5`, `local = tanh(local_decay * c)`, `local_incompetence = (1 - global) * (1 - local)`. Mean of tanh is not tanh of mean, and the gain is 5 instead of 3. The global learning-rate modulation is commented out (`# No global modulation of LR and sigma`).
7. **Success threshold and stopping.** Draft: success iff `psi_t > max_psi`; run `while C_tot < 1`. Code adds `match_incr_th` (0.001 default, 0.01 in recent runs), `cum_match_stop_th` (successes needed before switching goal), and runs a fixed number of epochs (1000 default, 3000 in recent runs). The draft's figures run to 1500 epochs while the text says 1000.
8. **Environment description.** Draft text: three objects (blue square fixed, red triangle movable, green square graspable and movable). Code: 3 colour contexts x 2 stretches x 3 rotations = 18 episode types (`build_episode_dataset`), and Fig. 3 already shows "Stretch 1.0, Rotation 90". The text needs the same update as the figures. Colour-to-affordance mapping depends on `context` 1/2/3 in the simulator; check it matches the text.
9. **Match includes `a` and `v` in 2023, not here.** Formalisation eq 10 (2023 form) averages all modalities including policy; the active definition and the code use touch and proprio only. Keep eq 10 only if the paper discusses the 2023 variant.
10. **Reported noise-free evaluation.** Code comment in `evaluation_episodes`: "without noise eval results are much worse than with normal noise", while evaluation defaults to `zero_noise=True`. The draft never mentions it. A reviewer will ask.
11. **Draft citation year.** Draft and formalisation cite Zubek et al. 2024; Topographic Gating cites Zubek et al. 2025.
12. **Predictor update.** Draft eq `c += alpha psi` (not a running mean); code `c += lr (psi - c)`. The code form is the sensible one; fix the draft. `competences[idx] += ...` with repeated idx applies only one increment per node per call (use `np.add.at` if every episode should count).
13. **"Internal alignment" metric.** Plotted in Figs 5-7 but not named in the code. Closest quantity: `grid_comp_mean` (wandb) = mean tanh(decay * c). Say which one the figure shows, and keep it identical to `C_tot` of the text.
14. **Limitation not tested.** The draft's limitation ("no failure condition, cannot deal with changing environments") and the claim that "learning resumes with new contexts" are not exercised by any script. A staged-context experiment (add contexts mid-training) would test it and links directly to Topographic Gating.

## 4. Relation to Topographic Gating

- Topographic Gating takes goal-formation as its multi-modal parent and leaves the cross-modal test to it. The draft is therefore the place where the shared-anchor claim is actually tested; the two papers should cite each other consistently (year, role).
- Anchor placement differs in kind: Topographic Gating fixes the anchor per class before training (supervised); goal-formation derives it from the current policy (self-supervised). Say so in both papers.
- The maturity (competence) modulation is used in Topographic Gating for `sigma_r` only. If item 2 above is confirmed, goal-formation effectively also relies on sigma (and noise) only; the two would then be consistent, and the draft should stop claiming a learning-rate effect.
- The leakage bound (decays with grid distance from the anchor) gives a principled way to choose `representation_sigma` (anchor width), grid size and the spacing of goals, and to predict interference between the 18 episode types on 100 nodes. Currently these are hand-set.
- Readout: Topographic Gating argues for smooth Gaussian-weighted readout; goal-formation code uses argmax (item 4).
- Generative replay and capacity: untested in goal-formation; relevant only if the continual-context experiment (item 14) is run.

## 5. How the toy HTML can be used

`stm_full_linear_ring.html` is a clean oracle for the formalisation: two 8x8 maps, linear data, ring anchors, SGD M-step, C_tot = mean exp(-8 |Ws - Wp|^2). Port it to a ~60-line Python test that runs the same data through (a) the formula as written and (b) `stm.SMSTM` with Adam, and compare the C_tot curves. This directly tests items 2, 3 and 5 on a problem where the answer is known, without running the simulator.

## 6. Suggested order before starting the planned refactor/speed work

1. Decide the reference: draft equations or current code behaviour. Items 1-7 need a decision from you (fix the text or fix the code).
2. Run the Adam-vs-SGD toy test (section 5) and the `local_lr` scaling check (item 2). These determine whether the learning-rate modulation is real.
3. Freeze behaviour with the seeded smoke run from `reports/03` before touching `SMMain.py`; the discrepancies above must not change silently during refactoring.
4. Then do the rationalisation (`03`) and speed (`04`) plans. Note two speed-plan items alter numerics (RNG for motor noise, `np.add.at` in the predictor) and should wait until the reference is decided.
5. Update the draft: environment text, match equation and width, predictor equation, citation year, evaluation-noise note, and say which readout (argmax vs blend) is used.

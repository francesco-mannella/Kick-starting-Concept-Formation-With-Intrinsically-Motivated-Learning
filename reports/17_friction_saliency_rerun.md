# Friction and saliency: Figs 5-7 sweeps, seeds 0-9 and a rerun on seeds 100-109

## Setup
All runs use the 5-output, amplitude-8 champion on the v3 map (block_rms, rbf_block, tanh, touch_gain 3, anchored policy and map winners, ema, touch_gate), 1500 epochs, evaluation every 50 epochs. Friction is `obj_damping=20.0`, `obj_friction=2.0` on the movable and controllable objects. Saliency is `saliency=True sal_gamma=3 sal_kappa=1 sal_temp=1.0`. Scoring window: epochs 1000-1500, paired by seed. A difference is called real if |z| > 2 (z = mean paired difference over its standard error, 10 seeds) and, for success, at least 0.05. The former parasite loads the parasite's final weights (`--pretrain-epoch 1499`); older sweeps loaded epoch 1000 (launcher default of commit 52f096f), so their former rows are not paired with these.

Scripts: `src/sweeps/` (`run_fig567_*_nosal.sh` for seeds 0-9, `r2_chain.sh`, `r2_condition.sh`, `r2_score.py` for the rerun, `stats_friction.sh` for counterfactuals). Results live in `~/tmp/kss/fig567_*` (not versioned).

## Round 1, seeds 0-9 (friction without saliency vs none)
Friction against no friction, both without saliency: agent success +0.045 (se 0.072, not real); agent touch +0.08 (z 4.3); former success +0.36 (z 3.4); former touch +0.165 (z 7.3); former MI context-policy -0.09 (z -2.8); alignment unchanged. Turning saliency off raised agent MI context-policy by about 0.10 (z 3) with and without friction, and, with friction, parasite success from 0.12 to 0.56 (z 4.3).

## Round 2, new seeds 100-109, 2x2 (abridged output of `r2_score.py`)
Folders `fig567_r2_{base,fric,sal,fricsal}`; counterfactuals on the 10 epoch-1499 agents of each.

```
Epochs 1000-1500, 10 seeds (100-109), mean +- sd

  series  metric no-fric/no-sal  no-fric/sal  fric/no-sal     fric/sal
   agent   align  0.614+-0.036 0.635+-0.043 0.661+-0.043 0.612+-0.022
   agent    succ  0.740+-0.192 0.727+-0.117 0.711+-0.195 0.760+-0.194
   agent   MIpol  0.906+-0.089 0.866+-0.047 0.899+-0.071 0.776+-0.095
   agent MItouch  0.181+-0.065 0.163+-0.026 0.169+-0.040 0.134+-0.053
   agent   touch  0.157+-0.040 0.162+-0.035 0.217+-0.044 0.220+-0.044
parasite   align  0.615+-0.060 0.637+-0.039 0.653+-0.037 0.640+-0.027
parasite    succ  0.271+-0.369 0.256+-0.311 0.267+-0.265 0.303+-0.267
parasite   MIpol  1.004+-0.088 0.983+-0.072 0.961+-0.105 0.902+-0.099
parasite MItouch  0.043+-0.071 0.066+-0.082 0.045+-0.061 0.059+-0.059
parasite   touch  0.053+-0.074 0.062+-0.072 0.061+-0.061 0.091+-0.080
  former   align  0.611+-0.045 0.638+-0.048 0.653+-0.050 0.639+-0.029
  former    succ  0.520+-0.353 0.376+-0.311 0.540+-0.396 0.696+-0.300
  former   MIpol  0.913+-0.097 0.923+-0.106 0.933+-0.091 0.856+-0.092
  former MItouch  0.116+-0.085 0.095+-0.081 0.105+-0.070 0.126+-0.061
  former   touch  0.115+-0.074 0.086+-0.070 0.155+-0.107 0.206+-0.107
stuck formers (success < 0.3), of 10: 3, 4, 4, 1

Paired contrasts (new minus reference), z = mean/se over 10 seeds; * = |z|>2

== friction effect, no saliency
  agent   align    diff +0.047 se 0.019 z +2.4 *
  agent   succ     diff -0.029 se 0.048 z -0.6
  agent   MIpol    diff -0.007 se 0.027 z -0.3
  agent   touch    diff +0.060 se 0.018 z +3.3 *
  parasitesucc     diff -0.004 se 0.131 z -0.0
  former  succ     diff +0.021 se 0.197 z +0.1
  former  touch    diff +0.040 se 0.047 z +0.8
== friction effect, with saliency
  agent   align    diff -0.023 se 0.014 z -1.7
  agent   succ     diff +0.033 se 0.067 z +0.5
  agent   MIpol    diff -0.090 se 0.024 z -3.7 *
  agent   touch    diff +0.059 se 0.021 z +2.8 *
  parasiteMIpol    diff -0.081 se 0.035 z -2.3 *
  former  succ     diff +0.320 se 0.089 z +3.6 *
  former  touch    diff +0.120 se 0.028 z +4.2 *
== saliency effect, no friction
  agent   align    diff +0.022 se 0.018 z +1.2
  agent   succ     diff -0.013 se 0.084 z -0.2
  agent   MIpol    diff -0.039 se 0.022 z -1.8
  former  succ     diff -0.144 se 0.116 z -1.2
== saliency effect, with friction
  agent   align    diff -0.049 se 0.016 z -3.0 *
  agent   succ     diff +0.049 se 0.071 z +0.7
  agent   MIpol    diff -0.123 se 0.029 z -4.3 *
  parasiteMIpol    diff -0.059 se 0.025 z -2.3 *
  former  succ     diff +0.156 se 0.138 z +1.1
  former  MIpol    diff -0.078 se 0.026 z -3.0 *
(only the rows with |z| > 2 and the success rows are shown; the full list is in the script output)

Counterfactual (agent, epoch 1499), success: intact | swap | shuffle_within_context | shuffle_all_used | constant_mean
  no-fric/no-sal  0.754 | 0.782 | 0.758 | 0.771 | 0.716   (constant cost +0.038 +- 0.041)
  no-fric/sal     0.692 | 0.678 | 0.682 | 0.699 | 0.672   (+0.019 +- 0.030)
  fric/no-sal     0.693 | 0.720 | 0.706 | 0.689 | 0.614   (+0.079 +- 0.042)
  fric/sal        0.706 | 0.711 | 0.691 | 0.710 | 0.657   (+0.049 +- 0.037)
constant-policy success cost by context (ctx 1 / 2 / 3):
  no-fric/no-sal  +0.064 / +0.039 / +0.011
  no-fric/sal     +0.017 / +0.039 / +0.003
  fric/no-sal     +0.075 / +0.022 / +0.139
  fric/sal        +0.083 / +0.017 / +0.047
hit2 cost of the constant policy: +0.040, +0.015, +0.055 (+-0.019), +0.019
```

## Reading
- Replicated: friction raises agent touch frequency (+0.06 with z 3.3 and 2.8; +0.08 in round 1).
- Not replicated: any friction or saliency effect on agent success (|z| < 1 in every contrast); the parasite success jump (0.56 in round 1, 0.27 now); the friction effect on former success without saliency (+0.36 then, +0.02 now).
- Appears in one pair only: former success +0.32 and former touch +0.12 from friction with saliency; agent MI context-policy lowered by saliency under friction (-0.12) and by friction with saliency (-0.09); agent alignment +0.047 from friction without saliency and -0.049 from saliency with friction. Round 1 shows the opposite sign for saliency on MI, so these are treated as seed noise until repeated.
- Counterfactual: swap and shuffle cost about 0; constant-mean-policy success cost +0.04, +0.02, +0.08, +0.05. The cost on context 3 is larger with friction (+0.139 and +0.047 with friction, +0.011 and +0.003 without in round 2; +0.095 in the old friction+saliency sweep), pooled about +0.09 against +0.02, z about 1.4 (hand estimate, not significant).
- Parasite success is evaluated by running the parasite's own policy map in a real environment (`SMMain.diagnose`, suffix `_par`), not by replay; it tracks the parasite's own touch frequency (r 0.8-0.99 across seeds), not the agent's.
- Per-seed spread is large (agent success sd 0.12-0.2, former success sd 0.3-0.4, 1-4 of 10 stuck formers), so with 10 seeds only effects of about 0.15 are resolved.

The complete script output with every contrast is kept in `~/tmp/kss/r2_scores.txt`.

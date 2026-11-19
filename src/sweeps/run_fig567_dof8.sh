#!/bin/bash
WT=/home/fmannella/Dropbox/Projects/current/Kick-starting-Sources/.claude/worktrees/noise-plan
export MPLBACKEND=agg OMP_NUM_THREADS=1
export PYTHONPATH=$WT/tools/box2dsim:$WT/src
cd $WT
python3 src/launch_fig567.py --root /home/fmannella/tmp/kss/fig567_dof8 --seeds 0-9 --epochs 1500 --pretrain-epoch 1000 --epochs-to-test 50 \
  -o "grip_input_norm='block_rms'" -o "grip_encoding='rbf_block'" -o "grip_out_fn='tanh'" -o "grip_touch_gain=3.0" \
  -o "policy_winner='anchored'" -o "policy_update='ema'" -o "map_winner='anchored'" -o "touch_gate=True" \
  >> /home/fmannella/tmp/kss/fig567_dof8.out 2>&1

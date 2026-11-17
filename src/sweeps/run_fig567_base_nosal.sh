#!/bin/bash
WT=/home/fmannella/Dropbox/Projects/current/Kick-starting-Sources
export MPLBACKEND=agg OMP_NUM_THREADS=1
export PYTHONPATH=$WT/tools/box2dsim:$WT/src
cd $WT
python3 src/launch_fig567.py --root /home/fmannella/tmp/kss/fig567_base_nosal --seeds 0-9 --epochs 1500 --pretrain-epoch 1499 --epochs-to-test 50 \
  -o "grip_map_name='data/StoredGripActuatorMap_v3'" -o "grip_norm_name='data/StoredGripActuatorNorm_v3'" \
  -o "grip_input_norm='block_rms'" -o "grip_encoding='rbf_block'" -o "grip_out_fn='tanh'" -o "grip_touch_gain=3.0" \
  -o "policy_winner='anchored'" -o "policy_update='ema'" -o "map_winner='anchored'" -o "touch_gate=True" \
  \
  >> /home/fmannella/tmp/kss/fig567_base_nosal.out 2>&1

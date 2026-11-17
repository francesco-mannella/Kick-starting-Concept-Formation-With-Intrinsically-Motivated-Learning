#!/bin/bash
label=$1; shift
WT=/home/fmannella/Dropbox/Projects/current/Kick-starting-Sources
export MPLBACKEND=agg OMP_NUM_THREADS=1
export PYTHONPATH=$WT/tools/box2dsim:$WT/src
ROOT=/home/fmannella/tmp/kss/fig567_r2_$label
cd $WT
python3 src/launch_fig567.py --root $ROOT --seeds 100-109 --epochs 1500 --pretrain-epoch 1499 --epochs-to-test 50 \
  -o "grip_map_name='data/StoredGripActuatorMap_v3'" -o "grip_norm_name='data/StoredGripActuatorNorm_v3'" \
  -o "grip_input_norm='block_rms'" -o "grip_encoding='rbf_block'" -o "grip_out_fn='tanh'" -o "grip_touch_gain=3.0" \
  -o "policy_winner='anchored'" -o "policy_update='ema'" -o "map_winner='anchored'" -o "touch_gate=True" \
  "$@" >> /home/fmannella/tmp/kss/fig567_r2_$label.out 2>&1
mkdir -p $ROOT/stats
cd $WT/src
job(){ i=$1; run=parasite_$i
  [ -f $ROOT/stats/agent_${i}_static.json ] && return
  python3 policy_stats.py --storage $ROOT/$run/simulations/$run/storage --epochs 1499 --counterfactual --out $ROOT/stats/agent_$i > $ROOT/stats/agent_$i.log 2>&1
}
export -f job; export ROOT
seq 100 109 | xargs -P 6 -n1 bash -c 'job $0'
echo "$label ALLDONE $(date +%T)" >> /home/fmannella/tmp/kss/r2.progress

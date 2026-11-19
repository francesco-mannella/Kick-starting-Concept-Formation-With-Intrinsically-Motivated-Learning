#!/bin/bash
WT=/home/fmannella/Dropbox/Projects/current/Kick-starting-Sources/.claude/worktrees/l2-objects
export MPLBACKEND=agg OMP_NUM_THREADS=1
export PYTHONPATH=$WT/tools/box2dsim:$WT/src
ROOT=/home/fmannella/tmp/kss/fig567_designB
cd $WT
python3 src/launch_fig567.py --root $ROOT --seeds 100-109 --epochs 1500 --pretrain-epoch 1499 --epochs-to-test 50 \
  -o "grip_map_name='data/StoredGripActuatorMap_v3'" -o "grip_norm_name='data/StoredGripActuatorNorm_v3'" \
  -o "grip_input_norm='block_rms'" -o "grip_encoding='rbf_block'" -o "grip_out_fn='tanh'" -o "grip_touch_gain=3.0" \
  -o "policy_winner='anchored'" -o "policy_update='ema'" -o "map_winner='anchored'" -o "touch_gate=True" \
  -o "obj_damping=20.0" -o "obj_friction=2.0" \
  -o "obj_set='mix'" -o "obj_pos_kind={'still':[14.0,12.0],'movable':[7.0,12.0],'controllable':[14.0,4.0]}" \
  >> /home/fmannella/tmp/kss/fig567_designB.out 2>&1
mkdir -p $ROOT/stats
cd $WT/src
job(){ i=$1; run=parasite_$i
  [ -f $ROOT/stats/agent_${i}_static.json ] && return
  python3 policy_stats.py --storage $ROOT/$run/simulations/$run/storage --epochs 1499 --counterfactual --out $ROOT/stats/agent_$i > $ROOT/stats/agent_$i.log 2>&1
}
export -f job; export ROOT
seq 100 109 | xargs -P 6 -n1 bash -c 'job $0'
echo "sweep and counterfactuals done $(date +%T)" >> /home/fmannella/tmp/kss/designB.progress

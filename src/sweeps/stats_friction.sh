#!/bin/bash
cd /home/fmannella/Dropbox/Projects/current/Kick-starting-Sources/src
export MPLBACKEND=agg OMP_NUM_THREADS=1 PYTHONPATH=/home/fmannella/Dropbox/Projects/current/Kick-starting-Sources/tools/box2dsim:/home/fmannella/Dropbox/Projects/current/Kick-starting-Sources/src
job(){ root=$1; i=$2; run=parasite_00$i
  [ -f $root/stats/agent_00$i\_static.json ] && return
  python3 policy_stats.py --storage $root/$run/simulations/$run/storage --epochs 1499 --counterfactual --out $root/stats/agent_00$i > $root/stats/agent_00$i.log 2>&1
  echo "$(basename $root) $i done $(date +%T)" >> /home/fmannella/tmp/kss/stats_friction.progress
}
export -f job
for r in fig567_fric_nosal fig567_base_nosal; do mkdir -p /home/fmannella/tmp/kss/$r/stats; done
for i in 0 1 2 3 4 5 6 7 8 9; do for r in fig567_fric_nosal fig567_base_nosal; do echo "/home/fmannella/tmp/kss/$r $i"; done; done | xargs -P 6 -L1 bash -c 'job $0 $1'
echo ALLDONE >> /home/fmannella/tmp/kss/stats_friction.progress

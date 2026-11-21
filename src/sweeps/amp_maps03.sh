#!/bin/bash
REPO=/home/fmannella/Dropbox/Projects/current/Kick-starting-Sources
K=/home/fmannella/tmp/kss
export MPLBACKEND=agg OMP_NUM_THREADS=1 PYTHONPATH=$REPO/src:$REPO/tools/box2dsim
cd $REPO/src
mkdir -p $K/policymap_amp
job(){ tag=$1; ck=$2; python3 policy_map_glyphs.py $ck --out $K/policymap_amp/$tag > $K/policymap_amp/$tag.log 2>&1; }
export -f job; export K
{
for s in 0 1 2 3 4; do
  echo "amp03_$s $K/satplan_probe/agent_amp03_eta0.05_00$s/simulations/agent_amp03_eta0.05_00$s/storage/000999"
done
} | xargs -P 5 -L1 bash -c 'job $0 $1'
ls $K/policymap_amp/*/policy_map.png | wc -l

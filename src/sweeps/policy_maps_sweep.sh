#!/bin/bash
# usage: policy_maps_sweep.sh SWEEP_ROOT   (agent of every parasite_NNN run, epoch 1499)
ROOT=$1
REPO=/home/fmannella/Dropbox/Projects/current/Kick-starting-Sources
export MPLBACKEND=agg OMP_NUM_THREADS=1 PYTHONPATH=$REPO/src:$REPO/tools/box2dsim
cd $REPO/src
mkdir -p $ROOT/policy_maps $ROOT/figures
job(){ s=$1
  ck=$ROOT/parasite_$s/simulations/parasite_$s/storage/001499
  python3 policy_map_glyphs.py $ck --out $ROOT/policy_maps/seed_$s > $ROOT/policy_maps/seed_$s.log 2>&1 \
    && cp $ROOT/policy_maps/seed_$s/policy_map.png $ROOT/figures/policy_map_seed_$s.png
}
export -f job; export ROOT
seq 100 109 | xargs -P 5 -n1 bash -c 'job $0'
ls $ROOT/figures | grep -c policy_map_seed

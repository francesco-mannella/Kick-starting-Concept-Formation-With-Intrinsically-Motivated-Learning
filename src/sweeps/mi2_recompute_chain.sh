#!/bin/bash
# usage: mi2_recompute_chain.sh SWEEP_ROOT CODE_ROOT [WORKERS]
# Re-runs the 360-episode evaluation at every stored checkpoint of a finished sweep and writes the mi2 values
# (one json per run, epoch and series); finished jobs are skipped, so it can be stopped and restarted.
# CODE_ROOT is the checkout whose src/ and tools/box2dsim match the sweep (object set, fovea); mi_standard.py comes from this repo.
ROOT=$1; WT=$2; WORKERS=${3:-6}
OUT=$ROOT/mi2_recompute; mkdir -p $OUT
export MPLBACKEND=agg OMP_NUM_THREADS=1 PYTHONPATH=$WT/tools/box2dsim:$WT/src
HERE=$(cd "$(dirname "$0")" && pwd)
job(){ run=$1; epoch=$2; series=$3; out=$OUT/${run}_${epoch}_${series}.json
  [ -f $out ] && return
  python3 $HERE/mi2_recompute.py --storage $ROOT/$run/simulations/$run/storage --epoch $((10#$epoch)) --series $series --out $out.tmp > $OUT/${run}_${epoch}_${series}.log 2>&1 && mv $out.tmp $out
}
export -f job; export ROOT OUT HERE
{
for run in $(ls $ROOT | grep "^parasite_\|^former_"); do
  for ep in $(ls $ROOT/$run/simulations/$run/storage); do
    echo "$run $ep agent"
    [[ $run == parasite_* ]] && echo "$run $ep par"
  done
done
} | xargs -P $WORKERS -L1 bash -c 'job $0 $1 $2'
echo "MI2 RECOMPUTE DONE $(date +%T)"

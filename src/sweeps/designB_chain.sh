#!/bin/bash
WT=/home/fmannella/Dropbox/Projects/current/Kick-starting-Sources/.claude/worktrees/l2-objects
K=/home/fmannella/tmp/kss
ROOT=$K/fig567_designB
export MPLBACKEND=agg OMP_NUM_THREADS=1
export PYTHONPATH=$WT/tools/box2dsim:$WT/src
cd $WT
python3 src/test_objects_l2.py > $K/designB_tests.log 2>&1
if ! tail -n 1 $K/designB_tests.log | grep -q "^ok"; then echo "TESTS FAILED, chain stopped $(date +%T)" >> $K/designB.progress; exit 1; fi
echo "unit tests passed $(date +%T)" >> $K/designB.progress
bash $K/designB_condition.sh
python3 $K/designB_checks.py > $K/designB_checks.txt 2>&1
echo "sanity checks written $(date +%T)" >> $K/designB.progress
python3 $K/designB_score.py > $K/designB_scores.txt 2>&1
echo "scoring written $(date +%T)" >> $K/designB.progress
python3 src/policy_demos.py --sweep $ROOT --work $K/designB_demos_work --out $ROOT/demos/gifs --seeds 100-109 \
  --labels blue-plate,red-triangle,green-wedge --parallel 8 > $K/designB_demos.log 2>&1
echo "policy demos done $(date +%T)" >> $K/designB.progress
echo "DESIGN B ALLDONE $(date +%T)" >> $K/designB.progress

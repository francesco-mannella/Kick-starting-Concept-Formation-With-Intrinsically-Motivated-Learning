#!/bin/bash
cd /home/fmannella/tmp/kss
SAL='-o saliency=True -o sal_gamma=3 -o sal_kappa=1 -o sal_temp=1.0'
FRIC='-o obj_damping=20.0 -o obj_friction=2.0'
bash r2_condition.sh base
bash r2_condition.sh fric $FRIC
bash r2_condition.sh sal $SAL
bash r2_condition.sh fricsal $FRIC $SAL
python3 r2_score.py > r2_scores.txt 2>&1
echo "ALL ALLDONE $(date +%T)" >> r2.progress

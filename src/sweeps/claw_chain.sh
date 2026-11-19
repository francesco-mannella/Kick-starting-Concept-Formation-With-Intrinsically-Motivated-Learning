#!/bin/bash
L=/home/fmannella/Dropbox/Projects/current/Kick-starting-Sources/.claude/worktrees/l2-objects
D=/home/fmannella/Dropbox/Projects/current/Kick-starting-Sources/.claude/worktrees/dof8-screens
export OMP_NUM_THREADS=1 MPLBACKEND=agg
cd /home/fmannella/tmp/kss
PYTHONPATH=$L/tools/box2dsim:$L/src python3 claw_sweeps.py /home/fmannella/tmp/kss/claw_5 > claw_5.log 2>&1
PYTHONPATH=$D/tools/box2dsim:$D/src python3 claw_sweeps.py /home/fmannella/tmp/kss/claw_8 > claw_8.log 2>&1
echo CLAW ALLDONE >> claw.progress

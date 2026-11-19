#!/bin/bash
cd /home/fmannella/tmp/kss
bash pilot_condition.sh ctrl &
bash pilot_condition.sh kind -o "obj_pos_kind={'still':[10.7,9.3],'movable':[14.0,6.0],'controllable':[14.0,12.0]}" &
wait
echo "PILOT ALLDONE $(date +%T)" >> pilot.progress

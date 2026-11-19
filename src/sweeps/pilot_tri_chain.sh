#!/bin/bash
cd /home/fmannella/tmp/kss
bash pilot_condition.sh tri -o "obj_set='tri'" -o "obj_pos_kind={'still':[7.0,12.0],'movable':[10.7,9.3],'controllable':[14.0,4.0]}"
echo "TRI PILOT ALLDONE $(date +%T)" >> pilot.progress

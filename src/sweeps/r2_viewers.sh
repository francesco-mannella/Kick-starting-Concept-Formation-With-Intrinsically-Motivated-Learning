#!/bin/bash
cd /home/fmannella/Dropbox/Projects/current/Kick-starting-Sources
start(){ label=$1; port=$2; db=/home/fmannella/tmp/kss/fig567_r2_$label/runs.db
  until [ -f $db ]; do sleep 60; done
  python3 src/localwandb_server.py --db $db --port $port --host 0.0.0.0 < /dev/null > /home/fmannella/tmp/kss/fig567_r2_${label}_viewer.log 2>&1
}
start base 8081 &
start fric 8082 &
start sal 8083 &
start fricsal 8084 &
wait

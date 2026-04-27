#!/bin/bash

for d in $(find -type d| grep 000999| xargs realpath); do 
    cd $d; 
    run_evaluation.py -s 100 -n 10 -p -g
    run_postures -r $(seq 0 9) -g
done

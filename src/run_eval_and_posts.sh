#!/bin/bash

for d in $(find -type d| grep 000999| xargs realpath|sort); do 
    cd $d; 
    echo $d | sed -E 's/\//\n/g'| grep last
    run_evaluation.py -s 100 -n 10 -p -g
    run_postures.py -r $(seq 0 9) -g
done

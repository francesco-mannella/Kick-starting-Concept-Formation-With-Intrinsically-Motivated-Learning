#!/bin/bash

set -euo pipefail

export PYTHONPATH=/home/fmannella/Projects/current/Kick-starting-Sources/src/
export PATH=$PATH:$PYTHONPATH

CUR="$(pwd)"

run_analysis=false
while getopts "rh" opt; do
    case $opt in
        r) run_analysis=true ;;
        h) echo "Usage: $0 [-r] [-h] [pattern]"
           echo "  -r  Run analysis"
           echo "  -h  Show this help"
           exit 0 ;;
        *) echo "Usage: $0 [-r] [-h] [pattern]"; exit 1 ;;
    esac
done
shift $((OPTIND - 1))

find_opts="-mindepth 1 -maxdepth 1 -type d"
if [[ -n "${1:-}" ]]; then
    find_opts="$find_opts -name '*$1*'"
fi

while IFS= read -r -d '' data_dir; do
    simdata="$(find $data_dir -type d -name '*2999*')"
    data_dir="${data_dir#CUR/}"
    simdata="${simdata#CUR/}"

    echo "folder: $data_dir"
    echo "simulation: $data_dir"

    if $run_analysis; then
        if [[ -z "$(find $simdata -type f -name "*goalgrid*")" ]]; then
            echo "Starting analysis ..."
            cd $simdata || (cd "$CUR" && exit 1)
            run_evaluation.py -s 200 -n 30 -g -p
            run_postures.py -r {0..9} -g
            cd "$CUR"
        fi
    fi
    echo "Compressing data ..."
    zip -r ${data_dir}.zip $data_dir
    rm -fr $data_dir

done < <(eval "find . $find_opts -print0" | sort -z | xargs -0 -r realpath -z)

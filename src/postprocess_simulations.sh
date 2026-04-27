#!/bin/bash
# Script to analyze and compress simulation data directories

set -euo pipefail

# Set up Python environment paths
export PYTHONPATH=/home/fmannella/Projects/current/Kick-starting-Sources/src/
export PATH=$PATH:$PYTHONPATH

readonly CUR="$(pwd)"
readonly SCRIPT_NAME="$(basename "$0")"

# Default options
run_analysis=false
suffix=2999
compress=false

usage() {
    cat <<EOF
Usage: $SCRIPT_NAME [-r] [-s <SUFFIX>] [-h] [pattern]
  -r          Run analysis
  -c          Compress
  -s <SUFFIX> Suffix of the internal data folder (use 'none' for main folder)
  -h          Show this help
EOF
}

# Parse command line arguments
while getopts "crs:h" opt; do
    case $opt in
        r) run_analysis=true ;;
        s) suffix=$OPTARG ;;
        c) compress=true ;;
        h) usage; exit 0 ;;
        *) usage >&2; exit 1 ;;
    esac
done
shift $((OPTIND - 1))

# Build find command with optional pattern filter
pattern="${1:-}"
find_args=(-mindepth 1 -maxdepth 1 -type d)
[[ -n "$pattern" ]] && find_args+=(-name "*${pattern}*")

# Process each data directory
while IFS= read -r -d '' data_dir; do
    simdata=""

    if [[ "$suffix" == "none" ]]; then
        # Use main folder directly
        simdata="$data_dir"
    elif [[ "$suffix" =~ ^[0-9]+$ ]]; then
        # Find subfolder matching numeric suffix
        simdata="$(find "$data_dir" -type d -name "*${suffix}*" | head -1)"
    else
        echo "Invalid suffix: $suffix" >&2
        exit 1
    fi

    if [[ -z "$simdata" ]]; then
        echo "No folder matching suffix '$suffix' in $data_dir" >&2
        exit 1
    fi

    # Convert to relative paths for display and zip
    rel_data="${data_dir#$CUR/}"
    rel_sim="${simdata#$CUR/}"
    echo "folder: $rel_data"
    echo "simulation: $rel_sim"

    # Run analysis if requested and not already done
    if $run_analysis; then
        if ! find "$simdata" -type f -name "*goalgrid*" -quit | grep -q .; then
            echo "Starting analysis ..."
            cd "$simdata" || exit 1
            run_evaluation.py -s 200 -n 30 -g -p
            run_postures.py -r {0..9} -g
            cd "$CUR"
        fi
    fi

    # Compress and remove original directory
    if [[ $compress = true ]]; then
        echo "Compressing data ..."
        zip -rq "${rel_data}.zip" "$rel_data"
        rm -rf "$rel_data"
    fi

done < <(find . "${find_args[@]}" -print0 | sort -z | xargs -0 -r realpath -z)

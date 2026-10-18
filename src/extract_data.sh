#!/bin/bash

set -e
set -euo pipefail

usage() {
    echo "Usage: $(basename $0) -d <data_file> [-o <dest_dir>] [-s <suffix>]" >&2
    exit 1
}

SUFFIX="2999"
DEST="out"
DATA=

while getopts "d:o:s:h" opt; do
    case $opt in
        d) DATA="$OPTARG" ;;
        o) DEST="$OPTARG" ;;
        s) SUFFIX="$OPTARG" ;;
        h|*) usage ;;
    esac
done

[[ -z "$DATA" ]] && usage
[[ ! -f "$DATA" ]] && { echo "Error: Data file not found" >&2; exit 1; }

base_dir=$(unzip -t "$DATA" "*${SUFFIX}*" 2>/dev/null | awk 'NR==2 {print $2}' \
    | sed -E "s/^(.*${SUFFIX}).*/\1/")

[[ -z "$base_dir" ]] && { echo "Error: No matching files in archive" >&2; exit 1; }

unzip "$DATA" "*${SUFFIX}*" -d /tmp || exit 1
echo "/tmp/$base_dir"
mv "/tmp/$base_dir" "$DEST" || exit 1
rm -fr /tmp/home

#!/bin/bash

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

base_dir=$(unzip -t "$DATA" "*${SUFFIX}*" | awk 'NR==2 {print $2}' | \
    sed -E "s/^(.*${SUFFIX}).*/\1/")

unzip "$DATA" "*${SUFFIX}*" -d /tmp
echo "/tmp/$base_dir"
mv "/tmp/$base_dir" "$DEST"
rm -fr /tmp/home

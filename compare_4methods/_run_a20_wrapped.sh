#!/usr/bin/env bash
set -o pipefail
LOG=/mnt/c/code/media_code/compare_4methods/_run_a20_wrapped.log
: > "$LOG"
{
    source ~/miniconda3/etc/profile.d/conda.sh
    conda activate querysplat
    cd /mnt/c/code/media_code
    python compare_4methods/A20_make_framework_figures.py
    rc=$?
    echo "A20 exit=$rc"
    exit $rc
} >> "$LOG" 2>&1

#!/usr/bin/env bash
# A21 出图包装：WSL conda querysplat env（matplotlib 所在），日志落 _run_a21_wrapped.log
set -o pipefail
LOG=/mnt/c/code/media_code/compare_4methods/_run_a21_wrapped.log
: > "$LOG"
{
    source ~/miniconda3/etc/profile.d/conda.sh
    conda activate querysplat
    cd /mnt/c/code/media_code
    python compare_4methods/A21_make_delta_figures.py
    rc=$?
    echo "A21 exit=$rc"
    exit $rc
} >> "$LOG" 2>&1

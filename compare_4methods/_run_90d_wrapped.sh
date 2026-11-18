#!/usr/bin/env bash
# 90d 的 WSL 包装（与 _run_90b_wrapped 同款）：日志落 _run_90d_wrapped.log
set -o pipefail
LOG=/mnt/c/code/media_code/compare_4methods/_run_90d_wrapped.log
: > "$LOG"
{
    cd /mnt/c/code/media_code
    bash compare_4methods/90d_run_2k_jiajia.sh
    rc=$?
    echo "90d exit=$rc"
    exit $rc
} >> "$LOG" 2>&1

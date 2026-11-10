#!/usr/bin/env bash
# _run_90b_wrapped.sh — 90b 的日志落盘包装（输出回 Windows 侧便于读取）
set -o pipefail
LOG=/mnt/c/code/media_code/compare_4methods/_run_90b_wrapped.log
bash /mnt/c/code/media_code/compare_4methods/90b_run_3view_da3gs.sh > "$LOG" 2>&1
RC=$?
echo "WRAPPED_EXIT=$RC" >> "$LOG"
exit $RC

#!/usr/bin/env bash
# _run_31_wrapped.sh — audit_31 评测的日志落盘包装
set -o pipefail
LOG=/mnt/c/code/media_code/compare_4methods/_run_31_wrapped.log
bash /mnt/c/code/media_code/compare_4methods/audit_31_eval_3view.sh > "$LOG" 2>&1
RC=$?
echo "WRAPPED_EXIT=$RC" >> "$LOG"
exit $RC

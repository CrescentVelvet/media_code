#!/usr/bin/env bash
# 32v 同口径对照跑：audit_31 V32=1（五方法 32 视角产物，GT 侧与 3v 评测完全相同）
# 环境照 3v 成功路径：走 audit_31_eval_3view.sh（querysplat env），只注入 V32/OUT
set -uo pipefail
LOG=/mnt/c/code/media_code/compare_4methods/_run_31_32v_wrapped.log
export V32=1
export OUT=/home/velvet/output/batch_4methods/eval_32view_jiajia.json
bash /mnt/c/code/media_code/compare_4methods/audit_31_eval_3view.sh > "$LOG" 2>&1
RC=$?
echo "WRAPPED_EXIT=$RC" >> "$LOG"
exit $RC

#!/bin/bash
# 中间档统一评测（VMID=1）
source /mnt/c/code/media_code/compare_4methods/_env_eval.sh
VMID=1 OUT=/home/velvet/output/batch_4methods/eval_mid_jiajia.json \
  python compare_4methods/audit_31_eval_3view.py 2>&1 | tail -45
echo "SCRIPT_EXIT=${PIPESTATUS[0]}"

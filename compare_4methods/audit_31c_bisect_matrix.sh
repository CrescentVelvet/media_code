#!/usr/bin/env bash
# 逐级二分矩阵：STAGE 0→5，GNU timeout 420s 兜底，定位活锁层级
set -o pipefail

LOG=/mnt/c/code/media_code/compare_4methods/audit_31c_bisect.log
exec > "$LOG" 2>&1

source ~/miniconda3/etc/profile.d/conda.sh
conda activate querysplat
cd /mnt/c/code/media_code/compare_4methods

for STAGE in 0 1 2 3 4 5; do
    echo "########## STAGE=$STAGE ##########"
    START=$(date +%s)
    STAGE=$STAGE timeout 420 python3 audit_31c_bisect_render.py
    RC=$?
    ELAPSED=$(($(date +%s) - START))
    if [ $RC -eq 124 ]; then
        echo ">>> STAGE=$STAGE 超时（420s）—— 活锁在这一层或更前 ✅"
    elif [ $RC -ne 0 ]; then
        echo ">>> STAGE=$STAGE 报错 exit=$RC"
    else
        echo ">>> STAGE=$STAGE 通过（${ELAPSED}s）"
    fi
    echo
    sleep 3
done
echo "BISECT_DONE"

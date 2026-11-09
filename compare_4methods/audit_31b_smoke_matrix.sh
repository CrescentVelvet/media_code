#!/usr/bin/env bash
# 冒烟矩阵：MODE ∈ {orig, san, halfres} × GNU timeout 600s，定位 audit_31 活锁充要条件
set -o pipefail

LOG=/mnt/c/code/media_code/compare_4methods/audit_31b_smoke.log
exec > "$LOG" 2>&1

source ~/miniconda3/etc/profile.d/conda.sh
conda activate querysplat
cd /mnt/c/code/media_code/compare_4methods

for MODE in orig san halfres; do
    echo "########## MODE=$MODE ##########"
    START=$(date +%s)
    timeout 600 python3 audit_31b_smoke_render.py
    RC=$?
    ELAPSED=$(($(date +%s) - START))
    if [ $RC -eq 124 ]; then
        echo ">>> MODE=$MODE 超时（600s）—— 活锁复现 ✅"
    elif [ $RC -ne 0 ]; then
        echo ">>> MODE=$MODE 报错 exit=$RC（非活锁）"
    else
        echo ">>> MODE=$MODE 渲完（${ELAPSED}s）—— 该条件不是充要"
    fi
    echo
    sleep 5  # 让 GPU 显存回收
done
echo "MATRIX_DONE"

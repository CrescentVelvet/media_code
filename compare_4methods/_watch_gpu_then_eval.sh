#!/usr/bin/env bash
# GPU 看护：等峰哥的 bench（08_bench_official_gs.py / scenesplatter.py）退出后，
# 自动起 audit_31 三图全量评测。背景运行，日志落盘可直接 tail。
#
# 判据（两个都要满足）：
#   1) WSL 内无其它 python GPU 重任务（[x] 技巧避免匹配自身）
#   2) nvidia-smi 空闲显存 ≥ 3 GiB（防死进程残留占满）
# 每 120s 轮询；评测起完后本脚本自行退出。
set -o pipefail

LOG=/mnt/c/code/media_code/compare_4methods/_watch_gpu_then_eval.log
exec >> "$LOG" 2>&1

echo "[$(date '+%F %T')] watcher 启动，等 GPU 空出来…"

while true; do
    BUSY=$(pgrep -af 'python.*([0][8]_bench_official_gs|[s]cenesplatter)' | grep -v $PPID || true)
    USED_MIB=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
    FREE_MIB=$((24576 - USED_MIB))
    echo "[$(date '+%F %T')] used=${USED_MIB}MiB free=${FREE_MIB}MiB busy='${BUSY:-无}'"
    if [ -z "$BUSY" ] && [ "$FREE_MIB" -ge 3072 ]; then
        echo "[$(date '+%F %T')] ✅ 条件满足，起 audit_31 全量评测"
        source ~/miniconda3/etc/profile.d/conda.sh
        conda activate querysplat
        cd /mnt/c/code/media_code/compare_4methods
        bash _run_31_wrapped.sh
        echo "[$(date '+%F %T')] 评测退出码=$?，watcher 退出"
        exit 0
    fi
    sleep 120
done

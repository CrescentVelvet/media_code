#!/bin/bash
#SBATCH -J hold_gpu
#SBATCH -p h100a011
#SBATCH --gres=gpu:8
#SBATCH -c 16
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH -o /data/00950754/logs/hold_%j.out

# H100 保活任务（真实 matmul 负载，非空转）
#
# 通常不直接 sbatch 本文件，而是用同目录的 10_hold.sh：
#   ./10_hold.sh start            8 卡 / 24h / high / 自动链式续期
#   ./10_hold.sh start 4 12h      4 卡 / 12h
#   ./10_hold.sh stop             停掉
#
# 直接提交也行：
#   sbatch 10_hold_gpu.sbatch
#   sbatch --gres=gpu:4 --time=08:00:00 10_hold_gpu.sbatch
#
# 环境变量（注意必须带 HOLD_ 前缀——裸 SIZE/DTYPE 会被 conda 的 binutils
# 激活钩子覆盖成 x86_64-conda-linux-gnu-size，argparse 直接报错）：
#   HOLD_INTENSITY=high|mid|low   占空比 100/75/35%   （默认 high）
#   HOLD_SIZE=4096                matmul 方阵边长    （默认 4096）
#   HOLD_DTYPE=bf16|fp16|fp32                        （默认 bf16）
#   HOLD_MEM_GIB=0                额外显存垫片 GiB    （默认 0）
#   HOLD_INTERVAL=30              状态打印间隔秒      （默认 30）
#   HOLD_CHAIN=1                  到期前自动提交下一个作业（无缝续期）

set -eo pipefail
echo "job=$SLURM_JOB_ID node=$(hostname) start=$(date -Is)"
echo "分配到的卡: ${CUDA_VISIBLE_DEVICES:-(未设置)}   SLURM_GPUS_ON_NODE=${SLURM_GPUS_ON_NODE:-?}"

# 脚本目录：优先用 10_hold.sh / 10_hold_gpu.py 传下来的，其次提交目录
SCRIPT_DIR="${HOLD_SCRIPT_DIR:-${SLURM_SUBMIT_DIR:-/data/00950754/code/media_code/canada_h100}}"
echo "脚本目录: $SCRIPT_DIR"

source /data/00950754/envs/activate_splatanything.sh

INTENSITY="${HOLD_INTENSITY:-high}"
SIZE="${HOLD_SIZE:-4096}"
DTYPE="${HOLD_DTYPE:-bf16}"
MEM_GIB="${HOLD_MEM_GIB:-0}"
INTERVAL="${HOLD_INTERVAL:-30}"
CHAIN_ARGS=""
if [ "${HOLD_CHAIN:-0}" = "1" ]; then CHAIN_ARGS="--chain"; fi

mkdir -p /data/00950754/runs/hold
cd /data/00950754/runs/hold

python "$SCRIPT_DIR/10_hold_gpu.py" \
    --intensity "$INTENSITY" \
    --size "$SIZE" \
    --dtype "$DTYPE" \
    --mem-gib "$MEM_GIB" \
    --interval "$INTERVAL" \
    --rundir "/data/00950754/runs/hold/job_${SLURM_JOB_ID}" \
    $CHAIN_ARGS

echo "end=$(date -Is)"

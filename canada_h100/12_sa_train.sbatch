#!/bin/bash
#SBATCH -J sa_train
#SBATCH -p h100a011
#SBATCH --gres=gpu:8
#SBATCH -c 16
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH -o /data/00950754/logs/sa_train_%j.out

# SplatAnything 常驻训练（真训练，无限 iter，吃满显存与算力）
#
# 通常不直接 sbatch 本文件，而用同目录的 12_sa_train.sh：
#   ./12_sa_train.sh start            8 卡 / 24h / 全参微调 / 无限 iter / 自动续期
#   ./12_sa_train.sh start 4 12h      4 卡 / 12h
#   ./12_sa_train.sh start 8 24h real ./12_sa_train.sh stop
#
# 直接提交也行：
#   sbatch 12_sa_train.sbatch
#   sbatch --gres=gpu:4 --time=08:00:00 12_sa_train.sbatch
#
# 环境变量（统一加 SA_ 前缀，避免与 conda 的激活钩子撞名）：
#   SA_SCENE=synthetic       synthetic = 程序化 360° 环绕场景（无数据也能真训）
#                            或真实场景 images 目录
#   SA_RES=504               训练分辨率长边（必须是 14 的倍数）
#   SA_NVIEWS=8              每步视角数
#   SA_ITERS=0               0 = 无限（默认）
#   SA_LR=1e-5               学习率
#   SA_SCOPE=all             all = 全参微调；gshead = 只训解码头（冒烟）
#   SA_MEM_RATIO=0.85        目标显存占用比
#   SA_CHAIN=1               到期前自动提交下一个作业
#   SA_RUNDIR=               输出目录（续期时由上一个作业传下来，保持同一目录）

set -eo pipefail
echo "job=$SLURM_JOB_ID node=$(hostname) start=$(date -Is)"
echo "分配到的卡: ${CUDA_VISIBLE_DEVICES:-(未设置)}   SLURM_GPUS_ON_NODE=${SLURM_GPUS_ON_NODE:-?}"

SCRIPT_DIR="${SA_SCRIPT_DIR:-${SLURM_SUBMIT_DIR:-/data/00950754/code/media_code/canada_h100}}"
echo "脚本目录: $SCRIPT_DIR"

source /data/00950754/envs/activate_splatanything.sh

SCENE="${SA_SCENE:-synthetic}"
RES="${SA_RES:-504}"
NVIEWS="${SA_NVIEWS:-8}"
ITERS="${SA_ITERS:-0}"
LR="${SA_LR:-1e-5}"
SCOPE="${SA_SCOPE:-all}"
MEM_RATIO="${SA_MEM_RATIO:-0.85}"
INTERVAL="${SA_INTERVAL:-0}"

# 全参训练（9.4 亿参数）的梯度与 AdamW 状态是巨型块分配，极易撞碎片。
# OOM 报错本身也建议开这个：分割段落由驱动统一管理，省掉大量空洞。
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

EXTRA=()
[ -n "${SA_RUNDIR:-}" ] && EXTRA+=(--rundir "$SA_RUNDIR")
[ "${SA_CHAIN:-0}" = "1" ] && EXTRA+=(--chain)
[ "${SA_POSE:-0}" = "1" ] && EXTRA+=(--pose-mode)

# 权重缺失时自动补齐（只在计算节点需读权重时才跑，登录节点也能跑）
if [ ! -f /data/00950754/models/splatanything/SplatAnything.ckpt ]; then
    echo "⚠️ 权重缺失，先跑 00_setup_sa.sh 下载（16 GB）"
    exit 1
fi

mkdir -p /data/00950754/runs/sa_train
cd /data/00950754/code/media_code/canada_h100

python "$SCRIPT_DIR/12_sa_train.py" \
    --scene "$SCENE" \
    --res "$RES" \
    --n-views "$NVIEWS" \
    --iters "$ITERS" \
    --lr "$LR" \
    --train-scope "$SCOPE" \
    --mem-ratio "$MEM_RATIO" \
    --interval "$INTERVAL" \
    "${EXTRA[@]}"

echo "end=$(date -Is)"

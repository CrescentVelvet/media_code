#!/bin/bash
#SBATCH -J sa_train
#SBATCH -p h100a011
#SBATCH --gres=gpu:8
#SBATCH -c 16
#SBATCH --mem=128G
#SBATCH --time=168:00:00
#SBATCH -o /data/00950754/logs/sa_train_%j.out

# SplatAnything 单节点多卡训练：torchrun 起 N 个进程，NCCL 数据并行。
#
# 通常不直接 sbatch 本文件，而用同目录的 12_sa_train.sh：
#   ./12_sa_train.sh start             8 卡 / 168h / 全参 / 25 万步
#   ./12_sa_train.sh start 4 168h      4 卡 / 168h
#   ./12_sa_train.sh scene <场景目录>   指定单场景
#   ./12_sa_train.sh stop
#
# 环境变量（SA_ 前缀）：
#   SA_SCENE=       单场景 images 目录，给了就全卡共用
#   SA_SCENES=      逗号分隔的多个场景目录；空 = examples/ 下 6 个按 rank 轮转
#   SA_RES=504      训练分辨率长边（14 的倍数）
#   SA_NVIEWS=8     每步视角数
#   SA_ROTATE=500   每 N 步轮换视角窗口
#   SA_ITERS=250000 总步数；0 = 不限
#   SA_LR=1e-5      学习率
#   SA_SCOPE=all    all = 全参微调；gshead = 只训解码头
#   SA_RUNDIR=      输出目录（缺省按作业号新建）

set -eo pipefail
echo "job=$SLURM_JOB_ID node=$(hostname) start=$(date -Is)"
echo "分配到的卡: ${CUDA_VISIBLE_DEVICES:-(未设置)}   SLURM_GPUS_ON_NODE=${SLURM_GPUS_ON_NODE:-?}"

SCRIPT_DIR="${SA_SCRIPT_DIR:-${SLURM_SUBMIT_DIR:-/data/00950754/code/media_code/canada_h100}}"
echo "脚本目录: $SCRIPT_DIR"

source /data/00950754/envs/activate_splatanything.sh

RUNDIR="${SA_RUNDIR:-/data/00950754/runs/sa_train/job_${SLURM_JOB_ID}}"
mkdir -p "$RUNDIR"

# 全参训练（9.4 亿参数）的梯度与 AdamW 状态是巨型块分配，分段由驱动统一管理
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"

ARGS=(
    --res "${SA_RES:-504}"
    --n-views "${SA_NVIEWS:-8}"
    --rotate-every "${SA_ROTATE:-500}"
    --iters "${SA_ITERS:-250000}"
    --lr "${SA_LR:-1e-5}"
    --train-scope "${SA_SCOPE:-all}"
    --seed "${SA_SEED:-1234}"
    --rundir "$RUNDIR"
    --ckpt-every "${SA_CKPT_EVERY:-20000}"
)
[ -n "${SA_SCENE:-}" ] && ARGS+=(--scene "$SA_SCENE")
[ -n "${SA_SCENES:-}" ] && ARGS+=(--scenes "$SA_SCENES")
[ "${SA_POSE:-0}" = "1" ] && ARGS+=(--pose-mode)
[ "${SA_RESUME:-0}" = "1" ] && ARGS+=(--resume)

NGPU="${SLURM_GPUS_ON_NODE:-8}"
if ! [[ "$NGPU" =~ ^[0-9]+$ ]]; then
    NGPU=8
fi

if [ ! -f /data/00950754/models/splatanything/SplatAnything.ckpt ]; then
    echo "❌ 权重缺失，先跑 00_setup_env.sh 下载"
    exit 1
fi

echo "启动 torchrun: ${NGPU} 进程  rundir=$RUNDIR  iters=${SA_ITERS:-250000}"
cd /data/00950754/code/media_code/canada_h100

# 单节点独立启动：--standalone 让 torchrun 自建 rendezvous，不依赖外部服务
torchrun --standalone --nnodes=1 --nproc_per_node="$NGPU" \
    "$SCRIPT_DIR/12_sa_train.py" "${ARGS[@]}"

echo "end=$(date -Is)"

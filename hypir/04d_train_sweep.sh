#!/usr/bin/env bash
# 04d_train_sweep.sh — 超参扫描训练包装：命令行传 LR_G(+可选 KEY=VAL 覆盖)，自动按超参
# 组合命名实验、后台调 04b 跑一个实验。便于一晚上并行跑多个 LR_G/数据集组合找最佳参数。
#
# 背景：04b 默认 LR_G=1e-5、steps=300000、ckpt_every=500，练久了 L2 坍缩会越练越糊
# (发布模型的 GAN/LPIPS 锐度先验被稀释)。本脚本默认 LR_G 透传、steps=30000、ckpt_every=100，
# 练完逐 checkpoint 评测找峰值(常在早期，如几百~几千步)，而非取终点。
#
# 用法(一个命令 = 一个后台实验，换 GPU/PARQUET/LR_G 跑多条即并行)：
#   前提：已 conda activate 你的训练 env（脚本不再切 env）。
#   GPU=0 PARQUET_PATH=.../rest.parquet        bash hypir/04d_train_sweep.sh 5e-6
#   GPU=1 PARQUET_PATH=.../rest.parquet        bash hypir/04d_train_sweep.sh 2e-6
#   GPU=2 PARQUET_PATH=.../rest_beauty.parquet bash hypir/04d_train_sweep.sh 5e-6
#   GPU=3 PARQUET_PATH=.../rest.parquet SWEEP_TAG=disc1e5 LR_D=1e-5 bash hypir/04d_train_sweep.sh 5e-6
#
# 参数：
#   $1 = LR_G (必填，如 5e-6 / 2e-6 / 1e-5)
# 环境变量(可选，未设则用下方默认；其余 04b 参数如 BATCH_SIZE/GRAD_ACCUM/CROP_TYPE/
#   OUT_SIZE/RESUME/CHECKPOINTS_TOTAL_LIMIT/N_TRAIN_GPU/LORA_WEIGHT_PATH 均透传给 04b)：
#   PARQUET_PATH        必填(用哪个 parquet：rest.parquet=A基线 / rest_beauty.parquet=B美颜)
#   MAX_TRAIN_STEPS     默认 30000
#   CHECKPOINTING_STEPS 默认 100(每100步存一个，便于找峰值)
#   LR_D                默认 = LR_G(D/G 同步；想让 D 固定 1e-5 就 LR_D=1e-5)
#   GPU / BG            默认 GPU=0 / BG=1(后台 nohup，一晚并行跑多个靠这个)
#   SWEEP_TAG           可选，加到实验名后(如 disc1e5 / bs8)
#
# 实验名自动 = experiments/<parquet名>_lrg<LR_G>[_<SWEEP_TAG>]，如 rest_lrg5e-6、rest_beauty_lrg2e-6_disc1e5
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
HYPIR_DIR="${HYPIR_DIR:-$REPO_DIR/../HYPIR}"

LR_G="${1:?用法: CONDA_ENV=hypir GPU=0 PARQUET_PATH=.../rest.parquet bash hypir/04d_train_sweep.sh <LR_G> (如 5e-6)}"
: "${PARQUET_PATH:?set PARQUET_PATH (如 .../rest.parquet 或 .../rest_beauty.parquet)}"
[ -f "$PARQUET_PATH" ] || { echo "ERROR: parquet not found: $PARQUET_PATH" >&2; exit 1; }
PARQUET_PATH="$(cd "$(dirname "$PARQUET_PATH")" && pwd)/$(basename "$PARQUET_PATH")"

# parquet 文件名(去 .parquet)当实验名前缀：rest / rest_beauty
PQ_STEM="$(basename "$PARQUET_PATH" .parquet)"

# 扫描默认(覆盖 04b 的 300000/500/1e-5)
export MAX_TRAIN_STEPS="${MAX_TRAIN_STEPS:-30000}"
export CHECKPOINTING_STEPS="${CHECKPOINTING_STEPS:-100}"
export LR_D="${LR_D:-$LR_G}"              # D/G 同步；想固定 D 就 LR_D=1e-5
GPU="${GPU:-0}"
BG="${BG:-1}"                             # 后台 nohup，便于一终端拉多个
SWEEP_TAG="${SWEEP_TAG:-}"

# 实验名 + 输出目录(自动按超参组合命名)
EXP_NAME="${PQ_STEM}_lrg${LR_G}"
[ -n "$SWEEP_TAG" ] && EXP_NAME="${EXP_NAME}_${SWEEP_TAG}"
OUTPUT_DIR="$HYPIR_DIR/experiments/$EXP_NAME"
export DATASET="$EXP_NAME"                # 让 04b 的日志文件名也带上实验名

echo "=== [04d] sweep train (wraps 04b) ==="
echo "  parquet:     $PARQUET_PATH"
echo "  实验名:      $EXP_NAME"
echo "  OUTPUT_DIR:  $OUTPUT_DIR"
echo "  params:      steps=$MAX_TRAIN_STEPS ckpt_every=$CHECKPOINTING_STEPS lr_G=$LR_G lr_D=$LR_D bs=${BATCH_SIZE:-6} GPU=$GPU BG=$BG"
[ -n "$SWEEP_TAG" ] && echo "  tag:         $SWEEP_TAG"
echo "  ⚠ 每 $CHECKPOINTING_STEPS 步存 1 个 × $MAX_TRAIN_STEPS 步 ≈ $((MAX_TRAIN_STEPS/CHECKPOINTING_STEPS)) 个 checkpoint；"
echo "    盘紧就设 CHECKPOINTS_TOTAL_LIMIT=N(留最近 N 个)，但找峰值会丢早期点——本扫描就是为找早期峰值，慎用。"
echo "  → 后台拉起后回终端；日志见 04b 打印的 LOG_FILE。"

export PARQUET_PATH OUTPUT_DIR LR_G
GPU="$GPU" BG="$BG" bash "$SCRIPT_DIR/04b_train_paired.sh"

echo "=== [04d] launched: $EXP_NAME (GPU=$GPU) ==="
echo "    跟踪: tail -f $OUTPUT_DIR/*-LoRA_*.log"
echo "    找峰值: 对几个 checkpoint 跑 05_eval(CKPT_STEP=...) 或 02(WEIGHT_PATH=.../checkpoint-N/state_dict.pth)"
echo "         常在早期(几百~几千步)，别只看终点 30000。"

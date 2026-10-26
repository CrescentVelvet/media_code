#!/usr/bin/env bash
# 02_train.sh — 训练 splannequin（双检测时间锚定正则）动态高斯。
#
# 默认 splannequin 模式（带 hidden/defective 锚定）；
# BASELINE=1 跑 4DGaussians 基线（arguments/baseline.py，用于消融对照）。
#
# 输出：$EXP_DIR/<mode>/<scene_name>/（官方默认写仓内 ./exp，这里外置）
set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/_env.sh"

SCENE_DIR="${SCENE_DIR:?❌ SCENE_DIR not set (dir with images/ + sparse/, run 01 first)}"
SCENE_NAME="${SCENE_NAME:-$(basename "$SCENE_DIR")}"
if [ "${BASELINE:-0}" = "1" ]; then TRAIN_MODE="baseline"; else TRAIN_MODE="splannequin"; fi
# CONFIG_FILE：覆盖默认配置（如加速方案 A 的半量 schedule，须为 WSL 内可读路径）
CONFIG_FILE="${CONFIG_FILE:-./arguments/${TRAIN_MODE}.py}"
PORT="${PORT:-6017}"

if [ ! -d "$SCENE_DIR/sparse" ]; then
    echo "❌ ERROR: $SCENE_DIR/sparse not found — run 01_colmap.sh first" >&2
    exit 1
fi

OUT_DIR="$EXP_DIR/$TRAIN_MODE/$SCENE_NAME"
mkdir -p "$OUT_DIR"

echo "🚀 [02] Training ($TRAIN_MODE)"
echo "  📁 scene:  $SCENE_DIR"
echo "  💾 output: $OUT_DIR"
echo "  🎮 GPU:    ${CUDA_VISIBLE_DEVICES:-auto}"
export PYTORCH_ALLOC_CONF="max_split_size_mb:512"

cd "$SPLANNEQUIN_DIR"
python train.py \
    -s "$SCENE_DIR" \
    -m "$OUT_DIR" \
    --port "$PORT" \
    --eval \
    --train_mode "$TRAIN_MODE" \
    --resolution "${RESOLUTION:-1}" \
    --expname "${SCENE_NAME}_${TRAIN_MODE}" \
    --configs "$CONFIG_FILE" \
    $EXTRA_TRAIN_ARGS
if [ $? -ne 0 ]; then
    echo "❌ FAILED" >&2
    exit 1
fi
echo "🎉 Done. Model at $OUT_DIR"
echo "  Next: SCENE_DIR=$SCENE_DIR bash $SCRIPT_DIR/03_render.sh"

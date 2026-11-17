#!/usr/bin/env bash
# 03_layer_ablation.sh — SHARP 两层深度层数消融（三图计划 S0.5）
#
# 不改权重、不训练：把 composer 输出的 2 x 768^2 高斯按层切开，比较
#   full（完整两层） / layer0（只留前表面层） / layer1（只留被遮挡层）
# 在 source 视角（extrinsic=I）下的 PSNR / alpha 覆盖 / 层间深度差。
#
# 这是「第二层到底贡献了什么」的最便宜检验，不依赖任何位姿估计。
#
# 用法：
#   GPU=0 INPUT=/path/to/portrait.heic bash sharp/03_layer_ablation.sh
#   GPU=0 INPUT=/path/to/a.jpg OUT_DIR=~/output/sharp_ablation/xxx bash sharp/03_layer_ablation.sh
set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/_env.sh"

INPUT="${INPUT:-}"
OUT_DIR="${OUT_DIR:-$RESULTS_DIR/03_layer_ablation}"
RENDER_SCALE="${RENDER_SCALE:-1.0}"

if [ -z "$INPUT" ]; then
    echo "❌ ERROR: INPUT 未设置（单张图）" >&2
    echo "   例: GPU=0 INPUT=/mnt/d/dataset/x.heic bash sharp/03_layer_ablation.sh" >&2
    exit 1
fi
if [ ! -e "$INPUT" ]; then
    echo "❌ ERROR: INPUT 不存在: $INPUT" >&2
    exit 1
fi
if [ ! -f "$SHARP_CKPT" ]; then
    echo "❌ ERROR: 权重不存在: $SHARP_CKPT（先跑 01_download_models.sh）" >&2
    exit 1
fi

echo "🚀 [03] SHARP 两层深度层数消融"
echo "  🖼️ 输入:   $INPUT"
echo "  💾 输出:   $OUT_DIR"
echo "  🏋️ 权重:   $SHARP_CKPT"
echo "  🎮 GPU:    ${CUDA_VISIBLE_DEVICES:-all}"
echo ""

_t0=$(date +%s)
python "$SCRIPT_DIR/03_layer_ablation.py" \
    --input "$INPUT" \
    --out "$OUT_DIR" \
    --ckpt "$SHARP_CKPT" \
    --render-scale "$RENDER_SCALE"
_rc=$?
_t1=$(date +%s)

if [ $_rc -ne 0 ]; then
    echo "" >&2
    echo "❌ FAILED (exit $_rc)" >&2
    exit 1
fi

echo ""
echo "⏱️  总耗时: $((_t1 - _t0))s"
echo "🎉 [03] Done."

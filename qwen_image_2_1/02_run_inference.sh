#!/usr/bin/env bash
# 02_run_inference.sh — Qwen-Image-2.1 推理（t2i / edit / rgba 三模式）。
set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/_env.sh"

# 参数（全用 ${VAR:-default}）
export MODEL_PATH="${MODEL_PATH:-/mnt/d/wheel/qwen_image21_ms}"
export MODE="${MODE:-t2i}"
export PROMPT="${PROMPT:-A neon shop sign that reads \"QWEN IMAGE 2.1\", rainy night, reflections on wet pavement}"
export IMAGES="${IMAGES:-}"
export ASPECT="${ASPECT:-1:1}"
export STEPS="${STEPS:-40}"
export SEED="${SEED:-42}"
export CPU_OFFLOAD="${CPU_OFFLOAD:-1}"
export RESULTS_DIR="${RESULTS_DIR:-$HOME/output/qwen_image_2_1_results}"
export OUTPUT="${OUTPUT:-$RESULTS_DIR/${MODE}_$(date +%Y%m%d_%H%M%S).png}"

# 前置检查
if [ ! -f "$MODEL_PATH/model_index.json" ]; then
    echo "❌ ERROR: 模型未就绪: $MODEL_PATH（缺 model_index.json，先跑 01_check_models.sh）" >&2
    exit 1
fi
if [ "$MODE" = "edit" ] && [ -z "$IMAGES" ]; then
    echo "❌ ERROR: edit 模式必须给 IMAGES（逗号分隔的输入图路径）" >&2
    exit 1
fi
mkdir -p "$RESULTS_DIR"

echo "🚀 [02] Qwen-Image-2.1 推理"
echo "  🤖 模型: $MODEL_PATH"
echo "  🎯 模式: $MODE"
echo "  📝 prompt: $PROMPT"
[ -n "$IMAGES" ] && echo "  🖼️ 输入: $IMAGES"
echo "  💾 输出: $OUTPUT"

python "$SCRIPT_DIR/run_inference.py"
if [ $? -ne 0 ]; then
    echo "❌ FAILED" >&2
    exit 1
fi
echo "🎉 Done."

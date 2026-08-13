#!/usr/bin/env bash
# 01_download_models.sh — 下载 World-R1 全部模型权重。
#
# 下载内容:
#   1. Wan2.1-T2V-14B-Diffusers  (基础视频模型, ~28GB; 训练 + 推理)
#   2. DA3-GIANT                 (Depth Anything 3, 3D reward 重建; HF cache)
#   3. Qwen3-VL-4B-Instruct      (reward scorer; HF cache)
#   LPIPS + HPSv2 权重在 reward server 首次启动时自动下载 (小文件)。
#
# 用 MODEL_FAMILY=wan_small 可下 1.3B 版本 (显存不够时)。
# 用 MODEL_FAMILY=cogvideox 可下 CogVideoX1.5-5B。
set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/_env.sh"

MODEL_FAMILY="${MODEL_FAMILY:-wan_large}"

echo "🚀 [01] 下载 World-R1 模型权重"
echo "  🤖 MODEL_FAMILY: $MODEL_FAMILY"
echo "  🏋️ MODEL_DIR:    $MODEL_DIR"
echo "  📁 WAN_MODEL_PATH: ${WAN_MODEL_PATH:-$MODEL_DIR/<auto>}"
echo "  📦 HF cache:       $HUGGINGFACE_HUB_CACHE"
echo ""

# 下载时关掉离线模式 (01 负责下载, 之后训练/推理走离线)
export HF_HUB_OFFLINE=0
export TRANSFORMERS_OFFLINE=0

python "$SCRIPT_DIR/download_models.py"
if [ $? -ne 0 ]; then
    echo "❌ download_models.py failed" >&2
    echo ""
    echo "🔧 手动下载 (代理封 HF 时):"
    echo "   1) Wan2.1:   https://huggingface.co/Wan-AI/Wan2.1-T2V-14B-Diffusers"
    echo "      下到 $WAN_MODEL_PATH (需有 model_index.json)"
    echo "   2) DA3:      https://huggingface.co/depth-anything/DA3-GIANT"
    echo "      用 huggingface-cli download depth-anything/DA3-GIANT --cache-dir $HUGGINGFACE_HUB_CACHE"
    echo "   3) Qwen3-VL: https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct"
    echo "      用 huggingface-cli download Qwen/Qwen3-VL-4B-Instruct --cache-dir $HUGGINGFACE_HUB_CACHE"
    echo "   或用 ModelScope: pip install modelscope; from modelscope import snapshot_download"
    exit 1
fi

echo ""
echo "🎉 [01] Done. 下一步:"
echo "   训练: bash $SCRIPT_DIR/03_run_training.sh"
echo "   推理: bash $SCRIPT_DIR/04_run_inference.sh"

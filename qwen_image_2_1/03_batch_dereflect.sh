#!/usr/bin/env bash
# 03_batch_dereflect.sh — Qwen-Image-2.1 批量去除图像反光（环绕拍摄数据集预处理）。
#
# 用法（WSL）：
#   bash qwen_image_2_1/03_batch_dereflect.sh                # 全量 574 张
#   LIMIT=2 bash qwen_image_2_1/03_batch_dereflect.sh        # 冒烟：只跑前 2 张
#   START=100 LIMIT=50 bash qwen_image_2_1/03_batch_dereflect.sh  # 分批
# 断点续跑：输出已存在则自动跳过，直接重跑同一命令即可。
set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/_env.sh"

# 参数（全用 ${VAR:-default}，可被外部覆盖）
export MODEL_PATH="${MODEL_PATH:-/mnt/d/wheel/qwen_image21_ms}"
export INPUT_DIR="${INPUT_DIR:-/mnt/d/dataset/物体反光数据/耳机白/image}"
export OUTPUT_DIR="${OUTPUT_DIR:-/mnt/d/output/qwen_image_2_1_dereflect/耳机白/image}"
export GLOB="${GLOB:-*.jpg}"
# 默认 prompt：去反光，同时强约束「除反光外一切不变」（重建需要跨视角一致性）
export PROMPT="${PROMPT:-去除图像中物体表面的镜面反光、眩光和光斑，恢复被反光遮挡的真实表面纹理和颜色。严格保持物体的形状、位置、拍摄视角、背景和其他所有内容完全不变，输出一张干净、无反光、写实的照片。}"
export STEPS="${STEPS:-40}"
export SEED="${SEED:-42}"
export CPU_OFFLOAD="${CPU_OFFLOAD:-1}"
export START="${START:-0}"
export LIMIT="${LIMIT:-0}"
export SAMPLE="${SAMPLE:-0}"
export STRIDE="${STRIDE:-1}"
export RESIZE_BACK="${RESIZE_BACK:-1}"
export JPG_QUALITY="${JPG_QUALITY:-98}"

# 前置检查
if [ ! -f "$MODEL_PATH/model_index.json" ]; then
    echo "❌ ERROR: 模型未就绪: $MODEL_PATH（先跑 01_check_models.sh）" >&2
    exit 1
fi
if [ ! -d "$INPUT_DIR" ]; then
    echo "❌ ERROR: 输入目录不存在: $INPUT_DIR" >&2
    exit 1
fi
mkdir -p "$OUTPUT_DIR"

echo "🚀 [03] Qwen-Image-2.1 批量去反光"
echo "  📂 $INPUT_DIR  →  $OUTPUT_DIR"

python "$SCRIPT_DIR/batch_dereflect.py"
if [ $? -ne 0 ]; then
    echo "❌ FAILED" >&2
    exit 1
fi
echo "🎉 All done."

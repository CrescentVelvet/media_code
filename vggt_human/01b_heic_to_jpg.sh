#!/usr/bin/env bash
# 01b_heic_to_jpg.sh — 将 HEIC/HEIF 图像批量转换为 JPG（喂给 step 02 用）。
# iPhone 拍的照片默认 .heic，VGGT-Omega / mediapipe 不认，需先转 JPG。
#
# 用法：
#   GPU=0 INPUT_DIR=/mnt/d/dataset/sample/image \
#     bash vggt_human/01b_heic_to_jpg.sh
#
# 输出：INPUT_DIR 同级 image_jpg/（默认；可用 OUTPUT_DIR 覆盖）
set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/_env.sh"

INPUT_DIR="${INPUT_DIR:-/mnt/d/dataset/测试数据sample/image}"
OUTPUT_DIR="${OUTPUT_DIR:-$(dirname "$INPUT_DIR")/image_jpg}"

export INPUT_DIR OUTPUT_DIR

echo "🚀 [01b] HEIC -> JPG 转换"
echo "  📁 输入: $INPUT_DIR"
echo "  💾 输出: $OUTPUT_DIR"

if [ ! -d "$INPUT_DIR" ]; then
    echo "❌ ERROR: INPUT_DIR not found: $INPUT_DIR" >&2
    exit 1
fi

python "$SCRIPT_DIR/heic_to_jpg.py"
if [ $? -ne 0 ]; then
    echo "❌ FAILED" >&2
    exit 1
fi

echo ""
echo "✅ [01b] Done. Next:"
echo "  GPU=0 INPUT_DIR=$OUTPUT_DIR \\"
echo "    MODEL_DIR=~/model/VGGT-Omega \\"
echo "    RESULTS_DIR=~/output/vggt_human_results \\"
echo "    bash vggt_human/02_run_inference.sh"

#!/usr/bin/env bash
# 07_rotate.sh — 生成 360° 旋转视频（MiniMax-H3，diffusers 直接跑，不起服务）。
#
# 参考 wan22_rotate 的思路，但 MiniMax-H3 不需要 LoRA——prompt 自带 360° 旋转
# 指令（H3-Context-IR 格式）。用 diffusers 的 ModularPipeline 直接跑，无需起服务。
#
# 两种输入方式：
#   1) 纯文生旋转（无图，prompt 描述主体旋转一圈）
#   2) 首帧生旋转（传入一张图作首帧，从首帧开始绕主体旋转一圈）
#      图可以是原始拍摄图 / 分割白底图 / 任意主体图
#
# Output: $OUTPUT_DIR/rotate_360.mp4
set -o pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# 默认旋转 prompt（H3-Context-IR 格式）
PROMPT_FILE="${PROMPT_FILE:-$SCRIPT_DIR/examples/rotate_prompt.txt}"

# 首帧图（传了就 fl2va，没传就 t2va）
FIRST_FRAME="${FIRST_FRAME:-}"

DURATION="${DURATION:-10}"            # 10s 足够一圈
SHORT_EDGE="${SHORT_EDGE:-768}"
SEED="${SEED:-0}"
OUTPUT_DIR="${OUTPUT_DIR:-../MiniMax-H3/results/rotate}"
OUTPUT_NAME="${OUTPUT_NAME:-rotate_360.mp4}"

if [ -n "$FIRST_FRAME" ]; then
    TASK=fl2va
    echo "🖼️ 首帧: $FIRST_FRAME"
else
    TASK=t2va
    echo "📝 纯文生旋转"
fi

GPU="${GPU:-0,1}" \
MODEL_PATH="${MODEL_PATH:-../../model/MiniMax-H3}" \
TRANSFORMER_DEVICE="${TRANSFORMER_DEVICE:-cuda:0}" \
TEXT_ENCODER_TRANSFORMER_DEVICE="${TEXT_ENCODER_TRANSFORMER_DEVICE:-cuda:1}" \
TASK="$TASK" \
PROMPT_FILE="$PROMPT_FILE" \
FIRST_FRAME="$FIRST_FRAME" \
NUM_FRAMES="$(python -c "print(int(${DURATION})*24+5)")" \
SEED="$SEED" \
OUTPUT_DIR="$OUTPUT_DIR" OUTPUT_NAME="$OUTPUT_NAME" \
bash "$SCRIPT_DIR/06_diffusers_inference.sh"

#!/usr/bin/env bash
# run_rotate.sh — 生成 360° 旋转视频（MiniMax-H3）。
#
# 参考 wan22_rotate 的思路（选正面图+分割 → Wan2.2+LoRA 旋转），但 MiniMax-H3
# 不需要 LoRA——prompt 自带 360° 旋转指令。支持三种输入方式（按需选一）：
#
#   1) t2va  纯文生旋转（无图，prompt 描述主体旋转）
#   2) fl2va 首帧生旋转（传入一张图作首帧，从首帧开始绕主体旋转一圈）
#   3) ref2va 参考生旋转（传入参考图，<Picture 1> 的主体 360° 旋转）
#
# 输入图可以是：原始拍摄图 / 人体分割白底图 / 任意主体图——MiniMax-H3 能理解
# 各种输入（不像 Wan2.2 需要 LoRA 训练 + 白底分割图）。
#
# Prereq: 服务已起（FL2VA :30010 或 Ref2VA :30011）。
# Output: ../MiniMax-H3/results/rotate/rotate_360.mp4
set -o pipefail
EX_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# 默认旋转 prompt（H3-Context-IR 格式；ref2va 用 rotate_ref_prompt.txt）
PROMPT_FILE="${PROMPT_FILE:-$EX_DIR/rotate_prompt.txt}"

# 输入图（按优先级决定 task）：
#   REF_IMAGES  -> ref2va（参考图，:30011）
#   FIRST_FRAME -> fl2va（首帧，:30010）
#   都没传      -> t2va（纯文，:30010）
REF_IMAGES="${REF_IMAGES:-}"
FIRST_FRAME="${FIRST_FRAME:-}"

DURATION="${DURATION:-10}"            # 10s 足够一圈
ASPECT_RATIO="${ASPECT_RATIO:-}"       # t2va=16:9 / fl2va·ref2va=auto
SHORT_EDGE="${SHORT_EDGE:-768}"
SEED="${SEED:-0}"
OUTPUT_NAME="${OUTPUT_NAME:-rotate_360.mp4}"

if [ -n "$REF_IMAGES" ]; then
    # 3) ref2va：参考图 + 旋转 prompt（引用 <Picture 1>）
    SERVER_URL="${SERVER_URL:-http://localhost:30011}" \
    TASK=ref2va \
    PROMPT_FILE="$EX_DIR/rotate_ref_prompt.txt" \
    REF_IMAGES="$REF_IMAGES" \
    DURATION="$DURATION" SHORT_EDGE="$SHORT_EDGE" SEED="$SEED" \
    OUTPUT_NAME="$OUTPUT_NAME" \
    bash "$EX_DIR/../03_generate.sh"
elif [ -n "$FIRST_FRAME" ]; then
    # 2) fl2va：首帧图 + 旋转 prompt
    SERVER_URL="${SERVER_URL:-http://localhost:30010}" \
    TASK=fl2va \
    FIRST_FRAME="$FIRST_FRAME" \
    PROMPT_FILE="$PROMPT_FILE" \
    DURATION="$DURATION" SHORT_EDGE="$SHORT_EDGE" SEED="$SEED" \
    OUTPUT_NAME="$OUTPUT_NAME" \
    bash "$EX_DIR/../03_generate.sh"
else
    # 1) t2va：纯文生旋转视频（无图）
    SERVER_URL="${SERVER_URL:-http://localhost:30010}" \
    TASK=t2va \
    PROMPT_FILE="$PROMPT_FILE" \
    DURATION="$DURATION" ASPECT_RATIO="16:9" \
    SHORT_EDGE="$SHORT_EDGE" SEED="$SEED" \
    OUTPUT_NAME="$OUTPUT_NAME" \
    bash "$EX_DIR/../03_generate.sh"
fi

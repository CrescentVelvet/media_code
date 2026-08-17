#!/usr/bin/env bash
# 08_context_ir.sh — 调 MiniMax H3-Context-IR API 把短 prompt 转长描述，打印出来。
#
# H3-Context-IR 是 MiniMax 的 hosted API（非开源），把自由短 prompt + 可选首帧图转成
# 结构化长描述（integrated_multimodal_description / overall_soundscape / ...），
# 效果比短 prompt 好（官方推荐）。本脚本调 API 拿长描述后打印 + 写临时文件，
# 你自己复制命令跑 06 生成视频（不自动调 06，方便先看 prompt 内容）。
#
# ⚠️ 需要 MiniMax API token（platform.minimaxi.com CN / platform.minimax.io Global 申请）
# ⚠️ FIRST_FRAME 必须是 http URL（API 不读本地路径；本地图先上传到公网）
#
# Usage:
#   MINIMAX_API_KEY=xxx \
#     PROMPT="a drone shot over alpine peaks" \
#     FIRST_FRAME=https://example.com/subject.png \
#     bash minimax_h3/08_context_ir.sh
set -o pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/_env.sh"

if [ -z "${MINIMAX_API_KEY:-}" ]; then
    echo "❌ ERROR: MINIMAX_API_KEY not set (apply at platform.minimaxi.com)" >&2; exit 1
fi
if [ -z "${PROMPT:-}" ]; then
    echo "❌ ERROR: PROMPT not set (short prompt to convert to H3-Context-IR)" >&2; exit 1
fi

# 调 08_context_ir.py 拿长描述（stdout 只输出长描述，调试信息到 stderr）
LONG_PROMPT=$(python "$SCRIPT_DIR/08_context_ir.py")
if [ $? -ne 0 ]; then
    echo "❌ ERROR: H3-Context-IR API call failed" >&2; exit 1
fi
if [ -z "$LONG_PROMPT" ]; then
    echo "❌ ERROR: got empty prompt from H3-Context-IR" >&2; exit 1
fi

# 写到临时文件（方便用 PROMPT_FILE 传给 06）
PROMPT_FILE="${PROMPT_FILE:-/tmp/h3_context_ir_prompt.txt}"
printf '%s' "$LONG_PROMPT" > "$PROMPT_FILE"

# 打印长描述
echo ""
echo "════════════════════════════════════════════════════════════════"
echo "✅ H3-Context-IR prompt (${#LONG_PROMPT} chars) -> $PROMPT_FILE"
echo "════════════════════════════════════════════════════════════════"
echo ""
echo "$LONG_PROMPT"
echo ""
echo "════════════════════════════════════════════════════════════════"
echo "复制下面命令跑 06 生成视频（按需改 GPU / FIRST_FRAME / OUTPUT_DIR）："
echo "════════════════════════════════════════════════════════════════"
echo ""
# 构建建议命令（透传当前环境的关键变量）
CMD="GPU=${GPU:-0,1} MODEL_PATH=${MODEL_PATH:-../../model/MiniMax-H3} \\"
CMD="$CMD\n  TRANSFORMER_DEVICE=${TRANSFORMER_DEVICE:-cuda:0} TEXT_ENCODER_DEVICE=${TEXT_ENCODER_DEVICE:-cuda:1} \\"
CMD="$CMD\n  PROMPT_FILE=$PROMPT_FILE \\"
if [ -n "${FIRST_FRAME:-}" ]; then
    CMD="$CMD\n  FIRST_FRAME=$FIRST_FRAME \\"
fi
CMD="$CMD\n  OUTPUT_DIR=${OUTPUT_DIR:-../MiniMax-H3/results} \\"
CMD="$CMD\n  bash minimax_h3/06_diffusers_inference.sh"
printf '%b\n' "$CMD"
echo ""

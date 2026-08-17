#!/usr/bin/env bash
# 08_context_ir.sh — 调 MiniMax H3-Context-IR API 把短 prompt 转长描述，再调 06 生成视频。
#
# H3-Context-IR 是 MiniMax 的 hosted API（非开源），把自由短 prompt + 可选首帧图转成
# 结构化长描述（integrated_multimodal_description / overall_soundscape / ...），
# 效果比短 prompt 好（官方推荐）。本脚本调 API 拿长描述后自动传给 06 生成视频。
#
# ⚠️ 需要 MiniMax API token（platform.minimaxi.com CN / platform.minimax.io Global 申请）
# ⚠️ FIRST_FRAME 必须是 http URL（API 不读本地路径；本地图先上传到公网）
#
# Usage:
#   MINIMAX_API_KEY=xxx \
#     GPU=0,1 MODEL_PATH=../../model/MiniMax-H3 \
#     TRANSFORMER_DEVICE=cuda:0 TEXT_ENCODER_DEVICE=cuda:1 \
#     PROMPT="a drone shot over alpine peaks" \
#     FIRST_FRAME=https://example.com/subject.png \
#     OUTPUT_DIR=../MiniMax-H3/results \
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
echo "✅ H3-Context-IR prompt ready (${#LONG_PROMPT} chars), passing to 06..."

# 用长描述调 06 生成视频（PROMPT 覆盖，FIRST_FRAME 透传给 06）
PROMPT="$LONG_PROMPT" bash "$SCRIPT_DIR/06_diffusers_inference.sh"

#!/usr/bin/env bash
# 02_run_inference.sh — Gen3R 推理（1view / 2view / allview）
#
# 输出（每次运行一个时间戳子目录）：
#   rgb.mp4        49 帧 560x560 生成视频（3D 一致的环绕/推移视角）
#   pcds.ply       全局点云（世界系，带 RGB；VGGT 深度反投影 + 体素降采样 + 离群过滤）
#   cameras.json   extrinsics [F,4,4] w2c（OpenCV 系）+ intrinsics [F,3,3]
#   prompts.txt    文本提示
#
# Env:
#   TASK=1view        1view | 2view | allview
#   FRAME_PATH        1view: 单图；2view: "first.png last.png"（空格分隔）；allview: 视频或帧目录
#   PROMPTS           文本提示（字符串或 txt 路径）
#   CAMERAS=free      free | zoom_in/zoom_out/arc_left/arc_right/translate_up/translate_down | cameras.json 路径
#   OUTPUT_DIR        输出根（每次运行再创建 <task>/<时间戳>/）
#   REMOVE_FAR_POINTS 1 = 截掉最远 20% 点（voxel 0.003），默认 0（voxel 0.005）
set -o pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/_env.sh"

TASK="${TASK:-1view}"
FRAME_PATH="${FRAME_PATH:-}"
PROMPTS="${PROMPTS:-}"
CAMERAS="${CAMERAS:-free}"
OUTPUT_DIR="${OUTPUT_DIR:-$RESULTS_DIR/02_infer}"
REMOVE_FAR_POINTS="${REMOVE_FAR_POINTS:-0}"

if [ -z "$FRAME_PATH" ]; then
    echo "❌ ERROR: FRAME_PATH 未设置" >&2
    echo "   1view:  FRAME_PATH=/path/to/img.png" >&2
    echo "   2view:  FRAME_PATH=\"/path/first.png /path/last.png\"" >&2
    echo "   allview: FRAME_PATH=/path/clip.mp4" >&2
    exit 1
fi
if [ -z "$PROMPTS" ]; then
    echo "❌ ERROR: PROMPTS 未设置（场景描述，英文短句效果最好）" >&2
    exit 1
fi
if [ ! -d "$MODEL_DIR/transformer" ]; then
    echo "❌ ERROR: 权重未就绪: $MODEL_DIR（先跑 01_download_models.sh）" >&2
    exit 1
fi

echo "🚀 [02] Gen3R 推理"
echo "  🎯 task:    $TASK"
echo "  🖼️  frames:  $FRAME_PATH"
echo "  📷 cameras: $CAMERAS"
echo "  💬 prompt:  $PROMPTS"
echo "  💾 output:  $OUTPUT_DIR"
echo "  🎮 GPU:     ${CUDA_VISIBLE_DEVICES:-all}"
echo ""

cd "$GEN3R_DIR" || { echo "❌ ERROR: $GEN3R_DIR 不存在" >&2; exit 1; }

EXTRA=()
[ "$REMOVE_FAR_POINTS" = "1" ] && EXTRA+=(--remove_far_points)

# shellcheck disable=SC2086  # FRAME_PATH 需按空格拆成多个位置参数
python infer.py \
    --pretrained_model_name_or_path "$MODEL_DIR" \
    --task "$TASK" \
    --prompts "$PROMPTS" \
    --frame_path $FRAME_PATH \
    --cameras "$CAMERAS" \
    --output_dir "$OUTPUT_DIR" \
    "${EXTRA[@]}"
STATUS=$?

if [ $STATUS -ne 0 ]; then
    echo "❌ FAILED (exit $STATUS)" >&2
    exit $STATUS
fi

LATEST="$(ls -1dt "$OUTPUT_DIR/$TASK"/*/ 2>/dev/null | head -1)"
echo "🎉 Done. 产物目录: $LATEST"
[ -n "$LATEST" ] && ls -lh "$LATEST"

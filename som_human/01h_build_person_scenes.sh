#!/usr/bin/env bash
# 01h_build_person_scenes.sh — 按 SAM3 track pid 拆出每人独立训练场景。
#
# 输入：<SCENE_DIR>/persons/masks/{stem}.p{pid}.alpha.png + masks_manifest.json（01g 产物）
# 输出：<SCENE_DIR>/persons/p{pid}/{images,sparse 软链, masks/, person.json}
#
# Env:
#   SCENE_NAME=human_seq
#   SCENE_DIR=      # 默认 $RESULTS_DIR/$SCENE_NAME/colmap_refined
#   MASKS_DIR=      # 默认 $SCENE_DIR/persons/masks
#   OUT_ROOT=       # 默认 $SCENE_DIR/persons
#   MIN_FRAMES=10   # 某人至少被跟踪到的帧数
#   MIN_COVERAGE=0.5  # 平均占画面百分比下限
#   FORCE=0
set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/_env.sh"

SCENE_NAME="${SCENE_NAME:-human_seq}"
SCENE_DIR="${SCENE_DIR:-$RESULTS_DIR/$SCENE_NAME/colmap_refined}"
MASKS_DIR="${MASKS_DIR:-$SCENE_DIR/persons/masks}"
OUT_ROOT="${OUT_ROOT:-$SCENE_DIR/persons}"
MIN_FRAMES="${MIN_FRAMES:-10}"
MIN_FRAME_RATIO="${MIN_FRAME_RATIO:-0.5}"
MIN_COVERAGE="${MIN_COVERAGE:-0.5}"
FORCE="${FORCE:-0}"

echo "🚀 [01h] 构建 per-person 训练场景"
echo "  📂 场景:   $SCENE_DIR"
echo "  🎭 masks:  $MASKS_DIR"
echo "  💾 输出:   $OUT_ROOT/p{pid}/"
echo ""

ARGS=(--scene "$SCENE_DIR" --masks-dir "$MASKS_DIR" --out-root "$OUT_ROOT"
      --min-frames "$MIN_FRAMES" --min-coverage "$MIN_COVERAGE"
      --min-frame-ratio "$MIN_FRAME_RATIO")
[ "$FORCE" = "1" ] && ARGS+=(--force)

python "$SCRIPT_DIR/01h_build_person_scenes.py" "${ARGS[@]}"
if [ $? -ne 0 ]; then
    echo "❌ FAILED: 01h 构建失败" >&2
    exit 1
fi

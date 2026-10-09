#!/usr/bin/env bash
# 01c_pose_adjust.sh — COLMAP 场景位姿调整（主体居中 + 重力对齐 + 尺度归一化）。
#
# 参考 vggt_human 的 POSE_ADJUST：核心算法复用 vggt_human/pose_adjuster.py
# （视线交点最小二乘求中心、相机 right 向量 SVD 估重力、相机距离中位数归一到 10）。
# 与 vggt_human 的差异：那里在 npz→COLMAP 时顺带做；这里做成独立步骤，
# 直接改写已有的 sparse/0（经 colmap model_converter TXT 中转，不写二进制解析）。
#
# 效果与性质：
#   - 世界系重参数化（相似变换），COLMAP 位姿与稀疏点同步变换，尺度/朝向一致
#   - 原始模型备份在 sparse/0_raw（已存在则幂等复用，不会重复备份）
#   - 变换参数存 colmap_scene/pose_adjuster.json（训练后反变换回原坐标用）
#
# Env (all optional, defaults shown):
#   SCENE_NAME=human_seq
#   SCENE_DIR=           # 默认 $RESULTS_DIR/$SCENE_NAME/colmap_scene
#   ENABLE_TRANS=1       # 视线交点居中
#   ENABLE_ROTATE=1      # 重力对齐
#   ENABLE_SCALE=1       # 尺度归一化（相机距离中位数 → 10）
#   GRAVITY_PRIOR=0      # 1=不估计，直接用世界系 -Y 当重力方向
set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/_env.sh"

SCENE_NAME="${SCENE_NAME:-human_seq}"
SCENE_DIR="${SCENE_DIR:-$RESULTS_DIR/$SCENE_NAME/colmap_scene}"
ENABLE_TRANS="${ENABLE_TRANS:-1}"
ENABLE_ROTATE="${ENABLE_ROTATE:-1}"
ENABLE_SCALE="${ENABLE_SCALE:-1}"
GRAVITY_PRIOR="${GRAVITY_PRIOR:-0}"

echo "🧭 [01c] POSE_ADJUST: 居中($ENABLE_TRANS) 重力对齐($ENABLE_ROTATE) 尺度($ENABLE_SCALE)"
echo "  📂 场景: $SCENE_DIR"

# ── 幂等源选择：已备份则永远以 sparse/0_raw 为准（重复跑不会叠加变换）────────
if [ -d "$SCENE_DIR/sparse/0_raw" ]; then
    SRC_SPARSE="$SCENE_DIR/sparse/0_raw"
    echo "  ♻️  检测到 sparse/0_raw，以其为源（幂等重跑）"
elif [ -d "$SCENE_DIR/sparse/0" ]; then
    SRC_SPARSE="$SCENE_DIR/sparse/0"
else
    echo "❌ ERROR: $SCENE_DIR/sparse/0 不存在（先跑 01_prepare_data.sh）" >&2
    exit 1
fi

command -v colmap >/dev/null 2>&1 || { echo "❌ ERROR: colmap 不可用" >&2; exit 1; }

TMP="$(mktemp -d /tmp/pose_adjust.XXXXXX)"
trap 'rm -rf "$TMP"' EXIT

echo "  1/3 BIN → TXT"
# colmap WriteText 不会自建输出目录（不报清晰错误，直接 abort）——必须先 mkdir
mkdir -p "$TMP/raw"
colmap model_converter --input_path "$SRC_SPARSE" --output_path "$TMP/raw" --output_type TXT >/dev/null
mkdir -p "$TMP/adj"

echo "  2/3 变换（pose_adjust_colmap.py）"
PY_FLAGS=(--txt_dir "$TMP/raw" --out_dir "$TMP/adj"
          --json "$SCENE_DIR/pose_adjuster.json")
[ "$ENABLE_TRANS" != "1" ]  && PY_FLAGS+=(--no-trans)
[ "$ENABLE_ROTATE" != "1" ] && PY_FLAGS+=(--no-rotate)
[ "$ENABLE_SCALE" != "1" ]  && PY_FLAGS+=(--no-scale)
[ "$GRAVITY_PRIOR" = "1" ]  && PY_FLAGS+=(--gravity-prior)
VGGT_HUMAN_DIR="$VGGT_HUMAN_DIR" python "$SCRIPT_DIR/pose_adjust_colmap.py" "${PY_FLAGS[@]}"
if [ $? -ne 0 ]; then
    echo "❌ FAILED: pose_adjust_colmap.py（sparse/0 未动）" >&2
    exit 1
fi

echo "  3/3 TXT → BIN，替换 sparse/0（原始备份到 sparse/0_raw）"
NEW0="$TMP/new0"
mkdir -p "$NEW0"
colmap model_converter --input_path "$TMP/adj" --output_path "$NEW0" --output_type BIN >/dev/null
[ -d "$SCENE_DIR/sparse/0_raw" ] || mv "$SCENE_DIR/sparse/0" "$SCENE_DIR/sparse/0_raw"
rm -rf "$SCENE_DIR/sparse/0"
mkdir -p "$SCENE_DIR/sparse"
mv "$NEW0" "$SCENE_DIR/sparse/0"

echo ""
echo "🎉 [01c] Done. sparse/0 已调整为居中+重力对齐+归一化坐标"
echo "  📦 原始模型: $SCENE_DIR/sparse/0_raw"
echo "  📄 变换记录: $SCENE_DIR/pose_adjuster.json"

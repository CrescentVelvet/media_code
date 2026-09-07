#!/usr/bin/env bash
# 03g_build_head_gs.sh — 3DMM 头网格 → 头部高斯点云 head_gs_p<id>.ply
#
# 头/身/景拆分 step 4A：把 03f 拟合出的头网格稠密化成高斯，并从输入视角取真实颜色。
# 产出直接给 render_closeup 用（覆盖率闭环推近 + 近景 loss mask），
# 替代原来的「椭球 K×spread 代理」——后者缺后脑、且太扁导致 40% 覆盖率不可达。
#
# 输入: $RESULTS_DIR/03e_head_3dmm/head_fit.json + head_mesh_p<id>.npz   (03f 产出)
#       $RESULTS_DIR/03_sam3_face_masks/  (取色过滤；注意 dedup 是就地清洗，
#                                          _cleaned/ 目录只放 manifest)
#       $RESULTS_DIR/03_source/                                           (COLMAP 相机 + 图像)
# 输出: $RESULTS_DIR/03e_head_3dmm/head_gs_p<id>.ply
#       $RESULTS_DIR/03e_head_3dmm/vis/head_{overlay,render}_mosaic_p<id>.png
#
# Env (all optional, defaults shown):
#   RESULTS_DIR=
#   SOURCE_DIR=        # 默认 $RESULTS_DIR/03_source
#   MASKS_DIR=         # 默认 $RESULTS_DIR/03_sam3_face_masks_cleaned
#   PERSONS=           # 只跑指定 pid，如 p00,p02
#   HEAD_GS_POINTS=60000
#   HEAD_GS_OPACITY=0.9
#   HEAD_GS_SCALE_K=1.0
#   VIS_FRAMES=6
set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/_env.sh"

RESULTS_DIR="${RESULTS_DIR:-$HOME/output/vggt_human_results}"
SOURCE_DIR="${SOURCE_DIR:-$RESULTS_DIR/03_source}"
FIT_DIR="${FIT_DIR:-$RESULTS_DIR/03e_head_3dmm}"
MASKS_DIR="${MASKS_DIR:-$RESULTS_DIR/03_sam3_face_masks}"
HEAD_GS_POINTS="${HEAD_GS_POINTS:-60000}"
HEAD_GS_OPACITY="${HEAD_GS_OPACITY:-0.9}"
HEAD_GS_SCALE_K="${HEAD_GS_SCALE_K:-1.0}"
VIS_FRAMES="${VIS_FRAMES:-6}"

echo "🎨 [03g] 头部高斯构建（3DMM 网格 → head_gs）"
echo "  📂 场景:   $SOURCE_DIR"
echo "  💾 输出:   $FIT_DIR"
echo "  ⚙️ points=$HEAD_GS_POINTS opacity=$HEAD_GS_OPACITY scale_k=$HEAD_GS_SCALE_K"
echo ""

if [ ! -f "$FIT_DIR/head_fit.json" ]; then
    echo "❌ 缺少拟合结果，先跑: bash 03f_fit_head_3dmm.sh" >&2
    exit 1
fi

mkdir -p "$FIT_DIR"

RESULTS_DIR="$RESULTS_DIR" \
SOURCE_DIR="$SOURCE_DIR" \
FIT_DIR="$FIT_DIR" \
MASKS_DIR="$MASKS_DIR" \
IMAGES_DIR="$SOURCE_DIR/images" \
PERSONS="$PERSONS" \
HEAD_GS_POINTS="$HEAD_GS_POINTS" \
HEAD_GS_OPACITY="$HEAD_GS_OPACITY" \
HEAD_GS_SCALE_K="$HEAD_GS_SCALE_K" \
VIS_FRAMES="$VIS_FRAMES" \
python "$SCRIPT_DIR/build_head_gs.py"
if [ $? -ne 0 ]; then
    echo "❌ FAILED" >&2
    exit 1
fi

echo ""
echo "✅ [03g] head_gs 完成"
echo "  🧱 $FIT_DIR/head_gs_p*.ply"
echo "  🖼️  $FIT_DIR/vis/head_overlay_mosaic_p*.png"

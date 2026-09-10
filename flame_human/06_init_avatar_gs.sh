#!/usr/bin/env bash
# 06_init_avatar_gs.sh — 阶段五：AvatarGaussian 初始化（重心绑定 + 自由高斯）。
#
# 产出：avatar_p{id}.ply（标准 3DGS 初始 PLY）、avatar_bind_p{id}.npz（face_id/bary）、
#       avatar_mesh_p{id}.npz（世界坐标中性脸，阶段七切分要用）。
set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/_env.sh"

echo "🚀 [flame_human 06] 阶段五 AvatarGaussian 初始化"
echo "  📥 输入: ${ALIGN_JSON:-$RESULTS_DIR/05_align/head_align.json}"
echo "  💾 输出: ${OUT_DIR:-$RESULTS_DIR/06_avatar_gs}"

if [ ! -f "${ALIGN_JSON:-$RESULTS_DIR/05_align/head_align.json}" ]; then
    echo "❌ 缺少阶段四输出，先跑 bash 05_align_3dmm.sh"
    exit 1
fi

ALIGN_JSON="${ALIGN_JSON:-$RESULTS_DIR/05_align/head_align.json}" \
OUT_DIR="${OUT_DIR:-$RESULTS_DIR/06_avatar_gs}" \
N_BOUND="${N_BOUND:-50000}" \
N_FREE="${N_FREE:-20000}" \
FREE_OFFSET_MM="${FREE_OFFSET_MM:-8}" \
SCALP_Y_PCT="${SCALP_Y_PCT:-60}" \
INIT_OPACITY="${INIT_OPACITY:-0.1}" \
SCALE_FACTOR="${SCALE_FACTOR:-1.0}" \
SEED="${SEED:-0}" \
    python "$SCRIPT_DIR/init_avatar_gs.py"
if [ $? -ne 0 ]; then
    echo "❌ FAILED: 阶段五失败"
    exit 1
fi

echo "🎉 Done. 下一步：bash 07_init_body_gs.sh"

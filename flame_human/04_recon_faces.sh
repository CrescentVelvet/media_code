#!/usr/bin/env bash
# 04_recon_faces.sh — 阶段三：各向同性裁剪 → 468 点 landmark + DECA/PnP 初值。
#
# 前置：03_match_faces.sh 产出 face_person_match.json；
#       01b_build_lm468_embedding.sh 产出 lm468 嵌入（PnP 位姿初值要用）。
set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/_env.sh"

echo "🚀 [flame_human 04] 阶段三 人脸重建输入准备"
echo "  🎯 匹配结果: ${MATCH_JSON:-$RESULTS_DIR/03_match/face_person_match.json}"
echo "  🗿 lm468 嵌入: $FLAME_LM468_EMBEDDING"
echo "  💾 输出      : ${OUT_JSON:-$RESULTS_DIR/04_recon/face_recon.json}"

if [ ! -f "$FLAME_LM468_EMBEDDING" ]; then
    echo "⚠️  缺少 lm468 嵌入: $FLAME_LM468_EMBEDDING"
    echo "   → PnP 位姿初值将不可用。先跑 bash 01b_build_lm468_embedding.sh"
fi

MATCH_JSON="${MATCH_JSON:-$RESULTS_DIR/03_match/face_person_match.json}" \
IMAGES_DIR="${IMAGES_DIR:-$SOURCE_DIR/images}" \
OUT_JSON="${OUT_JSON:-$RESULTS_DIR/04_recon/face_recon.json}" \
CROP_SCALE_LM="${CROP_SCALE_LM:-1.6}" \
CROP_SCALE_DECA="${CROP_SCALE_DECA:-1.25}" \
LM_SIZE="${LM_SIZE:-512}" \
USE_DECA="${USE_DECA:-1}" \
MIN_DET_CONF="${MIN_DET_CONF:-0.5}" \
    python "$SCRIPT_DIR/recon_faces.py"
if [ $? -ne 0 ]; then
    echo "❌ FAILED: 阶段三失败"
    exit 1
fi

echo "🎉 Done. 下一步：bash 05_align_3dmm.sh"

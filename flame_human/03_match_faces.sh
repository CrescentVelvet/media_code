#!/usr/bin/env bash
# 03_match_faces.sh — 阶段二：face bbox ↔ person track ID 关联（三层判据）。
#
# 三层：point-in-mask 主判据 → IoM 次判据 → 匈牙利一对一 + 时序连续性检查。
# 不用字面 IoU(face_box, person_box)：人脸框几乎完全包含在人体框内，
# IoU ≈ |face|/|person| 量级 0.02~0.1，两人靠近时区分度趋近于零。
set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/_env.sh"

echo "🚀 [flame_human 03] 阶段二 face ↔ person 关联"
echo "  🎯 人脸框  : ${FACE_BOXES:-$RESULTS_DIR/02_faces/face_boxes.json}"
echo "  🎭 人体 mask: $PERSON_MASKS_DIR"
echo "  💾 输出    : ${OUT_JSON:-$RESULTS_DIR/03_match/face_person_match.json}"

if [ ! -d "$PERSON_MASKS_DIR" ]; then
    echo "❌ person mask 目录不存在: $PERSON_MASKS_DIR"
    echo "   → 用 PERSON_MASKS_DIR= 指向 SegTrack/SAM3 的 person mask"
    exit 1
fi

FACE_BOXES="${FACE_BOXES:-$RESULTS_DIR/02_faces/face_boxes.json}" \
MASKS_DIR="$PERSON_MASKS_DIR" \
IMAGES_DIR="${IMAGES_DIR:-$SOURCE_DIR/images}" \
OUT_JSON="${OUT_JSON:-$RESULTS_DIR/03_match/face_person_match.json}" \
INSIDE_W="${INSIDE_W:-2.0}" \
MIN_IOM="${MIN_IOM:-0.3}" \
MAX_JUMP_PX="${MAX_JUMP_PX:-200}" \
DROP_SUSPECT="${DROP_SUSPECT:-0}" \
    python "$SCRIPT_DIR/match_faces.py"
if [ $? -ne 0 ]; then
    echo "❌ FAILED: 关联失败"
    exit 1
fi

echo "🎉 Done. 下一步：bash 04_recon_faces.sh"

#!/usr/bin/env bash
# 03_evaluate_jiajia.sh — 输入自洽与未参与重建的留出视角评测（含相机对齐诊断）
set -o pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/_env.sh"
RESULT="${RESULT:-$RESULTS_DIR/jiajia_16v}"
SOURCE_CAMERAS="${SOURCE_CAMERAS:-$RESULTS_DIR/_captures/jiajia_baozhen/cameras.npz}"
TARGET_CAMERAS="${TARGET_CAMERAS:-$RESULTS_DIR/_captures/jiajia_holdout/cameras.npz}"
python "$SCRIPT_DIR/evaluate_jiajia.py" --result "$RESULT" \
    --source-cameras "$SOURCE_CAMERAS" --target-cameras "$TARGET_CAMERAS" \
    ${PLY:+--ply "$PLY"} ${OUT:+--out "$OUT"}
RC=$?
if [ $RC -ne 0 ]; then echo "❌ 评测失败 exit=$RC" >&2; exit $RC; fi
echo "🎉 Done: $RESULT/evaluation"

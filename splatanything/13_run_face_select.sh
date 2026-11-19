#!/usr/bin/env bash
# 13_run_face_select.sh — 在 splatanything env 里跑 13_face_select.py
set -o pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONDA_ENV=splatanything
source "$SCRIPT_DIR/_env.sh"
source "$SCRIPT_DIR/_gsplat_env.sh"

RESULT="${RESULT:?❌ 需要 RESULT=<02输出目录>}"

echo "🚀 [13] face select | RESULT=$RESULT"
python "$SCRIPT_DIR/13_face_select.py" --result "$RESULT" ${MIN_SUPPORT:+--min-support "$MIN_SUPPORT"}
if [ $? -ne 0 ]; then
    echo "❌ FAILED" >&2
    exit 1
fi
echo "🎉 Done."

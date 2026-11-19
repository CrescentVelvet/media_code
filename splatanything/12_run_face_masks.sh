#!/usr/bin/env bash
# 12_run_face_masks.sh — 在 sam3 env 里跑 12_face_masks.py
set -o pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONDA_ENV=sam3
source "$SCRIPT_DIR/_env.sh"

RESULT="${RESULT:?❌ 需要 RESULT=<02输出目录>}"
PY="$HOME/miniconda3/envs/sam3/bin/python"

echo "🚀 [12] face masks | RESULT=$RESULT"
"$PY" "$SCRIPT_DIR/12_face_masks.py" --result "$RESULT" ${OUT:+--out "$OUT"}
if [ $? -ne 0 ]; then
    echo "❌ FAILED" >&2
    exit 1
fi
echo "🎉 Done."

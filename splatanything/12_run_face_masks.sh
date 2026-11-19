#!/usr/bin/env bash
# 12_run_face_masks.sh — 在 sam3 env 里跑 12_face_masks.py
set -o pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONDA_ENV=sam3
source "$SCRIPT_DIR/_env.sh"

PY="$HOME/miniconda3/envs/sam3/bin/python"

# 两种模式：RESULT=<02输出目录>（mask 落在 <result>/face_opt/masks）
# 或 IMAGES=<PNG目录> OUT=<输出目录>（任意图集，如 held-out target 帧）
if [ -n "${IMAGES:-}" ]; then
    OUT="${OUT:?❌ IMAGES 模式必须给 OUT=<mask输出目录>}"
    echo "🚀 [12] face masks | IMAGES=$IMAGES OUT=$OUT"
    "$PY" "$SCRIPT_DIR/12_face_masks.py" --images "$IMAGES" --out "$OUT"
else
    RESULT="${RESULT:?❌ 需要 RESULT=<02输出目录>（或 IMAGES+OUT）}"
    echo "🚀 [12] face masks | RESULT=$RESULT"
    "$PY" "$SCRIPT_DIR/12_face_masks.py" --result "$RESULT" ${OUT:+--out "$OUT"}
fi
if [ $? -ne 0 ]; then
    echo "❌ FAILED" >&2
    exit 1
fi
echo "🎉 Done."

#!/usr/bin/env bash
# 14_run_face_opt.sh — P0~P3 脸部局部优化（splatanything env）
set -o pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONDA_ENV=splatanything
source "$SCRIPT_DIR/_env.sh"
source "$SCRIPT_DIR/_gsplat_env.sh"

RESULT="${RESULT:?❌ 需要 RESULT=<02输出目录>}"
ARM="${ARM:?❌ 需要 ARM=p0|p1|p2|p3}"
ITERS="${ITERS:-100}"

echo "🚀 [14] face local opt | ARM=$ARM ITERS=$ITERS RESULT=$RESULT"
python "$SCRIPT_DIR/14_face_local_opt.py" \
  --result "$RESULT" --arm "$ARM" --iters "$ITERS" ${EXTRA_ARGS:-}
if [ $? -ne 0 ]; then
    echo "❌ FAILED" >&2
    exit 1
fi
echo "🎉 Done."

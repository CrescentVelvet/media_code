#!/usr/bin/env bash
# 19_run_f1.sh — F1 保守自由度对照（splatanything env）
set -o pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONDA_ENV=splatanything
source "$SCRIPT_DIR/_env.sh"
source "$SCRIPT_DIR/_gsplat_env.sh"

RESULT="${RESULT:?❌ 需要 RESULT=<02输出目录>}"
ARM="${ARM:?❌ 需要 ARM=f1a|f1b}"
ITERS="${ITERS:-300}"

echo "🚀 [19] F1 conservative opt | ARM=$ARM ITERS=$ITERS RESULT=$RESULT"
python "$SCRIPT_DIR/19_f1_conservative_opt.py" \
  --result "$RESULT" --arm "$ARM" --iters "$ITERS" ${EXTRA_ARGS:-}
if [ $? -ne 0 ]; then
    echo "❌ FAILED" >&2
    exit 1
fi
echo "🎉 Done."

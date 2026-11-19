#!/usr/bin/env bash
# 15_run_target_eval.sh — held-out target 视角评估 P0~P3 回退 PLY（splatanything env）
set -o pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONDA_ENV=splatanything
source "$SCRIPT_DIR/_env.sh"
source "$SCRIPT_DIR/_gsplat_env.sh"

RESULT="${RESULT:?❌ 需要 RESULT=<02输出目录>}"
TARGET="${TARGET:?❌ 需要 TARGET=<target场景目录>}"
MASKS="${MASKS:?❌ 需要 MASKS=<target mask目录>}"

echo "🚀 [15] target eval | RESULT=$RESULT TARGET=$TARGET"
python "$SCRIPT_DIR/15_face_target_eval.py" \
  --result "$RESULT" --target "$TARGET" --masks "$MASKS" ${EXTRA_ARGS:-}
if [ $? -ne 0 ]; then
    echo "❌ FAILED" >&2
    exit 1
fi
echo "🎉 Done."

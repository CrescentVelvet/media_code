#!/usr/bin/env bash
set -o pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/_env.sh"
source "$SCRIPT_DIR/_gsplat_env.sh"
ROOT="${ROOT:?Set ROOT to protocol_v5}"
TRACE="${TRACE:?Set TRACE to the 31 capture root}"
OUT="${OUT:?Set OUT to a separate alignment results directory}"
export JIA_ARCHIVE="${JIA_ARCHIVE:-$SPLATANYTHING_RESULTS_DIR/pose_face3v/jiajia_face3v}"
export PYTHONPATH="$SPLATANYTHING_DIR:${PYTHONPATH:-}"
export OPENBLAS_NUM_THREADS=2
for stage in ${STAGES:-fit evaluate audit}; do
    for name in ${PACKS:-jiajia real_black_3a8b3cc7 real_gray_0221a3dd rosemary daven_holdout}; do
        echo "🔍 $stage $name"
        "$CONDA_PREFIX/bin/python" "$SCRIPT_DIR/32_align_face.py" "$stage" "$ROOT" "$TRACE" "$OUT" "$name" || exit 1
    done
done
if [[ " ${STAGES:-fit evaluate audit} " == *" evaluate "* ]]; then
    "$CONDA_PREFIX/bin/python" "$SCRIPT_DIR/32_summarize_alignment.py" "$OUT" \
        ${PACKS:-jiajia real_black_3a8b3cc7 real_gray_0221a3dd rosemary daven_holdout} || exit 1
fi

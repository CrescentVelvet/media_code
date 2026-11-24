#!/usr/bin/env bash
set -o pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/_env.sh"
source "$SCRIPT_DIR/_gsplat_env.sh"
export JIA_ARCHIVE="${JIA_ARCHIVE:-$SPLATANYTHING_RESULTS_DIR/pose_face3v/jiajia_face3v}"
export PYTHONPATH="$SPLATANYTHING_DIR:${PYTHONPATH:-}"
export OPENBLAS_NUM_THREADS=2
for name in ${PACKS:-rosemary daven_holdout real_black_3a8b3cc7 jiajia real_gray_0221a3dd}; do
    echo "🔍 $name"
    "$CONDA_PREFIX/bin/python" "$SCRIPT_DIR/33_rigid_face.py" "${ROOT:?Set ROOT to protocol_v5}" \
        "${TRACE:?Set TRACE to 31 captures}" "${DEPTH:?Set DEPTH to 30 diagnostics}" \
        "${OUT:?Set OUT to a separate results folder}" "$name" || exit 1
done
"$CONDA_PREFIX/bin/python" "$SCRIPT_DIR/33_summarize_rigid.py" "$OUT" \
    ${PACKS:-rosemary daven_holdout real_black_3a8b3cc7 jiajia real_gray_0221a3dd} || exit 1

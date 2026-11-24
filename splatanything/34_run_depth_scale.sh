#!/usr/bin/env bash
set -o pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/_env.sh"
source "$SCRIPT_DIR/_gsplat_env.sh"
export PYTHONPATH="$SPLATANYTHING_DIR:${PYTHONPATH:-}"
export OPENBLAS_NUM_THREADS=2
echo "📐 Estimating scale and offsets from source cameras"
"$CONDA_PREFIX/bin/python" "$SCRIPT_DIR/34_calibrate_source_depth.py" "${ROOT:?set ROOT}" "${TRACE:?set TRACE}" "${PREVIOUS:?set PREVIOUS}" "${OUT:?set OUT}" || exit 1
echo "🔍 Evaluating Gaussian renders at held-out cameras"
"$CONDA_PREFIX/bin/python" "$SCRIPT_DIR/34_render_source_depth.py" "$ROOT" "$TRACE" "$OUT" || exit 1

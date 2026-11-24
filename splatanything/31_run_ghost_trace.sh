#!/usr/bin/env bash
# Replay three-source inference and render each held-out stage with the original weights.
set -o pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/_env.sh"
source "$SCRIPT_DIR/_gsplat_env.sh"
ROOT="${ROOT:?Set ROOT to the frozen protocol_v5 directory}"
OUT="${OUT:?Set OUT to the trace output directory}"
JIA_ARCHIVE="${JIA_ARCHIVE:-$SPLATANYTHING_RESULTS_DIR/pose_face3v/jiajia_face3v}"
SA_PY="$CONDA_PREFIX/bin/python"
FACE_PY="${FACE_PY:-$(dirname "$CONDA_PREFIX")/sam3/bin/python}"
export PYTHONPATH="$SPLATANYTHING_DIR:${PYTHONPATH:-}"
export OPENBLAS_NUM_THREADS="${TRACE_BLAS_THREADS:-2}"

for name in ${PACKS:-jiajia rosemary daven_holdout real_gray_0221a3dd real_black_3a8b3cc7}; do
    archive="$ROOT/models/sa_pose/${name}_source3"
    [ "$name" = jiajia ] && archive="$JIA_ARCHIVE"
    dest="$OUT/$name"
    mkdir -p "$dest" || exit 1
    echo "🔍 $name — $archive"
    if [ "${REUSE_CAPTURE:-0}" != 1 ]; then
        "$SA_PY" "$SCRIPT_DIR/31_capture_layers.py" "$SPLATANYTHING_DIR" "$archive" "$dest" || exit 1
    fi
    args=("$ROOT" "$dest" "$archive" "$name")
    case "$name" in
        jiajia|real_*)
            "$SA_PY" "$SCRIPT_DIR/31_trace_ghost.py" "${args[@]}" --prepare-faces || exit 1
            "$FACE_PY" "$SCRIPT_DIR/31_detect_canvas_faces.py" "$dest" || exit 1 ;;
    esac
    "$SA_PY" "$SCRIPT_DIR/31_trace_ghost.py" "${args[@]}" || exit 1
    echo "✅ $dest/trace.json"
done

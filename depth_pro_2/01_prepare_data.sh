#!/usr/bin/env bash
set -o pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/_env.sh"
export SCENE="${SCENE:?❌ SCENE=human|router is required}"
export RAW_DIR="${RAW_DIR:?❌ RAW_DIR is required}"
export OUT_DIR="${OUT_DIR:?❌ OUT_DIR is required}"
if [ "$SCENE" = router ]; then export POSE_DIR="${POSE_DIR:?❌ POSE_DIR is required}"; fi
python "$SCRIPT_DIR/prepare_data.py" || exit 1

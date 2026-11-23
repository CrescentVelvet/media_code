#!/usr/bin/env bash
set -o pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/_env.sh"
export SCENE_DIR="${SCENE_DIR:?❌ SCENE_DIR is required}"
export POSE_DIR="${POSE_DIR:?❌ POSE_DIR is required for the router COLMAP audit}"
echo "🚀 Native COLMAP track audit: $POSE_DIR -> $SCENE_DIR"
python "$SCRIPT_DIR/audit_colmap.py" || exit 1

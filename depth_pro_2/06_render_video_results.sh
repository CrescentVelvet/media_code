#!/usr/bin/env bash
set -o pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/_env.sh"
export SCENE_DIR="${SCENE_DIR:?❌ SCENE_DIR is required}"
[ -f "$SCENE_DIR/inference.json" ] || { echo "❌ Missing completed video inference: $SCENE_DIR/inference.json" >&2; exit 1; }
echo "🚀 Render video depth, local-camera PCDs and previews: $SCENE_DIR"
python -B "$SCRIPT_DIR/render_video_results.py" || exit 1

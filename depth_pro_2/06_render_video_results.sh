#!/usr/bin/env bash
set -o pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/_env.sh"
export SCENE_DIR="${SCENE_DIR:?❌ SCENE_DIR is required}"
[ -f "$SCENE_DIR/inference.json" ] || { echo "❌ Missing completed video inference: $SCENE_DIR/inference.json" >&2; exit 1; }
echo "🚀 Render video depth; VIDEO_ONLY=1 skips PCDs and extra previews: $SCENE_DIR"
python -B "$SCRIPT_DIR/render_video_results.py" || exit 1

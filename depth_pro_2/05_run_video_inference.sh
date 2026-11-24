#!/usr/bin/env bash
set -o pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/_env.sh"
export VIDEO_PATH="${VIDEO_PATH:?❌ VIDEO_PATH is required}"
export OUT_DIR="${OUT_DIR:?❌ OUT_DIR is required}"
[ -f "$VIDEO_PATH" ] || { echo "❌ Missing source video: $VIDEO_PATH" >&2; exit 1; }
[ -f "$CKPT_PATH" ] || { echo "❌ Missing H+ checkpoint: $CKPT_PATH" >&2; exit 1; }
[ -f "$FOV_CKPT_PATH" ] || { echo "❌ Missing FOV checkpoint: $FOV_CKPT_PATH" >&2; exit 1; }
export OFFICIAL_REVISION="${OFFICIAL_REVISION:-$(git -C "$DEPTH_PRO_2_DIR" rev-parse HEAD)}"
echo "🚀 Official buffered video inference: $VIDEO_PATH -> $OUT_DIR"
python -B "$SCRIPT_DIR/run_video_inference.py" || exit 1

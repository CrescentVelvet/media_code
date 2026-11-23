#!/usr/bin/env bash
set -o pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/_env.sh"
export INPUT_DIR="${INPUT_DIR:?❌ INPUT_DIR is required}"
export OUT_DIR="${OUT_DIR:?❌ OUT_DIR is required}"
[ -f "$CKPT_PATH" ] || { echo "❌ Missing H+ checkpoint: $CKPT_PATH" >&2; exit 1; }
[ -f "$FOV_CKPT_PATH" ] || { echo "❌ Missing FOV fallback checkpoint: $FOV_CKPT_PATH" >&2; exit 1; }
export OFFICIAL_REVISION="${OFFICIAL_REVISION:-$(git -C "$DEPTH_PRO_2_DIR" rev-parse HEAD)}"
echo "🚀 Joint H+ inference: $INPUT_DIR -> $OUT_DIR"
python "$SCRIPT_DIR/run_inference.py" || exit 1

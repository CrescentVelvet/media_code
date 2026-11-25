#!/usr/bin/env bash
set -o pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/_env.sh"
export INPUT_DIR="${INPUT_DIR:?❌ INPUT_DIR is required}"
export OUT_DIR="${OUT_DIR:?❌ OUT_DIR is required}"
[ -f "$INPUT_DIR/manifest.json" ] || { echo "❌ Missing sequence manifest: $INPUT_DIR" >&2; exit 1; }
[ -f "$CKPT_PATH" ] || { echo "❌ Missing H+ checkpoint: $CKPT_PATH" >&2; exit 1; }
[ -f "$FOV_CKPT_PATH" ] || { echo "❌ Missing FOV checkpoint: $FOV_CKPT_PATH" >&2; exit 1; }
export OFFICIAL_REVISION="${OFFICIAL_REVISION:-$(git -C "$DEPTH_PRO_2_DIR" rev-parse HEAD)}"
export WINDOW="${WINDOW:-16}"
export OVERLAP="${OVERLAP:-4}"
echo "🚀 Official buffered H+ inference; window=$WINDOW overlap=$OVERLAP"
echo "  📁 Input: $INPUT_DIR"
echo "  💾 Output: $OUT_DIR"
python -B "$SCRIPT_DIR/run_sequence_inference.py" || exit 1

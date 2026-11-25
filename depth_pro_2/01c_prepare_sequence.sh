#!/usr/bin/env bash
set -o pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/_env.sh"
export RAW_DIR="${RAW_DIR:?❌ RAW_DIR is required}"
export OUT_DIR="${OUT_DIR:?❌ OUT_DIR is required}"
[ -f "$RAW_DIR/transforms.json" ] || { echo "❌ Missing transforms.json: $RAW_DIR" >&2; exit 1; }
echo "🚀 Decode all ordered Remy cameras; preserve calibrated c2w: $RAW_DIR"
python -B "$SCRIPT_DIR/prepare_sequence.py" || exit 1

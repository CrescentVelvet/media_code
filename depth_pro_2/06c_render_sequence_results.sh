#!/usr/bin/env bash
set -o pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/_env.sh"
export SCENE_DIR="${SCENE_DIR:?❌ SCENE_DIR is required}"
[ -f "$SCENE_DIR/inference.json" ] || { echo "❌ Missing completed sequence inference: $SCENE_DIR" >&2; exit 1; }
[ -f "$SCENE_DIR/input/manifest.json" ] || { echo "❌ Missing calibrated sequence manifest: $SCENE_DIR" >&2; exit 1; }
export PCD_PIXEL_STRIDE="${PCD_PIXEL_STRIDE:-2}"
export NEIGHBORHOOD_RADIUS="${NEIGHBORHOOD_RADIUS:-8}"
export PCD_POINT_RADIUS="${PCD_POINT_RADIUS:-0.65}"
echo "🚀 Render RGB/depth and world-registered PCD splat/depth videos"
echo "  📁 Scene: $SCENE_DIR"
echo "  📐 PCD render stride: $PCD_PIXEL_STRIDE; neighborhood: +/-$NEIGHBORHOOD_RADIUS frames"
python -B "$SCRIPT_DIR/render_sequence_results.py" || exit 1

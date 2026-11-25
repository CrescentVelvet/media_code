#!/usr/bin/env bash
set -o pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/_env.sh"
export SCENE_DIR="${SCENE_DIR:?❌ SCENE_DIR is required}"
export RAW_DIR="${RAW_DIR:?❌ RAW_DIR containing the three selected HEICs and transforms.json is required}"
export OUT_DIR="${OUT_DIR:?❌ OUT_DIR is required; it must differ from SCENE_DIR}"
echo "🚀 Verify supplied poses and compare post-inference lifting: $SCENE_DIR"
echo "  📁 transforms: $RAW_DIR/transforms.json"
echo "  💾 output: $OUT_DIR"
echo "  ⏭️ Reuse cached joint depth; no model inference, ICP or per-frame scale"
python "$SCRIPT_DIR/audit_pose_conditioning.py" || exit 1
echo "🎉 Done."

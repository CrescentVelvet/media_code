#!/usr/bin/env bash
set -o pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/_env.sh"

echo "🚀 [02b] 佳佳抱枕与路由器逐帧独立 H+ 推理"
for SCENE in human router; do
  export INPUT_DIR="$RESULTS_DIR/$SCENE"
  export OUT_DIR="$INPUT_DIR/single_frame"
  echo "🖼️ 输入: $INPUT_DIR/rgb/00.png～02.png"
  echo "💾 输出: $OUT_DIR"
  python "$SCRIPT_DIR/run_single_frame_inference.py"
  if [ $? -ne 0 ]; then
    echo "❌ FAILED: $SCENE single-frame inference" >&2
    exit 1
  fi
done
echo "🎉 Done."

#!/usr/bin/env bash
# 02_run_inference.sh — image(s) -> textured GLB.
#
# MODE=single: IMAGE=<path>                      (2.1 shape, best quality)
# MODE=mv:     FRONT=/BACK=/LEFT=/RIGHT=<paths>  (2mv shape, any non-empty
#              subset; views should be background-removed square crops)
set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/_env.sh"

MODE="${MODE:-single}"
export MODE

export SHAPE_DIR="${SHAPE_DIR:-$MODEL_DIR/Hunyuan3D-2.1}"
export MV_DIR="${MV_DIR:-$MODEL_DIR/Hunyuan3D-2mv}"
export DINO_DIR="${DINO_DIR:-$MODEL_DIR/dinov2-giant}"
export OUTPUT_DIR="${OUTPUT_DIR:-$RESULTS_DIR}"

export SEED="${SEED:-0}"
export NUM_INFERENCE_STEPS="${NUM_INFERENCE_STEPS:-50}"
export GUIDANCE_SCALE="${GUIDANCE_SCALE:-5.0}"
export OCTREE_RESOLUTION="${OCTREE_RESOLUTION:-384}"
export NUM_CHUNKS="${NUM_CHUNKS:-8000}"
export TEX_MAX_NUM_VIEW="${TEX_MAX_NUM_VIEW:-6}"
export TEX_RESOLUTION="${TEX_RESOLUTION:-512}"
export SKIP_TEXTURE="${SKIP_TEXTURE:-0}"

# --- input validation ---
if [ "$MODE" = "single" ]; then
    if [ -z "${IMAGE:-}" ] || [ ! -f "$IMAGE" ]; then
        echo "❌ MODE=single requires IMAGE=<existing file>" >&2; exit 1
    fi
    export IMAGE
elif [ "$MODE" = "mv" ]; then
    n=0
    for v in FRONT BACK LEFT RIGHT; do
        p="${!v:-}"
        if [ -n "$p" ]; then
            [ -f "$p" ] || { echo "❌ $v=$p not found" >&2; exit 1; }
            export "$v=$p"
            n=$((n + 1))
        fi
    done
    [ "$n" -ge 1 ] || { echo "❌ MODE=mv requires at least one of FRONT/BACK/LEFT/RIGHT" >&2; exit 1; }
    echo "🖼️ [02] multi-view input: $n view(s)"
else
    echo "❌ unknown MODE='$MODE' (single|mv)" >&2; exit 1
fi

for f in "$SHAPE_DIR" "$HY3D_DIR/hy3dshape" "$HY3D_DIR/hy3dpaint"; do
    [ -e "$f" ] || { echo "❌ missing: $f (run 00/01 first)" >&2; exit 1; }
done
if [ "$MODE" = "mv" ] && [ ! -d "$MV_DIR" ]; then
    echo "❌ MV_DIR=$MV_DIR missing (run 01 without SKIP_MV=1)" >&2; exit 1
fi

echo "🚀 [02] Hunyuan3D-2.1 inference (MODE=$MODE)"
echo "  📁 output: $OUTPUT_DIR"
echo "  🎮 CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-<all>}"

python "$SCRIPT_DIR/run_inference.py"
rc=$?
[ $rc -ne 0 ] && { echo "❌ FAILED (rc=$rc)" >&2; exit $rc; }
echo "🎉 [02] Done."

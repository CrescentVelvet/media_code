#!/usr/bin/env bash
# 02_run_inference.sh — render a trained Deformable-GS scene + evaluate it.
#
# For a dynamic-3DGS method "inference" = rendering the trained Gaussian point
# cloud from held-out test cameras, then scoring the renders against GT. This
# wraps the official `render.py` + `metrics.py` (no official file is modified):
#   python render.py  -m <model_path> --mode <mode> [--iteration N] [...]
#   python metrics.py -m <model_path>
#
# train.py writes a cfg_args file into <model_path>; render.py / metrics.py
# read it (via get_combined_args) to recover --is_blender / --is_6dof /
# --source_path, so you only pass -m here. To render a checkpoint you placed by
# hand (no cfg_args), pass the ModelParams flags via EXTRA_RENDER_ARGS, e.g.
#   EXTRA_RENDER_ARGS="--is_blender --source_path /data/D-NeRF/hook".
#
# render.py writes, under <model_path>/<split>/ours_<iter>/:
#   renders/%05d.png  gt/%05d.png  depth/%05d.png   (+ video.mp4 for the
# interpolate_* modes). metrics.py writes <model_path>/test/results.json +
# per_view.json (PSNR / SSIM / LPIPS over the TEST split).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/_env.sh"

DG_DIR="${DG_DIR:-$REPO_DIR/../Deformable-3D-Gaussians}"
MODEL_DIR="${MODEL_DIR:-$REPO_DIR/../../model/deformable-3d-gaussians}"

# Trained scene output dir (what you passed as -m to train.py). Required.
MODEL_PATH="${MODEL_PATH:-}"
SCENE="${SCENE:-}"
if [ -z "$MODEL_PATH" ] && [ -n "$SCENE" ]; then
    MODEL_PATH="${DG_OUTPUT_ROOT:-$DG_DIR/output}/$SCENE"
fi

# --- render params (all overridable via env) ---
# iteration: -1 = latest saved point_cloud (searchForMaxIteration).
ITERATION="${ITERATION:--1}"
# mode: render (all test images) | time | all | view | pose | original.
# D-NeRF time-interpolation uses `time`/`all`; NeRF-DS/HyperNeRF use `original`.
MODE="${MODE:-render}"
SKIP_TRAIN="${SKIP_TRAIN:-1}"        # 1 = skip rendering the train split (faster; metrics only need test)
SKIP_TEST="${SKIP_TEST:-0}"          # 0 = render the test split (needed for metrics)
QUIET="${QUIET:-0}"
RUN_METRICS="${RUN_METRICS:-1}"      # 0 = render only, skip metrics.py
# forwarded verbatim to render.py (e.g. "--is_blender --source_path /data/...")
EXTRA_RENDER_ARGS="${EXTRA_RENDER_ARGS:-}"

echo "=== [02] Deformable-GS render + metrics ==="
echo "  代码路径:   $DG_DIR"
echo "  模型路径:   ${MODEL_PATH:-<unset>}"
echo "  iteration: $ITERATION (-1 = latest)"
echo "  mode:      $MODE"
echo "  skip_train=$SKIP_TRAIN skip_test=$SKIP_TEST  run_metrics=$RUN_METRICS"
if [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
    echo "  GPU:       physical $CUDA_VISIBLE_DEVICES (cuda:0 in-process)  [GPU=N to change]"
else
    echo "  GPU:       default cuda:0 (= first visible)  [set GPU=N to pin a card]"
fi

# --- checks ---
if [ ! -d "$DG_DIR" ]; then
    echo "ERROR: Deformable-GS code dir not found at $DG_DIR. Run run_all.sh first." >&2; exit 1
fi
if [ -z "$MODEL_PATH" ]; then
    echo "ERROR: MODEL_PATH not set. Point it at a trained scene output dir, e.g.:" >&2
    echo "         MODEL_PATH=$DG_DIR/output/hook GPU=0 bash $0" >&2
    echo "       (or set SCENE=hook and DG_OUTPUT_ROOT=$DG_DIR/output)." >&2
    exit 1
fi
if [ ! -d "$MODEL_PATH" ]; then
    echo "ERROR: trained model dir not found: $MODEL_PATH" >&2
    echo "       Train a scene first (run_all.sh or: python train.py -s <data> -m $MODEL_PATH --eval --is_blender)." >&2
    exit 1
fi
if ! python -c "import diff_gaussian_rasterization" 2>/dev/null; then
    echo "ERROR: diff_gaussian_rasterization not importable. Build the CUDA ext:" >&2
    echo "         BUILD_CUDA=1 bash deformable_gaussians/00_setup_env.sh" >&2
    exit 1
fi

# Build the render.py flag list from env. render.py uses --skip_train /
# --skip_test store_true flags, so only pass them when set to 1.
RENDER_FLAGS=(-m "$MODEL_PATH" --iteration "$ITERATION" --mode "$MODE")
[ "$SKIP_TRAIN" = "1" ] && RENDER_FLAGS+=(--skip_train)
[ "$SKIP_TEST"  = "1" ] && RENDER_FLAGS+=(--skip_test)
[ "$QUIET"      = "1" ] && RENDER_FLAGS+=(--quiet)
# shellcheck disable=SC2086
[ -n "$EXTRA_RENDER_ARGS" ] && RENDER_FLAGS+=($EXTRA_RENDER_ARGS)

echo "--- render.py ${RENDER_FLAGS[*]} ---"
( cd "$DG_DIR" && python render.py "${RENDER_FLAGS[@]}" )

if [ "$RUN_METRICS" = "1" ]; then
    # metrics.py only scores the TEST split (<model_path>/test/ours_<iter>/).
    if [ "$SKIP_TEST" = "1" ]; then
        echo "--- SKIP metrics: test split was not rendered (SKIP_TEST=1) ---"
    else
        echo "--- metrics.py -m $MODEL_PATH ---"
        ( cd "$DG_DIR" && python metrics.py -m "$MODEL_PATH" )
        echo "--- metrics -> $MODEL_PATH/test/results.json (PSNR/SSIM/LPIPS) ---"
        python - "$MODEL_PATH" <<'PY' 2>/dev/null || true
import json, sys, os
p = sys.argv[1]
f = os.path.join(p, "test", "results.json")
if os.path.isfile(f):
    d = json.load(open(f))
    for scene, m in d.items():
        print(f"    {scene}: PSNR {m.get('PSNR', 0):.4f}  SSIM {m.get('SSIM', 0):.4f}  LPIPS {m.get('LPIPS', 0):.4f}")
PY
    fi
fi

echo "=== [02] Done. Renders under: $MODEL_PATH/{train,test}/ours_$ITERATION/ ==="
echo "    test images: $MODEL_PATH/test/ours_$ITERATION/renders/"
echo "    metrics:    $MODEL_PATH/test/results.json"

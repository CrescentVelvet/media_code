#!/usr/bin/env bash
# 02_run_inference.sh — Real-World Image Super-Resolution with OSEDiff.
#
# Wraps the official test_osediff.py. Defaults to the repo's preset test images
# and the shipped osediff.pkl. Point INPUT_DIR at your own folder to SR a batch.
#
# Usage:
#   GPU=0 bash osediff/02_run_inference.sh
#   GPU=0 INPUT_DIR=/path/to/lq OUTPUT_DIR=/path/to/out bash osediff/02_run_inference.sh
#   GPU=0 MODE=face INPUT_DIR=/path/to/face bash osediff/02_run_inference.sh   # face restore
set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/_env.sh"

# ── params (all overridable via env) ─────────────────────────────────────
# MODE: "sr" (default, x4 super-res) or "face" (x1 face restoration).
MODE="${MODE:-sr}"
UPSCALE="${UPSCALE:-4}"
if [ "$MODE" = "face" ]; then
    UPSCALE="${UPSCALE:-1}"
    OSEDIFF_PKL="${OSEDIFF_PKL:-$OSEDIFF_FACE_PKL}"
    ALIGN_METHOD="${ALIGN_METHOD:-nofix}"
    INPUT_DIR="${INPUT_DIR:-$OSEDIFF_DIR/preset/datasets/test_dataset/input_face}"
else
    INPUT_DIR="${INPUT_DIR:-$OSEDIFF_DIR/preset/datasets/test_dataset/input}"
fi
OUTPUT_DIR="${OUTPUT_DIR:-$RESULTS_DIR/output}"
export OUTPUT_DIR

echo "🚀 [02] OSEDiff inference (mode=$MODE)"
echo "  🤖 OSEDiff code: $OSEDIFF_DIR"
echo "  🏋️ osediff.pkl:  $OSEDIFF_PKL"
echo "  🤖 SD2.1-Base:   $SD21_BASE_DIR"
echo "  🤖 RAM:          $RAM_PATH"
echo "  🤖 DAPE:         $DAPE_PATH"
echo "  🖼️  input:       $INPUT_DIR"
echo "  💾 output:       $OUTPUT_DIR"
echo "  📐 upscale:      $UPSCALE"
if [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
    echo "  🎮 GPU: physical $CUDA_VISIBLE_DEVICES (cuda:0 in-process)  [GPU=N to change]"
else
    echo "  🎮 GPU: default cuda:0  [set GPU=N to pin a card]"
fi
echo ""

# ── checks ────────────────────────────────────────────────────────────────
if [ ! -d "$OSEDIFF_DIR" ]; then
    echo "❌ ERROR: OSEDiff code not found: $OSEDIFF_DIR" >&2
    echo "       Run: bash osediff/00a_setup_env.sh  (or 00_setup_env.sh on server)" >&2
    exit 1
fi
if [ ! -f "$OSEDIFF_PKL" ]; then
    echo "❌ ERROR: osediff.pkl not found: $OSEDIFF_PKL" >&2
    echo "       Should ship inside the repo (preset/models/). Re-clone." >&2
    exit 1
fi
if [ ! -d "$SD21_BASE_DIR" ] || [ ! -f "$SD21_BASE_DIR/unet/diffusion_pytorch_model.safetensors" ]; then
    echo "❌ ERROR: SD2.1-Base not found: $SD21_BASE_DIR" >&2
    echo "       Run: bash osediff/01_download_models.sh" >&2
    exit 1
fi
if [ ! -f "$RAM_PATH" ]; then
    echo "❌ ERROR: RAM not found: $RAM_PATH" >&2
    echo "       Run: bash osediff/01_download_models.sh" >&2
    exit 1
fi
if [ ! -f "$DAPE_PATH" ]; then
    echo "❌ ERROR: DAPE not found: $DAPE_PATH" >&2
    echo "       Run: bash osediff/01_download_models.sh" >&2
    exit 1
fi
if [ ! -d "$INPUT_DIR" ]; then
    echo "❌ ERROR: input dir not found: $INPUT_DIR" >&2
    exit 1
fi

mkdir -p "$OUTPUT_DIR"

# ── run official test_osediff.py from the repo dir (rel imports + preset) ─
cd "$OSEDIFF_DIR"

EXTRA_ARGS=()
if [ "$MODE" = "face" ]; then
    EXTRA_ARGS+=(--align_method "${ALIGN_METHOD}")
fi

echo "🎬 running test_osediff.py ..."
python test_osediff.py \
    -i "$INPUT_DIR" \
    -o "$OUTPUT_DIR" \
    --osediff_path "$OSEDIFF_PKL" \
    --pretrained_model_name_or_path "$SD21_BASE_DIR" \
    --ram_ft_path "$DAPE_PATH" \
    --ram_path "$RAM_PATH" \
    --upscale "$UPSCALE" \
    "${EXTRA_ARGS[@]}"
if [ $? -ne 0 ]; then
    echo "❌ FAILED: test_osediff.py exited non-zero" >&2
    exit 1
fi

echo ""
echo "🎉 [02] Done. Super-resolved images in: $OUTPUT_DIR"
echo "  (WSL: run 08_move_output.sh to move them to /mnt/d/output/)"

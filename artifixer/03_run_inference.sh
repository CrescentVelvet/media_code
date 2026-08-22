#!/usr/bin/env bash
# 03_run_inference.sh — ArtiFixer inference on a prepared scene.
#
# Runs model_eval.run_inference with --evalset reconstructed_colmap.
# Corrects 3DGRUT renders using the ArtiFixer diffusion model.
#
# This is the first stage of the full ArtiFixer pipeline:
#   ArtiFixer (03) → ArtiFixer3D (04) → ArtiFixer3D+ (05)
set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/_env.sh"

# Inference uses local model paths, no network needed.
export HF_HUB_OFFLINE=1

# --- params ---
SCENE_ROOT="${SCENE_ROOT:-$RESULTS_DIR/prep/my_scene}"
SAVE_DIR="${SAVE_DIR:-$RESULTS_DIR/corrected}"
SPLIT_PATH="${SPLIT_PATH:-$SCENE_ROOT/split.json}"
RENDER_TRAJECTORY="${RENDER_TRAJECTORY:-all_frames}"

# Inference params
NUM_INFERENCE_STEPS="${NUM_INFERENCE_STEPS:-4}"
FRAMES_PER_BLOCK="${FRAMES_PER_BLOCK:-7}"
LOCAL_ATTN_SIZE="${LOCAL_ATTN_SIZE:-21}"
SINK_SIZE="${SINK_SIZE:-7}"
OUTPUT_FPS="${OUTPUT_FPS:-15}"
INFERENCE_PIPELINE="${INFERENCE_PIPELINE:-kv_cache}"
NUM_VIEWS="${NUM_VIEWS:-}"

# Memory-constrained: set to 1 to encode neighbors one at a time
MAX_NEIGHBORS_PER_ENCODE="${MAX_NEIGHBORS_PER_ENCODE:-}"

# Multi-GPU: set NUM_GPUS > 1 for torchrun distribution across GPUs
NUM_GPUS="${NUM_GPUS:-1}"

echo "🚀 [03] ArtiFixer inference"
echo "  🏋️ checkpoint:  $ARTIFIXER_CHECKPOINT"
echo "  🤖 model_id:    $WAN_MODEL_ID  (variant: $MODEL_VARIANT)"
echo "  📁 scene root:  $SCENE_ROOT"
echo "  📁 split:       $SPLIT_PATH"
echo "  💾 save dir:    $SAVE_DIR"
echo "  🎬 trajectory:  $RENDER_TRAJECTORY"
echo "  ⏱️  steps:       $NUM_INFERENCE_STEPS  (pipeline: $INFERENCE_PIPELINE)"
echo "  🎮 GPUs:        $NUM_GPUS"
if [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
    echo "  🎮 GPU:         physical $CUDA_VISIBLE_DEVICES"
fi

# --- checks ---
if [ ! -f "$ARTIFIXER_CHECKPOINT" ]; then
    echo "❌ ERROR: checkpoint not found: $ARTIFIXER_CHECKPOINT" >&2
    echo "   Run bash 01_download_models.sh first." >&2
    exit 1
fi
if [ ! -d "$WAN_MODEL_ID" ]; then
    echo "❌ ERROR: base model not found: $WAN_MODEL_ID" >&2
    echo "   Run bash 01_download_models.sh first." >&2
    exit 1
fi
if [ ! -f "$SPLIT_PATH" ]; then
    echo "❌ ERROR: split.json not found: $SPLIT_PATH" >&2
    echo "   Run bash 02_prepare_data.sh first (or set SPLIT_PATH)." >&2
    exit 1
fi
if ! python -c "import diffusers" 2>/dev/null; then
    echo "❌ ERROR: diffusers not importable. Run INSTALL_DEPS=1 bash 00_setup_env.sh" >&2
    exit 1
fi

mkdir -p "$SAVE_DIR"

# --- build command ---
CMD=(python -m model_eval.run_inference
    --evalset reconstructed_colmap
    --checkpoint_pt "$ARTIFIXER_CHECKPOINT"
    --model_id "$WAN_MODEL_ID"
    --save_dir "$SAVE_DIR"
    --split_path "$SPLIT_PATH"
    --render_trajectory "$RENDER_TRAJECTORY"
    --num_inference_steps "$NUM_INFERENCE_STEPS"
    --frames_per_block "$FRAMES_PER_BLOCK"
    --local_attn_size "$LOCAL_ATTN_SIZE"
    --sink_size "$SINK_SIZE"
    --output_fps "$OUTPUT_FPS"
    --inference_pipeline "$INFERENCE_PIPELINE"
    --default_negative_prompt_path "$ARTIFIXER_DIR/default_negative_prompt.pt"
)

if [ -n "$NUM_VIEWS" ]; then
    CMD+=(--num_views "$NUM_VIEWS")
fi
if [ -n "$MAX_NEIGHBORS_PER_ENCODE" ]; then
    CMD+=(--max_neighbors_per_encode "$MAX_NEIGHBORS_PER_ENCODE")
fi
if [ "${REPLACE_IF_EXISTS:-0}" = "1" ]; then
    CMD+=(--replace_if_exists)
fi
if [ "${SAVE_FRAME_OUTPUTS_ONLY:-0}" = "1" ]; then
    CMD+=(--save_frame_outputs_only)
fi

# --- run ---
cd "$ARTIFIXER_DIR"
export PYTHONPATH="$ARTIFIXER_DIR:${PYTHONPATH:-}"

if [ "$NUM_GPUS" -gt 1 ]; then
    echo "🎮 running with $NUM_GPUS GPUs (torchrun)..."
    torchrun --nproc_per_node "$NUM_GPUS" \
        -m model_eval.run_inference \
        "${CMD[@]:3}"  # skip "python -m model_eval.run_inference" prefix
else
    echo "🔍 running ArtiFixer inference..."
    "${CMD[@]}"
fi

if [ $? -ne 0 ]; then
    echo "❌ FAILED: ArtiFixer inference" >&2
    exit 1
fi
cd "$SCRIPT_DIR"

# --- find output ---
# Output dir pattern: $SAVE_DIR/<ckpt_stem>/distilled_views_reconstructed_colmap_<views>_...
CKPT_STEM=$(basename "$ARTIFIXER_CHECKPOINT" .pt)
echo "✅ ArtiFixer inference done."
echo "  📁 outputs: $SAVE_DIR/$CKPT_STEM/"
echo ""
echo "  For ArtiFixer3D, the predicted frames are at:"
echo "    $SAVE_DIR/$CKPT_STEM/*/frames/batch_0000/pred/"
echo ""
echo "🎉 [03] Done."
echo "    Next: bash $SCRIPT_DIR/04_run_artifixer3d.sh  (ArtiFixer3D distillation)"

#!/usr/bin/env bash
# 02_run_inference.sh — run HunyuanVideo-1.5 generate.py (T2V or I2V) via
# torchrun. One prompt per run; output is a single .mp4.
# Set IMAGE_PATH=none (default) for text-to-video, or a real image path for i2v.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/_env.sh"

HYVIDEO_DIR="${HYVIDEO_DIR:-$REPO_DIR/../HunyuanVideo-1.5}"
MODEL_PATH="${MODEL_PATH:-$HYVIDEO_DIR/ckpts}"
OUTPUT_DIR="${OUTPUT_DIR:-$HYVIDEO_DIR/outputs}"

# --- generation params (all overridable via env) ---
PROMPT="${PROMPT:-A cinematic shot of a panda eating bamboo in a misty forest, soft morning light, highly detailed.}"
IMAGE_PATH="${IMAGE_PATH:-none}"            # 'none' = text-to-video; a path = image-to-video
RESOLUTION="${RESOLUTION:-480p}"            # 480p | 720p
ASPECT_RATIO="${ASPECT_RATIO:-16:9}"
SEED="${SEED:-1}"
VIDEO_LENGTH="${VIDEO_LENGTH:-121}"
NUM_INFERENCE_STEPS="${NUM_INFERENCE_STEPS:-50}"
DTYPE="${DTYPE:-bf16}"
N_INFERENCE_GPU="${N_INFERENCE_GPU:-1}"     # torchrun --nproc_per_node

# --- acceleration / quality toggles (defaults = light, dependency-free first run) ---
REWRITE="${REWRITE:-false}"                       # needs a vLLM rewrite endpoint (T2V_REWRITE_*/I2V_REWRITE_*)
CFG_DISTILLED="${CFG_DISTILLED:-false}"           # true -> use cfg-distilled transformer (2x faster)
ENABLE_STEP_DISTILL="${ENABLE_STEP_DISTILL:-false}" # true -> 480p i2v step-distill (8/12 steps)
SPARSE_ATTN="${SPARSE_ATTN:-false}"               # true -> needs flex-block-attn (720p only)
SAGE_ATTN="${SAGE_ATTN:-false}"                   # true -> needs SageAttention built
ENABLE_CACHE="${ENABLE_CACHE:-false}"             # true -> deepcache/teacache/taylorcache
CACHE_TYPE="${CACHE_TYPE:-deepcache}"
OFFLOADING="${OFFLOADING:-true}"                  # CPU offloading (fit ~14GB VRAM)
OVERLAP_GROUP_OFFLOADING="${OVERLAP_GROUP_OFFLOADING:-true}"
ENABLE_SR="${ENABLE_SR:-false}"                   # super-resolution to 720p/1080p (needs sr model, slower)
SAVE_PRE_SR_VIDEO="${SAVE_PRE_SR_VIDEO:-0}"       # 1 -> also save the pre-SR clip

# OOM on a >=14GB card? Try: export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:128
[ -n "${PYTORCH_CUDA_ALLOC_CONF:-}" ] && export PYTORCH_CUDA_ALLOC_CONF

OUTPUT_PATH="${OUTPUT_PATH:-}"
if [ -z "$OUTPUT_PATH" ]; then
    mkdir -p "$OUTPUT_DIR"
    STEM="$(echo "$IMAGE_PATH" | tr '/' '_' | sed 's/^none$/t2v/')"
    OUTPUT_PATH="$OUTPUT_DIR/${RESOLUTION}_${STEM}_seed${SEED}.mp4"
fi

echo "=== [02] HunyuanVideo-1.5 inference ==="
echo "  code:       $HYVIDEO_DIR"
echo "  model:      $MODEL_PATH"
echo "  prompt:     $PROMPT"
echo "  image:      $IMAGE_PATH   (none = T2V)"
echo "  resolution: $RESOLUTION  $ASPECT_RATIO  ${VIDEO_LENGTH}f  steps=$NUM_INFERENCE_STEPS  dtype=$DTYPE"
echo "  accel:      cfg_distill=$CFG_DISTILLED step_distill=$ENABLE_STEP_DISTILL sparse=$SPARSE_ATTN sage=$SAGE_ATTN cache=$ENABLE_CACHE($CACHE_TYPE) sr=$ENABLE_SR offload=$OFFLOADING rewrite=$REWRITE"
echo "  output:     $OUTPUT_PATH"
echo "  GPUs:       nproc=$N_INFERENCE_GPU  $([ -n "${CUDA_VISIBLE_DEVICES:-}" ] && echo "(CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES)" || echo "(all visible)")"

if [ ! -d "$HYVIDEO_DIR" ]; then
    echo "ERROR: code dir not found at $HYVIDEO_DIR. Run run_all.sh first." >&2; exit 1
fi
if [ ! -e "$HYVIDEO_DIR/ckpts" ]; then
    echo "ERROR: $HYVIDEO_DIR/ckpts missing. Run 01_download_models.sh first." >&2; exit 1
fi

# Build the generate.py argument list (bools passed as explicit true/false,
# matching the official README examples).
ARGS=(
    --prompt "$PROMPT"
    --image_path "$IMAGE_PATH"
    --resolution "$RESOLUTION"
    --aspect_ratio "$ASPECT_RATIO"
    --seed "$SEED"
    --video_length "$VIDEO_LENGTH"
    --num_inference_steps "$NUM_INFERENCE_STEPS"
    --dtype "$DTYPE"
    --model_path "$MODEL_PATH"
    --output_path "$OUTPUT_PATH"
    --rewrite "$REWRITE"
    --cfg_distilled "$CFG_DISTILLED"
    --enable_step_distill "$ENABLE_STEP_DISTILL"
    --sparse_attn "$SPARSE_ATTN"
    --use_sageattn "$SAGE_ATTN"
    --enable_cache "$ENABLE_CACHE"
    --cache_type "$CACHE_TYPE"
    --offloading "$OFFLOADING"
    --overlap_group_offloading "$OVERLAP_GROUP_OFFLOADING"
    --sr "$ENABLE_SR"
)
[ "${ENABLE_CACHE}" = "true" ] && ARGS+=(--no_cache_block_id "${NO_CACHE_BLOCK_ID:-53}" \
    --cache_start_step "${CACHE_START_STEP:-11}" --cache_end_step "${CACHE_END_STEP:-45}" \
    --total_steps "${TOTAL_STEPS:-50}" --cache_step_interval "${CACHE_STEP_INTERVAL:-4}")
[ "$SAVE_PRE_SR_VIDEO" = "1" ] && ARGS+=(--save_pre_sr_video)
[ -n "${SAGE_BLOCKS_RANGE:-}" ] && ARGS+=(--sage_blocks_range "$SAGE_BLOCKS_RANGE")

cd "$HYVIDEO_DIR"
torchrun --nproc_per_node="$N_INFERENCE_GPU" generate.py "${ARGS[@]}"

echo "=== [02] Done. Video: $OUTPUT_PATH ==="

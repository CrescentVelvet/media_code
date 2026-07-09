#!/usr/bin/env bash
# 02_run_inference.sh — batch FLUX.1 text-to-image inference. Calls
# run_inference.py (a thin wrapper that loads the FluxPipeline ONCE, loops over
# prompts, and prints model-load + per-image timing + a summary). No
# per-prompt relaunch.
#
# This is the "文生图" (text-to-image) path: the diffusion model turns each
# text prompt into a PNG. Prompts come from a single PROMPT (one image) or a
# PROMPTS_FILE (one prompt per line -> batch). For each prompt, produces:
#   $OUTPUT_DIR/result/<idx>_<slug>.png   (generated image)
#   $OUTPUT_DIR/prompt/<idx>_<slug>.txt   (the prompt used)
#
# Defaults target FLUX.1-schnell (4-step, guidance 0.0). For FLUX.1-dev set
# NUM_INFERENCE_STEPS=28 GUIDANCE_SCALE=3.5 (and download dev with HF_TOKEN).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/_env.sh"

MODEL_DIR="${MODEL_DIR:-$REPO_DIR/../../model/FLUX1}"
HF_REPO_ID="${HF_REPO_ID:-black-forest-labs/FLUX.1-schnell}"
REPO_BASE="$(basename "$HF_REPO_ID")"
MODEL_PATH="${MODEL_PATH:-$MODEL_DIR/$REPO_BASE}"

# --- input / output ---
PROMPT="${PROMPT:-A cinematic shot of a panda eating bamboo in a misty forest, soft morning light, highly detailed.}"
PROMPTS_FILE="${PROMPTS_FILE:-}"          # a file, one prompt per line; unset -> use single PROMPT
INPUT_LABEL="${INPUT_LABEL:-prompt}"
OUTPUT_DIR="${OUTPUT_DIR:-$REPO_DIR/../FLUX1/results/$INPUT_LABEL}"

# --- generation params (all overridable via env) ---
NUM_INFERENCE_STEPS="${NUM_INFERENCE_STEPS:-4}"   # schnell=4; dev=28
GUIDANCE_SCALE="${GUIDANCE_SCALE:-0.0}"           # schnell=0.0 (no CFG); dev=3.5
HEIGHT="${HEIGHT:-1024}"
WIDTH="${WIDTH:-1024}"
MAX_SEQUENCE_LENGTH="${MAX_SEQUENCE_LENGTH:-256}" # schnell=256; dev=512
SEED="${SEED:-231}"
NUM_IMAGES_PER_PROMPT="${NUM_IMAGES_PER_PROMPT:-1}"

# --- model loading / VRAM ---
DTYPE="${DTYPE:-bf16}"                     # bf16 | fp16 | fp32 (bf16 recommended for Flux)
OFFLOAD="${OFFLOAD:-model}"                # model=enable_model_cpu_offload(~12GB) | sequential | none(whole on GPU)
VAE_SLICING="${VAE_SLICING:-1}"            # 1 -> enable_vae_slicing (lower VRAM at decode)
VAE_TILING="${VAE_TILING:-0}"              # 1 -> enable_vae_tiling (large images / very low VRAM)
ATTN_IMPL="${ATTN_IMPL:-}"                 # "" = diffusers default(sdpa); flash_attention_2 | eager (advanced)

echo "=== [02] FLUX.1 text-to-image inference ==="
echo "  模型路径:  $MODEL_PATH  ($HF_REPO_ID)"
if [ -n "$PROMPTS_FILE" ]; then
    echo "  提示词文件: $PROMPTS_FILE  (每行一条)"
else
    echo "  单条PROMPT: ${PROMPT:0:80}$( [ ${#PROMPT} -gt 80 ] && echo '...' )"
fi
echo "  输出图像:  $OUTPUT_DIR  (result/ + prompt/)"
echo "  生成参数:  steps=$NUM_INFERENCE_STEPS cfg=$GUIDANCE_SCALE ${HEIGHT}x${WIDTH} max_seq_len=$MAX_SEQUENCE_LENGTH seed=$SEED imgs/prompt=$NUM_IMAGES_PER_PROMPT"
echo "  加载参数:  dtype=$DTYPE offload=$OFFLOAD vae_slicing=$VAE_SLICING vae_tiling=$VAE_TILING attn=${ATTN_IMPL:-default}"
if [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
    echo "  GPU:       physical $CUDA_VISIBLE_DEVICES (cuda:0 in-process)  [GPU=N to change; Flux 单卡跑, 不自动多卡切分]"
else
    echo "  GPU:       default cuda:0 (= first visible)  [set GPU=N to pin a card]"
fi

# --- checks ---
if [ ! -d "$MODEL_PATH" ]; then
    echo "ERROR: model not found at $MODEL_PATH. Run 01_download_models.sh first (or set MODEL_PATH/HF_REPO_ID)." >&2; exit 1
fi
if [ -n "$PROMPTS_FILE" ] && [ ! -f "$PROMPTS_FILE" ]; then
    echo "ERROR: PROMPTS_FILE not found: $PROMPTS_FILE" >&2; exit 1
fi

mkdir -p "$OUTPUT_DIR"

# Forward params to run_inference.py (reads env; loads pipeline ONCE, loops,
# prints model-load + per-prompt timing).
export MODEL_PATH
export PROMPT PROMPTS_FILE OUTPUT_DIR
export NUM_INFERENCE_STEPS GUIDANCE_SCALE HEIGHT WIDTH MAX_SEQUENCE_LENGTH SEED NUM_IMAGES_PER_PROMPT
export DTYPE OFFLOAD VAE_SLICING VAE_TILING ATTN_IMPL

python "$SCRIPT_DIR/run_inference.py"

echo "=== [02] Done. Images in: $OUTPUT_DIR/result ==="

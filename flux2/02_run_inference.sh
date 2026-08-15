#!/usr/bin/env bash
# 02_run_inference.sh — batch FLUX.2 text-to-image AND image-editing inference.
# Calls run_inference.py (loads the pipeline ONCE, loops over prompts, prints
# model-load + per-image timing + a summary). No per-prompt relaunch.
#
# Two modes (same script, same model):
#   - 文生图 (T2I):     no INPUT_IMAGES -> prompt -> PNG.
#   - 图像编辑/多参考:  INPUT_IMAGES="a.jpg,b.jpg" -> prompt + ref image(s) -> PNG.
# Both FLUX.2-dev and FLUX.2-klein-9B support T2I + single/multi-ref editing.
#
# MODEL_TYPE selects the pipeline + per-type defaults:
#   klein (default, fast): Flux2KleinPipeline, steps=4,  guidance=1.0 (distilled)
#   dev   (max quality):    Flux2Pipeline,       steps=50, guidance=4.0
set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/_env.sh"

MODEL_TYPE="${MODEL_TYPE:-klein}"            # klein | dev
MODEL_DIR="${MODEL_DIR:-$REPO_DIR/../../model/FLUX2}"
case "$MODEL_TYPE" in
    dev)   DEF_BASE="FLUX.2-dev" ;;
    klein) DEF_BASE="FLUX.2-klein-9B" ;;
    *) echo "❌ MODEL_TYPE must be 'dev' or 'klein' (got '$MODEL_TYPE')" >&2; exit 1 ;;
esac
MODEL_PATH="${MODEL_PATH:-$MODEL_DIR/$DEF_BASE}"

# --- input / output ---
PROMPT="${PROMPT:-A cinematic shot of a panda eating bamboo in a misty forest, soft morning light, highly detailed.}"
PROMPTS_FILE="${PROMPTS_FILE:-}"             # a file, one prompt per line; unset -> single PROMPT
INPUT_IMAGES="${INPUT_IMAGES:-}"             # comma-separated ref image paths (editing/multi-ref)
INPUT_LABEL="${INPUT_LABEL:-$([ -n "$INPUT_IMAGES" ] && echo edit || echo prompt)}"
OUTPUT_DIR="${OUTPUT_DIR:-$REPO_DIR/../FLUX2/results/$INPUT_LABEL}"

# --- generation params (per-type defaults; overridable via env) ---
if [ "$MODEL_TYPE" = "dev" ]; then
    DEF_STEPS=50;  DEF_GUIDANCE=4.0
else
    DEF_STEPS=4;   DEF_GUIDANCE=1.0           # klein: step+guidance distilled (guidance ignored)
fi
NUM_INFERENCE_STEPS="${NUM_INFERENCE_STEPS:-$DEF_STEPS}"
GUIDANCE_SCALE="${GUIDANCE_SCALE:-$DEF_GUIDANCE}"
HEIGHT="${HEIGHT:-1024}"
WIDTH="${WIDTH:-1024}"
MAX_SEQUENCE_LENGTH="${MAX_SEQUENCE_LENGTH:-512}"
SEED="${SEED:-231}"
NUM_IMAGES_PER_PROMPT="${NUM_IMAGES_PER_PROMPT:-1}"

# --- model loading / VRAM ---
DTYPE="${DTYPE:-bf16}"                       # bf16 recommended (fp16 may overflow)
OFFLOAD="${OFFLOAD:-model}"                  # model(~fits w/ offload) | sequential(最省最慢) | none(整管线上GPU)
QUANT="${QUANT:-}"                           # "" | 4bit | 8bit (dev on consumer GPU; needs bitsandbytes)

echo "=== [02] FLUX.2 [$MODEL_TYPE] $([ -n "$INPUT_IMAGES" ] && echo editing || echo text-to-image) inference ==="
echo "  模型路径:  $MODEL_PATH  ($MODEL_TYPE)"
if [ -n "$PROMPTS_FILE" ]; then
    echo "  提示词文件: $PROMPTS_FILE  (每行一条)"
else
    echo "  单条PROMPT: ${PROMPT:0:80}$( [ ${#PROMPT} -gt 80 ] && echo '...' )"
fi
[ -n "$INPUT_IMAGES" ] && echo "  参考图像:  $INPUT_IMAGES"
echo "  输出图像:  $OUTPUT_DIR  (result/ + prompt/)"
echo "  生成参数:  steps=$NUM_INFERENCE_STEPS cfg=$GUIDANCE_SCALE ${HEIGHT}x${WIDTH} max_seq_len=$MAX_SEQUENCE_LENGTH seed=$SEED imgs/prompt=$NUM_IMAGES_PER_PROMPT"
echo "  加载参数:  dtype=$DTYPE offload=$OFFLOAD quant=${QUANT:-none}"
if [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
    echo "  GPU:       physical $CUDA_VISIBLE_DEVICES (cuda:0)  [GPU=N to change; Flux2 单卡跑, 不自动多卡切分]"
else
    echo "  GPU:       default cuda:0  [set GPU=N to pin a card]"
fi

# --- checks ---
if [ ! -d "$MODEL_PATH" ]; then
    echo "❌ model not found at $MODEL_PATH. Run 01_download_models.sh (or set MODEL_PATH)." >&2
    exit 1
fi
if [ ! -f "$MODEL_PATH/model_index.json" ]; then
    echo "⚠️ WARNING: $MODEL_PATH/model_index.json not found." >&2
    echo "   Flux2Pipeline.from_pretrained needs the diffusers-format snapshot (model_index.json + subfolders)." >&2
    echo "   If you have single-file *.safetensors, re-download via 01_download_models.sh (hf download <repo>)." >&2
fi
if [ -n "$PROMPTS_FILE" ] && [ ! -f "$PROMPTS_FILE" ]; then
    echo "❌ PROMPTS_FILE not found: $PROMPTS_FILE" >&2; exit 1
fi

mkdir -p "$OUTPUT_DIR"

# Forward params to run_inference.py (reads env; loads pipeline ONCE, loops).
export MODEL_TYPE MODEL_PATH
export PROMPT PROMPTS_FILE INPUT_IMAGES OUTPUT_DIR
export NUM_INFERENCE_STEPS GUIDANCE_SCALE HEIGHT WIDTH MAX_SEQUENCE_LENGTH SEED NUM_IMAGES_PER_PROMPT
export DTYPE OFFLOAD QUANT

python "$SCRIPT_DIR/run_inference.py"

echo "=== [02] Done. Images in: $OUTPUT_DIR/result ==="

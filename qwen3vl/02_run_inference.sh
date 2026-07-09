#!/usr/bin/env bash
# 02_run_inference.sh — batch Qwen3-VL image-to-text inference over a folder
# of images. Calls run_inference.py (a thin wrapper that loads the model +
# processor ONCE, loops over all images, and prints model-load + per-image
# timing + a summary). No per-image relaunch.
#
# For each image in IMAGE_DIR (walked recursively), produces:
#   $OUTPUT_DIR/result/<same-relative-path>.txt   (model-generated text)
#   $OUTPUT_DIR/prompt/<same-relative-path>.txt   (the prompt/question used)
#
# This is the "图生文" (image-to-text) path: the VLM looks at each image and
# answers the PROMPT (default: a detailed captioning instruction). For VQA,
# set PROMPT to your question, or pass TXT_DIR (a folder mirroring IMAGE_DIR's
# structure, .txt per image) to ask a different question per image.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/_env.sh"

MODEL_DIR="${MODEL_DIR:-$REPO_DIR/../../model/Qwen3-VL}"
HF_REPO_ID="${HF_REPO_ID:-Qwen/Qwen3-VL-7B-Instruct}"
REPO_BASE="$(basename "$HF_REPO_ID")"
MODEL_PATH="${MODEL_PATH:-$MODEL_DIR/$REPO_BASE}"

# --- input / output ---
IMAGE_DIR="${IMAGE_DIR:-$REPO_DIR/../Qwen3-VL/examples/images}"
# Per-image questions: a folder mirroring IMAGE_DIR, .txt per image. If unset,
# every image uses the global PROMPT below.
TXT_DIR="${TXT_DIR:-}"
INPUT_NAME="$(basename "$IMAGE_DIR")"
OUTPUT_DIR="${OUTPUT_DIR:-$REPO_DIR/../Qwen3-VL/results/$INPUT_NAME}"

# --- generation params (all overridable via env) ---
PROMPT="${PROMPT:-Describe this image in detail, including objects, scene, colors, text, and any notable details.}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-512}"
TEMPERATURE="${TEMPERATURE:-0.7}"
TOP_P="${TOP_P:-0.8}"
TOP_K="${TOP_K:-20}"
REPETITION_PENALTY="${REPETITION_PENALTY:-1.05}"
DO_SAMPLE="${DO_SAMPLE:-true}"          # false -> greedy (temperature ignored)
SEED="${SEED:-231}"

# --- model loading ---
DTYPE="${DTYPE:-bf16}"                  # bf16 | fp16 | fp32
DEVICE_MAP="${DEVICE_MAP:-auto}"        # auto (shard across visible GPUs) | cpu | a device map JSON
ATTN_IMPL="${ATTN_IMPL:-}"              # "" = config default; sdpa | flash_attention_2 | eager
LOAD_IN_4BIT="${LOAD_IN_4BIT:-0}"       # 1 -> bitsandbytes 4-bit (needs bitsandbytes)
LOAD_IN_8BIT="${LOAD_IN_8BIT:-0}"       # 1 -> bitsand 8-bit (needs bitsandbytes)

# --- thinking mode (Qwen3-VL thinking variants emit <think>...</think> blocks) ---
THINKING="${THINKING:-0}"               # 1 -> prepend enable_thinking=True system handling
STRIP_THINKING="${STRIP_THINKING:-0}"   # 1 -> strip <think>...</think>, keep only the final answer

echo "=== [02] Qwen3-VL image-to-text inference ==="
echo "  模型路径:  $MODEL_PATH"
echo "  输入图像:  $IMAGE_DIR"
echo "  每图提问:  ${TXT_DIR:-<none -> use global PROMPT>}"
echo "  全局PROMPT: ${PROMPT:0:80}$( [ ${#PROMPT} -gt 80 ] && echo '...' )"
echo "  输出文本:  $OUTPUT_DIR  (result/ + prompt/)"
echo "  生成参数:  max_new_tokens=$MAX_NEW_TOKENS do_sample=$DO_SAMPLE temp=$TEMPERATURE top_p=$TOP_P top_k=$TOP_K rep=$REPETITION_PENALTY seed=$SEED"
echo "  加载参数:  dtype=$DTYPE device_map=$DEVICE_MAP attn=$ATTN_IMPL 4bit=$LOAD_IN_4BIT 8bit=$LOAD_IN_8BIT thinking=$THINKING strip_thinking=$STRIP_THINKING"
if [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
    echo "  GPU:       physical $CUDA_VISIBLE_DEVICES (visible to device_map)  [GPU=N to change; GPU=0,1 to shard]"
else
    echo "  GPU:       all visible  [set GPU=N to pin a single card]"
fi

# --- checks ---
if [ ! -d "$MODEL_PATH" ]; then
    echo "ERROR: model not found at $MODEL_PATH. Run 01_download_models.sh first (or set MODEL_PATH/HF_REPO_ID)." >&2; exit 1
fi
if [ ! -d "$IMAGE_DIR" ]; then
    echo "ERROR: image dir not found: $IMAGE_DIR" >&2; exit 1
fi

mkdir -p "$OUTPUT_DIR"

# Forward params to run_inference.py (reads env; loads pipeline ONCE, loops,
# prints model-load + per-image timing).
export MODEL_PATH
export IMAGE_DIR TXT_DIR OUTPUT_DIR
export PROMPT MAX_NEW_TOKENS TEMPERATURE TOP_P TOP_K REPETITION_PENALTY DO_SAMPLE SEED
export DTYPE DEVICE_MAP ATTN_IMPL LOAD_IN_4BIT LOAD_IN_8BIT THINKING STRIP_THINKING

python "$SCRIPT_DIR/run_inference.py"

echo "=== [02] Done. Results in: $OUTPUT_DIR/result ==="

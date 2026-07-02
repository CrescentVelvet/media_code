#!/usr/bin/env bash
# 02_run_inference.sh — batch HYPIR image restoration over a folder of LQ
# images. Calls run_inference.py (a thin wrapper over SD2Enhancer that mirrors
# the official test.py: recursive walk, relative-path output, prompt handling)
# but loads the pipeline ONCE, loops over all images, and prints model-load +
# per-image timing + a summary. No per-image relaunch.
#
# For each image in LQ_DIR (walked recursively), produces:
#   $OUTPUT_DIR/result/<same-relative-path>.png   (restored)
#   $OUTPUT_DIR/prompt/<same-relative-path>.txt   (prompt used)
#
# Prompts: pass TXT_DIR (a folder mirroring LQ_DIR's structure, .txt per image)
# to use per-image captions; otherwise an empty captioner is used.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/_env.sh"

HYPIR_DIR="${HYPIR_DIR:-$REPO_DIR/../HYPIR}"
MODEL_DIR="${MODEL_DIR:-$REPO_DIR/../../model/HYPIR}"
BASE_MODEL_DIR="${BASE_MODEL_DIR:-$MODEL_DIR/sd2_base}"
WEIGHT_PATH="${WEIGHT_PATH:-$MODEL_DIR/HYPIR_sd2.pth}"

LQ_DIR="${LQ_DIR:-$HYPIR_DIR/examples/lq}"
# Default the prompt dir to the official example prompts only when LQ_DIR is the
# default example dir; for a custom LQ_DIR without prompts, fall back to empty
# captions (--captioner empty) so any image folder works out of the box.
if [ -z "${TXT_DIR+x}" ]; then
    if [ "$LQ_DIR" = "$HYPIR_DIR/examples/lq" ]; then
        TXT_DIR="$HYPIR_DIR/examples/prompt"
    else
        TXT_DIR=""
    fi
fi

INPUT_NAME="$(basename "$LQ_DIR")"
OUTPUT_DIR="${OUTPUT_DIR:-$HYPIR_DIR/results/$INPUT_NAME}"

# --- restoration params (all overridable via env) ---
SCALE_BY="${SCALE_BY:-factor}"            # factor | longest_side
UPSCALE="${UPSCALE:-4}"
TARGET_LONGEST_SIDE="${TARGET_LONGEST_SIDE:-}"   # required when SCALE_BY=longest_side
PATCH_SIZE="${PATCH_SIZE:-512}"
STRIDE="${STRIDE:-256}"
SEED="${SEED:-231}"
DEVICE="${DEVICE:-cuda}"
MODEL_T="${MODEL_T:-200}"
COEFF_T="${COEFF_T:-200}"

# Fixed LoRA module list / rank from the official HYPIR-SD2 config.
LORA_RANK="${LORA_RANK:-256}"
LORA_MODULES="${LORA_MODULES:-to_k,to_q,to_v,to_out.0,conv,conv1,conv2,conv_shortcut,conv_out,proj_in,proj_out,ff.net.2,ff.net.0.proj}"

echo "=== [02] HYPIR batch inference ==="
echo "  代码路径:      $HYPIR_DIR"
echo "  模型路径:      $BASE_MODEL_DIR"
echo "  lora路径:      $WEIGHT_PATH  (rank=$LORA_RANK)"
echo "  输入低质量图像:         $LQ_DIR"
echo "  输入文本:               ${TXT_DIR:-<none -> empty caption>}"
echo "  输出高质量图像:         $OUTPUT_DIR  (result/ + prompt/)"
echo "  参数:   scale_by=$SCALE_BY upscale=$UPSCALE patch=$PATCH_SIZE stride=$STRIDE seed=$SEED"
if [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
    echo "  GPU:       physical $CUDA_VISIBLE_DEVICES (cuda:0 in-process)  [GPU=N to change]"
else
    echo "  GPU:       default cuda:0 (= first visible)  [set GPU=N to pin a card]"
fi

# --- checks ---
if [ ! -d "$HYPIR_DIR" ]; then
    echo "ERROR: HYPIR code dir not found at $HYPIR_DIR. Run run_all.sh first." >&2; exit 1
fi
if [ ! -d "$BASE_MODEL_DIR" ]; then
    echo "ERROR: base model not found at $BASE_MODEL_DIR. Run 01_download_models.sh first." >&2; exit 1
fi
if [ ! -f "$WEIGHT_PATH" ]; then
    echo "ERROR: lora weight not found at $WEIGHT_PATH. Run 01_download_models.sh first." >&2; exit 1
fi
if [ ! -d "$LQ_DIR" ]; then
    echo "ERROR: lq dir not found: $LQ_DIR" >&2; exit 1
fi

mkdir -p "$OUTPUT_DIR"

# Make `from HYPIR...` importable when running run_inference.py from outside the repo.
export PYTHONPATH="$HYPIR_DIR:${PYTHONPATH:-}"

# Forward params to run_inference.py (reads env; loads pipeline ONCE, loops,
# prints model-load + per-image timing). Replaces the direct test.py call —
# test.py has no --config flag and prints no per-image timing.
export HYPIR_DIR
export BASE_MODEL_PATH="$BASE_MODEL_DIR"
export WEIGHT_PATH LORA_RANK LORA_MODULES MODEL_T COEFF_T
export LQ_DIR OUTPUT_DIR SCALE_BY UPSCALE PATCH_SIZE STRIDE SEED DEVICE
export TXT_DIR TARGET_LONGEST_SIDE
export CAPTIONER="${CAPTIONER:-}" FIXED_CAPTION="${FIXED_CAPTION:-}"

python "$SCRIPT_DIR/run_inference.py"

echo "=== [02] Done. Results in: $OUTPUT_DIR/result ==="

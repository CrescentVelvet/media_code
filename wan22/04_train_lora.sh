#!/usr/bin/env bash
# 04_train_lora.sh — LoRA fine-tune Wan2.2-TI2V-5B on your own video dataset.
#
# Wraps `accelerate launch train.py` (the official DiffSynth-Studio wanvideo
# trainer). LoRA is added to the DiT (dit) on q,k,v,o,ffn.0,ffn.2 layers.
# For TI2V (text+image-to-video), --extra_inputs "input_image" tells the model
# to use the first frame of each training video as the image condition.
#
# REQUIRED:
#   DATASET_BASE_PATH=/path/to/dataset   (must contain metadata.csv, from 03)
#
# Common overrides (all optional):
#   METADATA_PATH=/path/to/metadata.csv  (default: $DATASET_BASE_PATH/metadata.csv)
#   OUTPUT_DIR=../wan22_experiments/exp1
#   HEIGHT=480 WIDTH=832 NUM_FRAMES=49   (training resolution; smaller = less VRAM)
#   LEARNING_RATE=1e-4 NUM_EPOCHS=5
#   LORA_RANK=32 LORA_TARGET_MODULES="q,k,v,o,ffn.0,ffn.2"
#   GRAD_ACCUM=1                        (increase for bigger effective batch)
#   SAVE_STEPS=500                      (save every N steps; empty = every epoch)
#   LOW_VRAM_TRAIN=1                    (enable CPU offload training on small GPUs)
set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/_env.sh"

DATASET_BASE_PATH="${DATASET_BASE_PATH:-}"
if [ -z "$DATASET_BASE_PATH" ]; then
    echo "ERROR: set DATASET_BASE_PATH (from 03_build_dataset.sh) or point at a folder with metadata.csv." >&2
    echo "       e.g. DATASET_BASE_PATH=/data/wan22_dataset bash $0" >&2
    exit 1
fi
if [ ! -d "$DATASET_BASE_PATH" ]; then
    echo "ERROR: dataset dir not found: $DATASET_BASE_PATH" >&2; exit 1
fi

METADATA_PATH="${METADATA_PATH:-$DATASET_BASE_PATH/metadata.csv}"
if [ ! -f "$METADATA_PATH" ]; then
    echo "ERROR: metadata.csv not found at $METADATA_PATH." >&2
    echo "       Run 03_build_dataset.sh first, or set METADATA_PATH." >&2
    exit 1
fi

# Make paths absolute (train.py runs from DIFFSYNTH_DIR).
DATASET_BASE_PATH="$(cd "$DATASET_BASE_PATH" && pwd)"
METADATA_PATH="$(cd "$(dirname "$METADATA_PATH")" && pwd)/$(basename "$METADATA_PATH")"

# --- training hyperparams (override via env; defaults match the official example) ---
HEIGHT="${HEIGHT:-480}"
WIDTH="${WIDTH:-832}"
NUM_FRAMES="${NUM_FRAMES:-49}"
DATASET_REPEAT="${DATASET_REPEAT:-100}"
LEARNING_RATE="${LEARNING_RATE:-1e-4}"
NUM_EPOCHS="${NUM_EPOCHS:-5}"
LORA_RANK="${LORA_RANK:-32}"
LORA_TARGET_MODULES="${LORA_TARGET_MODULES:-q,k,v,o,ffn.0,ffn.2}"
GRAD_ACCUM="${GRAD_ACCUM:-1}"
SAVE_STEPS="${SAVE_STEPS:-}"
LOW_VRAM_TRAIN="${LOW_VRAM_TRAIN:-0}"
ENABLE_MODEL_CPU_OFFLOAD_FLAG=""
if [ "$LOW_VRAM_TRAIN" = "1" ]; then
    ENABLE_MODEL_CPU_OFFLOAD_FLAG="--enable_model_cpu_offload"
fi
EXTRA_INPUTS="${EXTRA_INPUTS:-input_image}"

OUTPUT_DIR="${OUTPUT_DIR:-$REPO_DIR/../wan22_experiments/exp1}"
OUTPUT_DIR="$(mkdir -p "$OUTPUT_DIR" && cd "$OUTPUT_DIR" && pwd)"

# The DiT model files (glob pattern matches sharded safetensors).
MODEL_ID_PATHS="Wan-AI/Wan2.2-TI2V-5B:diffusion_pytorch_model*.safetensors,Wan-AI/Wan2.2-TI2V-5B:models_t5_umt5-xxl-enc-bf16.pth,Wan-AI/Wan2.2-TI2V-5B:Wan2.2_VAE.pth"

echo "=== [04] Wan2.2-TI2V-5B LoRA training ==="
echo "  code:        $DIFFSYNTH_DIR"
echo "  models:      $MODEL_DIR (DIFFSYNTH_SKIP_DOWNLOAD=$DIFFSYNTH_SKIP_DOWNLOAD)"
echo "  dataset:     $DATASET_BASE_PATH"
echo "  metadata:    $METADATA_PATH"
echo "  shape:       ${WIDTH}x${HEIGHT}  frames=$NUM_FRAMES  repeat=$DATASET_REPEAT"
echo "  train:       lr=$LEARNING_RATE  epochs=$NUM_EPOCHS  grad_accum=$GRAD_ACCUM"
echo "  lora:        base=dit  rank=$LORA_RANK  targets=$LORA_TARGET_MODULES"
echo "  extra_inputs: $EXTRA_INPUTS  (first frame as image condition)"
echo "  output:      $OUTPUT_DIR"
echo "  low_vram:    $LOW_VRAM_TRAIN"
if [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
    echo "  GPU:         physical $CUDA_VISIBLE_DEVICES (cuda:0 in-process)  [GPU=N to change]"
else
    echo "  GPU:         all visible  [set GPU=N to pin a single card]"
fi

# --- checks ---
if [ ! -d "$DIFFSYNTH_DIR" ]; then
    echo "ERROR: DiffSynth-Studio code dir not found at $DIFFSYNTH_DIR. Run 00_setup_env.sh first." >&2; exit 1
fi
if ! python -c "import diffsynth, accelerate" 2>/dev/null; then
    echo "ERROR: training deps missing. Run: INSTALL_DEPS=1 bash $SCRIPT_DIR/00_setup_env.sh" >&2; exit 1
fi

cd "$DIFFSYNTH_DIR"

# --- build accelerate args ---
ACCEL_ARGS=()
# Multi-GPU: run `accelerate config` once, then leave GPU unset so all visible
# cards are used. Single-GPU (default) just needs GPU=N or the first visible card.
if [ -n "${N_TRAIN_GPU:-}" ] && [ "${N_TRAIN_GPU:-1}" -gt 1 ]; then
    ACCEL_ARGS+=(--num_processes "$N_TRAIN_GPU")
fi

# Count visible GPUs (for multi-GPU auto-detect via CUDA_VISIBLE_DEVICES).
if [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
    _NGPU=$(echo "$CUDA_VISIBLE_DEVICES" | awk -F, '{print NF}')
else
    _NGPU=$(nvidia-smi -L 2>/dev/null | wc -l)
    [ "$_NGPU" -eq 0 ] && _NGPU=1
fi
_MULTI=0
[ "$_NGPU" -gt 1 ] && _MULTI=1
if [ -n "${N_TRAIN_GPU:-}" ] && [ "${N_TRAIN_GPU:-1}" -gt 1 ]; then _MULTI=1; fi

# Multi-GPU needs a free port (avoid default 29500 collision).
if [ "$_MULTI" = "1" ]; then
    if [ -z "${PORT:-}" ]; then
        PORT=$(python -c 'import socket; s=socket.socket(); s.bind(("",0)); print(s.getsockname()[1]); s.close()' 2>/dev/null)
    fi
    : "${PORT:=29530}"
    ACCEL_ARGS+=(--main_process_port "$PORT")
fi

# --- save_steps: if set, pass it; otherwise checkpoints are saved every epoch ---
SAVE_ARGS=()
if [ -n "$SAVE_STEPS" ]; then
    SAVE_ARGS+=(--save_steps "$SAVE_STEPS")
fi

# --- launch the official trainer ---
# train.py is at examples/wanvideo/model_training/train.py in the DiffSynth-Studio repo.
TRAIN_PY="$DIFFSYNTH_DIR/examples/wanvideo/model_training/train.py"

accelerate launch "${ACCEL_ARGS[@]}" "$TRAIN_PY" \
    --dataset_base_path "$DATASET_BASE_PATH" \
    --dataset_metadata_path "$METADATA_PATH" \
    --height "$HEIGHT" \
    --width "$WIDTH" \
    --num_frames "$NUM_FRAMES" \
    --dataset_repeat "$DATASET_REPEAT" \
    --model_id_with_origin_paths "$MODEL_ID_PATHS" \
    --learning_rate "$LEARNING_RATE" \
    --num_epochs "$NUM_EPOCHS" \
    --remove_prefix_in_ckpt "pipe.dit." \
    --output_path "$OUTPUT_DIR" \
    --lora_base_model "dit" \
    --lora_target_modules "$LORA_TARGET_MODULES" \
    --lora_rank "$LORA_RANK" \
    --extra_inputs "$EXTRA_INPUTS" \
    --gradient_accumulation_steps "$GRAD_ACCUM" \
    --use_gradient_checkpointing \
    $ENABLE_MODEL_CPU_OFFLOAD_FLAG \
    "${SAVE_ARGS[@]}" \
    ${ENABLE_TENSORBOARD:+--enable_tensorboard_log}

echo "=== [04] Done. LoRA checkpoints in: $OUTPUT_DIR ==="
echo "    Each epoch saves epoch-N.safetensors (LoRA weights)."
echo "    Use for inference:"
echo "      WEIGHT_PATH=$OUTPUT_DIR/epoch-N.safetensors bash $SCRIPT_DIR/02_run_inference.sh"

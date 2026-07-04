#!/usr/bin/env bash
# 04_train.sh — LoRA fine-tune HYPIR-SD2 on your own image dataset.
#
# Wraps `accelerate launch train.py --config <filled>`: it generates a derived
# config (gen_train_config.py fills the TODOs in configs/sd2_train.yaml — no
# official file is modified) and launches the official trainer.
#
# REQUIRED:
#   PARQUET_PATH=/path/to/hypir_train.parquet   (from 03_build_dataset.sh)
#
# Common overrides (all optional):
#   DATA_DIR=/path/to/images        (if set, runs 03 first to build the parquet)
#   OUTPUT_DIR=../HYPIR/experiments/exp1
#   CROP_TYPE=none|random|center    (use random if your GT images are >512 and
#                                    you did NOT pre-crop with 03 CROP=1)
#   OUT_SIZE=512
#   MAX_TRAIN_STEPS=30000  BATCH_SIZE=6  LR_G=1e-5
#   RESUME=/path/to/checkpoint-N
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/_env.sh"

HYPIR_DIR="${HYPIR_DIR:-$REPO_DIR/../HYPIR}"
MODEL_DIR="${MODEL_DIR:-$REPO_DIR/../../model/HYPIR}"
BASE_MODEL_PATH="${BASE_MODEL_PATH:-$MODEL_DIR/sd2_base}"

OUTPUT_DIR="${OUTPUT_DIR:-$HYPIR_DIR/experiments/exp1}"
OUTPUT_DIR="$(mkdir -p "$OUTPUT_DIR" && cd "$OUTPUT_DIR" && pwd)"

PARQUET_PATH="${PARQUET_PATH:-}"

# If the user gave DATA_DIR but no parquet, build one first (CROP=1 by default).
if [ -z "$PARQUET_PATH" ] && [ -n "${DATA_DIR:-}" ]; then
    echo "--- no PARQUET_PATH; building one from DATA_DIR=$DATA_DIR (03_build_dataset.sh) ---"
    PARQUET_OUT="$OUTPUT_DIR/hypir_train.parquet" \
        bash "$SCRIPT_DIR/03_build_dataset.sh"
    PARQUET_PATH="$OUTPUT_DIR/hypir_train.parquet"
fi

if [ -z "$PARQUET_PATH" ]; then
    echo "ERROR: set PARQUET_PATH (from 03_build_dataset.sh) or DATA_DIR." >&2
    echo "       e.g. PARQUET_PATH=/data/hypir_train.parquet bash $0" >&2
    exit 1
fi
if [ ! -f "$PARQUET_PATH" ]; then
    echo "ERROR: parquet not found: $PARQUET_PATH" >&2; exit 1
fi
PARQUET_PATH="$(cd "$(dirname "$PARQUET_PATH")" && pwd)/$(basename "$PARQUET_PATH")"

# --- dataset shape (optional overrides) ---
CROP_TYPE="${CROP_TYPE:-none}"      # none|random|center ; none needs 512x512 GTs
OUT_SIZE="${OUT_SIZE:-512}"
IMAGE_PATH_PREFIX="${IMAGE_PATH_PREFIX:-}"
IMAGE_PATH_KEY="${IMAGE_PATH_KEY:-image_path}"
PROMPT_KEY="${PROMPT_KEY:-prompt}"

# --- training hyperparams (override via env; defaults match sd2_train.yaml) ---
export MAX_TRAIN_STEPS="${MAX_TRAIN_STEPS:-30000}"
export BATCH_SIZE="${BATCH_SIZE:-6}"
export LR_G="${LR_G:-1e-5}"
export LR_D="${LR_D:-1e-5}"
export GRAD_ACCUM="${GRAD_ACCUM:-1}"
export SEED="${SEED:-231}"
export CHECKPOINTING_STEPS="${CHECKPOINTING_STEPS:-500}"
export LOG_IMAGE_STEPS="${LOG_IMAGE_STEPS:-100}"
export LOG_GRAD_STEPS="${LOG_GRAD_STEPS:-100}"
RESUME="${RESUME:-}"

echo "=== [04] HYPIR-SD2 LoRA training ==="
echo "  code:      $HYPIR_DIR"
echo "  base:      $BASE_MODEL_PATH"
echo "  parquet:   $PARQUET_PATH"
echo "  dataset:   crop_type=$CROP_TYPE out_size=$OUT_SIZE"
echo "  train:     steps=$MAX_TRAIN_STEPS bs=$BATCH_SIZE grad_accum=$GRAD_ACCUM lr_G=$LR_G lr_D=$LR_D seed=$SEED"
echo "  ckpt:      every $CHECKPOINTING_STEPS steps; log_img=$LOG_IMAGE_STEPS log_grad=$LOG_GRAD_STEPS"
echo "  output:    $OUTPUT_DIR"
[ -n "$RESUME" ] && echo "  resume:    $RESUME"
if [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
    echo "  GPU:       physical $CUDA_VISIBLE_DEVICES (cuda:0 in-process)  [GPU=N to change]"
else
    echo "  GPU:       all visible  [set GPU=N to pin a single card]"
fi

# --- checks ---
if [ ! -d "$HYPIR_DIR" ]; then
    echo "ERROR: HYPIR code dir not found at $HYPIR_DIR. Run run_all.sh first." >&2; exit 1
fi
if [ ! -d "$BASE_MODEL_PATH" ]; then
    echo "ERROR: base model not found at $BASE_MODEL_PATH. Run 01_download_models.sh first." >&2; exit 1
fi
if ! python -c "import accelerate, omegaconf, peft, diffusers" 2>/dev/null; then
    echo "ERROR: training deps missing. Run: INSTALL_DEPS=1 bash $SCRIPT_DIR/00_setup_env.sh" >&2
    exit 1
fi

# --- 1. generate the filled config (no official file is modified) ---
TEMPLATE="$HYPIR_DIR/configs/sd2_train.yaml"
CONFIG_OUT="$OUTPUT_DIR/sd2_train_filled.yaml"

export TEMPLATE CONFIG_OUT OUTPUT_DIR PARQUET_PATH BASE_MODEL_PATH
export CROP_TYPE OUT_SIZE IMAGE_PATH_PREFIX IMAGE_PATH_KEY PROMPT_KEY
python "$SCRIPT_DIR/gen_train_config.py"

# --- 2. patch the scalar hyperparams the trainer reads from the top-level config ---
python - <<PY
from omegaconf import OmegaConf
p = "$CONFIG_OUT"
cfg = OmegaConf.load(p)
cfg.max_train_steps = int("$MAX_TRAIN_STEPS")
cfg.data_config.train.batch_size = int("$BATCH_SIZE")
cfg.lr_G = float("$LR_G")
cfg.lr_D = float("$LR_D")
cfg.gradient_accumulation_steps = int("$GRAD_ACCUM")
cfg.seed = int("$SEED")
cfg.checkpointing_steps = int("$CHECKPOINTING_STEPS")
cfg.log_image_steps = int("$LOG_IMAGE_STEPS")
cfg.log_grad_steps = int("$LOG_GRAD_STEPS")
res = "$RESUME"
cfg.resume_from_checkpoint = res if res else None
OmegaConf.save(cfg, p)
print(f"[*] patched hyperparams -> {p}")
PY

# --- 3. launch the official trainer ---
# Multi-GPU: run `accelerate config` once, then leave GPU unset so all visible
# cards are used. Single-GPU (default) just needs GPU=N or the first visible card.
cd "$HYPIR_DIR"
ACCEL_ARGS=()
# If multiple GPUs are visible and the user set N_TRAIN_GPU, use torchrun-style
# multi-process via accelerate.
if [ -n "${N_TRAIN_GPU:-}" ] && [ "${N_TRAIN_GPU:-1}" -gt 1 ]; then
    ACCEL_ARGS+=(--num_processes "$N_TRAIN_GPU")
fi
[ -n "${MIXED_PRECISION:-}" ] && ACCEL_ARGS+=(--mixed_precision "$MIXED_PRECISION")
# 多卡时指定分布式端口，避开默认 29500 被占(0=自动找空闲端口；单卡时被忽略)。
ACCEL_ARGS+=(--main_process_port "${PORT:-0}")

accelerate launch "${ACCEL_ARGS[@]}" train.py --config "$CONFIG_OUT"

echo "=== [04] Done. Checkpoints in: $OUTPUT_DIR ==="
echo "    Each checkpoint-N/ holds state_dict.pth (LoRA params)."
echo "    Use it for inference: WEIGHT_PATH=$OUTPUT_DIR/checkpoint-N/state_dict.pth bash $SCRIPT_DIR/02_run_inference.sh"

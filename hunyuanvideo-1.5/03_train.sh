#!/usr/bin/env bash
# 03_train.sh — fine-tune / continue-train HunyuanVideo-1.5 on your own
# video+caption dataset. Wraps torchrun + train_run.py (which injects a real
# dataloader into the official train.py without forking it).
#
# REQUIRED:
#   DATA_DIR=/path/to/dataset     (see README / train_dataset.py for layout)
#
# Common overrides (all optional):
#   N_TRAIN_GPU=8  PRETRAINED_TRANSFORMER_VERSION=480p_t2v
#   LEARNING_RATE=1e-5  BATCH_SIZE=1  MAX_STEPS=10000  WARMUP_STEPS=500
#   GRAD_ACCUM=1  DTYPE=bf16  SEED=42  I2V_PROB=0.3
#   USE_MUON=true  ENABLE_FSDP=true  ENABLE_GRAD_CKPT=true  SP_SIZE=8
#   USE_LORA=true  LORA_R=8  LORA_ALPHA=16
#   TRAIN_VIDEO_LENGTH=41 (4n+1)  TRAIN_RESOLUTION=480p  TRAIN_HEIGHT/TRAIN_WIDTH
#   OUTPUT_DIR=...  SAVE_INTERVAL=1000  LOG_INTERVAL=10  VALIDATION_INTERVAL=100
#   RESUME=/path/to/checkpoint-N
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/_env.sh"

HYVIDEO_DIR="${HYVIDEO_DIR:-$REPO_DIR/../HunyuanVideo-1.5}"
MODEL_PATH="${MODEL_PATH:-$HYVIDEO_DIR/ckpts}"

DATA_DIR="${DATA_DIR:-}"
if [ -z "$DATA_DIR" ]; then
    echo "ERROR: set DATA_DIR to your dataset root." >&2
    echo "       Layout: captions.jsonl OR videos/*.mp4 (+ optional prompts/<stem>.txt)." >&2
    echo "       See hunyuanvideo-1.5/README.md and train_dataset.py." >&2
    exit 1
fi
if [ ! -d "$DATA_DIR" ]; then
    echo "ERROR: DATA_DIR not found: $DATA_DIR" >&2; exit 1
fi
DATA_DIR="$(cd "$DATA_DIR" && pwd)"   # absolute so workers resolve it regardless of cwd

# --- model / data shape ---
PRETRAINED_TRANSFORMER_VERSION="${PRETRAINED_TRANSFORMER_VERSION:-480p_t2v}"
export TRAIN_VIDEO_LENGTH="${TRAIN_VIDEO_LENGTH:-41}"          # must be 4n+1
export TRAIN_RESOLUTION="${TRAIN_RESOLUTION:-480p}"            # 480p|720p (or TRAIN_HEIGHT/TRAIN_WIDTH)
export TRAIN_HEIGHT="${TRAIN_HEIGHT:-}"
export TRAIN_WIDTH="${TRAIN_WIDTH:-}"

# --- training hyperparams ---
LEARNING_RATE="${LEARNING_RATE:-1e-5}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.01}"
BATCH_SIZE="${BATCH_SIZE:-1}"
MAX_STEPS="${MAX_STEPS:-10000}"
WARMUP_STEPS="${WARMUP_STEPS:-500}"
GRAD_ACCUM="${GRAD_ACCUM:-1}"
MAX_GRAD_NORM="${MAX_GRAD_NORM:-1.0}"
DTYPE="${DTYPE:-bf16}"
SEED="${SEED:-42}"
I2V_PROB="${I2V_PROB:-0.3}"
NUM_WORKERS="${NUM_WORKERS:-4}"

# --- optimizer / parallelism ---
USE_MUON="${USE_MUON:-true}"
ENABLE_FSDP="${ENABLE_FSDP:-true}"
ENABLE_GRAD_CKPT="${ENABLE_GRAD_CKPT:-true}"
SP_SIZE="${SP_SIZE:-1}"            # must divide world_size; 1 is always safe
DP_REPLICATE="${DP_REPLICATE:-1}"
N_TRAIN_GPU="${N_TRAIN_GPU:-1}"

# --- output / schedule ---
OUTPUT_DIR="${OUTPUT_DIR:-$HYVIDEO_DIR/outputs_train}"
OUTPUT_DIR="$(mkdir -p "$OUTPUT_DIR" && cd "$OUTPUT_DIR" && pwd)"
SAVE_INTERVAL="${SAVE_INTERVAL:-1000}"
LOG_INTERVAL="${LOG_INTERVAL:-10}"
VALIDATION_INTERVAL="${VALIDATION_INTERVAL:-100}"
TRAIN_TIMESTEP_SHIFT="${TRAIN_TIMESTEP_SHIFT:-3.0}"
FLOW_SNR_TYPE="${FLOW_SNR_TYPE:-lognorm}"

# --- LoRA ---
USE_LORA="${USE_LORA:-false}"
LORA_R="${LORA_R:-8}"
LORA_ALPHA="${LORA_ALPHA:-16}"

# --- resume ---
RESUME="${RESUME:-}"

echo "=== [03] HunyuanVideo-1.5 training ==="
echo "  code:        $HYVIDEO_DIR"
echo "  model:       $MODEL_PATH  (transformer: $PRETRAINED_TRANSFORMER_VERSION)"
echo "  dataset:     $DATA_DIR  (len via train_dataset)"
SHAPE="video_length=$TRAIN_VIDEO_LENGTH  res=$TRAIN_RESOLUTION"
[ -n "$TRAIN_HEIGHT" ] && SHAPE="$SHAPE  h=$TRAIN_HEIGHT"
[ -n "$TRAIN_WIDTH" ]  && SHAPE="$SHAPE  w=$TRAIN_WIDTH"
echo "  data shape:  $SHAPE"
echo "  train:       lr=$LEARNING_RATE  bs=$BATCH_SIZE  max_steps=$MAX_STEPS  warmup=$WARMUP_STEPS  grad_accum=$GRAD_ACCUM  dtype=$DTYPE  i2v_prob=$I2V_PROB"
echo "  optim/par:   muon=$USE_MUON  fsdp=$ENABLE_FSDP  grad_ckpt=$ENABLE_GRAD_CKPT  sp=$SP_SIZE  dp_replicate=$DP_REPLICATE  world=$N_TRAIN_GPU"
echo "  lora:        $USE_LORA  ${USE_LORA:+r=$LORA_R alpha=$LORA_ALPHA}"
echo "  output:      $OUTPUT_DIR  (save=$SAVE_INTERVAL log=$LOG_INTERVAL val=$VALIDATION_INTERVAL)"
[ -n "$RESUME" ] && echo "  resume:      $RESUME"
if [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
    echo "  GPU:         physical $CUDA_VISIBLE_DEVICES (cuda:0 in-process)  [GPU=N to change]"
else
    echo "  GPU:         all visible  [set GPU=N to pin a single card]"
fi

if [ ! -d "$HYVIDEO_DIR" ]; then
    echo "ERROR: code dir not found at $HYVIDEO_DIR. Run run_all.sh first." >&2; exit 1
fi
if [ ! -e "$HYVIDEO_DIR/ckpts" ]; then
    echo "ERROR: $HYVIDEO_DIR/ckpts missing. Run 01_download_models.sh first." >&2; exit 1
fi

# sp_size must divide world_size (N_TRAIN_GPU). SP_SIZE=1 is always valid.
if [ "$(( N_TRAIN_GPU % SP_SIZE ))" -ne 0 ]; then
    echo "ERROR: SP_SIZE=$SP_SIZE must divide N_TRAIN_GPU=$N_TRAIN_GPU." >&2; exit 1
fi

ARGS=(
    --pretrained_model_root "$MODEL_PATH"
    --pretrained_transformer_version "$PRETRAINED_TRANSFORMER_VERSION"
    --learning_rate "$LEARNING_RATE"
    --weight_decay "$WEIGHT_DECAY"
    --batch_size "$BATCH_SIZE"
    --max_steps "$MAX_STEPS"
    --warmup_steps "$WARMUP_STEPS"
    --gradient_accumulation_steps "$GRAD_ACCUM"
    --max_grad_norm "$MAX_GRAD_NORM"
    --dtype "$DTYPE"
    --seed "$SEED"
    --i2v_prob "$I2V_PROB"
    --num_workers "$NUM_WORKERS"
    --output_dir "$OUTPUT_DIR"
    --save_interval "$SAVE_INTERVAL"
    --log_interval "$LOG_INTERVAL"
    --validation_interval "$VALIDATION_INTERVAL"
    --train_timestep_shift "$TRAIN_TIMESTEP_SHIFT"
    --flow_snr_type "$FLOW_SNR_TYPE"
    --use_muon "$USE_MUON"
    --enable_fsdp "$ENABLE_FSDP"
    --enable_gradient_checkpointing "$ENABLE_GRAD_CKPT"
    --sp_size "$SP_SIZE"
    --dp_replicate "$DP_REPLICATE"
    --use_lora "$USE_LORA"
)
[ "$USE_LORA" = "true" ] && ARGS+=(--lora_r "$LORA_R" --lora_alpha "$LORA_ALPHA")
[ -n "$RESUME" ] && ARGS+=(--resume_from_checkpoint "$RESUME")
[ -n "${VALIDATION_PROMPTS:-}" ] && ARGS+=(--validation_prompts $VALIDATION_PROMPTS)

# Pass HYVIDEO_DIR so train_run.py can `import train` + `from hyvideo...` without
# forking the official repo; DATA_DIR is already exported for train_dataset.
export HYVIDEO_DIR
export DATA_DIR

cd "$HYVIDEO_DIR"
torchrun --nproc_per_node="$N_TRAIN_GPU" "$SCRIPT_DIR/train_run.py" "${ARGS[@]}"

echo "=== [03] Done. Checkpoints in: $OUTPUT_DIR ==="

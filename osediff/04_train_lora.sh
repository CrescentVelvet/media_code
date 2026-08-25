#!/usr/bin/env bash
# 04_train_lora.sh — LoRA fine-tune OSEDiff (Real-World Image Super-Resolution).
#
# Wraps the official train_osediff.py (accelerate launch). The paper uses 4 GPUs
# with batch_size=4; this script defaults to single-card training (batch_size=1,
# gradient_accumulation=4 = equivalent batch 4) which fits a 24GB RTX 3090 thanks
# to fp16 + xformers + gradient checkpointing + LoRA(rank=4).
#
# For face restoration, set MODE=face (uses train_osediff_face.py + codeformer).
#
# Usage:
#   GPU=0 DATASET_TXT=/path/to/lsdir.txt bash osediff/04_train_lora.sh
#   GPU=0 DATASET_TXT=/path/to/lsdir.txt DATASET_TXT2=/path/to/ffhq.txt \
#       bash osediff/04_train_lora.sh
#   GPU=0 MODE=face DATASET_TXT=/path/to/ffhq.txt bash osediff/04_train_lora.sh
set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/_env.sh"

# ── required: dataset txt (one absolute image path per line) ──────────────
DATASET_TXT="${DATASET_TXT:-}"
if [ -z "$DATASET_TXT" ]; then
    echo "❌ ERROR: DATASET_TXT not set." >&2
    echo "       Create a txt with one absolute image path per line (LSDIR/FFHQ)." >&2
    echo "       Example: DATASET_TXT=/path/to/lsdir.txt bash $0" >&2
    exit 1
fi
if [ ! -f "$DATASET_TXT" ]; then
    echo "❌ ERROR: dataset txt not found: $DATASET_TXT" >&2
    exit 1
fi
DATASET_TXT="$(cd "$(dirname "$DATASET_TXT")" && pwd)/$(basename "$DATASET_TXT")"

# Optional second dataset (e.g. FFHQ) mixed with the first.
DATASET_TXT2="${DATASET_TXT2:-}"
if [ -n "$DATASET_TXT2" ]; then
    if [ ! -f "$DATASET_TXT2" ]; then
        echo "❌ ERROR: DATASET_TXT2 not found: $DATASET_TXT2" >&2
        exit 1
    fi
    DATASET_TXT2="$(cd "$(dirname "$DATASET_TXT2")" && pwd)/$(basename "$DATASET_TXT2")"
fi

# ── mode: sr (default) or face ────────────────────────────────────────────
MODE="${MODE:-sr}"
if [ "$MODE" = "face" ]; then
    TRAIN_PY="$OSEDIFF_DIR/train_osediff_face.py"
    DEG_FILE="${DEG_FILE:-params_codeformer.yml}"
else
    TRAIN_PY="$OSEDIFF_DIR/train_osediff.py"
    DEG_FILE="${DEG_FILE:-params_realesrgan.yml}"
fi

# ── training hyperparams (all overridable via env) ────────────────────────
LEARNING_RATE="${LEARNING_RATE:-5e-5}"
# Single card: bs=1 + grad_accum=4 ≈ paper's bs=4. Multi-card: bs=4 accum=1.
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-1}"
GRAD_ACCUM="${GRAD_ACCUM:-4}"
LORA_RANK="${LORA_RANK:-4}"
CFG_VSD="${CFG_VSD:-7.5}"
LAMBDA_LPIPS="${LAMBDA_LPIPS:-2}"
LAMBDA_L2="${LAMBDA_L2:-1}"
LAMBDA_VSD="${LAMBDA_VSD:-1}"
LAMBDA_VSD_LORA="${LAMBDA_VSD_LORA:-1}"
MIXED_PRECISION="${MIXED_PRECISION:-fp16}"
CHECKPOINTING_STEPS="${CHECKPOINTING_STEPS:-500}"
SEED="${SEED:-123}"
NEG_PROMPT="${NEG_PROMPT:-painting, oil painting, illustration, drawing, art, sketch, cartoon, CG Style, 3D render, unreal engine, blurring, dirty, messy, worst quality, low quality, frames, watermark, signature, jpeg artifacts, deformed, lowres, over-smooth}"

OUTPUT_DIR="${OUTPUT_DIR:-$EXPERIMENTS_DIR/exp1}"
OUTPUT_DIR="$(mkdir -p "$OUTPUT_DIR" && cd "$OUTPUT_DIR" && pwd)"

echo "🚀 [04] Train OSEDiff LoRA (mode=$MODE)"
echo "  🤖 OSEDiff code: $OSEDIFF_DIR"
echo "  🏋️ SD2.1-Base:   $SD21_BASE_DIR"
echo "  🤖 RAM:          $RAM_PATH"
echo "  📁 dataset txt:  $DATASET_TXT"
[ -n "$DATASET_TXT2" ] && echo "  📁 dataset txt2: $DATASET_TXT2"
echo "  💾 output:       $OUTPUT_DIR"
echo "  ⚙️  lr=$LEARNING_RATE  bs=$TRAIN_BATCH_SIZE  accum=$GRAD_ACCUM  lora_rank=$LORA_RANK  precision=$MIXED_PRECISION"
if [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
    echo "  🎮 GPU: physical $CUDA_VISIBLE_DEVICES  [GPU=N to change]"
else
    echo "  🎮 GPU: default cuda:0  [set GPU=N to pin a card]"
fi
echo ""

# ── checks ────────────────────────────────────────────────────────────────
if [ ! -d "$OSEDIFF_DIR" ]; then
    echo "❌ ERROR: OSEDiff code not found: $OSEDIFF_DIR" >&2; exit 1
fi
if [ ! -f "$TRAIN_PY" ]; then
    echo "❌ ERROR: trainer not found: $TRAIN_PY" >&2; exit 1
fi
if [ ! -d "$SD21_BASE_DIR" ]; then
    echo "❌ ERROR: SD2.1-Base not found: $SD21_BASE_DIR" >&2; exit 1
fi
if [ ! -f "$RAM_PATH" ]; then
    echo "❌ ERROR: RAM not found: $RAM_PATH" >&2; exit 1
fi
if ! python -c "import accelerate" 2>/dev/null; then
    echo "❌ ERROR: accelerate not installed (pip install accelerate)" >&2; exit 1
fi
if ! python -c "import xformers.ops" 2>/dev/null; then
    echo "⚠️  xformers missing — training will be slower / may OOM. Install xformers==0.0.20." >&2
fi

# ── build accelerate args (multi-card auto port) ──────────────────────────
ACCEL_ARGS=()
# Count visible GPUs for multi-card detection.
if [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
    _NGPU=$(echo "$CUDA_VISIBLE_DEVICES" | awk -F, '{print NF}')
else
    _NGPU=$(nvidia-smi -L 2>/dev/null | wc -l)
    [ "$_NGPU" -eq 0 ] && _NGPU=1
fi
if [ -n "${N_TRAIN_GPU:-}" ] && [ "${N_TRAIN_GPU:-1}" -gt 1 ]; then
    _NGPU="$N_TRAIN_GPU"
    ACCEL_ARGS+=(--num_processes "$N_TRAIN_GPU")
    # find a free port (avoid the default 29500 clash)
    if [ -z "${PORT:-}" ]; then
        PORT=$(python -c 'import socket; s=socket.socket(); s.bind(("",0)); print(s.getsockname()[1]); s.close()' 2>/dev/null)
    fi
    : "${PORT:=29530}"
    ACCEL_ARGS+=(--main_process_port "$PORT")
fi
echo "  🎮 accelerate: ${_NGPU} process(es)"

# dataset list args (1 or 2 txt files + matching probabilities)
if [ -n "$DATASET_TXT2" ]; then
    DATASET_LIST_ARGS=(
        --dataset_txt_paths_list "$DATASET_TXT","$DATASET_TXT2"
        --dataset_prob_paths_list 1,1
    )
else
    DATASET_LIST_ARGS=(
        --dataset_txt_paths_list "$DATASET_TXT"
        --dataset_prob_paths_list 1
    )
fi

# ── launch the official trainer (run from repo dir for relative imports) ───
cd "$OSEDIFF_DIR"

echo ""
echo "🏋️ accelerate launch ${ACCEL_ARGS[*]} $TRAIN_PY ..."
accelerate launch "${ACCEL_ARGS[@]}" "$TRAIN_PY" \
    --pretrained_model_name_or_path="$SD21_BASE_DIR" \
    --ram_path="$RAM_PATH" \
    --learning_rate="$LEARNING_RATE" \
    --train_batch_size="$TRAIN_BATCH_SIZE" \
    --gradient_accumulation_steps="$GRAD_ACCUM" \
    --enable_xformers_memory_efficient_attention \
    --checkpointing_steps "$CHECKPOINTING_STEPS" \
    --mixed_precision="$MIXED_PRECISION" \
    --report_to "tensorboard" \
    --seed "$SEED" \
    --output_dir="$OUTPUT_DIR" \
    "${DATASET_LIST_ARGS[@]}" \
    --neg_prompt="$NEG_PROMPT" \
    --cfg_vsd="$CFG_VSD" \
    --lora_rank="$LORA_RANK" \
    --lambda_lpips="$LAMBDA_LPIPS" \
    --lambda_l2="$LAMBDA_L2" \
    --lambda_vsd="$LAMBDA_VSD" \
    --lambda_vsd_lora="$LAMBDA_VSD_LORA" \
    --deg_file_path="$DEG_FILE" \
    --tracker_project_name "train_osediff"
if [ $? -ne 0 ]; then
    echo "❌ FAILED: $TRAIN_PY exited non-zero" >&2
    exit 1
fi

echo ""
echo "🎉 [04] Done. LoRA checkpoints in: $OUTPUT_DIR"
echo "  To run inference with your trained LoRA:"
echo "    GPU=0 OSEDIFF_PKL=$OUTPUT_DIR/osediff.pkl bash $SCRIPT_DIR/02_run_inference.sh"

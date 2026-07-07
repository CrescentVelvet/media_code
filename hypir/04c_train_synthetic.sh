#!/usr/bin/env bash
# 04c_train_synthetic.sh — 在 HQ-only 数据上用 HYPIR 官方在线退化 + 暖启动发布 LoRA 微调。
#
# 与 04b_train_paired.sh 的区别：config 用官方 configs/sd2_train.yaml
# （RealESRGANDataset + RealESRGANBatchTransform，在线合成 LQ：blur/sinc/noise/jpeg
#  两阶段，每 epoch 随机刷新）而非 sd2_train_paired.yaml（真实配对，不退化）。
# 暖启动机制相同：FineTuneSD2Trainer（train_paired.py）从 config.lora_weight_path
#  加载发布 LoRA（你改过的 sd2.py 会 torch.load(config.weight_path)）。
#
# 输入：03c 产出的 HQ-only parquet（image_path + prompt）。
# 流程：gen_train_config.py 填官方 sd2_train.yaml 的 TODO -> heredoc 补
#       lora_weight_path + resume_ema=false + 标量超参 -> accelerate launch train_paired.py
#
# 必填(二选一)：PARQUET_PATH=.../hypir_synthetic.parquet  或  HQ_DIR=.../hq(自动先跑 03c)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/_env.sh"   # 激活 conda、设代理/CA、按 GPU=N 选卡

HYPIR_DIR="${HYPIR_DIR:-$REPO_DIR/../HYPIR}"
MODEL_DIR="${MODEL_DIR:-$REPO_DIR/../../model/HYPIR}"
BASE_MODEL_PATH="${BASE_MODEL_PATH:-$MODEL_DIR/sd2_base}"
LORA_WEIGHT_PATH="${LORA_WEIGHT_PATH:-$MODEL_DIR/HYPIR_sd2.pth}"   # 暖启动；04c 默认暖启动发布 LoRA

OUTPUT_DIR="${OUTPUT_DIR:-$HYPIR_DIR/experiments/synthetic_exp1}"
OUTPUT_DIR="$(mkdir -p "$OUTPUT_DIR" && cd "$OUTPUT_DIR" && pwd)"

# 默认数据集(用环境变量覆盖)；没给 parquet 就从 HQ_DIR 自动建一张
DATASET_ROOT="${DATASET_ROOT:-/data_3d/w00950754/code/HYPIR/dataset/ppr10k_faces_20260703}"
HQ_DIR="${HQ_DIR:-$DATASET_ROOT/hq}"

MODEL="${MODEL:-hypir}"
DATASET="${DATASET:-synthetic}"
LOG_FILE="$OUTPUT_DIR/${MODEL}-${DATASET}-LoRA_$(date +%Y%m%d_%H%M%S).log"

PARQUET_PATH="${PARQUET_PATH:-}"
if [ -z "$PARQUET_PATH" ] && [ -n "${HQ_DIR:-}" ]; then
    echo "--- no PARQUET_PATH; building one from HQ=$HQ_DIR (03c) ---"
    PARQUET_OUT="$OUTPUT_DIR/hypir_synthetic.parquet" HQ_DIR="$HQ_DIR" \
        bash "$SCRIPT_DIR/03c_build_synthetic_dataset.sh"
    PARQUET_PATH="$OUTPUT_DIR/hypir_synthetic.parquet"
fi
[ -f "$PARQUET_PATH" ] || { echo "ERROR: parquet not found: $PARQUET_PATH" >&2; exit 1; }
PARQUET_PATH="$(cd "$(dirname "$PARQUET_PATH")" && pwd)/$(basename "$PARQUET_PATH")"

# --- 数据集形状 ---
CROP_TYPE="${CROP_TYPE:-random}"      # HQ>512 用 random(在线裁 512 patch)；HQ=512 用 none
OUT_SIZE="${OUT_SIZE:-512}"
IMAGE_PATH_PREFIX="${IMAGE_PATH_PREFIX:-}"
IMAGE_PATH_KEY="${IMAGE_PATH_KEY:-image_path}"
PROMPT_KEY="${PROMPT_KEY:-prompt}"

# --- 训练超参(面向 HQ-only 在线退化微调；都可被环境变量覆盖) ---
export MAX_TRAIN_STEPS="${MAX_TRAIN_STEPS:-300000}"
export BATCH_SIZE="${BATCH_SIZE:-6}"
export LR_G="${LR_G:-1e-5}"
export LR_D="${LR_D:-1e-5}"
export GRAD_ACCUM="${GRAD_ACCUM:-1}"
export SEED="${SEED:-231}"
export CHECKPOINTING_STEPS="${CHECKPOINTING_STEPS:-500}"
export LOG_IMAGE_STEPS="${LOG_IMAGE_STEPS:-100}"
export LOG_GRAD_STEPS="${LOG_GRAD_STEPS:-100}"
export CHECKPOINTS_TOTAL_LIMIT="${CHECKPOINTS_TOTAL_LIMIT:-}"   # 空=全留(None); 数字=留 N; ⚠️0=只留最新1个(非全留)
RESUME="${RESUME:-}"

echo "=== [04c] HYPIR-SD2 LoRA fine-tune (synthetic online degradation + warm-start) ==="
echo "  💎代码路径:   $HYPIR_DIR"
echo "  💎基座模型:   $BASE_MODEL_PATH"
echo "  💎暖启动LoRA: ${LORA_WEIGHT_PATH:-<from scratch>}"
echo "  💎parquet:   $PARQUET_PATH  (HQ-only, 在线退化)"
echo "  💎数据集:     crop_type=$CROP_TYPE out_size=$OUT_SIZE"
echo "  💎训练参数:   steps=$MAX_TRAIN_STEPS bs=$BATCH_SIZE grad_accum=$GRAD_ACCUM lr_G=$LR_G lr_D=$LR_D seed=$SEED"
echo "  💎存档点:    every $CHECKPOINTING_STEPS steps"
echo "  💎输出路径:   $OUTPUT_DIR"
echo "  💎日志:       $LOG_FILE  (BG=${BG:-1})"
[ -n "$RESUME" ] && echo "  resume:    $RESUME"
if [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
    echo "  GPU:       physical $CUDA_VISIBLE_DEVICES  [GPU=N to change]"
else
    echo "  GPU:       all visible  [set GPU=N to pin a single card]"
fi

# --- 前置检查 ---
[ -d "$HYPIR_DIR" ] || { echo "ERROR: HYPIR code dir not found at $HYPIR_DIR." >&2; exit 1; }
[ -d "$BASE_MODEL_PATH" ] || { echo "ERROR: base model not found at $BASE_MODEL_PATH." >&2; exit 1; }
if [ -n "$LORA_WEIGHT_PATH" ] && [ ! -f "$LORA_WEIGHT_PATH" ]; then
    echo "ERROR: lora_weight_path not found: $LORA_WEIGHT_PATH" >&2; exit 1; fi
python -c "import accelerate, omegaconf, peft, diffusers, polars" 2>/dev/null || {
    echo "ERROR: training deps missing. Run: INSTALL_DEPS=1 bash $SCRIPT_DIR/00_setup_env.sh  and  pip install polars" >&2; exit 1; }

# --- 1. 填官方 sd2_train.yaml 的 TODO（RealESRGANDataset + RealESRGANBatchTransform）---
TEMPLATE="$HYPIR_DIR/configs/sd2_train.yaml"
CONFIG_OUT="$OUTPUT_DIR/sd2_train_synthetic_filled.yaml"
export TEMPLATE CONFIG_OUT OUTPUT_DIR PARQUET_PATH BASE_MODEL_PATH
export CROP_TYPE OUT_SIZE IMAGE_PATH_PREFIX IMAGE_PATH_KEY PROMPT_KEY
python "$SCRIPT_DIR/gen_train_config.py"

# --- 2. 补标量超参 + lora_weight_path(暖启动) + resume_ema=false ---
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
ctl = "$CHECKPOINTS_TOTAL_LIMIT"
cfg.checkpoints_total_limit = int(ctl) if ctl.strip() else None   # 空=None(全留) / N(留N) / 0(只留最新1)
# 暖启动：官方 sd2_train.yaml 没有 lora_weight_path 字段，这里加上。
# FineTuneSD2Trainer 会把它映射到 config.weight_path，你改过的 sd2.py 会 torch.load 它。
lwp = "$LORA_WEIGHT_PATH"
OmegaConf.set_struct(cfg, False)
cfg.lora_weight_path = lwp if lwp else None
cfg.resume_ema = False        # 从原始 LoRA 暖启动：没有 EMA state 可恢复
OmegaConf.set_struct(cfg, True)
OmegaConf.save(cfg, p)
_ctl_desc = "全留(None)" if not ctl.strip() else ctl
print(f"[*] patched -> {p}  (lora_weight_path={lwp or '<from scratch>'}, resume_ema=False, checkpoints_total_limit={_ctl_desc})")
PY

# --- 3. 启动训练(train_paired.py = FineTuneSD2Trainer 暖启动；官方 config = 在线退化)---
export HYPIR_DIR
export PYTHONPATH="$SCRIPT_DIR:$HYPIR_DIR${PYTHONPATH:+:$PYTHONPATH}"
cd "$HYPIR_DIR"

ACCEL_ARGS=()
if [ -n "${N_TRAIN_GPU:-}" ] && [ "${N_TRAIN_GPU:-1}" -gt 1 ]; then
    ACCEL_ARGS+=(--num_processes "$N_TRAIN_GPU")
fi
[ -n "${MIXED_PRECISION:-}" ] && ACCEL_ARGS+=(--mixed_precision "$MIXED_PRECISION")

if [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
    _NGPU=$(echo "$CUDA_VISIBLE_DEVICES" | awk -F, '{print NF}')
else
    _NGPU=$(nvidia-smi -L 2>/dev/null | wc -l); [ "$_NGPU" -eq 0 ] && _NGPU=1
fi
_MULTI=0; [ "$_NGPU" -gt 1 ] && _MULTI=1
if [ -n "${N_TRAIN_GPU:-}" ] && [ "${N_TRAIN_GPU:-1}" -gt 1 ]; then _MULTI=1; fi
if [ "$_MULTI" = "1" ]; then
    if [ -z "${PORT:-}" ]; then
        PORT=$(python -c 'import socket; s=socket.socket(); s.bind(("",0)); print(s.getsockname()[1]); s.close()' 2>/dev/null)
    fi
    : "${PORT:=29530}"
    ACCEL_ARGS+=(--main_process_port "$PORT")
fi

if [ "${BG:-1}" = "0" ]; then
    accelerate launch "${ACCEL_ARGS[@]}" "$SCRIPT_DIR/train_paired.py" --config "$CONFIG_OUT" 2>&1 | tee "$LOG_FILE"
    echo "=== [04c] Done. Checkpoints in: $OUTPUT_DIR ==="
    echo "    💎Inference: WEIGHT_PATH=$OUTPUT_DIR/checkpoint-N/state_dict.pth bash $SCRIPT_DIR/02_run_inference.sh"
else
    nohup accelerate launch "${ACCEL_ARGS[@]}" "$SCRIPT_DIR/train_paired.py" --config "$CONFIG_OUT" > "$LOG_FILE" 2>&1 &
    TRAIN_PID=$!
    echo "=== [04c] 训练已在后台启动 (PID=$TRAIN_PID) ==="
    echo "    💎日志: $LOG_FILE"
    echo "    💎跟踪: tail -f $LOG_FILE"
    echo "    💎停止: kill $TRAIN_PID  (或 pkill -f train_paired.py)"
    echo "    💎推理: WEIGHT_PATH=$OUTPUT_DIR/checkpoint-N/state_dict.pth bash $SCRIPT_DIR/02_run_inference.sh"
    sleep 5
    if ! kill -0 "$TRAIN_PID" 2>/dev/null; then
        echo "    [警告] 进程 5 秒内已退出，多半是配置/依赖报错，查日志: tail -50 $LOG_FILE"
    fi
fi

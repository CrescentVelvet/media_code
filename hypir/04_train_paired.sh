#!/usr/bin/env bash
# 04_train_paired.sh — 在「真实配对 LQ+HQ 人脸」上微调 HYPIR-SD2 的 LoRA。
#
# 流程：建配对 parquet(若缺) -> 填 sd2_train_paired.yaml -> 打标量超参 ->
#       accelerate launch train_paired.py --config <填好的配置>
#
# train_paired.py 用的是 FineTuneSD2Trainer：它先把 LoRA 从发布的
# HYPIR_sd2.pth 暖启动(config.lora_weight_path)，再跑官方那套一步去噪 +
# L2/LPIPS/GAN 训练循环——等于「在已发布模型上继续练、适配到人脸域」，
# 而不是从零重学(7k 张数据从零练不够，发布模型是 bs1024 大数据训出来的)。
#
# 默认数据集路径(可改)：
#   /data_3d/w00950754/code/HYPIR/dataset/ppr10k_faces_20260703/{hq,lq}
# 任何参数都能用环境变量覆盖(见 README 的 Config 表)。
#
# 必填(二选一)：
#   PARQUET_PATH=/path/to/paired.parquet   (由 03b_build_paired_dataset.sh 生成)
#   HQ_DIR=.../hq  LQ_DIR=.../lq           (没给 parquet 就自动先建一张)
#
# 日志：默认后台运行(BG=1)，stdout/stderr 写入
#       $OUTPUT_DIR/${MODEL}-${DATASET}-LoRA_<时间>.log；BG=0 改前台(tee 同步存日志)。
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/_env.sh"   # 激活 conda 环境、设代理/CA、按 GPU=N 选卡

# 官方代码与权重目录(默认与仓库布局一致，可被 HYPIR_DIR/MODEL_DIR 覆盖)
HYPIR_DIR="${HYPIR_DIR:-$REPO_DIR/../HYPIR}"
MODEL_DIR="${MODEL_DIR:-$REPO_DIR/../../model/HYPIR}"
BASE_MODEL_PATH="${BASE_MODEL_PATH:-$MODEL_DIR/sd2_base}"            # SD2 基座(本地 diffusers)
LORA_WEIGHT_PATH="${LORA_WEIGHT_PATH:-$MODEL_DIR/HYPIR_sd2.pth}"    # 暖启动 LoRA；设 "" 则从零训

# 实验输出目录(checkpoint + 日志)，默认 HYPIR_DIR/experiments/ppr10k_faces_paired
OUTPUT_DIR="${OUTPUT_DIR:-$HYPIR_DIR/experiments/ppr10k_faces_paired}"
OUTPUT_DIR="$(mkdir -p "$OUTPUT_DIR" && cd "$OUTPUT_DIR" && pwd)"

# 默认数据集位置(用环境变量覆盖即可)
DATASET_ROOT="${DATASET_ROOT:-/data_3d/w00950754/code/HYPIR/dataset/ppr10k_faces_20260703}"
HQ_DIR="${HQ_DIR:-$DATASET_ROOT/hq}"
LQ_DIR="${LQ_DIR:-$DATASET_ROOT/lq}"

# 日志文件名里的模型/数据集标签(可被环境变量覆盖)；日志最终存到 OUTPUT_DIR
MODEL="${MODEL:-hypir}"
DATASET="${DATASET:-$(basename "$DATASET_ROOT")}"
LOG_FILE="$OUTPUT_DIR/${MODEL}-${DATASET}-LoRA_$(date +%Y%m%d_%H%M%S).log"

# --- 1. 取得配对 parquet(没给就先用 03 建一张) ---
PARQUET_PATH="${PARQUET_PATH:-}"
if [ -z "$PARQUET_PATH" ]; then
    echo "--- no PARQUET_PATH; building one from HQ=$HQ_DIR LQ=$LQ_DIR (03) ---"
    PARQUET_OUT="$OUTPUT_DIR/hypir_paired.parquet" \
        HQ_DIR="$HQ_DIR" LQ_DIR="$LQ_DIR" \
        bash "$SCRIPT_DIR/03_build_paired_dataset.sh"
    PARQUET_PATH="$OUTPUT_DIR/hypir_paired.parquet"
fi
[ -f "$PARQUET_PATH" ] || { echo "ERROR: parquet not found: $PARQUET_PATH" >&2; exit 1; }
PARQUET_PATH="$(cd "$(dirname "$PARQUET_PATH")" && pwd)/$(basename "$PARQUET_PATH")"

# --- 2. 数据集形状(一般不用改) ---
CROP_TYPE="${CROP_TYPE:-none}"        # none=把整张脸 resize 到 512 | center | random(配对随机裁剪增强)
OUT_SIZE="${OUT_SIZE:-512}"           # HQ/LQ 都缩放到此尺寸(HYPIR 的 VAE patch 就是 512)

# --- 3. 训练超参(默认值面向 ~7k 配对的微调；都可被环境变量覆盖) ---
export MAX_TRAIN_STEPS="${MAX_TRAIN_STEPS:-15000}"   # ~2 个 epoch(bs=6)；10k–30k 可调
export BATCH_SIZE="${BATCH_SIZE:-6}"                 # 每卡 batch；A100/H100 可提到 8–12
export LR_G="${LR_G:-1e-5}"                          # 想更温和别忘掉发布模型就用 5e-6
export LR_D="${LR_D:-1e-5}"
export GRAD_ACCUM="${GRAD_ACCUM:-1}"                 # 显存不够时调大以等效增 batch
export SEED="${SEED:-231}"
export CHECKPOINTING_STEPS="${CHECKPOINTING_STEPS:-500}"   # 每 N 步存一个 checkpoint-N/state_dict.pth
export LOG_IMAGE_STEPS="${LOG_IMAGE_STEPS:-100}"
export LOG_GRAD_STEPS="${LOG_GRAD_STEPS:-100}"
RESUME="${RESUME:-}"                                 # 断点续训：填一个 04 的 checkpoint-N 目录

echo "=== [04-paired] HYPIR-SD2 LoRA fine-tune on REAL paired faces ==="
echo "  代码路径:      $HYPIR_DIR"
echo "  模型路径:      $BASE_MODEL_PATH"
echo "  LoRA路径: ${LORA_WEIGHT_PATH:-<from scratch>}"
echo "  parquet:   $PARQUET_PATH"
echo "  数据集路径:   crop_type=$CROP_TYPE out_size=$OUT_SIZE"
echo "  训练参数:     steps=$MAX_TRAIN_STEPS bs=$BATCH_SIZE grad_accum=$GRAD_ACCUM lr_G=$LR_G lr_D=$LR_D seed=$SEED"
echo "  存档点:      every $CHECKPOINTING_STEPS steps"
echo "  输出路径:    $OUTPUT_DIR"
echo "  日志路径:       $LOG_FILE  (BG=${BG:-1})"
[ -n "$RESUME" ] && echo "  resume:    $RESUME"
if [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
    echo "  GPU:       physical $CUDA_VISIBLE_DEVICES  [GPU=N to change]"
else
    echo "  GPU:       all visible  [set GPU=N to pin a single card]"
fi

# --- 前置检查：官方代码、基座、(若暖启动)LoRA 文件、训练依赖 ---
[ -d "$HYPIR_DIR" ] || { echo "ERROR: HYPIR code dir not found at $HYPIR_DIR. Run run_all.sh first." >&2; exit 1; }
[ -d "$BASE_MODEL_PATH" ] || { echo "ERROR: base model not found at $BASE_MODEL_PATH. Run 01_download_models.sh first." >&2; exit 1; }
if [ -n "$LORA_WEIGHT_PATH" ] && [ ! -f "$LORA_WEIGHT_PATH" ]; then
    echo "ERROR: lora_weight_path not found: $LORA_WEIGHT_PATH (run 01_download_models.sh, or LORA_WEIGHT_PATH='' for from-scratch)" >&2; exit 1
fi
python -c "import accelerate, omegaconf, peft, diffusers, polars" 2>/dev/null || {
    echo "ERROR: training deps missing. Run: INSTALL_DEPS=1 bash $SCRIPT_DIR/00_setup_env.sh  and  pip install polars" >&2; exit 1; }

# --- 4. 填写配对配置(不动官方文件：生成一份填好的副本到 OUTPUT_DIR) ---
TEMPLATE="$SCRIPT_DIR/sd2_train_paired.yaml"
CONFIG_OUT="$OUTPUT_DIR/sd2_train_paired_filled.yaml"
export TEMPLATE CONFIG_OUT OUTPUT_DIR PARQUET_PATH BASE_MODEL_PATH LORA_WEIGHT_PATH
export CROP_TYPE OUT_SIZE
python "$SCRIPT_DIR/gen_paired_config.py"

# --- 5. 打官方 trainer 从顶层 config 读的那些标量超参 ---
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

# --- 6. 启动训练(把本目录与 HYPIR_DIR 都加进 PYTHONPATH，让 paired_face_plugin 和 HYPIR.* 都可 import) ---
export HYPIR_DIR                    # 供 train_paired.py 兜底 sys.path 用
export PYTHONPATH="$SCRIPT_DIR:$HYPIR_DIR${PYTHONPATH:+:$PYTHONPATH}"
cd "$HYPIR_DIR"

ACCEL_ARGS=()
if [ -n "${N_TRAIN_GPU:-}" ] && [ "${N_TRAIN_GPU:-1}" -gt 1 ]; then
    ACCEL_ARGS+=(--num_processes "$N_TRAIN_GPU")   # 多卡：显式指定进程数(不必先跑 accelerate config)
fi
[ -n "${MIXED_PRECISION:-}" ] && ACCEL_ARGS+=(--mixed_precision "$MIXED_PRECISION")
# 多卡时指定分布式端口，避开默认 29500 被别的进程占用(0=自动找空闲端口，单机多卡适用)。
# 单卡(GPU=N)时 accelerate 跑单进程，此参数被忽略，无副作用。
ACCEL_ARGS+=(--main_process_port "${PORT:-0}")

# --- 7. 启动训练 ---
# 默认后台运行、stdout/stderr 全存到 LOG_FILE(BG=1)；想前台看实时输出就 BG=0(用 tee 同时存日志)。
if [ "${BG:-1}" = "0" ]; then
    accelerate launch "${ACCEL_ARGS[@]}" "$SCRIPT_DIR/train_paired.py" --config "$CONFIG_OUT" 2>&1 | tee "$LOG_FILE"
    echo "=== [04-paired] Done. Checkpoints in: $OUTPUT_DIR ==="
    echo "    Inference: WEIGHT_PATH=$OUTPUT_DIR/checkpoint-N/state_dict.pth bash $SCRIPT_DIR/02_run_inference.sh"
else
    nohup accelerate launch "${ACCEL_ARGS[@]}" "$SCRIPT_DIR/train_paired.py" --config "$CONFIG_OUT" > "$LOG_FILE" 2>&1 &
    TRAIN_PID=$!
    echo "=== [04-paired] 训练已在后台启动 (PID=$TRAIN_PID) ==="
    echo "    日志: $LOG_FILE"
    echo "    跟踪: tail -f $LOG_FILE"
    echo "    停止: kill $TRAIN_PID  (或 pkill -f train_paired.py)"
    echo "    推理: WEIGHT_PATH=$OUTPUT_DIR/checkpoint-N/state_dict.pth bash $SCRIPT_DIR/02_run_inference.sh"
    # 等 5 秒看是否秒退(配置/依赖报错会立刻挂)
    sleep 5
    if ! kill -0 "$TRAIN_PID" 2>/dev/null; then
        echo "    [警告] 进程 5 秒内已退出，多半是配置/依赖报错，查日志: tail -50 $LOG_FILE"
    fi
fi

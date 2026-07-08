#!/usr/bin/env bash
# 05_eval.sh — 评测训练好的 HYPIR-SD2 LoRA 在配对人脸上的复原效果。
#
# 用训练好的 checkpoint (state_dict.pth) 复原 LQ 测试图，与同名 HQ 计算
# PSNR / SSIM / LPIPS，并给出 bicubic 基线（无模型）作对比，看模型增益。
# 同时存：复原结果图、三联对比图(LQ|result|HQ)、metrics.csv。
#
# 默认指向 04b_train_paired.sh 的产物：
#   权重:  $TRAIN_DIR/checkpoint-$CKPT_STEP/state_dict.pth
#   数据:  $DATASET_ROOT/{lq,hq}  (按同名文件配对)
#   输出:  $TRAIN_DIR/eval_ckpt$CKPT_STEP/{result,compare,metrics.csv}
#
# 注：默认数据即训练集——指标反映"训练拟合"程度；要客观评测请把
#     TEST_LQ_DIR/TEST_HQ_DIR 指向留出的测试集。EVAL_LIMIT 控制评测张数。
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/_env.sh"   # 激活 conda、设代理/CA、按 GPU=N 选卡

HYPIR_DIR="${HYPIR_DIR:-$REPO_DIR/../HYPIR}"
MODEL_DIR="${MODEL_DIR:-$REPO_DIR/../../model/HYPIR}"
BASE_MODEL_PATH="${BASE_MODEL_PATH:-$MODEL_DIR/sd2_base}"

# 训练产物目录与 checkpoint 步数(权重默认从这里取)
TRAIN_DIR="${TRAIN_DIR:-$HYPIR_DIR/experiments/ppr10k_faces_paired}"
CKPT_STEP="${CKPT_STEP:-65000}"
WEIGHT_PATH="${WEIGHT_PATH:-$TRAIN_DIR/checkpoint-$CKPT_STEP/state_dict.pth}"

# 评测数据(默认即 04 的数据集；指向留出集做客观评测)
DATASET_ROOT="${DATASET_ROOT:-/data_3d/w00xxxxxx/code/HYPIR/dataset/ppr10k_faces_20260703}"
TEST_LQ_DIR="${TEST_LQ_DIR:-$DATASET_ROOT/lq}"
TEST_HQ_DIR="${TEST_HQ_DIR:-$DATASET_ROOT/hq}"     # 设为空则只复原不算指标

# 评测输出目录
EVAL_DIR="${EVAL_DIR:-$TRAIN_DIR/eval_ckpt$CKPT_STEP}"

# 复原参数(人脸配对是 512->512 复原，故默认 upscale=1；做超分改 UPSCALE)
SCALE_BY="${SCALE_BY:-factor}"
UPSCALE="${UPSCALE:-1}"
PATCH_SIZE="${PATCH_SIZE:-512}"
STRIDE="${STRIDE:-256}"
SEED="${SEED:-231}"
DEVICE="${DEVICE:-cuda}"
MODEL_T="${MODEL_T:-200}"
COEFF_T="${COEFF_T:-200}"
LORA_RANK="${LORA_RANK:-256}"
LORA_MODULES="${LORA_MODULES:-to_k,to_q,to_v,to_out.0,conv,conv1,conv2,conv_shortcut,conv_out,proj_in,proj_out,ff.net.2,ff.net.0.proj}"

EVAL_LIMIT="${EVAL_LIMIT:-50}"      # 评测张数；0=全部
SAVE_COMPARE="${SAVE_COMPARE:-1}"   # 1=存三联对比图

echo "=== [05] HYPIR 评测 ==="
echo "  代码路径:   $HYPIR_DIR"
echo "  基座模型:   $BASE_MODEL_PATH"
echo "  评测权重:   $WEIGHT_PATH  (rank=$LORA_RANK)"
echo "  测试LQ:    $TEST_LQ_DIR"
echo "  测试HQ:    ${TEST_HQ_DIR:-<none -> 只复原, 不算指标>}"
echo "  输出路径:   $EVAL_DIR  (result/ + compare/ + metrics.csv)"
echo "  参数:   scale_by=$SCALE_BY upscale=$UPSCALE patch=$PATCH_SIZE stride=$STRIDE seed=$SEED"
echo "  评测张数:   ${EVAL_LIMIT} (0=全部)   存对比图: $SAVE_COMPARE"
if [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
    echo "  GPU:       physical $CUDA_VISIBLE_DEVICES  [GPU=N to change]"
else
    echo "  GPU:       all visible  [set GPU=N to pin a single card]"
fi

# --- 前置检查 ---
[ -d "$HYPIR_DIR" ] || { echo "ERROR: HYPIR code dir not found at $HYPIR_DIR." >&2; exit 1; }
[ -d "$BASE_MODEL_PATH" ] || { echo "ERROR: base model not found at $BASE_MODEL_PATH. Run 01_download_models.sh first." >&2; exit 1; }
[ -f "$WEIGHT_PATH" ] || { echo "ERROR: weight not found: $WEIGHT_PATH (check CKPT_STEP / TRAIN_DIR)." >&2; exit 1; }
[ -d "$TEST_LQ_DIR" ] || { echo "ERROR: TEST_LQ_DIR not found: $TEST_LQ_DIR" >&2; exit 1; }
if [ -n "$TEST_HQ_DIR" ] && [ ! -d "$TEST_HQ_DIR" ]; then
    echo "WARNING: TEST_HQ_DIR not found: $TEST_HQ_DIR — 将只复原不算指标。" >&2
    TEST_HQ_DIR=""
fi
python -c "import accelerate, omegaconf, peft, diffusers, numpy" 2>/dev/null || {
    echo "ERROR: deps missing. Run: INSTALL_DEPS=1 bash $SCRIPT_DIR/00_setup_env.sh" >&2; exit 1; }

mkdir -p "$EVAL_DIR"

# --- 透传参数给 eval.py (读 env；加载一次、循环、算指标、存对比图) ---
export PYTHONPATH="$HYPIR_DIR:${PYTHONPATH:-}"
export HYPIR_DIR BASE_MODEL_PATH WEIGHT_PATH LORA_RANK LORA_MODULES MODEL_T COEFF_T
export TEST_LQ_DIR TEST_HQ_DIR EVAL_DIR
export SCALE_BY UPSCALE PATCH_SIZE STRIDE SEED DEVICE EVAL_LIMIT SAVE_COMPARE

python "$SCRIPT_DIR/eval.py"

echo "=== [05] Done. 结果在: $EVAL_DIR ==="
echo "    复原图:   $EVAL_DIR/result/"
echo "    对比图:   $EVAL_DIR/compare/  (LQ|result|HQ)"
echo "    指标表:   $EVAL_DIR/metrics.csv"

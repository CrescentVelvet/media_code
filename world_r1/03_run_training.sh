#!/usr/bin/env bash
# 03_run_training.sh — World-R1 Flow-GRPO RL 训练。
#
# 默认调官方 scripts/run_single_node.sh (自动启 reward server + 训练)。
# 设 EXTERNAL_REWARD=1 则只跑训练 (假设 reward server 已在跑, 见 02)。
#
# GPU 分配 (默认 8 GPU: 2 server + 6 train):
#   SERVER_VISIBLE_DEVICES=0,1  TRAIN_VISIBLE_DEVICES=2,3,4,5,6,7
# 4 GPU 示例:
#   SERVER_VISIBLE_DEVICES=0,1  TRAIN_VISIBLE_DEVICES=2,3  NUM_PROCESSES=2
#
# 模型选择:
#   MODEL_FAMILY=wan_large   (默认, 14B, 需 ~40GB+ 显存)
#   MODEL_FAMILY=wan_small    (1.3B, 需 ~16GB 显存)
#   MODEL_FAMILY=cogvideox    (CogVideoX1.5-5B)
set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/_env.sh"

MODEL_FAMILY="${MODEL_FAMILY:-wan_large}"

# 根据 MODEL_FAMILY 设模型路径
if [ "$MODEL_FAMILY" = "wan_small" ]; then
    WAN_MODEL_PATH="${WAN_MODEL_PATH:-$MODEL_DIR/Wan2.1-T2V-1.3B-Diffusers}"
elif [ "$MODEL_FAMILY" = "cogvideox" ]; then
    WAN_MODEL_PATH="${WAN_MODEL_PATH:-$MODEL_DIR/CogVideoX1.5-5B}"
else
    WAN_MODEL_PATH="${WAN_MODEL_PATH:-$MODEL_DIR/Wan2.1-T2V-14B-Diffusers}"
fi

# 训练配置名 (config/world_r1.py 里的函数名)
case "$MODEL_FAMILY" in
    wan_large)   TRAIN_CONFIG_DEFAULT="config/world_r1.py:world_r1_large" ;;
    wan_small)   TRAIN_CONFIG_DEFAULT="config/world_r1.py:world_r1_small" ;;
    cogvideox)  TRAIN_CONFIG_DEFAULT="config/world_r1.py:world_r1_cogvideox_5b" ;;
    *)           TRAIN_CONFIG_DEFAULT="config/world_r1.py:world_r1_large" ;;
esac

# GPU 分配
SERVER_VISIBLE_DEVICES="${SERVER_VISIBLE_DEVICES:-0,1}"
TRAIN_VISIBLE_DEVICES="${TRAIN_VISIBLE_DEVICES:-2,3,4,5,6,7}"

# torchrun 进程数 (默认 = TRAIN_VISIBLE_DEVICES GPU 数)
if [ -z "${NUM_PROCESSES:-}" ]; then
    NUM_PROCESSES=$(echo "$TRAIN_VISIBLE_DEVICES" | tr ',' '\n' | grep -c .)
fi

# 官方脚本路径
export WAN_PYTHON="${WAN_PYTHON:-$(which python)}"
export WAN_TORCHRUN="${WAN_TORCHRUN:-$(which torchrun)}"
export PYTHONPATH="${WORLD_R1_DIR}:${PYTHONPATH:-}"
export PYTHONNOUSERSITE=1
export no_proxy="${no_proxy:-127.0.0.1,localhost}"
export NO_PROXY="${NO_PROXY:-127.0.0.1,localhost}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export WANDB_PROJECT="${WANDB_PROJECT:-world-r1}"
export WANDB_MODE="${WANDB_MODE:-offline}"

# 输出路径
export OUTPUT_ROOT="${OUTPUT_ROOT:-$EXPERIMENTS_DIR}"

echo "🚀 [03] World-R1 RL 训练"
echo "  🤖 MODEL_FAMILY:       $MODEL_FAMILY"
echo "  🏋️ base model:         $WAN_MODEL_PATH"
echo "  🎮 server GPU:         $SERVER_VISIBLE_DEVICES"
echo "  🎮 train GPU:          $TRAIN_VISIBLE_DEVICES ($NUM_PROCESSES processes)"
echo "  📁 output:             $OUTPUT_ROOT"
echo "  📐 config:             ${TRAIN_CONFIG:-$TRAIN_CONFIG_DEFAULT}"
echo ""

# 前置检查
if [ ! -d "$WAN_MODEL_PATH" ] && [ ! -d "$HUGGINGFACE_HUB_CACHE/models--${WAN_MODEL_PATH%%/*}--${WAN_MODEL_PATH#*/}" ]; then
    echo "❌ base model 不存在: $WAN_MODEL_PATH" >&2
    echo "   Run: bash $SCRIPT_DIR/01_download_models.sh" >&2
    exit 1
fi
if [ ! -d "$WORLD_R1_DIR/scripts" ]; then
    echo "❌ World-R1 代码不在 $WORLD_R1_DIR" >&2
    echo "   Run: INSTALL_DEPS=1 bash $SCRIPT_DIR/00_setup_env.sh" >&2
    exit 1
fi

cd "$WORLD_R1_DIR"

if [ "${EXTERNAL_REWARD:-0}" = "1" ]; then
    # --- 外部 reward server 模式 (02 已启动 reward server) ---
    export REWARD_3D_SERVER_URL="${REWARD_3D_SERVER_URL:-http://127.0.0.1:${SERVER_PORT:-18089}}"
    export GENERAL_REWARD_SERVER_URL="${GENERAL_REWARD_SERVER_URL:-http://127.0.0.1:${GENERAL_REWARD_PORT:-18090}}"

    echo "🔗 EXTERNAL_REWARD=1, 只跑训练 (reward server 已在跑)"
    echo "  📐 3D reward:     $REWARD_3D_SERVER_URL"
    echo "  🤖 general reward: $GENERAL_REWARD_SERVER_URL"
    echo ""

    export MODEL_PATH="$WAN_MODEL_PATH"
    export SERVER_VISIBLE_DEVICES TRAIN_VISIBLE_DEVICES NUM_PROCESSES
    export TRAIN_CONFIG="${TRAIN_CONFIG:-$TRAIN_CONFIG_DEFAULT}"

    bash scripts/run_training.sh
else
    # --- 一体化模式 (run_single_node.sh 自动启 reward server + 训练) ---
    echo "🔗 一体化模式 (run_single_node.sh: 自动启 reward server + 训练)"
    echo ""

    export MODEL_PATH="$WAN_MODEL_PATH"
    export SERVER_VISIBLE_DEVICES TRAIN_VISIBLE_DEVICES NUM_PROCESSES
    export TRAIN_CONFIG="${TRAIN_CONFIG:-$TRAIN_CONFIG_DEFAULT}"

    bash scripts/run_single_node.sh
fi

if [ $? -ne 0 ]; then
    echo "❌ [03] 训练失败" >&2
    exit 1
fi

echo ""
echo "🎉 [03] 训练完成"
echo "  📁 checkpoint: $OUTPUT_ROOT/*/"
echo "  📊 wandb:      $WANDB_PROJECT (offline mode)"
echo ""
echo "下一步推理:"
echo "  LORA_PATH=$OUTPUT_ROOT/<run_name>/checkpoint-*/ bash $SCRIPT_DIR/04_run_inference.sh"

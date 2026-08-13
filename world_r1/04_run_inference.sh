#!/usr/bin/env bash
# 04_run_inference.sh — World-R1 训练好的 LoRA 推理 (Wan2.1 + LoRA → 视频)。
#
# 支持单 prompt (PROMPT=...) 或 prompt 文件 (PROMPT_FILE=..., 每行一个)。
# 不设 LORA_PATH 则只跑基础模型 (baseline 对比用)。
set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/_env.sh"

MODEL_FAMILY="${MODEL_FAMILY:-wan_large}"
if [ "$MODEL_FAMILY" = "wan_small" ]; then
    WAN_MODEL_PATH="${WAN_MODEL_PATH:-$MODEL_DIR/Wan2.1-T2V-1.3B-Diffusers}"
elif [ "$MODEL_FAMILY" = "cogvideox" ]; then
    WAN_MODEL_PATH="${WAN_MODEL_PATH:-$MODEL_DIR/CogVideoX1.5-5B}"
else
    WAN_MODEL_PATH="${WAN_MODEL_PATH:-$MODEL_DIR/Wan2.1-T2V-14B-Diffusers}"
fi

LORA_PATH="${LORA_PATH:-}"
HEIGHT="${HEIGHT:-480}"
WIDTH="${WIDTH:-832}"
NUM_FRAMES="${NUM_FRAMES:-81}"
NUM_STEPS="${NUM_STEPS:-20}"
GUIDANCE_SCALE="${GUIDANCE_SCALE:-5.0}"
FPS="${FPS:-12}"
SEED="${SEED:-42}"
DTYPE="${DTYPE:-bf16}"
NEGATIVE_PROMPT="${NEGATIVE_PROMPT:-}"

# 输出目录
OUT_DIR="${RESULTS_DIR:-$REPO_DIR/../world_r1_results}"
mkdir -p "$OUT_DIR"

echo "🚀 [04] World-R1 推理"
echo "  🤖 base model:  $WAN_MODEL_PATH"
if [ -n "$LORA_PATH" ]; then
    echo "  🏋️ LoRA:        $LORA_PATH"
else
    echo "  🏋️ LoRA:        (none, base model only)"
fi
echo "  📐 resolution:  ${HEIGHT}x${WIDTH}x${NUM_FRAMES}"
echo "  🎬 steps/cfg:   $NUM_STEPS / $GUIDANCE_SCALE"
echo "  📁 output:      $OUT_DIR"
echo ""

# 前置检查
if [ ! -d "$WAN_MODEL_PATH" ] && [ ! -d "$HUGGINGFACE_HUB_CACHE/models--${WAN_MODEL_PATH%%/*}--${WAN_MODEL_PATH#*/}" ]; then
    echo "❌ base model 不存在: $WAN_MODEL_PATH" >&2
    echo "   Run: bash $SCRIPT_DIR/01_download_models.sh" >&2
    exit 1
fi
if [ -n "$LORA_PATH" ] && [ ! -d "$LORA_PATH" ]; then
    echo "❌ LoRA 不存在: $LORA_PATH" >&2
    exit 1
fi
if [ ! -d "$WORLD_R1_DIR/scripts" ]; then
    echo "❌ World-R1 代码不在 $WORLD_R1_DIR" >&2
    echo "   Run: INSTALL_DEPS=1 bash $SCRIPT_DIR/00_setup_env.sh" >&2
    exit 1
fi

# --- 准备 prompt 文件 ---
PROMPT_FILE="${PROMPT_FILE:-}"
if [ -z "$PROMPT_FILE" ]; then
    if [ -n "${PROMPT:-}" ]; then
        PROMPT_FILE="$OUT_DIR/_prompts.txt"
        echo "$PROMPT" > "$PROMPT_FILE"
        echo "📝 single prompt -> $PROMPT_FILE"
    else
        # 默认 prompt (含相机运动, 适合 World-R1)
        PROMPT_FILE="$OUT_DIR/_prompts_default.txt"
        cat > "$PROMPT_FILE" <<'PROMPTS'
A camera slowly orbits around a detailed ceramic vase on a wooden table, soft natural lighting
A camera pans left across a snowy mountain landscape, golden hour
A camera dollies forward through a forest path, autumn leaves falling
A camera tilts up to reveal a towering waterfall in a tropical jungle
A camera tracks right past a row of vintage books on a shelf, warm library lighting
PROMPTS
        echo "📝 no PROMPT/PROMPT_FILE given; using default prompts -> $PROMPT_FILE"
    fi
fi

if [ ! -f "$PROMPT_FILE" ]; then
    echo "❌ prompt file 不存在: $PROMPT_FILE" >&2
    exit 1
fi

# --- 调用官方 infer_wan_lora.py ---
export WORLD_R1_WAN_MODEL="$WAN_MODEL_PATH"
export PYTHONPATH="${WORLD_R1_DIR}:${PYTHONPATH:-}"
export PYTHONNOUSERSITE=1

# 用数组传参, 避免 word splitting
ARGS=()
if [ -n "$LORA_PATH" ]; then
    ARGS+=(--lora-path "$LORA_PATH")
fi
if [ -n "${DEVICES:-}" ]; then
    ARGS+=(--devices "$DEVICES")
fi
if [ -n "$NEGATIVE_PROMPT" ]; then
    ARGS+=(--negative-prompt "$NEGATIVE_PROMPT")
fi

cd "$WORLD_R1_DIR"

echo "🎬 generating videos..."
python scripts/infer_wan_lora.py \
    --model "$WAN_MODEL_PATH" \
    --prompt-file "$PROMPT_FILE" \
    --out-dir "$OUT_DIR" \
    --height "$HEIGHT" \
    --width "$WIDTH" \
    --num-frames "$NUM_FRAMES" \
    --num-steps "$NUM_STEPS" \
    --guidance-scale "$GUIDANCE_SCALE" \
    --fps "$FPS" \
    --seed "$SEED" \
    --dtype "$DTYPE" \
    "${ARGS[@]}"

if [ $? -ne 0 ]; then
    echo "❌ [04] 推理失败" >&2
    exit 1
fi

echo ""
echo "🎉 [04] 推理完成"
echo "  📁 videos: $OUT_DIR/*.mp4"
echo "  🔍 ls $OUT_DIR/*.mp4"

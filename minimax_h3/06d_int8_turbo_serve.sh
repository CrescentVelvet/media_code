#!/usr/bin/env bash
# 06d_int8_turbo_serve.sh — MiniMax-H3 int8 + Turbo LoRA 常驻 HTTP 服务（3090 最优解）。
#
# = 06c 的 int8 + block offload 加载（~62GB RAM，3090 24GB 够）
# + 06b 的 Turbo 4 步蒸馏 LoRA（50 步→4 步，~10× 快）。
# 06c 能跑但 50 步慢；06b 4 步快但 bf16 全量 ~124GB RAM 3090+64GB 跑不动。
# 06d 取两者之长：int8 压 RAM + Turbo 4 步省算力。
#
# LoRA checkpoint 配方（同 06b.sh，详见 https://huggingface.co/lightx2v/Minimax-h3-Turbo）：
#   checkpoint                                   NFE  VIDEO_SHIFT  LORA_ALPHA  MAX_PIXELS
#   ─────────────────────────────────────────────────────────────────────────────────────
#   minimax_h3_fl2v_turbo_4step_v1.0_768p_bf16    4    6            128         1032192 (1344x768)  ← 默认
#   minimax_h3_fl2v_turbo_4step_v1.1_768p_bf16    4    6            128*        1032192 (1344x768)  v1.1 质量改进版（同配方）
#   minimax_h3_fl2v_turbo_8step_v1.0_768p_bf16    8    6            128*        1032192 (1344x768)  质量优先（Studio 用，比 4 步慢 2×）
#   minimax_h3_fl2v_turbo_4step_v0.1              4    12           8           522240  (960x544)   ← 3090 OOM 兜底
#   minimax_h3_ref2v_turbo_4step_v0.1_bf16        4    12           8           522240  (Ref2VA，06d 不支持，用 06b)
# ⚠️ shift/alpha 与 checkpoint 绑定，换 checkpoint 必须同步改 VIDEO_SHIFT/LORA_ALPHA/MAX_PIXELS。
# * v1.1 / 8-step 768p 的 alpha=128 按 768p 系列推断（官方 model-specs 表未列）；v1.0 768p 的 6/128 官方确认。
# ⚠️ 768p LoRA 在远低于 1344x768 的分辨率上跑会掉质（蒸馏是分辨率敏感的）；3090 若 768p VAE
#    解码 OOM，降 NUM_FRAMES 或换 544p checkpoint（VIDEO_SHIFT=12 LORA_ALPHA=8 MAX_PIXELS=522240）。
#
# 用法：
#   GPU=0 MODEL_PATH=/mnt/d/wheel/minimaxh3_ms \
#     LORA_PATH=/mnt/d/wheel/minimaxh3_ms/minimax_h3_turbo/minimax_h3_fl2v_turbo_4step_v1.0_768p_bf16.safetensors \
#     bash minimax_h3/06d_int8_turbo_serve.sh
#   # 另一个终端：
#   curl -X POST http://localhost:8000/generate -H 'Content-Type: application/json' \
#     -d '{"prompt":"...","first_frame":"/mnt/d/img.jpg","seed":42}'
set -o pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/_env.sh"

PIP_FLAGS=(--trusted-host pypi.org --trusted-host pypi.python.org \
    --trusted-host files.pythonhosted.org --timeout 600 --retries 10)

# 检测 diffusers MiniMax-H3 模块（同 06c，int8 要 MiniMaxH3Transformer3DModel）
if ! python -c "from diffusers.modular_pipelines.minimax_h3 import MiniMaxH3ModularPipeline" 2>/dev/null; then
    echo "📦 diffusers too old or missing MiniMax-H3 module, installing -U diffusers ---"
    python -m pip install "${PIP_FLAGS[@]}" -U diffusers
fi
if ! python -c "from diffusers.modular_pipelines.minimax_h3 import MiniMaxH3ModularPipeline" 2>/dev/null; then
    echo "📦 PyPI diffusers still no MiniMax-H3 — cloning git main + editable install ---"
    DIFFUSERS_SRC="${DIFFUSERS_SRC:-/tmp/diffusers-src}"
    if [ ! -d "$DIFFUSERS_SRC/src/diffusers" ]; then
        LD_LIBRARY_PATH= git clone --depth 1 https://github.com/huggingface/diffusers.git "$DIFFUSERS_SRC" || \
            LD_LIBRARY_PATH= git -c http.sslVerify=false clone --depth 1 https://github.com/huggingface/diffusers.git "$DIFFUSERS_SRC"
    fi
    if [ -d "$DIFFUSERS_SRC/src/diffusers" ]; then
        python -m pip install "${PIP_FLAGS[@]}" -e "$DIFFUSERS_SRC" --no-deps --config-settings editable_mode=compat || \
            python -m pip install "${PIP_FLAGS[@]}" -e "$DIFFUSERS_SRC" --no-deps
    else
        echo "❌ clone diffusers failed" >&2; exit 1
    fi
fi
# 检测 torchao（int8 量化 + 06d peft torchao LoRA dispatcher 要 >=0.16.0；
#   peft is_torchao_available 在 <0.16 会 raise → add_adapter 炸；06a/06c/06d 统一 0.16）
if ! python -c "import torchao; assert tuple(map(int, torchao.__version__.split('.')[:2])) >= (0, 16)" 2>/dev/null; then
    echo "📦 torchao too old or missing (need >=0.16.0 for peft torchao LoRA dispatcher), installing torchao==0.16.0 ---"
    python -m pip install "${PIP_FLAGS[@]}" "torchao==0.16.0" || \
        { echo "❌ pip install torchao==0.16.0 failed" >&2; exit 1; }
fi
# 检测 peft（LoRA 注入要 LoraConfig + add_adapter + set_adapters，同 06b）
if ! python -c "import peft; assert tuple(map(int, peft.__version__.split('.')[:2])) >= (0, 8)" 2>/dev/null; then
    echo "📦 peft too old or missing (need >=0.8.0), installing -U peft ---"
    python -m pip install "${PIP_FLAGS[@]}" -U peft || \
        { echo "❌ pip install peft failed" >&2; exit 1; }
fi
python -c "from diffusers.modular_pipelines.minimax_h3 import MiniMaxH3ModularPipeline; import torchao, peft; print('✅ diffusers + torchao + peft ok')" 2>/dev/null || \
    { echo "❌ diffusers/torchao/peft not ready" >&2; exit 1; }

# 前置检查：LoRA 权重（默认放模型子目录 minimax_h3_turbo/，可被 LORA_PATH 覆盖）
LORA_PATH="${LORA_PATH:-/mnt/d/wheel/minimaxh3_ms/minimax_h3_turbo/minimax_h3_fl2v_turbo_4step_v1.0_768p_bf16.safetensors}"
if [ ! -f "$LORA_PATH" ]; then
    echo "❌ ERROR: LoRA checkpoint not found: $LORA_PATH" >&2
    echo "   modelscope 迅雷直链（~1.4GB，本机可达）:" >&2
    echo "     https://modelscope.cn/models/lightx2v/Minimax-h3-Turbo/resolve/master/minimax_h3_fl2v_turbo_4step_v1.0_768p_bf16.safetensors" >&2
    echo "   下完放到: /mnt/d/wheel/minimaxh3_ms/minimax_h3_turbo/minimax_h3_fl2v_turbo_4step_v1.0_768p_bf16.safetensors" >&2
    echo "   （期望大小 1383677808 字节；或 hf download lightx2v/Minimax-h3-Turbo \\" >&2
    echo "     minimax_h3_fl2v_turbo_4step_v1.0_768p_bf16.safetensors --local-dir /mnt/d/wheel/minimaxh3_ms/minimax_h3_turbo)" >&2
    echo "   then set LORA_PATH=/your/lora.safetensors" >&2
    exit 1
fi
export LORA_PATH

# 默认路径 + Turbo 配方（默认 768p 4-step v1.0）
export MODEL_PATH="${MODEL_PATH:-/mnt/d/wheel/minimaxh3_ms}"
export PORT="${PORT:-8000}"
export DEVICE="${DEVICE:-cuda:0}"
export OUTPUT_DIR="${OUTPUT_DIR:-/mnt/d/output/minimaxh3_rotate_results/results_int8turbo}"
export MAX_PIXELS="${MAX_PIXELS:-1032192}"          # 768p 配方 1344x768；3090 OOM 改 522240 + 544p checkpoint
export FPS="${FPS:-24}"
export NUM_FRAMES="${NUM_FRAMES:-124}"
export NUM_INFERENCE_STEPS="${NUM_INFERENCE_STEPS:-4}"
export VIDEO_SHIFT="${VIDEO_SHIFT:-6.0}"             # 768p 用 6；544p 用 12
export AUDIO_SHIFT="${AUDIO_SHIFT:-3.0}"
export LORA_ALPHA="${LORA_ALPHA:-128}"               # 768p 4-step v1.0 用 128；544p 用 8
export LORA_SCALE="${LORA_SCALE:-1.0}"
export FUSE_LORA="${FUSE_LORA:-0}"                   # int8 不支持 fuse（融不进量化权重），保持 0

echo "🚀 [06d] MiniMax-H3 int8 + Turbo 常驻服务"
echo "  🤖 model:  $MODEL_PATH"
echo "  🏋️ lora:   $LORA_PATH"
echo "  ⏱️ NFE:    $NUM_INFERENCE_STEPS (grid=$((NUM_INFERENCE_STEPS+1)))  shifts: video=$VIDEO_SHIFT audio=$AUDIO_SHIFT"
echo "  📡 port:   $PORT"
echo "  🎮 device: $DEVICE"
echo "  💾 output:  $OUTPUT_DIR"
echo "  📐 max_pixels=$MAX_PIXELS  fps=$FPS  num_frames=$NUM_FRAMES"

python "$SCRIPT_DIR/06d_int8_turbo_serve.py"
if [ $? -ne 0 ]; then
    echo "❌ FAILED" >&2; exit 1
fi
echo "🎉 [06d] Done."

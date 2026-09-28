#!/usr/bin/env bash
# 06b_turbo_lora_inference.sh — MiniMax-H3 Turbo LoRA 加速推理（单卡，4/8 步去噪）。
#
# 06a 的 Turbo 变体：用 lightx2v/Minimax-h3-Turbo 蒸馏的 LoRA checkpoint，把
# 官方 50 步去噪压到 4/8 步，推理快 ~10×。bf16 原版权重 + LoRA adapter（不量化），
# 单卡 80GB 用 ComponentsManager auto CPU offload 自动搬运（官方 Turbo 推理路径）。
#
# LoRA checkpoint 选择（详见 https://huggingface.co/lightx2v/Minimax-h3-Turbo）：
#   checkpoint                              NFE  VIDEO_SHIFT  alpha(文件)  PEFT scale   MAX_PIXELS
#   ────────────────────────────────────────────────────────────────────────────────────────────────
#   fl2v_turbo_4step_v1.0_768p_bf16          4    6            128         1.0          1032192 (1344x768) ← 默认
#   fl2v_turbo_4step_v1.1_768p_bf16          4    6            128         1.0          1032192 (1344x768)
#   fl2v_turbo_4step_v1.2_768p_bf16          4    6            8 ⚠️        0.0625 ⚠️    1032192 (1344x768) 新版(2026-09-06，音频更干净)
#   fl2v_turbo_8step_v1.0_768p_bf16          8    6            8 ⚠️        0.0625 ⚠️    1032192 (1344x768)
#   fl2v_turbo_4step_v0.1                    4    12           无记录      1.0          522240  (960x544)
#   fl2v_turbo_8step_v1.0_bf16               8    12           无记录      1.0          522240  (960x544)
#   ref2v_turbo_4step_v0.1_bf16              4    12           无记录      1.0          522240  (Ref2VA，需 TASK=ref2va)
#
# ⚠️ LORA_ALPHA 默认 auto：按 checkpoint 自身的 __metadata__['alpha'] 推导 PEFT scale
#    = alpha/rank（rank 均为 128）。规则照 diffusers 0.40 的 MiniMaxH3LoraLoaderMixin：
#    有该字段用它、没有则按 alpha==rank（scale 1.0）。
#    ⚠️ 但 768p 系列 metadata 自相矛盾（v1.0/v1.1=128，v1.2/8step=8，而实测 ||B@A||
#    幅度同量级，不是 16×），**至少一个记录是错的**。标 ⚠️ 的两支必须 A/B：
#    LORA_ALPHA=auto（按记录 0.0625） vs LORA_ALPHA=128（scale 1.0）。
#    另：旧版本脚本在这里写死 128，会把 8-step / v1.2 放大 16×、把 544p v0.1 缩小 16×。
# ⚠️ 768p checkpoint 必须传 VIDEO_SHIFT=6（训练 shift=6，不是 12）。
# ⚠️ Ref2VA checkpoint 必须配 TASK=ref2va（走 transformer_ref），不能复用 FL2VA LoRA。
#
# 需要装 peft + diffusers git main（有 MiniMaxH3Transformer3DModel + ComponentsManager）。
set -o pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/_env.sh"

PIP_FLAGS=(--trusted-host pypi.org --trusted-host pypi.python.org \
    --trusted-host files.pythonhosted.org --timeout 600 --retries 10)

# 检测 diffusers MiniMax-H3 模块（同 06a）
if ! python -c "from diffusers.modular_pipelines.minimax_h3 import MiniMaxH3ModularPipeline; from diffusers import ComponentsManager" 2>/dev/null; then
    echo "📦 diffusers too old or missing MiniMax-H3 module, installing -U diffusers ---"
    python -m pip install "${PIP_FLAGS[@]}" -U diffusers
fi
if ! python -c "from diffusers.modular_pipelines.minimax_h3 import MiniMaxH3ModularPipeline; from diffusers import ComponentsManager" 2>/dev/null; then
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
# 检测 peft（LoRA 注入要 LoraConfig + add_adapter + set_adapters + fuse_lora）
if ! python -c "import peft; assert tuple(map(int, peft.__version__.split('.')[:2])) >= (0, 8)" 2>/dev/null; then
    echo "📦 peft too old or missing (need >=0.8.0), installing -U peft ---"
    python -m pip install "${PIP_FLAGS[@]}" -U peft || \
        { echo "❌ pip install peft failed" >&2; exit 1; }
fi
python -c "from diffusers.modular_pipelines.minimax_h3 import MiniMaxH3ModularPipeline; from diffusers import ComponentsManager; import peft; print('✅ diffusers + peft ok')" 2>/dev/null || \
    { echo "❌ diffusers/peft not ready" >&2; exit 1; }

# 前置检查：LoRA 权重必须存在（默认指向 768p 4-step v1.0，可被 LORA_PATH 覆盖）
LORA_PATH="${LORA_PATH:-/mnt/d/wheel/minimaxh3_ms/minimax_h3_turbo/minimax_h3_fl2v_turbo_4step_v1.0_768p_bf16.safetensors}"
if [ ! -f "$LORA_PATH" ]; then
    echo "❌ ERROR: LoRA checkpoint not found: $LORA_PATH" >&2
    echo "   download from https://huggingface.co/lightx2v/Minimax-h3-Turbo" >&2
    echo "   e.g. hf download lightx2v/Minimax-h3-Turbo \\" >&2
    echo "          minimax_h3_fl2v_turbo_4step_v1.0_768p_bf16.safetensors --local-dir /mnt/d/wheel/minimaxh3_ms/minimax_h3_turbo" >&2
    echo "   then set LORA_PATH=/your/lora.safetensors" >&2
    exit 1
fi
export LORA_PATH

python "$SCRIPT_DIR/06b_turbo_lora_inference.py"
if [ $? -ne 0 ]; then
    echo "❌ FAILED" >&2; exit 1
fi
echo "🎉 [06b] Done."

#!/usr/bin/env bash
# 06_diffusers_inference.sh — 用 diffusers 直接跑 MiniMax-H3（无需起服务）。
#
# 与 02_serve+03_generate 不同，这个脚本直接用 diffusers 的 ModularPipeline
# 跑，输入图像+文字生成视频+音频。不起 HTTP 服务，适合单次生成。
#
# 显存：transformer 61.7GB + Qwen3-VL 62.1GB，单卡 80GB 放不下，用 ComponentsManager
# auto offload（CPU↔GPU 搬运，比 SGLang 慢但简单）。SGLang 用 Ulysses/TP 多卡并行
# 更快（8 卡跑满）；diffusers 单卡 offload 或两卡分拆（参考 diffusers 文档）。
#
# Usage:
#   TASK=t2va PROMPT="a drone shot over alpine peaks" \
#     MODEL_PATH=../../model/MiniMax-H3 \
#     OUTPUT_DIR=../MiniMax-H3/results/diffusers \
#     bash minimax_h3/06_diffusers_inference.sh
#   TASK=fl2va FIRST_FRAME=/data/first.png PROMPT="..." \
#     MODEL_PATH=../../model/MiniMax-H3 \
#     bash minimax_h3/06_diffusers_inference.sh
set -o pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/_env.sh"

PIP_FLAGS=(--trusted-host pypi.org --trusted-host pypi.python.org \
    --trusted-host files.pythonhosted.org --timeout 600 --retries 10)

# diffusers 需要 ModularPipeline（MiniMax-H3 支持），PyPI 版本可能还没有，
# 先试 PyPI -U，还不行从 git main clone + editable --no-deps 装。
if ! python -c "from diffusers import ComponentsManager, ModularPipeline" 2>/dev/null; then
    echo "📦 diffusers too old or missing (need ModularPipeline for MiniMax-H3), installing -U diffusers ---"
    python -m pip install "${PIP_FLAGS[@]}" -U diffusers
fi
if ! python -c "from diffusers import ModularPipeline" 2>/dev/null; then
    echo "📦 PyPI diffusers still no ModularPipeline — cloning git main + editable install (--no-deps) ---"
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
if ! python -c "from diffusers import ComponentsManager, ModularPipeline" 2>/dev/null; then
    echo "❌ diffusers still no ModularPipeline after install" >&2; exit 1
fi
echo "✅ diffusers ModularPipeline ok"

python "$SCRIPT_DIR/06_diffusers_inference.py"
if [ $? -ne 0 ]; then
    echo "❌ FAILED" >&2
    exit 1
fi
echo "🎉 [06] Done."

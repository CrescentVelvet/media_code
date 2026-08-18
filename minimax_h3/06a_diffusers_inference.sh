#!/usr/bin/env bash
# 06a_diffusers_inference.sh — MiniMax-H3 int8 量化推理（单卡常驻，无需起服务）。
#
# 06 的量化变体：用 int8 weight-only 量化（torchao），transformer+text_encoder
# 各 ~31GB，单卡 80GB 常驻（不走两卡分拆/offload，速度最快）。
# 关键模块不量化（proj_in/out, AdaLN, time_embedder 等保留 bf16），效果基本无损。
#
# 需要装 torchao（pip install torchao）+ diffusers git main（有 MiniMaxH3Transformer3DModel）。
set -o pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/_env.sh"

PIP_FLAGS=(--trusted-host pypi.org --trusted-host pypi.python.org \
    --trusted-host files.pythonhosted.org --timeout 600 --retries 10)

# 检测 diffusers MiniMax-H3 模块（同 06）
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
# 检测 torchao（int8 量化要 >=0.15.0，TorchAoConfig 要求）
if ! python -c "import torchao; assert tuple(map(int, torchao.__version__.split('.')[:2])) >= (0, 15)" 2>/dev/null; then
    echo "📦 torchao too old or missing (need >=0.15.0), installing torchao==0.15.0 ---"
    python -m pip install "${PIP_FLAGS[@]}" "torchao==0.15.0" || \
        { echo "❌ pip install torchao==0.15.0 failed" >&2; exit 1; }
fi
python -c "from diffusers.modular_pipelines.minimax_h3 import MiniMaxH3ModularPipeline; import torchao; print('✅ diffusers + torchao ok')" 2>/dev/null || \
    { echo "❌ diffusers/torchao not ready" >&2; exit 1; }

# diffusers 的 MiniMaxH3Transformer3DModel.from_pretrained 期望 diffusion_pytorch_model.safetensors.index.json，
# 但 MiniMax-H3 权重用 transformers 命名（model.safetensors.index.json），建符号链接让 diffusers 找到。
MODEL_PATH="${MODEL_PATH:-../../model/MiniMax-H3}"
TRANSFORMER_DIR="$MODEL_PATH/FL2VA/transformer"
if [ -f "$TRANSFORMER_DIR/model.safetensors.index.json" ] && [ ! -f "$TRANSFORMER_DIR/diffusion_pytorch_model.safetensors.index.json" ]; then
    echo "📦 creating symlink: diffusion_pytorch_model.safetensors.index.json -> model.safetensors.index.json"
    ln -sf model.safetensors.index.json "$TRANSFORMER_DIR/diffusion_pytorch_model.safetensors.index.json"
fi

python "$SCRIPT_DIR/06a_diffusers_inference.py"
if [ $? -ne 0 ]; then
    echo "❌ FAILED" >&2; exit 1
fi
echo "🎉 [06a] Done."

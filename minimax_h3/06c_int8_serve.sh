#!/usr/bin/env bash
# 06c_int8_serve.sh — MiniMax-H3 int8 常驻 HTTP 服务。
#
# 与 06a 相同的 int8 + block offload 加载逻辑，但加载后启动 HTTP 服务，
# 保持模型在 RAM 中，多次 POST /generate 请求不重新加载。
# 适合从 D 盘（HDD）加载的场景——20-40min 加载成本只付一次。
#
# 用法：
#   # 启动服务（加载 20-40min 从 D 盘，然后监听端口）
#   GPU=0 MODEL_PATH=/mnt/d/wheel/minimaxh3_ms bash minimax_h3/06c_int8_serve.sh
#
#   # 另一个终端发请求：
#   curl -X POST http://localhost:8000/generate \
#     -H 'Content-Type: application/json' \
#     -d '{"prompt":"...","first_frame":"/mnt/d/img.jpg","seed":42}'
#
#   # 健康检查：
#   curl http://localhost:8000/health
#
#   # 关闭服务：
#   curl -X POST http://localhost:8000/shutdown
set -o pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/_env.sh"

PIP_FLAGS=(--trusted-host pypi.org --trusted-host pypi.python.org \
    --trusted-host files.pythonhosted.org --timeout 600 --retries 10)

# 检测 diffusers MiniMax-H3 模块（同 06a）
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
python -c "from diffusers.modular_pipelines.minimax_h3 import MiniMaxH3ModularPipeline; import torchao; print('✅ diffusers + torchao ok')" 2>/dev/null || \
    { echo "❌ diffusers/torchao not ready" >&2; exit 1; }

# 默认路径：模型在 D 盘（HDD），输出在 ~/output（Linux fs）
export MODEL_PATH="${MODEL_PATH:-/mnt/d/wheel/minimaxh3_ms}"
export PORT="${PORT:-8000}"
export DEVICE="${DEVICE:-cuda:0}"
export OUTPUT_DIR="${OUTPUT_DIR:-/mnt/d/output/minimaxh3_rotate_results/results_int8}"
export MAX_PIXELS="${MAX_PIXELS:-133120}"
export FPS="${FPS:-24}"
export NUM_FRAMES="${NUM_FRAMES:-124}"

echo "🚀 [06c] MiniMax-H3 int8 常驻服务"
echo "  🤖 model:  $MODEL_PATH"
echo "  📡 port:   $PORT"
echo "  🎮 device: $DEVICE"
echo "  💾 output:  $OUTPUT_DIR"
echo "  📐 max_pixels=$MAX_PIXELS  fps=$FPS  num_frames=$NUM_FRAMES"

python "$SCRIPT_DIR/06c_int8_serve.py"

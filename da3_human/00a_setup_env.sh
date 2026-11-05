#!/usr/bin/env bash
# 00a_setup_env.sh — WSL 变体：从零建 conda env da3_human + 安装 Depth-Anything-3
# 用法: INSTALL_DEPS=1 bash da3_human/00a_setup_env.sh   （INSTALL_DEPS=0 只建 env 不装依赖）
set -o pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/_env.sh"

export INSTALL_DEPS="${INSTALL_DEPS:-1}"
export PYTHON_VERSION="${PYTHON_VERSION:-3.10}"

echo "🚀 [00a] da3_human 环境搭建 (WSL)"
echo "  🐍 conda env: $CONDA_ENV (python $PYTHON_VERSION)"
echo "  📁 DA3 官方仓: $DA3_DIR"
echo "  🎮 GPU: ${GPU:-all}"

# 0) clone 官方仓（Linux fs，编译/IO 快）
if [ ! -d "$DA3_DIR/.git" ]; then
    echo "📦 clone ByteDance-Seed/Depth-Anything-3 -> $DA3_DIR"
    git clone https://github.com/ByteDance-Seed/Depth-Anything-3.git "$DA3_DIR"
    if [ $? -ne 0 ]; then echo "❌ clone failed" >&2; exit 1; fi
else
    echo "⏭️ DA3 仓库已存在 ($DA3_DIR)，跳过 clone"
fi

# 1) 建 env（幂等）
if [ ! -d "$(conda info --base)/envs/$CONDA_ENV" ]; then
    echo "📦 创建 conda env $CONDA_ENV (python $PYTHON_VERSION)"
    conda create -y -n "$CONDA_ENV" python="$PYTHON_VERSION"
    if [ $? -ne 0 ]; then echo "❌ conda create failed" >&2; exit 1; fi
else
    echo "⏭️ env $CONDA_ENV 已存在，跳过创建"
fi
conda activate "$CONDA_ENV"

if [ "$INSTALL_DEPS" = "1" ]; then
    # torch 走 tuna（pypi 官方 wheel 自带 CUDA；download.pytorch.org 403 目录页可忽略）
    echo "📦 安装 torch/torchvision/xformers（pip 一次解析，保证版本匹配）"
    pip install torch torchvision xformers
    if [ $? -ne 0 ]; then echo "❌ torch install failed" >&2; exit 1; fi
    echo "📦 pip install -e $DA3_DIR（DA3 本体 + requirements）"
    pip install -e "$DA3_DIR"
    if [ $? -ne 0 ]; then echo "❌ pip install -e DA3 failed" >&2; exit 1; fi
fi

# 2) 验证 import（torch CUDA 可用 + DA3 包可导入）
python - <<'PYEOF'
import torch
import depth_anything_3
import xformers
print(f"✅ torch {torch.__version__} cuda_available={torch.cuda.is_available()}")
print(f"✅ xformers {xformers.__version__}")
print(f"✅ depth_anything_3 @ {depth_anything_3.__file__}")
PYEOF
if [ $? -ne 0 ]; then echo "❌ import 验证失败" >&2; exit 1; fi
echo "🎉 Done. 下一步: bash da3_human/01_download_models.sh"

#!/usr/bin/env bash
# 00a_setup_env.sh — WSL 从零/快速建 querysplat env
# 策略：clone gen3r env（python 3.11 + torch 2.8.0+cu128 + triton），再补装
# QuerySplat requirements；fused-ssim 源码编译需 conda 版 nvcc + gxx（WSL 无系统 gcc）。
# ⚠️ 官方测试配置是 Python 3.12 + torch 2.11.0 + cu128；本机复用 torch 2.8 是
#    已知偏差，若 import/前向不兼容再按官方配置重建（见 README_wsl.md 风险节）。
set -o pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/_env.sh"

export SRC_ENV="${SRC_ENV:-gen3r}"
export QUERYSPLAT_REPO="${QUERYSPLAT_REPO:-$HOME/repos/QuerySplat}"

echo "🚀 [00a] 建 querysplat env（clone $SRC_ENV）"

# 0) 官方仓已就绪（缺失则 clone）
if [ ! -d "$QUERYSPLAT_REPO/scripts" ]; then
    echo "📦 clone 官方仓: github.com/inspatio/QuerySplat -> $QUERYSPLAT_REPO"
    git clone --depth 1 https://github.com/inspatio/QuerySplat.git "$QUERYSPLAT_REPO"
    if [ $? -ne 0 ]; then echo "❌ clone 失败（网络问题先看 proxy.env）" >&2; exit 1; fi
fi

# 1) env：优先 clone gen3r（已有 torch 2.8.0+cu128），已存在则跳过
if conda env list | grep -qw querysplat; then
    echo "⏭️ env querysplat 已存在，跳过创建"
else
    if conda env list | grep -qw "$SRC_ENV"; then
        echo "📦 conda create -n querysplat --clone $SRC_ENV"
        conda create -n querysplat --clone "$SRC_ENV" -y
    else
        echo "📦 $SRC_ENV 不存在，从零建 python 3.11 env"
        conda create -n querysplat python=3.11 -y
    fi
    if [ $? -ne 0 ]; then echo "❌ conda create 失败" >&2; exit 1; fi
fi
conda activate querysplat

# 2) 编译工具链（WSL 无系统 gcc/nvcc；gsplat/fused-ssim 需 nvcc 12.8 对齐 torch cu128）
echo "📦 装编译工具链: cuda-nvcc=12.8 + gxx_linux-64=12"
conda install -n querysplat -c nvidia -c conda-forge cuda-nvcc=12.8 gxx_linux-64=12 -y
if [ $? -ne 0 ]; then echo "❌ 工具链安装失败" >&2; exit 1; fi
export CUDA_HOME="${CONDA_PREFIX}"

# 3) python 依赖（torch 已随 clone 带上；torchvision 版本与 torch 2.8 配套）
PIP_SRC="-i https://mirrors.aliyun.com/pypi/simple/ --find-links /mnt/d/wheel"
echo "📦 pip install python deps"
pip install $PIP_SRC einops plyfile pyyaml matplotlib scipy scikit-image tqdm lpips safetensors "numpy<2"
if [ $? -ne 0 ]; then echo "❌ pip deps 失败" >&2; exit 1; fi
echo "📦 pip install gsplat（torch 2.8 会源码编译 CUDA 扩展，耗时几分钟）"
pip install $PIP_SRC --no-build-isolation gsplat
if [ $? -ne 0 ]; then echo "❌ gsplat 编译失败（查 CUDA_HOME=$CUDA_HOME 是否指向 conda nvcc）" >&2; exit 1; fi
echo "📦 pip install fused-ssim（官方钉死的源码 tar）"
pip install $PIP_SRC --no-build-isolation \
    "fused-ssim @ https://github.com/rahul-goel/fused-ssim/archive/a7c48d6dd7ac6dc39a7958c7c4452e0b10418f38.tar.gz"
if [ $? -ne 0 ]; then echo "❌ fused-ssim 编译失败" >&2; exit 1; fi

# 4) 验证
echo "🔍 验证 import"
CUDA_HOME="${CONDA_PREFIX}" python - <<'PYEOF'
import sys
sys.path.insert(0, __import__("os").environ["QUERYSPLAT_REPO"])
import torch, gsplat
import fused_ssim  # noqa
from scripts.models import QuerySplat  # noqa
print(f"✅ torch {torch.__version__} cuda={torch.version.cuda}")
print("✅ gsplat / fused_ssim / scripts.models import OK")
PYEOF
if [ $? -ne 0 ]; then echo "❌ import 验证失败" >&2; exit 1; fi
echo "🎉 Done. 下一步: bash querysplat/01_download_models.sh"

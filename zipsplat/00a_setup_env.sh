#!/usr/bin/env bash
# 00a_setup_env.sh — WSL：clone 官方 ZipSplat + 独立 env（复用本机已验证的 torch 栈）
# 官方推理只要求 Python >=3.10 + CUDA GPU。BASE_ENV 提供 torch 2.8.0+cu128 / Python 3.11，
# 版本差异会写入 metrics.json，不把实测当成官方基准复现。
set -o pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/_env.sh"

BASE_ENV="${BASE_ENV:-querysplat}"
INSTALL_DEPS="${INSTALL_DEPS:-1}"
PYPI_INDEX="${PYPI_INDEX:-https://pypi.tuna.tsinghua.edu.cn/simple}"

echo "🚀 [00a] ZipSplat WSL 环境: $CONDA_ENV (base=$BASE_ENV)"
if [ ! -d "$ZIPSPLAT_REPO/zipsplat" ]; then
    mkdir -p "$(dirname "$ZIPSPLAT_REPO")"
    LD_LIBRARY_PATH= git clone https://github.com/cvg/ZipSplat.git "$ZIPSPLAT_REPO" || exit 1
fi
if ! conda env list | awk '{print $1}' | grep -qx "$CONDA_ENV"; then
    if [ "$CONDA_ENV" = "$BASE_ENV" ]; then echo "❌ 不允许原地修改 BASE_ENV" >&2; exit 1; fi
    if [ -d "$HOME/miniconda3/envs/$BASE_ENV" ]; then
        echo "📦 创建独立 env（clone $BASE_ENV）"
        conda create -n "$CONDA_ENV" --clone "$BASE_ENV" -y || exit 1
    else
        echo "📦 无 BASE_ENV，从零创建 Python 3.11 + torch 2.8.0+cu128"
        conda create -n "$CONDA_ENV" python=3.11 -y || exit 1
        conda activate "$CONDA_ENV" || exit 1
        pip install --find-links /mnt/d/wheel --index-url https://download.pytorch.org/whl/cu128 torch==2.8.0 torchvision==0.23.0 || exit 1
    fi
fi
conda activate "$CONDA_ENV" || exit 1
if [ "$INSTALL_DEPS" = "1" ]; then
    # 明确排除 torch/triton，避免安装 zipsplat 时替换已有的 cu128 torch。
    pip install --index-url "$PYPI_INDEX" --find-links "$WHEEL_DIR" --timeout 600 \
        "numpy<2" einops tqdm matplotlib plyfile pillow "imageio[ffmpeg]" || exit 1
    python -c 'import gsplat' 2>/dev/null || \
        pip install --index-url "$PYPI_INDEX" --find-links "$WHEEL_DIR" --no-build-isolation gsplat || exit 1
    pip install --no-deps -e "$ZIPSPLAT_REPO" || exit 1
fi

export CUDA_HOME="${CUDA_HOME:-$CONDA_PREFIX}"
echo "🔍 验证官方包与 GPU"
python - <<'PY'
import platform
import torch
import zipsplat
from zipsplat import predictor
print("Python:", platform.python_version())
print("Torch:", torch.__version__, "CUDA:", torch.version.cuda)
print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none")
assert platform.python_implementation() == "CPython"
assert torch.cuda.is_available()
assert str(predictor.__file__).startswith(str(__import__("pathlib").Path.home() / "repos" / "ZipSplat")), predictor.__file__
print("ZipSplat:", predictor.__file__)
PY
if [ $? -ne 0 ]; then echo "❌ 环境自检失败" >&2; exit 1; fi
echo "🎉 Done. 下一步: bash zipsplat/01_download_models.sh"

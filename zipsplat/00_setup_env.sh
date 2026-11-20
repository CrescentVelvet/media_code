#!/usr/bin/env bash
# 00_setup_env.sh — 服务器：clone 官方 ZipSplat + 建独立 Python/CUDA 环境
set -o pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/_env.sh"

INSTALL_DEPS="${INSTALL_DEPS:-1}"
ZIPSPLAT_REPO="${ZIPSPLAT_REPO:-$REPO_DIR/../ZipSplat}"
export ZIPSPLAT_REPO

echo "🚀 [00] ZipSplat 服务器环境: $CONDA_ENV"
if [ ! -d "$ZIPSPLAT_REPO/zipsplat" ]; then
    mkdir -p "$(dirname "$ZIPSPLAT_REPO")"
    git clone https://github.com/cvg/ZipSplat.git "$ZIPSPLAT_REPO" || exit 1
fi
if ! conda env list | awk '{print $1}' | grep -qx "$CONDA_ENV"; then
    conda create -n "$CONDA_ENV" python=3.11 -y || exit 1
fi
conda activate "$CONDA_ENV" || exit 1
if [ "$INSTALL_DEPS" = "1" ]; then
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128 || exit 1
    conda install -y -c nvidia/label/cuda-12.8.1 cuda-toolkit || exit 1
    pip install -e "$ZIPSPLAT_REPO" || exit 1
fi
python -c 'import torch; from zipsplat import predictor; print("✅", torch.__version__, torch.cuda.is_available(), predictor.__file__)' || exit 1
echo "🎉 Done. 下一步: bash zipsplat/01_download_models.sh"

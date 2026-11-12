#!/usr/bin/env bash
# 00_setup_env.sh — clone 官方仓（稀疏）、建 conda env、装依赖、自检。
#
# 幂等：已存在的 env / 仓会跳过。装完自检两个模型类能否 import。
#
# Env (all optional):
#   CONDA_ENV=splatanything       # conda 环境名
#   PY_VER=3.11                   # 官方要求 3.10+
#   TORCH_VER=2.7.1               # 与 infinisplat_v2 实测可用的版本对齐
#   PIP_INDEX=https://pypi.tuna.tsinghua.edu.cn/simple
#   CUDA_INDEX=                   # 非空则从 pytorch.org 官方源装（本机不通，默认走 pypi）
#   FORCE_RECREATE=1              # 重建 env
set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/_env.sh"

: "${PY_VER:=3.11}"
: "${TORCH_VER:=2.7.1}"
: "${PIP_INDEX:=https://pypi.tuna.tsinghua.edu.cn/simple}"

echo "🚀 [00] 准备环境 ($CONDA_ENV, py$PY_VER, torch $TORCH_VER)"
echo "  📁 官方仓: $SPLATANYTHING_DIR"

# ---------- 1) 官方仓（稀疏：examples/assets 共 400 MB，按需再拉） ----------
if [ ! -d "$SPLATANYTHING_DIR/.git" ]; then
    echo "📦 clone zju3dv/SplatAnything（blob:none + sparse）"
    mkdir -p "$(dirname "$SPLATANYTHING_DIR")"
    git clone --depth 1 --filter=blob:none --sparse \
        https://github.com/zju3dv/SplatAnything.git "$SPLATANYTHING_DIR" || exit 1
    git -C "$SPLATANYTHING_DIR" sparse-checkout set \
        src configs scripts tests docs checkpoints || exit 1
else
    echo "⏭️  官方仓已存在"
fi
echo "✅ commit: $(git -C "$SPLATANYTHING_DIR" rev-parse --short HEAD)"

# ---------- 2) conda env ----------
if [ "${FORCE_RECREATE:-0}" = "1" ] && conda env list | awk '{print $1}' | grep -qx "$CONDA_ENV"; then
    echo "🗑️  FORCE_RECREATE=1，删除旧 env"
    conda deactivate 2>/dev/null || true
    conda env remove -n "$CONDA_ENV" -y || exit 1
fi
if ! conda env list | awk '{print $1}' | grep -qx "$CONDA_ENV"; then
    echo "📦 创建 conda env $CONDA_ENV (python $PY_VER)"
    conda create -n "$CONDA_ENV" "python=$PY_VER" -y || exit 1
else
    echo "⏭️  conda env 已存在"
fi
conda activate "$CONDA_ENV" || exit 1
echo "✅ python: $(which python) $(python -V 2>&1)"

# ---------- 3) PyTorch ----------
if python -c "import torch,sys; sys.exit(0 if torch.__version__.startswith('$TORCH_VER') else 1)" 2>/dev/null; then
    echo "⏭️  torch $TORCH_VER 已装"
else
    echo "📦 安装 torch $TORCH_VER"
    if [ -n "${CUDA_INDEX:-}" ]; then
        python -m pip install "torch==$TORCH_VER" torchvision --index-url "$CUDA_INDEX" || exit 1
    else
        python -m pip install "torch==$TORCH_VER" torchvision -i "$PIP_INDEX" || exit 1
    fi
fi
python -c "import torch;print('  torch',torch.__version__,'cuda',torch.version.cuda,'avail',torch.cuda.is_available(),torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')" || exit 1

# ---------- 4) 官方 requirements ----------
# torch 行已单独处理（pip 会认已装版本）；其余按 requirements.txt 原样装。
echo "📦 pip install -r requirements.txt"
python -m pip install -r "$SPLATANYTHING_DIR/requirements.txt" -i "$PIP_INDEX" || exit 1

# ---------- 5) 自检（能建出两个模型类 = 代码与依赖就位） ----------
echo "🔍 自检"
cd "$SPLATANYTHING_DIR" || exit 1
python -c "
from src.models.models.geometry_worldmirror import GeometryWorldMirror
from src.models.models.gaussian_worldmirror import GaussianWorldMirror
from src.utils.scene_io import build_views
import torch, cv2, omegaconf, safetensors, plyfile
print('  imports ok; torch', torch.__version__, '| cv2', cv2.__version__, '| numpy', __import__('numpy').__version__)
" || exit 1

# 单元测试（CPU，秒级）。只跑自包含的那个：
#   test_inference.py 首行 `from validation import ...`，但发布包没有 validation.py
#     （docs 声称"remains available"，git ls-tree 实测无此文件）→ ImportError。
#   test_panorama.py 有一条用例读 examples/pano/prior_camera.json，稀疏克隆未检出该目录。
python -m unittest tests.test_camera_conversion 2>&1 | tail -4 \
    || echo "⚠️  测试失败，见上"

echo "🎉 Done. 下一步: bash $SCRIPT_DIR/01_download_models.sh"

#!/usr/bin/env bash
# 00_setup_env.sh — 建 conda env、装 PyTorch / InfiniSplat / DA3、克隆官方仓。
#
# 幂等：已存在的 env / 目录会跳过。装完用 `infinisplat --help` 自检。
#
# Env (all optional):
#   CONDA_ENV=infinisplat_v2      # conda 环境名
#   PY_VER=3.11                   # python 版本（包要求 >=3.10,<3.13）
#   TORCH_VER=2.7.1               # 官方在 2.7.1 + H20 上实测过
#   PIP_INDEX=https://pypi.tuna.tsinghua.edu.cn/simple
#   CUDA_INDEX=https://download.pytorch.org/whl/cu128  # 留空则用 PIP_INDEX
#   WITH_GSPLAT=1                 # 顺带装 gsplat（03 自接渲染器要用）
#   FORCE_RECREATE=1              # 重建 env
set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/_env.sh"

: "${PY_VER:=3.11}"
: "${TORCH_VER:=2.7.1}"
: "${PIP_INDEX:=https://pypi.tuna.tsinghua.edu.cn/simple}"
DA3_COMMIT=41736238f5bced4debf3f2a12375d2466874866d

echo "🚀 [00] 准备环境 ($CONDA_ENV, py$PY_VER, torch $TORCH_VER)"
echo "  📁 官方仓: $INFINISPLAT_REPO"
echo "  📁 DA3 仓: $DA3_REPO"

# ---------- 1) 官方仓 ----------
if [ ! -d "$INFINISPLAT_V2" ]; then
    echo "📦 clone zju3dv/InfiniSplat"
    mkdir -p "$(dirname "$INFINISPLAT_REPO")"
    git clone --depth 1 https://github.com/zju3dv/InfiniSplat.git "$INFINISPLAT_REPO" || exit 1
else
    echo "⏭️  官方仓已存在"
fi

# ---------- 2) DA3 仓（钉死 commit，官方用了其网络内部结构） ----------
if [ ! -d "$DA3_REPO/.git" ]; then
    echo "📦 clone Depth-Anything-3"
    mkdir -p "$(dirname "$DA3_REPO")"
    git clone https://github.com/ByteDance-Seed/Depth-Anything-3.git "$DA3_REPO" || exit 1
fi
if ! git -C "$DA3_REPO" cat-file -e "$DA3_COMMIT^{commit}" 2>/dev/null; then
    echo "📦 拉取 DA3 指定 commit"
    git -C "$DA3_REPO" fetch --depth 1 origin "$DA3_COMMIT" || exit 1
fi
git -C "$DA3_REPO" checkout -q "$DA3_COMMIT" || exit 1
echo "✅ DA3 commit: $(git -C "$DA3_REPO" rev-parse --short HEAD)"

# ---------- 3) conda env ----------
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

# ---------- 4) PyTorch ----------
if python -c "import torch,sys; sys.exit(0 if torch.__version__.startswith('$TORCH_VER') else 1)" 2>/dev/null; then
    echo "⏭️  torch $TORCH_VER 已装"
else
    echo "📦 安装 torch $TORCH_VER"
    if [ -n "${CUDA_INDEX:-}" ]; then
        python -m pip install "torch==$TORCH_VER" "torchvision" --index-url "$CUDA_INDEX" || exit 1
    else
        # 走国内 pypi 镜像（默认 cu12x wheel，3090 可用）
        python -m pip install "torch==$TORCH_VER" "torchvision" -i "$PIP_INDEX" || exit 1
    fi
fi
python -c "import torch;print('  torch',torch.__version__,'cuda',torch.version.cuda,'avail',torch.cuda.is_available(),torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')" || exit 1

# ---------- 5) InfiniSplat 包 ----------
cd "$INFINISPLAT_V2" || exit 1
echo "📦 pip install . (infinisplat)"
python -m pip install -e . -i "$PIP_INDEX" || exit 1

# ---------- 6) DA3（--no-deps，官方要求：只用其网络内部结构） ----------
echo "📦 pip install --no-deps depth-anything-3@$DA3_COMMIT"
python -m pip install --no-deps "depth-anything-3 @ git+https://github.com/ByteDance-Seed/Depth-Anything-3.git@$DA3_COMMIT" \
    -i "$PIP_INDEX" || exit 1

# ---------- 7) 可选：gsplat（发布包刻意不带渲染器，03 用） ----------
if [ "${WITH_GSPLAT:-1}" = "1" ]; then
    if python -c "import gsplat" 2>/dev/null; then
        echo "⏭️  gsplat 已装"
    else
        echo "📦 安装 gsplat（自接渲染器用）"
        python -m pip install gsplat -i "$PIP_INDEX" || echo "⚠️ gsplat 安装失败，03 渲染会跳过"
    fi
fi

# ---------- 8) 自检 ----------
echo "🔍 自检"
python -c "import infinisplat, depth_anything_3; print('  infinisplat ok')" || exit 1
infinisplat --help >/dev/null && echo "  CLI ok"
python -c "
import gsplat, torch
print('  gsplat', gsplat.__version__ if hasattr(gsplat,'__version__') else 'ok')
" 2>/dev/null || echo "  ⚠️ gsplat 不可用"

echo "🎉 Done. 下一步: bash $SCRIPT_DIR/01_download_models.sh"

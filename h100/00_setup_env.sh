#!/usr/bin/env bash
# 00_setup_env.sh — 在 H100 集群（ca-uae / 迪拜）建个人 conda 环境。
#
# 集群约束（实测 2026-11-13）：
#   * /data/conda 是共享只读安装 → 不能 conda create -n，必须用 -p 建到自己目录
#   * 服务器在迪拜，pypi.org / download.pytorch.org 直连很快（0.9~2.2s）
#     → 国内镜像（清华 12s 超时）反而慢，一律用官方源
#   * 家目录和系统盘一体，只剩 51G → 环境、缓存、数据全部放 /data
#
# 用法：
#   bash 00_setup_env.sh                    # 建默认 splatanything 环境
#   ENV_NAME=myenv PY_VER=3.10 bash 00_setup_env.sh
#   FORCE_RECREATE=1 bash 00_setup_env.sh   # 删掉重建
set -o pipefail

# ---------- 可配置项 ----------
: "${ENV_NAME:=splatanything}"
: "${PY_VER:=3.11}"
: "${TORCH_VER:=2.7.1}"
: "${CUDA_WHEEL:=cu126}"          # torch 2.7.1 官方 wheel 的 CUDA 档位
: "${CONDA_ROOT:=/data/conda}"
: "${MY_ROOT:=/data/00950754}"
: "${PIP_INDEX:=https://pypi.org/simple}"   # 迪拜机房，官方源最快
: "${TORCH_INDEX:=https://download.pytorch.org/whl}"
: "${FORCE_RECREATE:=0}"

ENV_PREFIX="$MY_ROOT/envs/$ENV_NAME"
PKG_CACHE="$MY_ROOT/conda_pkgs"

echo "============================================================"
echo "🚀 建 env: $ENV_NAME"
echo "   python $PY_VER / torch $TORCH_VER+$CUDA_WHEEL"
echo "   路径   $ENV_PREFIX"
echo "   源     $PIP_INDEX"
echo "============================================================"

# ---------- 0) conda 初始化 ----------
if ! command -v conda >/dev/null 2>&1; then
    # shellcheck disable=SC1091
    source "$CONDA_ROOT/etc/profile.d/conda.sh" || { echo "❌ 找不到 $CONDA_ROOT/etc/profile.d/conda.sh"; exit 1; }
fi
echo "✅ conda $(conda --version)"

mkdir -p "$MY_ROOT/envs" "$PKG_CACHE"

# ---------- 1) 建 env（-p 指定前缀，避开只读的共享 envs/） ----------
if [ "$FORCE_RECREATE" = "1" ] && [ -d "$ENV_PREFIX" ]; then
    echo "🗑️  FORCE_RECREATE=1 → 删除 $ENV_PREFIX"
    conda deactivate 2>/dev/null || true
    rm -rf "$ENV_PREFIX" || exit 1
fi

if [ ! -d "$ENV_PREFIX" ]; then
    echo "📦 conda create -p $ENV_PREFIX python=$PY_VER"
    conda create -p "$ENV_PREFIX" "python=$PY_VER" -y \
        --override-channels -c conda-forge \
        --no-default-packages 2>&1 | tail -15 || exit 1
else
    echo "⏭️  env 已存在：$ENV_PREFIX"
fi

conda activate "$ENV_PREFIX" || exit 1
export PATH="$ENV_PREFIX/bin:$PATH"
echo "✅ python: $(which python) — $(python -V 2>&1)"

# pip 缓存也放 /data，别塞爆 51G 的家目录
export PIP_CACHE_DIR="$MY_ROOT/.pip_cache"
mkdir -p "$PIP_CACHE_DIR"

# ---------- 2) PyTorch（先装，其余包才能编译/对齐 ABI） ----------
if python -c "import torch,sys; sys.exit(0 if torch.__version__.startswith('$TORCH_VER') else 1)" 2>/dev/null; then
    echo "⏭️  torch $TORCH_VER 已装"
else
    echo "📦 安装 torch $TORCH_VER ($CUDA_WHEEL)"
    python -m pip install --no-cache-dir \
        "torch==$TORCH_VER" "torchvision==0.22.1" \
        --index-url "$TORCH_INDEX/$CUDA_WHEEL" 2>&1 | tail -8 || exit 1
fi

python - <<'PY' || exit 1
import torch
print(f"  torch {torch.__version__} | cuda {torch.version.cuda} | "
      f"avail {torch.cuda.is_available()} | {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU only'}")
PY

# ---------- 3) SplatAnything 官方 requirements ----------
# numpy 必须 <2（官方要求），且要在装其它包之前钉住，否则会被顶上去导致 ABI 崩
echo "📦 基础依赖（钉 numpy<2）"
python -m pip install --no-cache-dir -i "$PIP_INDEX" \
    "numpy>=1.24,<2" \
    "Pillow>=9.5" \
    "opencv-python>=4.8,<4.12" \
    "omegaconf>=2.3,<3" \
    "safetensors>=0.4" \
    "plyfile>=1.0,<1.1.3" \
    "onnxruntime>=1.17" \
    "scipy" "tqdm" "matplotlib" "pyyaml" "rich" 2>&1 | tail -6 || exit 1

# ---------- 4) CUDA 编译工具链（gsplat 需要 nvcc + gcc） ----------
# 系统只有 gcc 13、无 nvcc；用 conda 装一套匹配 torch cu126 的 nvcc
echo "📦 CUDA 工具链（cuda-nvcc 12.6 + gcc 12）"
conda install -p "$ENV_PREFIX" -y --override-channels -c nvidia -c conda-forge \
    cuda-nvcc=12.6 "gxx_linux-64=12" 2>&1 | tail -10 || echo "⚠️  工具链装失败，gsplat JIT 可能报错"

# ---------- 5) gsplat（渲染 / 训练都要） ----------
# 注意：1.5.2 与 torch 2.7 JIT 接口兼容，1.5.3 不兼容（splatanything 实测结论）
echo "📦 gsplat 1.5.2"
python -m pip install --no-cache-dir "gsplat==1.5.2" -i "$PIP_INDEX" 2>&1 | tail -5 || exit 1

# gsplat 1.5.2 编译要 glm 头文件，不自带
GLM_DIR="$MY_ROOT/repos/glm_src"
if [ ! -d "$GLM_DIR/glm" ]; then
    echo "📦 clone glm 头文件"
    mkdir -p "$(dirname "$GLM_DIR")"
    git clone --depth 1 https://github.com/g-truc/glm.git "$GLM_DIR" 2>&1 | tail -3 || echo "⚠️  glm clone 失败"
fi

# ---------- 6) 写环境激活脚本 ----------
cat > "$MY_ROOT/envs/activate_$ENV_NAME.sh" <<EOF
# 用法: source $MY_ROOT/envs/activate_$ENV_NAME.sh
source $CONDA_ROOT/etc/profile.d/conda.sh
conda activate $ENV_PREFIX
export PIP_CACHE_DIR="$PIP_CACHE_DIR"
# gsplat JIT 编译环境
export CUDA_HOME="$ENV_PREFIX"
export PATH="\$CUDA_HOME/bin:\$PATH"
export CPATH="$GLM_DIR:\$CUDA_HOME/targets/x86_64-linux/include:\$CUDA_HOME/include:\$CPATH"
unset TORCH_CUDA_ARCH_LIST   # 让 gsplat 自动探测（H100=9.0）
EOF

# ---------- 7) 自检 ----------
echo "🔍 自检"
cd "$MY_ROOT" || exit 1
python - <<'PY'
import importlib, sys
mods = ["torch", "numpy", "cv2", "omegaconf", "safetensors", "plyfile", "onnxruntime", "gsplat", "scipy"]
bad = []
for m in mods:
    try:
        mod = importlib.import_module(m)
        print(f"  ✅ {m:12s} {getattr(mod, '__version__', 'n/a')}")
    except Exception as e:
        bad.append((m, str(e)[:100]))
        print(f"  ❌ {m:12s} {str(e)[:100]}")
import numpy, torch
print(f"\n  numpy {numpy.__version__} (需 <2: {'✅' if int(numpy.__version__.split('.')[0]) < 2 else '❌'})")
print(f"  torch CUDA avail: {torch.cuda.is_available()}")
sys.exit(1 if bad else 0)
PY

echo
echo "============================================================"
echo "🎉 完成。以后激活："
echo "   source $MY_ROOT/envs/activate_$ENV_NAME.sh"
echo "============================================================"

#!/usr/bin/env bash
# 00b_fix_env.sh — 00a 跑完后的环境修复（幂等，可重复执行）。
#
# 修 00a 遗留问题（2026-09-09 首次跑通时发现）：
#   1. numpy 2.x vs torch 2.1.2 / chumpy 0.70 不兼容 → 降 1.23.5
#      （torch 2.1.2 按 numpy 1.x 编译；chumpy 用了 numpy 1.x 独有别名）
#   2. charset_normalizer 混装（gdown 曾装 3.x，后覆盖装 2.1.1，残留 3.x 的
#      mypyc 编译 .so，与 2.1.1 纯 py 模块冲突 → requests/torchvision 全挂）
#   3. DECA 被 00a clone 到了 /mnt/c/code/DECA（启动时 ~/repos/DECA 尚未
#      clone 完成，_env.sh 路径判定失效）→ 移到 ~/repos/DECA
#   4. pytorch3d 编译链：setuptools<81（torch 2.1.2 的 cpp_extension 需要
#      pkg_resources）+ gcc 11（CUDA 11.8 nvcc 上限）+ CUDA 11.8 dev 头文件
#      （cusparse/cublas/curand/cusolver/cufft/nvjitlink）+ 源码编译 pytorch3d
#      （fbaipublicfiles wheel 被 403/超时挡，走 ghfast 镜像 clone）
#   5. DECA 运行时依赖（00a 的 DECA 段没装）：yacs / kornia 0.6.8 / iopath
set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/_env.sh"

echo "🚀 [flame_human 00b] env 修复"

# ── 1. numpy 降级 ──────────────────────────────────────────────────────────
echo "📦 1/5 numpy==1.23.5（torch 2.1.2 / chumpy 兼容线）..."
pip install -q "numpy==1.23.5" 2>&1 | grep -v "^$" | tail -2 || true

# ── 2. charset_normalizer 混装修复 ────────────────────────────────────────
# 先试 import；挂了才删目录重装（幂等）。
if python -W ignore -c "import charset_normalizer" 2>/dev/null; then
    echo "⏭️  2/5 charset_normalizer 正常，跳过"
else
    echo "📦 2/4 修复 charset_normalizer 混装（删残留 .so + 干净重装）..."
    SP="$(python -c 'import site; print(site.getsitepackages()[0])')"
    rm -rf "$SP"/charset_normalizer "$SP"/charset_normalizer-* "$SP"/charset_normalizer-*.dist-info
    pip install -q --no-deps --no-cache-dir "charset-normalizer==2.1.1" \
        || { echo "❌ charset-normalizer 重装失败"; exit 1; }
fi

# ── 3. DECA 目录归位 ──────────────────────────────────────────────────────
# _env.sh 优先 ~/repos/DECA；00a 首跑时把它 clone 到了 $REPO_DIR/../DECA（drvfs）。
if [ -d "$REPO_DIR/../DECA/.git" ] && [ ! -d "$HOME/repos/DECA/.git" ]; then
    echo "📦 3/4 移动 DECA: $REPO_DIR/../DECA -> $HOME/repos/DECA"
    mkdir -p "$HOME/repos"
    mv "$REPO_DIR/../DECA" "$HOME/repos/DECA" \
        || { echo "❌ DECA 移动失败（检查 ~/repos 可写）"; exit 1; }
elif [ -d "$HOME/repos/DECA/.git" ]; then
    echo "⏭️  3/5 DECA 已在 ~/repos/DECA"
else
    echo "⏭️  3/4 无需处理（两处均无 DECA）"
fi

# ── 4. pytorch3d 编译链 + 源码编译 ─────────────────────────────────────────
if python -W ignore -c "import pytorch3d" 2>/dev/null; then
    echo "⏭️  4/5 pytorch3d 已装，跳过"
else
    echo "📦 4/5 pytorch3d 编译链（gcc11 + CUDA dev 头文件 + setuptools<81）..."
    # setuptools<81：torch 2.1.2 cpp_extension 依赖 pkg_resources（setuptools 83 删除）
    pip install -q "setuptools<81" || true
    # gcc/gxx 11：CUDA 11.8 的 nvcc 上限（gcc 15 会报 unsupported GNU version）
    conda install -y -n "$CONDA_ENV" -c conda-forge "gxx_linux-64=11" "gcc_linux-64=11" \
        || { echo "❌ gcc 11 安装失败"; exit 1; }
    # nvcc + CUDA 11.8 dev 头文件（torch CUDAContext.h 会 include 整套）
    conda install -y -n "$CONDA_ENV" -c "nvidia/label/cuda-11.8.0" \
        cuda-nvcc cuda-cudart-dev cuda-cccl \
        libcusparse-dev libcublas-dev libcurand-dev libcusolver-dev libcufft-dev \
        libnvjitlink-dev \
        || { echo "❌ CUDA dev 头文件安装失败"; exit 1; }
    # conda 装的是 x86_64-conda-linux-gnu-*，无裸 gcc/g++ 命令（pytorch3d setup
    # 用 `which g++` 探测）——建符号链接。
    conda activate "$CONDA_ENV"
    ln -sf x86_64-conda-linux-gnu-gcc "$CONDA_PREFIX/bin/gcc"
    ln -sf x86_64-conda-linux-gnu-g++ "$CONDA_PREFIX/bin/g++"
    ln -sf x86_64-conda-linux-gnu-gcc "$CONDA_PREFIX/bin/cc"
    ln -sf x86_64-conda-linux-gnu-g++ "$CONDA_PREFIX/bin/c++"
    # 源码编译（fbaipublicfiles wheel 被墙；GitHub 直连超时，走 ghfast 镜像）
    rm -rf /tmp/pytorch3d-src
    git clone --depth 1 https://ghfast.top/https://github.com/facebookresearch/pytorch3d.git /tmp/pytorch3d-src \
        || git clone --depth 1 https://github.com/facebookresearch/pytorch3d.git /tmp/pytorch3d-src \
        || { echo "❌ pytorch3d 源码 clone 失败"; exit 1; }
    ( cd /tmp/pytorch3d-src \
      && MAX_JOBS=8 TORCH_CUDA_ARCH_LIST="8.6" CUDA_HOME="$CONDA_PREFIX" \
         pip install --no-build-isolation --no-deps . ) \
        || { echo "❌ pytorch3d 编译失败（3090 sm86，检查 CUDA_HOME/头文件）"; exit 1; }
fi

# ── 5. DECA 运行时依赖 ─────────────────────────────────────────────────────
# DECA import 链：yacs(config) → kornia(tensor_cropper) → iopath(pytorch3d.io)。
# kornia 0.6.8 是 torch 2.1 兼容线（新版要求 torch>=2.2）。
if python -W ignore -c "import yacs, kornia, iopath" 2>/dev/null; then
    echo "⏭️  5/5 DECA 运行时依赖已装，跳过"
else
    echo "📦 5/5 DECA 运行时依赖（yacs / kornia==0.6.8 / iopath）..."
    pip install -q yacs "kornia==0.6.8" iopath \
        || { echo "❌ DECA 依赖安装失败"; exit 1; }
fi

# ── 验证 ─────────────────────────────────────────────────────────────────
echo "🔍 验证："
python -W ignore "$SCRIPT_DIR/verify_env.py"
if [ $? -ne 0 ]; then
    echo "❌ 验证未通过（见上）" >&2
    exit 1
fi
echo "🎉 env 修复完成。"

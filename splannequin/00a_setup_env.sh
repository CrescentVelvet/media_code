#!/usr/bin/env bash
# 00a_setup_env.sh — WSL Ubuntu 24.04 从零建 splannequin 环境。
#
# 官方 README pin：python=3.8 + torch==1.13.1(cu117) + gcc-10 + CUDA 11.7。
# WSL 实现（沿用 vggt_human/00a 验证过的模式，无 sudo）：
#   - conda 建 env（python=3.8）
#   - pip 装 torch 1.13.1+cu117（download.pytorch.org 直连）
#   - conda 装 cuda-nvcc 11.7（nvidia channel）+ gcc 11（conda-forge；
#     nvcc 11.7 的 host compiler 上限是 gcc 11，官方 README 说 gcc-10，
#     gcc 11.x 实测可编 4DGaussians 系 rasterizer）
#   - pip 装其余依赖（tuna 镜像；mmcv==1.6.0 走 openmmlab cu117/torch1.13 预编译轮）
#   - 编译 submodules/depth-diff-gaussian-rasterization + simple-knn
#
# 官方仓获取：GitHub 直连不稳（2026-10-24 实测 WSL 不通），优先 git clone，
# 失败则回退到 SRC_CACHE（/mnt/d/wheel/splannequin_src）里的 tarball：
#   splannequin-gs-main.tar.gz                          （codeload main）
#   depth-diff-gaussian-rasterization-e495066.tar.gz    （子模块锁定 e495066）
#   simple-knn-44f7642.tar.gz                           （子模块锁定 44f7642，gitlab.inria.fr）
#
# Usage:
#   bash splannequin/00a_setup_env.sh            # full setup
#   INSTALL_DEPS=0 bash splannequin/00a_setup_env.sh   # 只建仓+编译，不装 pip 依赖
set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"

# WSL 路径默认值（Linux fs，编译快），在 source _env.sh 之前导出
export SPLANNEQUIN_DIR="${SPLANNEQUIN_DIR:-$HOME/repos/splannequin-gs}"
export SRC_CACHE="${SRC_CACHE:-/mnt/d/wheel/splannequin_src}"

# shellcheck disable=SC1091
source "$SCRIPT_DIR/_env.sh"

echo "=== [00a] Setup splannequin env on WSL ==="
echo "  📁 official repo: $SPLANNEQUIN_DIR"
echo "  🏠 conda env:     $CONDA_ENV"
echo "  💾 data:          $DATA_DIR"
echo "  💾 exp:           $EXP_DIR"
echo ""

# ── 1. 建 conda env（python=3.8）─────────────────────────────────────────
_new_prefix="$(conda info --base 2>/dev/null)/envs/$CONDA_ENV"
if [ -d "$_new_prefix/conda-meta" ]; then
    echo "⏭️  conda env '$CONDA_ENV' already exists"
else
    echo "📦 creating conda env '$CONDA_ENV' (python=3.8)..."
    conda create -y -n "$CONDA_ENV" python=3.8 || {
        echo "❌ conda create failed" >&2; exit 1; }
fi
# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"
echo "  🐍 python: $(python --version 2>&1)"

_impl="$(python -c "import platform; print(platform.python_implementation())" 2>/dev/null)"
if [ "$_impl" != "CPython" ]; then
    echo "❌ ERROR: python is $_impl, not CPython. Recreate env:" >&2
    echo "       conda env remove -n $CONDA_ENV && bash $0" >&2
    exit 1
fi

# ── 2. PyTorch 1.13.1 + cu117（官方 pin）─────────────────────────────────
TORCH_INDEX_URL="${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu117}"
if ! python -c "import torch; assert torch.__version__.startswith('1.13')" 2>/dev/null; then
    echo "📦 installing torch==1.13.1+cu117 ..."
    pip install --index-url "$TORCH_INDEX_URL" --timeout 600 --retries 5 \
        torch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 || {
        echo "❌ torch install failed" >&2; exit 1; }
fi
python -c "import torch; print(f'  ✅ torch {torch.__version__}  cuda={torch.version.cuda}  available={torch.cuda.is_available()}')"

# ── 3. CUDA toolkit (nvcc 11.7) + gcc 11 via conda ───────────────────────
CUDA_TOOLKIT_LABEL="${CUDA_TOOLKIT_LABEL:-nvidia/label/cuda-11.7.1}"
if ! command -v nvcc >/dev/null 2>&1; then
    echo "📦 installing CUDA toolkit ($CUDA_TOOLKIT_LABEL)..."
    conda install -y -c "$CUDA_TOOLKIT_LABEL" cuda-toolkit || {
        echo "❌ cuda-toolkit install failed" >&2; exit 1; }
fi
echo "  ✅ nvcc: $(nvcc --version 2>/dev/null | tail -1 | xargs)"

if ! command -v x86_64-conda-linux-gnu-gcc >/dev/null 2>&1; then
    echo "📦 installing gcc 11 (gxx_linux-64, python=3.8 pinned)..."
    conda install -y -c conda-forge gxx_linux-64=11 python=3.8 || {
        echo "  ⚠️ conda gcc install failed; will try system gcc" >&2; }
fi
_gcc="$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-gcc"
_gpp="$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-g++"
if [ -x "$_gcc" ] && [ -x "$_gpp" ]; then
    echo "  ✅ gcc: $($_gcc --version | head -1)"
else
    echo "  ⚠️ conda gcc not found; trying system gcc/g++" >&2
    command -v gcc >/dev/null 2>&1 && echo "  gcc: $(gcc --version | head -1)" || echo "  ❌ no gcc" >&2
fi

# ── 4. 官方仓（clone 优先，tarball 兜底）─────────────────────────────────
_TARBALL_MAIN="$SRC_CACHE/splannequin-gs-main.tar.gz"
_TARBALL_DGR="$SRC_CACHE/depth-diff-gaussian-rasterization-e495066.tar.gz"
_TARBALL_SK="$SRC_CACHE/simple-knn-44f7642.tar.gz"

if [ ! -f "$SPLANNEQUIN_DIR/train.py" ]; then
    echo "📦 fetching splannequin-gs -> $SPLANNEQUIN_DIR"
    if ! git clone "$SPLANNEQUIN_REPO" "$SPLANNEQUIN_DIR" 2>/dev/null; then
        echo "  ⚠️ git clone failed (GitHub unreachable); falling back to tarball"
        [ -f "$_TARBALL_MAIN" ] || {
            echo "❌ tarball missing: $_TARBALL_MAIN" >&2
            echo "   Download on Windows side (迅雷/proxy):" >&2
            echo "   https://codeload.github.com/chien90190/splannequin-gs/tar.gz/refs/heads/main" >&2
            exit 1; }
        mkdir -p "$SPLANNEQUIN_DIR"
        tar xzf "$_TARBALL_MAIN" -C "$SPLANNEQUIN_DIR" --strip-components=1
    fi
else
    echo "⏭️  repo already present: $SPLANNEQUIN_DIR"
fi

# 子模块（编译必需）：优先 git submodule，失败回退 tarball
_DGR="$SPLANNEQUIN_DIR/submodules/depth-diff-gaussian-rasterization"
_SK="$SPLANNEQUIN_DIR/submodules/simple-knn"
if [ ! -f "$_DGR/setup.py" ] || [ ! -f "$_SK/setup.py" ]; then
    ( cd "$SPLANNEQUIN_DIR" && git submodule update --init \
        submodules/depth-diff-gaussian-rasterization submodules/simple-knn ) 2>/dev/null || true
fi
if [ ! -f "$_DGR/setup.py" ]; then
    echo "  📦 depth-diff-gaussian-rasterization from tarball"
    [ -f "$_TARBALL_DGR" ] || { echo "❌ missing $_TARBALL_DGR" >&2; exit 1; }
    mkdir -p "$_DGR"
    tar xzf "$_TARBALL_DGR" -C "$_DGR" --strip-components=1
fi
if [ ! -f "$_SK/setup.py" ]; then
    echo "  📦 simple-knn from tarball"
    [ -f "$_TARBALL_SK" ] || { echo "❌ missing $_TARBALL_SK" >&2; exit 1; }
    mkdir -p "$_SK"
    tar xzf "$_TARBALL_SK" -C "$_SK" --strip-components=1
fi

# GLM：depth-diff-gaussian-rasterization 的 third_party/glm 子模块
# （锁定 commit 5c46b9c；codeload tarball 不含子模块，必须单独补）
_TARBALL_GLM="$SRC_CACHE/glm-5c46b9c.tar.gz"
if [ ! -f "$_DGR/third_party/glm/glm/glm.hpp" ]; then
    if [ -f "$_TARBALL_GLM" ]; then
        echo "  📦 glm from tarball"
        mkdir -p "$_DGR/third_party/glm"
        tar xzf "$_TARBALL_GLM" -C "$_DGR/third_party/glm" --strip-components=1
    else
        echo "  ⚠️ glm missing — rasterizer build WILL fail (glm/glm.hpp not found)" >&2
        echo "     Download: https://codeload.github.com/g-truc/glm/tar.gz/5c46b9c07008ae65cb81ab79cd677ecc1934b903" >&2
        echo "     -> $_TARBALL_GLM" >&2
    fi
fi

# ── 4b. 官方代码补丁（train.py 与 gaussian_model.py 调用/定义不一致）──────
python3 "$SCRIPT_DIR/00c_patch_official.py" || {
    echo "❌ official patch failed" >&2; exit 1; }

# ── 5. pip 依赖 ──────────────────────────────────────────────────────────
PIP_FLAGS=(-i https://pypi.tuna.tsinghua.edu.cn/simple --timeout 600 --retries 5)

if [ "${INSTALL_DEPS:-1}" = "1" ]; then
    echo "📦 installing mmcv==1.6.0 (openmmlab prebuilt cu117/torch1.13)..."
    # mmcv 只被用来读 config（mmcv.Config.fromfile），但必须装全版；
    # 源码编译要 1 小时级，优先预编译轮
    pip install mmcv==1.6.0 \
        -f https://download.openmmlab.com/mmcv/dist/cu117/torch1.13/index.html \
        --timeout 600 --retries 5 || {
        echo "  ⚠️ prebuilt mmcv failed; trying tuna source build" >&2
        pip install "${PIP_FLAGS[@]}" mmcv==1.6.0 || echo "  ❌ mmcv failed" >&2; }

    echo "📦 installing runtime deps (tuna)..."
    pip install "${PIP_FLAGS[@]}" \
        matplotlib lpips plyfile pyiqa yt-dlp pandas pytorch_msssim \
        open3d "imageio[ffmpeg]" || echo "  ⚠️ some deps failed" >&2

    # py3.8 兼容 pin（conda 装 gcc 时可能动了 numpy）
    pip install "${PIP_FLAGS[@]}" --force-reinstall --no-deps "numpy==1.24.4" || true
fi

# ── 6. 编译 CUDA 扩展 ────────────────────────────────────────────────────
export CUDA_HOME="${CUDA_HOME:-$CONDA_PREFIX}"
export PATH="$CUDA_HOME/bin:$PATH"
[ -x "$_gcc" ] && [ -x "$_gpp" ] && { export CC="$_gcc"; export CXX="$_gpp"; }
# RTX 3090 = sm_86；只编本机架构省一半时间
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.6}"

echo "🔨 building CUDA ext: depth-diff-gaussian-rasterization"
rm -rf "$_DGR/build" "$_DGR/dist" "$_DGR"/*.egg-info
pip install --no-build-isolation --no-deps "$_DGR" || \
    echo "  ❌ depth-diff-gaussian-rasterization build failed" >&2

echo "🔨 building CUDA ext: simple-knn"
pip install --no-build-isolation --no-deps "$_SK" || \
    echo "  ❌ simple-knn build failed" >&2

# ── 7. 验证 ──────────────────────────────────────────────────────────────
echo ""
echo "--- verification ---"
python -c "import torch; print(f'  ✅ torch {torch.__version__} cuda={torch.cuda.is_available()}')" 2>/dev/null || echo "  [MISS] torch"
python -c "import diff_gaussian_rasterization; print('  ✅ depth-diff-gaussian-rasterization')" 2>/dev/null || echo "  [MISS] depth-diff-gaussian-rasterization"
python -c "import simple_knn; print('  ✅ simple-knn')" 2>/dev/null || echo "  [MISS] simple-knn"
python -c "import mmcv; print(f'  ✅ mmcv {mmcv.__version__}')" 2>/dev/null || echo "  [MISS] mmcv"
python -c "import open3d; print('  ✅ open3d')" 2>/dev/null || echo "  [MISS] open3d"
python -c "import pyiqa; print('  ✅ pyiqa')" 2>/dev/null || echo "  [MISS] pyiqa"
[ -f "$SPLANNEQUIN_DIR/train.py" ] && echo "  ✅ official repo" || echo "  [MISS] official repo"

echo ""
echo "🎉 [00a] Done. Next:"
echo "  1. 自备数据: 把图片放到 <my_video>/images/*.png"
echo "  2. SCENE_DIR=<my_video> bash splannequin/01_colmap.sh"
echo "  3. SCENE_DIR=<my_video> bash splannequin/02_train.sh"

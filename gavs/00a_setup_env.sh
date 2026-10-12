#!/usr/bin/env bash
# 00a_setup_env.sh — GaVS WSL Ubuntu 24.04 从零建环境。
#
# 与官方 README 的两处刻意偏离（原因见 README_wsl.md「环境决策」）：
#   1. torch 用 cu121 而非官方 cu126 —— xformers==0.0.25.post1（UniDepthV1
#      NystromBlock 的运行时硬依赖）PyPI wheel 按 torch 2.2.2+cu121 编译，
#      混 cu126 会在 import xformers.ops 时报 CUDA 扩展加载失败。
#   2. CUDA toolkit 用 conda nvidia/label/cuda-12.1.1 —— 与 torch cu121 对齐，
#      且 thirdparty/diff-gaussian-rasterization-w-pose/setup.py 硬编码了
#      -D__CUDACC_VER_MAJOR__=12 -D__CUDACC_VER_MINOR__=1。
#
# 用法：
#   bash gavs/00a_setup_env.sh          # 全流程（幂等，已装步骤自动跳过）
set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"

CONDA_BASE="$HOME/miniconda3"
if [ ! -f "$CONDA_BASE/etc/profile.d/conda.sh" ]; then
    echo "❌ ERROR: miniconda3 not found at $CONDA_BASE" >&2
    exit 1
fi
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main 2>/dev/null || true
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r 2>/dev/null || true

# 默认路径（写不进 proxy.env —— 该文件全仓共享，gavs 专属默认值只放这里和 _env.sh）
export GAVS_DIR="${GAVS_DIR:-$HOME/repos/GaVS}"
export GAVS_DATA_ROOT="${GAVS_DATA_ROOT:-$HOME/model/gavs/gavs-data}"
export GAVS_RESULTS_DIR="${GAVS_RESULTS_DIR:-$HOME/output/gavs_results}"
export GAVS_STAGING_DIR="${GAVS_STAGING_DIR:-/mnt/d/wheel/gavs_ms}"
export UNIDEPTH_REPO="${UNIDEPTH_REPO:-$HOME/repos/UniDepth}"

source "$SCRIPT_DIR/_env.sh"

echo "=== [00a] Setup gavs env on WSL ==="
echo "  📁 repo:     $GAVS_DIR"
echo "  🏋️ data:     $GAVS_DATA_ROOT"
echo "  💾 output:   $GAVS_RESULTS_DIR"
echo "  🏠 conda:    $CONDA_ENV"
echo ""

# ── 1. 建 env ─────────────────────────────────────────────────────────────
_new_prefix="$(conda info --base 2>/dev/null)/envs/$CONDA_ENV"
if [ -d "$_new_prefix/conda-meta" ]; then
    echo "⏭️  env '$CONDA_ENV' already exists"
else
    echo "📦 creating conda env '$CONDA_ENV' (python=3.10)..."
    conda create -y -n "$CONDA_ENV" python=3.10 || { echo "❌ conda create failed" >&2; exit 1; }
fi
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"
echo "  🐍 python: $(python --version 2>&1)"

# ── 2. PyTorch 2.2.2 + cu121 ──────────────────────────────────────────────
TORCH_INDEX_URL="${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu121}"
if ! python -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
    echo "📦 installing PyTorch 2.2.2 (cu121)..."
    pip install --index-url "$TORCH_INDEX_URL" --timeout 600 --retries 5 \
        numpy==1.26.4 torch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2 || {
        echo "❌ torch install failed" >&2; exit 1; }
fi
python -c "import torch; print(f'  ✅ torch {torch.__version__}  cuda={torch.version.cuda}  available={torch.cuda.is_available()}')"

# ── 3. CUDA toolkit (nvcc 12.1) + gcc 12 + ffmpeg，全走 conda 免 sudo ──────
CUDA_TOOLKIT_LABEL="${CUDA_TOOLKIT_LABEL:-nvidia/label/cuda-12.1.1}"
if ! command -v nvcc >/dev/null 2>&1; then
    echo "📦 installing CUDA toolkit ($CUDA_TOOLKIT_LABEL)..."
    conda install -y -c "$CUDA_TOOLKIT_LABEL" cuda-toolkit || { echo "❌ cuda-toolkit failed" >&2; exit 1; }
fi
echo "  ✅ nvcc: $(nvcc --version 2>/dev/null | tail -1 | xargs)"

if ! command -v x86_64-conda-linux-gnu-gcc >/dev/null 2>&1; then
    echo "📦 installing gcc 12 (gxx_linux-64)..."
    conda install -y -c conda-forge gxx_linux-64=12 python=3.10 || echo "  ⚠️ gxx install failed" >&2
fi

# ffmpeg：evaluate.py 用它把渲染帧合成 mp4（os.system 调用，必须在 PATH）
if ! command -v ffmpeg >/dev/null 2>&1; then
    echo "📦 installing ffmpeg (conda-forge)..."
    conda install -y -c conda-forge ffmpeg || echo "  ⚠️ ffmpeg install failed（eval 的 mp4 合成会跳过）" >&2
fi
command -v ffmpeg >/dev/null 2>&1 && echo "  ✅ ffmpeg: $(ffmpeg -version 2>/dev/null | head -1)"

# ── 4. pip 依赖（清华镜像）────────────────────────────────────────────────
PIP_FLAGS=(-i https://pypi.tuna.tsinghua.edu.cn/simple --timeout 600 --retries 5)

# xformers 0.0.25.post1：PyPI wheel 按 torch 2.2.2+cu121 编译，与本 env 匹配。
# ⚠️ 不要让它升级/替换 torch —— 用 --no-deps 不行（它需要精确 torch==2.2.2，
#    已满足则不会动）。装完立刻验证 import。
if ! python -c "import xformers" 2>/dev/null; then
    echo "📦 installing xformers==0.0.25.post1..."
    pip install "${PIP_FLAGS[@]}" xformers==0.0.25.post1 || { echo "❌ xformers failed" >&2; exit 1; }
fi
python -c "import xformers, xformers.ops; print(f'  ✅ xformers {xformers.__version__}')" || {
    echo "❌ xformers import/ops failed（通常是 torch CUDA 版本不匹配）" >&2; exit 1; }
python -c "import torch; v=torch.__version__; assert v.startswith('2.2.2'), f'torch 被替换成了 {v}！重装 torch cu121'"
pip install "${PIP_FLAGS[@]}" --force-reinstall --no-deps numpy==1.26.4

# ── 5. Clone 官方仓 ───────────────────────────────────────────────────────
mkdir -p "$(dirname "$GAVS_DIR")"
if [ ! -d "$GAVS_DIR/.git" ]; then
    echo "📦 cloning GaVS -> $GAVS_DIR"
    git clone https://github.com/sinoyou/GaVS.git "$GAVS_DIR" || \
        git -c http.sslVerify=false clone https://github.com/sinoyou/GaVS.git "$GAVS_DIR" || {
        echo "❌ GaVS clone failed（github 不通时走 ghfast.top 镜像）" >&2
        git clone https://ghfast.top/https://github.com/sinoyou/GaVS.git "$GAVS_DIR"
    }
else
    echo "⏭️  GaVS repo already present: $GAVS_DIR"
fi

# UniDepth 仓：torch.hub.load("lpiccinelli-eth/UniDepth", ...) 会从 GitHub 下载，
# 预置到 torch hub 缓存目录即可完全离线（代码里 force_reload=False）。
if [ ! -d "$UNIDEPTH_REPO/.git" ]; then
    echo "📦 cloning UniDepth -> $UNIDEPTH_REPO"
    git clone --depth 1 https://github.com/lpiccinelli-eth/UniDepth.git "$UNIDEPTH_REPO" || \
        git -c http.sslVerify=false clone --depth 1 https://github.com/lpiccinelli-eth/UniDepth.git "$UNIDEPTH_REPO"
fi
TORCH_HUB_DIR="${TORCH_HOME:-$HOME/.cache/torch}/hub"
HUB_UNIDEPTH="$TORCH_HUB_DIR/lpiccinelli-eth_UniDepth_main"
if [ ! -d "$HUB_UNIDEPTH" ]; then
    echo "📦 seeding torch hub cache: $HUB_UNIDEPTH"
    mkdir -p "$TORCH_HUB_DIR"
    cp -r "$UNIDEPTH_REPO" "$HUB_UNIDEPTH"
else
    echo "⏭️  torch hub cache already seeded"
fi

# ── 6. 装 GaVS 依赖 ───────────────────────────────────────────────────────
echo "📦 installing GaVS requirements..."
pip install "${PIP_FLAGS[@]}" -r "$GAVS_DIR/requirements.txt" || \
    echo "  ⚠️ some deps failed（若关键包缺失后面 verify 会报）" >&2
# requirements 里有未钉版本的包会把 numpy 带到 2.x —— torch 2.2.2 是按 numpy 1.x
# 编译的，numpy 2.x 下 import torch 报 "_ARRAY_API not found" 且 tensor.numpy() 会炸。
# 必须在 requirements 之后强制钉回（第二轮实测踩坑，2026-10-12）。
pip install "${PIP_FLAGS[@]}" --force-reinstall --no-deps numpy==1.26.4
python -c "import numpy, torch; assert numpy.__version__.startswith('1.'), numpy.__version__" || {
    echo "❌ numpy 钉回 1.x 失败" >&2; exit 1; }

# ── 7. 编译 diff-gaussian-rasterization-w-pose ────────────────────────────
# 前置补丁：torch 2.2.2 捆绑的 pybind11 cast.h 与 nvcc 12.1 不兼容（nvcc 前端
# 无法解析 `caster.operator typename ...::template cast_op_type<T>()`，报
# "expected template-name before '<' token"）。torch 2.5.1 已按 pybind11
# PR #4893 修复（先 using result_t 再调 operator result_t()）。此处把 env 内
# torch 头文件打成同样写法；幂等，patch 后重跑自动跳过。
python - <<'PYEOF'
import pathlib, sys
cast = pathlib.Path(sys.prefix) / "lib/python3.10/site-packages/torch/include/pybind11/cast.h"
src = cast.read_text()
if "PR #4893" in src:
    print("  ⏭️  pybind11 cast.h already patched")
else:
    old1 = "    return caster.operator typename make_caster<T>::template cast_op_type<T>();"
    new1 = ("    using result_t = typename make_caster<T>::template cast_op_type<T>; // See PR #4893\n"
            "    return caster.operator result_t();")
    old2 = ("    return std::move(caster).operator typename make_caster<T>::\n"
            "        template cast_op_type<typename std::add_rvalue_reference<T>::type>();")
    new2 = ("    using result_t = typename make_caster<T>::template cast_op_type<\n"
            "        typename std::add_rvalue_reference<T>::type>; // See PR #4893\n"
            "    return std::move(caster).operator result_t();")
    assert src.count(old1) == 1 and src.count(old2) == 1, "cast.h pattern not found（torch 版本变了？）"
    cast.write_text(src.replace(old1, new1).replace(old2, new2))
    print("  🔧 patched torch pybind11 cast.h for nvcc 12.1 (PR #4893)")
PYEOF

export CUDA_HOME="${CUDA_HOME:-$CONDA_PREFIX}"
export PATH="$CUDA_HOME/bin:$PATH"
_gcc="$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-gcc"
_gpp="$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-g++"
[ -x "$_gcc" ] && [ -x "$_gpp" ] && export CC="$_gcc" CXX="$_gpp"
# 3090 = sm_86；显式指定避免为全架构编译（快很多）
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.6}"

_DGR="$GAVS_DIR/thirdparty/diff-gaussian-rasterization-w-pose"
if ! python -c "import diff_gaussian_rasterization" 2>/dev/null; then
    echo "🔨 building CUDA ext: diff-gaussian-rasterization-w-pose"
    rm -rf "$_DGR/build" "$_DGR/dist" "$_DGR"/*.egg-info
    pip install --no-build-isolation --no-deps "$_DGR" || {
        echo "❌ rasterizer build failed" >&2; exit 1; }
fi
python -c "import diff_gaussian_rasterization; print('  ✅ diff_gaussian_rasterization importable')" || {
    echo "❌ rasterizer not importable" >&2; exit 1; }

# ── 8. Verify ─────────────────────────────────────────────────────────────
echo ""
echo "--- verification ---"
[ -f "$GAVS_DIR/train.py" ] && echo "  ✅ GaVS code" || echo "  [MISS] GaVS: $GAVS_DIR"
python -c "import torch; print(f'  ✅ torch {torch.__version__} cuda={torch.cuda.is_available()}')"
python -c "import xformers.ops; print('  ✅ xformers')"
python -c "import diff_gaussian_rasterization; print('  ✅ rasterizer')"
python -c "import hydra, lightning, wandb; print('  ✅ hydra/lightning/wandb')" 2>/dev/null || echo "  [MISS] hydra/lightning/wandb"
command -v ffmpeg >/dev/null 2>&1 && echo "  ✅ ffmpeg" || echo "  [MISS] ffmpeg"
[ -d "$HUB_UNIDEPTH" ] && echo "  ✅ torch hub: UniDepth" || echo "  [MISS] torch hub UniDepth"

echo ""
echo "🎉 [00a] Done. 下一步："
echo "  1. 峰哥用迅雷下载权重/数据到 D:\\wheel\\gavs_ms\\（链接见 README_wsl.md）"
echo "  2. bash gavs/01_download_models.sh   # 搬入 Linux fs + 解压 + 种 HF 缓存"
echo "  3. SCENE=dynamic_dance bash gavs/02_run_finetune.sh"

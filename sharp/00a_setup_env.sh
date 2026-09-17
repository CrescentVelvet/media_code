#!/usr/bin/env bash
# 00a_setup_env.sh — WSL Ubuntu 24.04 环境安装（SHARP 推理链路）。
#
# 做什么：
#   1. 建 conda env `sharp`（python=3.13，与官方 .python-version 一致）
#   2. pip 装 torch 2.8.0 + torchvision 0.23.0（PyPI 默认 = cu128）
#   3. conda 装 CUDA toolkit 12.8 + gcc 12（**不需要 sudo**，全进 env 前缀）
#      —— gsplat 1.5.3 是纯 Python wheel，CUDA 光栅化是 JIT 编译，
#         所以 nvcc + g++ 必须在运行时可用（本机无系统 gcc，必须装）
#   4. clone 官方仓 apple-aiml-research/ml-sharp 到 ~/repos/ml-sharp
#   5. pip install -r requirements.txt（含 -e .，装 `sharp` CLI）
#   6. 把 SHARP_* 路径追加进 proxy.env（若尚未存在）
#
# 用法：
#   bash sharp/00a_setup_env.sh
#   SKIP_TOCLONE=1 bash sharp/00a_setup_env.sh      # 已有 ~/repos/ml-sharp，跳过 clone
#   CUDA_LABEL=nvidia/label/cuda-12.6.3 bash sharp/00a_setup_env.sh   # 换 CUDA 版本
set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"

# ── 1. Source conda ────────────────────────────────────────────────────────
CONDA_BASE="$HOME/miniconda3"
if [ ! -f "$CONDA_BASE/etc/profile.d/conda.sh" ]; then
    echo "❌ ERROR: miniconda3 not found at $CONDA_BASE" >&2
    echo "       Install first:" >&2
    echo "         curl -fsSL https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -o /tmp/miniconda.sh" >&2
    echo "         bash /tmp/miniconda.sh -b -p $CONDA_BASE" >&2
    exit 1
fi
# shellcheck disable=SC1091
source "$CONDA_BASE/etc/profile.d/conda.sh"

# Accept ToS for default channels (conda 26.x requires it; older versions ignore).
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main 2>/dev/null || true
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r 2>/dev/null || true

# ── 2. WSL 路径默认值（Linux fs：仓 ~/repos，权重 D 盘，输出 ~/output）─────
# 在 source _env.sh 之前 export，让它的 ${VAR:-default} 取到这些值。
export SHARP_DIR="${SHARP_DIR:-$HOME/repos/ml-sharp}"
export SHARP_MODEL_DIR="${SHARP_MODEL_DIR:-/mnt/d/wheel/sharp_ms}"
export SHARP_RESULTS_DIR="${SHARP_RESULTS_DIR:-$HOME/output/sharp_results}"

# ── 3. 追加 SHARP_* 路径到 proxy.env（已有则不动）─────────────────────────
# proxy.env 在仓根，gitignored，全仓共享。只追加缺失的 SHARP_* 段，
# 不覆盖其它项目（vggt_human / flame_human）已经写进去的内容。
PROXY_ENV="$REPO_DIR/proxy.env"
if ! grep -q "SHARP_MODEL_DIR" "$PROXY_ENV" 2>/dev/null; then
    echo "📦 appending SHARP_* path overrides -> $PROXY_ENV"
    cat >> "$PROXY_ENV" <<'ENVEOF'

# ── sharp（Apple SHARP 单图→3DGS；仅 sharp/_env.sh 读取）────────────────────
# 官方仓（Linux fs，JIT 编译快）；权重在 D 盘（读一次，不必拷到 Linux fs）
export SHARP_DIR="$HOME/repos/ml-sharp"
export SHARP_MODEL_DIR="/mnt/d/wheel/sharp_ms"
export SHARP_RESULTS_DIR="${SHARP_RESULTS_DIR:-$HOME/output/sharp_results}"
ENVEOF
    echo "  ✅ appended"
else
    echo "⏭️  proxy.env 已有 SHARP_* 配置: $PROXY_ENV"
fi

# ── 4. Source _env.sh（拿到 proxy.env + conda activate + 路径）──────────────
# shellcheck disable=SC1091
source "$SCRIPT_DIR/_env.sh"

echo "=== [00a] Setup SHARP env on WSL ==="
echo "  🤖 SHARP code:  $SHARP_DIR"
echo "  🏋️ weights:     $SHARP_CKPT"
echo "  💾 output:      $RESULTS_DIR"
echo "  🏠 conda env:   $CONDA_ENV"
echo ""

# ── 5. 建 conda env ───────────────────────────────────────────────────────
# python=3.13 对齐官方 .python-version / pyproject（pyright pythonVersion=3.13）。
_new_prefix="$(conda info --base 2>/dev/null)/envs/$CONDA_ENV"
if [ -d "$_new_prefix/conda-meta" ] && [ -f "$_new_prefix/conda-meta/history" ]; then
    echo "--- conda env '$CONDA_ENV' already exists at $_new_prefix ---"
else
    echo "📦 creating conda env '$CONDA_ENV' (python=3.13)..."
    conda create -y -n "$CONDA_ENV" python=3.13 || {
        echo "❌ conda create failed" >&2
        exit 1
    }
    echo "  ✅ created"
fi

# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"
echo "  🐍 python: $(python --version 2>&1) at $(which python)"

_impl="$(python -c "import platform; print(platform.python_implementation())" 2>/dev/null)"
if [ "$_impl" != "CPython" ]; then
    echo "❌ ERROR: python is $_impl, not CPython. Recreate env:" >&2
    echo "       conda env remove -n $CONDA_ENV && bash $0" >&2
    exit 1
fi

# ── 6. PyTorch 2.8.0 + torchvision 0.23.0（PyPI 默认 wheel = cu128）────────
# 本地 /mnt/d/wheel 里若有同名 wheel 优先用（省流量）。
WHEELS_DIR="${WHEELS_DIR:-/mnt/d/wheel}"
mkdir -p "$WHEELS_DIR"
PIP_FLAGS=(-i "${PIP_INDEX_URL:-https://pypi.tuna.tsinghua.edu.cn/simple}"
    --find-links "$WHEELS_DIR" --timeout 600 --retries 5)
TORCH_VERSION="${TORCH_VERSION:-2.8.0}"
TORCHVISION_VERSION="${TORCHVISION_VERSION:-0.23.0}"

if ! python -c "import torch; assert torch.__version__.startswith('$TORCH_VERSION'); assert torch.cuda.is_available()" 2>/dev/null; then
    echo "📦 installing PyTorch $TORCH_VERSION + torchvision $TORCHVISION_VERSION ..."
    pip install "${PIP_FLAGS[@]}" "torch==$TORCH_VERSION" "torchvision==$TORCHVISION_VERSION" || {
        echo "❌ torch install failed" >&2
        echo "  备用：TORCH_INDEX_URL 走官方源" >&2
        echo "    pip install --index-url https://download.pytorch.org/whl/cu128 torch==$TORCH_VERSION torchvision==$TORCHVISION_VERSION" >&2
        exit 1
    }
fi
python -c "import torch; print(f'  ✅ torch {torch.__version__}  cuda={torch.version.cuda}  available={torch.cuda.is_available()}')" || exit 1

# ── 7. CUDA toolkit 12.8（nvcc）+ gcc 12 via conda（无需 sudo）────────────
# gsplat 1.5.3 = py3-none-any wheel → CUDA 光栅化在**首次使用时 JIT 编译**，
# 需要 nvcc 与 g++ 同时可用；本机没有系统 gcc/g++，必须装进 env。
CUDA_LABEL="${CUDA_LABEL:-nvidia/label/cuda-12.8.0}"
if ! command -v nvcc >/dev/null 2>&1; then
    echo "📦 installing CUDA toolkit ($CUDA_LABEL) ..."
    conda install -y -c "$CUDA_LABEL" cuda-toolkit || {
        echo "  ⚠️ label 安装失败，回退到 nvidia 主频道 cuda-nvcc ..." >&2
        conda install -y -c nvidia cuda-nvcc=12.8 || {
            echo "❌ cuda-toolkit install failed" >&2
            exit 1
        }
    }
fi
echo "  ✅ nvcc: $(nvcc --version 2>/dev/null | tail -2 | head -1 | xargs)"

if ! command -v x86_64-conda-linux-gnu-gcc >/dev/null 2>&1; then
    echo "📦 installing gcc 12 (gxx_linux-64, python=3.13 pinned) ..."
    conda install -y -c conda-forge gxx_linux-64=12 python=3.13 || {
        echo "  ⚠️ conda gcc 安装失败；可改用系统编译器: sudo apt install build-essential" >&2
    }
fi
_gcc="$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-gcc"
_gpp="$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-g++"
if [ -x "$_gcc" ] && [ -x "$_gpp" ]; then
    echo "  ✅ gcc: $($_gcc --version | head -1)"
else
    echo "  ⚠️ conda gcc not found; 回退系统 gcc/g++" >&2
    command -v gcc >/dev/null 2>&1 && echo "  gcc: $(gcc --version | head -1)" || \
        echo "  ❌ 无编译器 —— gsplat JIT 编译会失败。请装: sudo apt install build-essential" >&2
fi

# ── 8. Clone 官方仓 ───────────────────────────────────────────────────────
if [ "${SKIP_TOCLONE:-0}" != "1" ] && [ ! -d "$SHARP_DIR/.git" ]; then
    mkdir -p "$(dirname "$SHARP_DIR")"
    echo "📦 cloning ml-sharp -> $SHARP_DIR"
    git clone "$SHARP_REPO" "$SHARP_DIR" || \
        git -c http.sslVerify=false clone "$SHARP_REPO" "$SHARP_DIR" || {
        echo "❌ clone failed" >&2
        exit 1
    }
else
    echo "⏭️  ml-sharp 已存在: $SHARP_DIR"
fi

# ── 9. 装 SHARP 依赖 + 自身（pip install -r requirements.txt 含 -e .）─────
# requirements.txt 固定了 torch==2.8.0 / gsplat==1.5.3 / timm / plyfile 等。
# ⚠️ timm 只用来搭 ViT 架构（TimmViT(timm.models.VisionTransformer)），
#    代码里没有 pretrained=True，**不会**额外从 HF 拉权重。
if [ ! -f "$SHARP_DIR/requirements.txt" ]; then
    echo "❌ ERROR: $SHARP_DIR/requirements.txt 不存在（clone 未完成？）" >&2
    exit 1
fi
echo "📦 installing SHARP requirements (torch/gsplat/nvidia-* 约 3-4GB，走镜像) ..."
pip install "${PIP_FLAGS[@]}" -r "$SHARP_DIR/requirements.txt" || {
    echo "❌ requirements 安装失败" >&2
    exit 1
}

# CLI 入口（pip install -e . 装出来的控制台脚本）
if command -v sharp >/dev/null 2>&1; then
    echo "  ✅ sharp CLI: $(command -v sharp)"
else
    echo "  ⚠️ sharp CLI 未在 PATH；手动: pip install -e \"$SHARP_DIR\"" >&2
fi

# ── 10. conda init bash ───────────────────────────────────────────────────
if ! grep -q "miniconda3" "$HOME/.bashrc" 2>/dev/null; then
    echo "📦 running 'conda init bash' ..."
    conda init bash >/dev/null 2>&1 && echo "  ✅ done（重开 shell 或 source ~/.bashrc 后生效）"
else
    echo "⏭️  conda already initialized in ~/.bashrc"
fi

# ── 11. 验证 ──────────────────────────────────────────────────────────────
echo ""
echo "--- verification ---"
python -c "import torch; print(f'  ✅ torch {torch.__version__} cuda={torch.cuda.is_available()}')" 2>/dev/null || echo "  [MISS] torch"
nvcc --version >/dev/null 2>&1 && echo "  ✅ nvcc ($(nvcc --version | tail -2 | head -1 | xargs))" || echo "  [MISS] nvcc（gsplat JIT 编译会失败）"
[ -x "$_gpp" ] && echo "  ✅ g++ (conda gcc 12)" || (command -v g++ >/dev/null 2>&1 && echo "  ✅ g++ (system)" || echo "  [MISS] g++")
command -v sharp >/dev/null 2>&1 && echo "  ✅ sharp CLI" || echo "  [MISS] sharp CLI"
[ -d "$SHARP_DIR/src/sharp" ] && echo "  ✅ SHARP code: $SHARP_DIR" || echo "  [MISS] SHARP code: $SHARP_DIR"
if [ -f "$SHARP_CKPT" ]; then
    echo "  ✅ weights: $SHARP_CKPT"
else
    echo "  [---] weights 未下载 —— 跑 sharp/01_download_models.sh（或迅雷下好放进去）"
fi

echo ""
echo "🎉 [00a] Done. Next:"
echo "  1. bash sharp/01_download_models.sh        # 下权重（2.62GB，或迅雷→$SHARP_CKPT）"
echo "  2. GPU=0 INPUT=~/my_images bash sharp/02_run_inference.sh"

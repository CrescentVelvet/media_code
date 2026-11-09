#!/usr/bin/env bash
# Scene Splatter 共享环境设置。被本目录所有 .sh source。
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"
export SCRIPT_DIR REPO_DIR

# 1. 代理 / 镜像（从 proxy.env 读，含 HF_ENDPOINT / PIP_INDEX_URL）
if [ -f "$REPO_DIR/proxy.env" ]; then
    set -a; source "$REPO_DIR/proxy.env"; set +a
fi
[ -n "${http_proxy:-}" ]  && export HTTP_PROXY="$http_proxy"
[ -n "${https_proxy:-}" ] && export HTTPS_PROXY="$https_proxy"
HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"
export HF_HUB_DISABLE_XET

# 2. conda env 激活（本机 conda 未 init 时自动找安装位置）
CONDA_ENV="${CONDA_ENV:-scene_splatter}"
export CONDA_ENV
if ! command -v conda >/dev/null 2>&1; then
    for _cb in "$HOME/miniconda3" "$HOME/anaconda3" "/opt/conda"; do
        if [ -f "$_cb/etc/profile.d/conda.sh" ]; then
            source "$_cb/etc/profile.d/conda.sh"
            break
        fi
    done
    unset _cb
fi
if ! command -v conda >/dev/null 2>&1; then
    echo "ERROR: conda not found on PATH (need env '$CONDA_ENV')." >&2
    exit 1
fi
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV" 2>/dev/null || true  # env 可能还没建

# 2b. xformers 运行时：PyPI 版 xformers（0.0.22/0.0.23）全是 cu121 构建，本机 torch 是
# cu118——缺的 libcudart.so.12/libcublas.so.12 由 pip 包 nvidia-cuda-runtime-cu12 /
# nvidia-cublas-cu12 补齐（驱动 595.95 满足 CUDA 12.x 前向兼容），这里挂进 LD_LIBRARY_PATH。
# 2026-11-09 实测：不挂则 xformers._C ImportError libcudart.so.12，dinov2 注意力崩。
_NVIDIA_LIBS="$(ls -d "$CONDA_PREFIX"/lib/python*/site-packages/nvidia 2>/dev/null | head -1)"
if [ -n "$_NVIDIA_LIBS" ] && [ -d "$_NVIDIA_LIBS" ]; then
    export LD_LIBRARY_PATH="$_NVIDIA_LIBS/cuda_runtime/lib:$_NVIDIA_LIBS/cublas/lib:${LD_LIBRARY_PATH:-}"
fi
unset _NVIDIA_LIBS

# 3. GPU 选卡
if [ -n "${GPU:-}" ]; then
    export CUDA_VISIBLE_DEVICES="$GPU"
fi

# 4. 路径（全部允许外部覆盖）
#    官方仓：WSL 放 ~/repos/Scene-Splatter（Linux fs，编译/IO 快）
#    权重复用池：D:\wheel\scene_splatter（峰哥约定 2026-11-05），消费侧一律软链
SS_DIR="${SS_DIR:-$HOME/repos/Scene-Splatter}"
MODEL_DIR="${MODEL_DIR:-$HOME/model/scene_splatter}"
RESULTS_DIR="${RESULTS_DIR:-$HOME/output/scene_splatter_results}"
WHEEL_SS_DIR="${WHEEL_SS_DIR:-/mnt/d/wheel/scene_splatter}"
export SS_DIR MODEL_DIR RESULTS_DIR WHEEL_SS_DIR

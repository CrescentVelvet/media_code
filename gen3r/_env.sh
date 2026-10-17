#!/usr/bin/env bash
# Gen3R 共享环境设置。被本目录所有 .sh source。
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
CONDA_ENV="${CONDA_ENV:-gen3r}"
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
conda activate "$CONDA_ENV" 2>/dev/null || true

# 3. GPU 选卡
if [ -n "${GPU:-}" ]; then
    export CUDA_VISIBLE_DEVICES="$GPU"
fi

# 3b. open3d 需要 libgomp.so.1（系统未装，conda env 里装了 libgomp）。
#     直接用 env 的 python 二进制时不会自动搜索 env/lib，必须显式加。
if [ -n "${CONDA_PREFIX:-}" ] && [ -d "$CONDA_PREFIX/lib" ]; then
    export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
fi

# 4. 路径（全部允许外部覆盖）
#    官方仓：WSL 放 ~/repos/Gen3R（Linux fs，编译/IO 快）；服务器为 media_code 的 sibling
GEN3R_DIR="${GEN3R_DIR:-$HOME/repos/Gen3R}"
MODEL_DIR="${MODEL_DIR:-$HOME/model/gen3r}"
RESULTS_DIR="${RESULTS_DIR:-$HOME/output/gen3r_results}"
export GEN3R_DIR MODEL_DIR RESULTS_DIR

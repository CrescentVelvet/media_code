# gavs/_env.sh — GaVS（3D 稳像）共享环境设置，被本目录所有 .sh source。
# 结构同仓级模板：proxy.env → CA → conda → GPU → 路径。
SCRIPT_DIR="${SCRIPT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
REPO_DIR="$(dirname "$SCRIPT_DIR")"

# 1. 代理 / 镜像（proxy.env 全仓共享；本机无代理，主要是 HF_ENDPOINT / PIP 镜像）
if [ -f "$REPO_DIR/proxy.env" ]; then
    set -a; source "$REPO_DIR/proxy.env"; set +a
fi
[ -n "${http_proxy:-}" ]  && export HTTP_PROXY="$http_proxy"
[ -n "${https_proxy:-}" ] && export HTTPS_PROXY="$https_proxy"

# 2. CA bundle（公司代理 TLS 拦截兜底；本机一般走不到）
SYS_CA=/etc/ssl/certs/ca-certificates.crt
USER_CA="$HOME/.ca-bundle.crt"
if [ -f "$USER_CA" ]; then CA_FILE="$USER_CA"
elif [ -f "$SYS_CA" ]; then CA_FILE="$SYS_CA"
else CA_FILE=""; fi
if [ -n "$CA_FILE" ]; then
    : "${REQUESTS_CA_BUNDLE:=$CA_FILE}"
    : "${SSL_CERT_FILE:=$CA_FILE}"
    : "${GIT_SSL_CAINFO:=$CA_FILE}"
    : "${PIP_CERT:=$CA_FILE}"
    export REQUESTS_CA_BUNDLE SSL_CERT_FILE GIT_SSL_CAINFO PIP_CERT
fi
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"

# 本项目所有 HF 资产（UniDepth v1 权重）由 01 脚本预置进本地 HF 缓存；
# 本机 huggingface.co 被 RST、hf-mirror 不稳定，强制离线防止 hf_hub_download
# 卡在 HEAD 请求上。
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"

# 3. conda env 激活
CONDA_ENV="${CONDA_ENV:-gavs}"
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
    echo "       Install miniconda or run: source ~/miniconda3/etc/profile.d/conda.sh" >&2
    exit 1
fi
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV" 2>/dev/null || true

# 4. GPU 选卡
if [ -n "${GPU:-}" ]; then
    export CUDA_VISIBLE_DEVICES="$GPU"
fi

# 5. 路径（${VAR:-default} 允许外部覆盖）
# 官方仓放 Linux fs（要编译 CUDA 扩展）；数据/权重放 ~/model；输出放 ~/output
GAVS_DIR="${GAVS_DIR:-$HOME/repos/GaVS}"
GAVS_DATA_ROOT="${GAVS_DATA_ROOT:-$HOME/model/gavs/gavs-data}"
GAVS_RESULTS_DIR="${GAVS_RESULTS_DIR:-$HOME/output/gavs_results}"
# 迅雷下载暂存区（Windows 盘，峰哥下载后放这里，01 脚本搬入 Linux fs）
GAVS_STAGING_DIR="${GAVS_STAGING_DIR:-/mnt/d/wheel/gavs_ms}"
export REPO_DIR GAVS_DIR GAVS_DATA_ROOT GAVS_RESULTS_DIR GAVS_STAGING_DIR

# 默认场景
SCENE="${SCENE:-dynamic_dance}"
export SCENE

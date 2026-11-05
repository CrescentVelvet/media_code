# da3_human/_env.sh — 共享环境设置（被本目录所有 .sh 脚本 source）
# 服务器默认路径按 ../AGENTS.md 第 2 节；WSL 路径覆盖见仓根 proxy.env（DA3_DIR / DA3_HUMAN_*）

SCRIPT_DIR_ENV="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MEDIA_CODE_ROOT="$(cd "$SCRIPT_DIR_ENV/.." && pwd)"
CODE_DIR="$(cd "$MEDIA_CODE_ROOT/.." && pwd)"

# 1. 代理（从仓根 proxy.env 读）
if [ -f "$MEDIA_CODE_ROOT/proxy.env" ]; then
    set -a; source "$MEDIA_CODE_ROOT/proxy.env"; set +a
fi
[ -n "${http_proxy:-}" ]  && export HTTP_PROXY="$http_proxy"
[ -n "${https_proxy:-}" ] && export HTTPS_PROXY="$https_proxy"

# 2. CA bundle（公司代理 TLS 拦截）
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

# 3. conda env 激活
CONDA_ENV="${CONDA_ENV:-da3_human}"
export CONDA_ENV
# Fallback: conda 不在 PATH 时自动找常见安装位置（WSL 本机未 conda init 时触发）
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
conda activate "$CONDA_ENV" 2>/dev/null || true  # env 可能还没建（00 负责创建）

# 4. GPU 选卡
if [ -n "${GPU:-}" ]; then
    export CUDA_VISIBLE_DEVICES="$GPU"
fi

# 5. 路径（都用 ${VAR:-default} 允许外部覆盖；WSL 由 proxy.env 覆盖）
export DA3_DIR="${DA3_DIR:-$CODE_DIR/Depth-Anything-3}"
export DA3_MODEL_DIR="${DA3_HUMAN_MODEL_DIR:-$CODE_DIR/../../model/da3_human}"
export RESULTS_DIR="${DA3_HUMAN_RESULTS_DIR:-$CODE_DIR/../da3_human_results}"

# _env.sh — 共享环境设置：proxy + CA bundle + conda 激活 + 路径默认值。
# 被 00a/01/02 source。调用方需先设 SCRIPT_DIR。
REPO_DIR="$(dirname "$SCRIPT_DIR")"

# 1. 代理（从 proxy.env 读）
if [ -f "$REPO_DIR/proxy.env" ]; then
    set -a; source "$REPO_DIR/proxy.env"; set +a
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

# 3. conda env 激活（不存在时不中断，00a 负责创建）
CONDA_ENV="${CONDA_ENV:-qwen_image_2_1}"
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
conda activate "$CONDA_ENV" 2>/dev/null || true  # env 可能还没建

# 4. GPU 选卡
if [ -n "${GPU:-}" ]; then
    export CUDA_VISIBLE_DEVICES="$GPU"
fi

# 5. 路径（都用 ${VAR:-default} 允许外部覆盖）
# 权重在 Windows 盘 D:\wheel（全算法共享的下载缓存），只读访问，慢但不影响推理
MODEL_DIR="${MODEL_DIR:-/mnt/d/wheel}"
MODEL_PATH="${MODEL_PATH:-/mnt/d/wheel/qwen_image21_ms}"
RESULTS_DIR="${RESULTS_DIR:-$HOME/output/qwen_image_2_1_results}"
export REPO_DIR MODEL_DIR MODEL_PATH RESULTS_DIR

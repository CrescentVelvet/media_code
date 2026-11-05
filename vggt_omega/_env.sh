# vggt_omega/_env.sh — 共享环境设置（被本目录所有 .sh 脚本 source）

SCRIPT_DIR_ENV="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MEDIA_CODE_ROOT="$(cd "$SCRIPT_DIR_ENV/.." && pwd)"

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
# 注意：推理/COLMAP/3DGS 链路整体转发给 vggt_human（其脚本自带 env 切换），
# 本 env 仅用于独立调用 vggt_omega 官方 API 的场景
CONDA_ENV="${CONDA_ENV:-vggt_omega}"
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

# 4. GPU 选卡
if [ -n "${GPU:-}" ]; then
    export CUDA_VISIBLE_DEVICES="$GPU"
fi

# 5. 路径
export VGGT_OMEGA_REPO="${VGGT_OMEGA_REPO:-$HOME/repos/vggt-omega}"
export MODEL_DIR="${VGGT_OMEGA_MODEL_DIR:-$HOME/model/VGGT-Omega}"
# 权重复用池（2026-11-05 峰哥约定）：下载模型前先在 D:\wheel 搜，
# 新下载的模型也落盘到 D:\wheel 下对应子目录，消费侧一律软链，不复制
export WHEEL_DIR="${WHEEL_DIR:-/mnt/d/wheel}"
export WHEEL_VGGT_OMEGA_DIR="${WHEEL_VGGT_OMEGA_DIR:-$WHEEL_DIR/vggt_human_ms/VGGT-Omega}"
export RESULTS_DIR="${VGGT_OMEGA_RESULTS_DIR:-$HOME/output/vggt_omega_results}"

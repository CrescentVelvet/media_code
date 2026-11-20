# _env.sh — shared setup: proxy + CA bundle + conda env activation + GPU + paths.
# Sourced by all pi3x/*.sh. Expects SCRIPT_DIR (this dir) to be set by the caller.
#
# Pi3X (yyfz/Pi3, π³ 工程升级版, 2025-12-28 released) — 无序多图 → 位姿+稠密点云 前馈回归。
# 相比原版 Pi3：卷积头（去网格伪影）、连续置信度、可注入 pose/intrinsic/depth 条件、近似 metric 尺度。
# 权重: yyfz233/Pi3X model.safetensors (~4GB, CC BY-NC 4.0 禁商用)。
REPO_DIR="$(dirname "$SCRIPT_DIR")"

# Optional proxy (gitignored proxy.env at repo root).
if [ -f "$REPO_DIR/proxy.env" ]; then
    set -a; # shellcheck disable=SC1091
    source "$REPO_DIR/proxy.env"; set +a
fi

[ -n "${http_proxy:-}" ]  && export HTTP_PROXY="$http_proxy"
[ -n "${https_proxy:-}" ] && export HTTPS_PROXY="$https_proxy"

# --- Corporate proxy TLS interception workaround (pip/hf/git) ---
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

# Activate the pi3x conda env (00a creates it: python=3.10 + torch 2.5.1+cu124, 官方 requirements 钉版).
CONDA_ENV="${CONDA_ENV:-pi3x}"
export CONDA_ENV
if ! command -v conda >/dev/null 2>&1; then
    # Fallback: try common conda locations (WSL without `conda init`).
    for _cb in "$HOME/miniconda3" "$HOME/anaconda3" "/opt/conda"; do
        if [ -f "$_cb/etc/profile.d/conda.sh" ]; then
            # shellcheck disable=SC1091
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
# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV" 2>/dev/null || true  # env may not exist yet (00a creates it)

# Pin GPU (0-indexed) via GPU=N.
if [ -n "${GPU:-}" ]; then
    export CUDA_VISIBLE_DEVICES="$GPU"
fi

# --- Paths ---
# Official code (WSL: Linux fs 编译/加载快; 00a clones it).
PI3_DIR="${PI3_DIR:-$HOME/repos/Pi3}"
PI3_REPO="${PI3_REPO:-https://github.com/yyfz/Pi3.git}"

# Weight root + output. proxy.env 可写 PI3X_MODEL_DIR / PI3X_RESULTS_DIR 覆盖（项目专属名）。
MODEL_DIR="${MODEL_DIR:-${PI3X_MODEL_DIR:-$HOME/model/pi3x}}"
RESULTS_DIR="${RESULTS_DIR:-${PI3X_RESULTS_DIR:-$HOME/output/pi3x_results}}"

# 权重复用池（AGENTS.md：下载前先搜 D:\wheel，新下载落 wheel + 消费侧软链）。
PI3X_WHEEL_DIR="${PI3X_WHEEL_DIR:-/mnt/d/wheel/pi3x_ms}"
PI3X_CKPT_NAME="model.safetensors"
if [ -z "${PI3X_CKPT:-}" ]; then
    if [ -f "$MODEL_DIR/$PI3X_CKPT_NAME" ]; then
        PI3X_CKPT="$MODEL_DIR/$PI3X_CKPT_NAME"
    elif [ -f "$PI3X_WHEEL_DIR/$PI3X_CKPT_NAME" ]; then
        PI3X_CKPT="$PI3X_WHEEL_DIR/$PI3X_CKPT_NAME"
    else
        PI3X_CKPT="$MODEL_DIR/$PI3X_CKPT_NAME"   # 01 负责下载落盘
    fi
fi
PI3X_CKPT_URL="${PI3X_CKPT_URL:-https://huggingface.co/yyfz233/Pi3X/resolve/main/model.safetensors}"

export REPO_DIR PI3_DIR PI3_REPO MODEL_DIR RESULTS_DIR PI3X_WHEEL_DIR PI3X_CKPT PI3X_CKPT_URL

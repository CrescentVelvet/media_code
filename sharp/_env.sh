# _env.sh — shared setup: proxy + CA bundle + conda env activation + GPU + paths.
# Sourced by 00a/01/02/08. Expects SCRIPT_DIR (this dir) to be set by the caller.
#
# SHARP (Apple, ICLR 2026) — 单图 → 3DGS 前馈回归，官方仓只含推理代码。
# 权重: sharp_2572gikvuh.pt (2,809,738,232 bytes ≈ 2.62 GiB)，Apple CDN 直链，非 gated。
REPO_DIR="$(dirname "$SCRIPT_DIR")"

# Optional proxy (gitignored proxy.env at repo root).
if [ -f "$REPO_DIR/proxy.env" ]; then
    set -a; # shellcheck disable=SC1090
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

# Activate the sharp conda env (00a creates it: python=3.13 + torch 2.8 cu128).
CONDA_ENV="${CONDA_ENV:-sharp}"
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

# --- CUDA toolkit for gsplat JIT compile ---
# gsplat 1.5.3 on PyPI is a PURE-PYTHON wheel (py3-none-any): the CUDA rasterizer
# is JIT-compiled on first use, so nvcc + gcc must be reachable at RUNTIME.
# 00a installs cuda-toolkit + gxx_linux-64 INTO the env (no sudo), so point
# CUDA_HOME at the conda prefix when nvcc lives there.
if [ -n "${CONDA_PREFIX:-}" ] && [ -x "$CONDA_PREFIX/bin/nvcc" ]; then
    export CUDA_HOME="${CUDA_HOME:-$CONDA_PREFIX}"
elif [ -d /usr/local/cuda ]; then
    export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
fi
[ -n "${CUDA_HOME:-}" ] && export PATH="$CUDA_HOME/bin:$PATH"
for _cuda_lib in "$CONDA_PREFIX/lib" "$CUDA_HOME/lib64"; do
    [ -d "$_cuda_lib" ] && export LD_LIBRARY_PATH="${_cuda_lib}:${LD_LIBRARY_PATH:-}"
done
unset _cuda_lib
# 3090 = sm_86. Pinning the arch skips compiling for every arch (minutes -> seconds).
# 换卡请覆盖: TORCH_CUDA_ARCH_LIST="8.9" bash sharp/02_run_inference.sh
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.6}"

# --- Paths ---
# Official code (Linux fs on WSL; 00a clones it).
SHARP_DIR="${SHARP_DIR:-$REPO_DIR/../ml-sharp}"
SHARP_REPO="${SHARP_REPO:-https://github.com/apple-aiml-research/ml-sharp.git}"

# Weight root + output.
# WSL: proxy.env 写 SHARP_MODEL_DIR / SHARP_RESULTS_DIR（项目专属名，避免污染其它项目）。
MODEL_DIR="${MODEL_DIR:-${SHARP_MODEL_DIR:-$REPO_DIR/../../model/sharp}}"
RESULTS_DIR="${RESULTS_DIR:-${SHARP_RESULTS_DIR:-$REPO_DIR/../sharp_results}}"

# Checkpoint (single file — the whole model, encoder included).
SHARP_CKPT="${SHARP_CKPT:-$MODEL_DIR/sharp_2572gikvuh.pt}"
SHARP_CKPT_URL="${SHARP_CKPT_URL:-https://ml-site.cdn-apple.com/models/sharp/sharp_2572gikvuh.pt}"
# 迅雷备份镜像 (HF, 非 gated, 无需 token)
SHARP_CKPT_URL_HF="${SHARP_CKPT_URL_HF:-https://huggingface.co/apple/Sharp/resolve/main/sharp_2572gikvuh.pt}"
# 官方文件字节数，下载后校验用（HTTP HEAD 实测值）
SHARP_CKPT_SIZE="${SHARP_CKPT_SIZE:-2809738232}"

export REPO_DIR SHARP_DIR SHARP_REPO MODEL_DIR RESULTS_DIR \
       SHARP_CKPT SHARP_CKPT_URL SHARP_CKPT_URL_HF SHARP_CKPT_SIZE CUDA_HOME

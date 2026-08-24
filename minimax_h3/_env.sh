# _env.sh — shared setup: proxy + CA bundle + conda env activation.
# Sourced by 00/01/02/03. Expects SCRIPT_DIR (this dir) to be set by the caller.
REPO_DIR="$(dirname "$SCRIPT_DIR")"

# Optional proxy (gitignored proxy.env at repo root; see proxy.env.example).
if [ -f "$REPO_DIR/proxy.env" ]; then
    set -a; # shellcheck disable=SC1090
    source "$REPO_DIR/proxy.env"; set +a
fi

# Some clients (Rust reqwest, git, curl) only read uppercase proxy vars.
[ -n "${http_proxy:-}" ]  && export HTTP_PROXY="$http_proxy"
[ -n "${https_proxy:-}" ] && export HTTPS_PROXY="$https_proxy"

# --- Corporate proxy TLS interception workaround (pip/hf/git) ---
# Prefer a user-built bundle (run setup_ca_bundle.sh once), then the system bundle.
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

# Disable HuggingFace Xet/CAS (Rust reqwest) download path — it doesn't honor
# REQUESTS_CA_BUNDLE / lowercase proxy and fails behind a TLS-intercepting
# corporate proxy. Falls back to the legacy Python downloader (honors proxy +
# CA bundle). If Xet still engages, also run: pip uninstall -y hf_xet
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"

# Activate the existing conda env (torch already installed; reuse to avoid
# re-downloading torch). 默认专用 env `minimax_h3`（SGLang pin 与其他算法冲突，
# 务必专用 env）。想用别的 env 就显式 `CONDA_ENV=xxx` 覆盖。
# SGLang pulls its own torch + flashinfer + cuda kernels; its version pins may
# CONFLICT with other algos in this repo (e.g. hunyuanvideo wants diffusers 0.35,
# hypir pins diffusers 0.32 / transformers 4.49). Use a DEDICATED env:
#   conda create -n minimax_h3 python=3.11 -y
CONDA_ENV="${CONDA_ENV:-minimax_h3}"
export CONDA_ENV
if ! command -v conda >/dev/null 2>&1; then
    # Fallback: try common conda locations (WSL/local dev without `conda init`).
    # On servers conda is already on PATH, so this block never triggers.
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
conda activate "$CONDA_ENV" 2>/dev/null || true  # env may not exist yet (00/00a creates it)

# Pin to a specific physical GPU (0-indexed) via GPU=N. It remaps
# CUDA_VISIBLE_DEVICES so cuda:0 inside the process == physical GPU N.
# NOTE: for SGLang multi-GPU serve (--num-gpus 4) you usually want ALL visible
# GPUs, so leave GPU unset and control the count with --num-gpus / NUM_GPUS.
# Only set GPU=N to restrict the server to a subset of physical cards.
if [ -n "${GPU:-}" ]; then
    export CUDA_VISIBLE_DEVICES="$GPU"
fi

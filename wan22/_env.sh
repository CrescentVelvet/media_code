# _env.sh — shared setup: proxy + CA bundle + conda env + GPU pinning.
# Sourced by 00/01/02/03/04. Expects SCRIPT_DIR (this dir) to be set by the caller.
REPO_DIR="$(dirname "$SCRIPT_DIR")"

# Optional proxy (gitignored proxy.env at repo root; see proxy.env.example).
if [ -f "$REPO_DIR/proxy.env" ]; then
    set -a; # shellcheck disable=SC1090
    source "$REPO_DIR/proxy.env"; set +a
fi

# Some clients (Rust reqwest, git, curl) only read uppercase proxy vars.
[ -n "${http_proxy:-}" ]  && export HTTP_PROXY="$http_proxy"
[ -n "${https_proxy:-}" ] && export HTTPS_PROXY="$https_proxy"

# --- Corporate proxy TLS interception workaround (pip/modelscope/git) ---
# Prefer a user-built bundle (run hypir/setup_ca_bundle.sh once), then the system bundle.
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

# Activate the conda env (torch already installed; reuse to avoid re-downloading).
# 不强制切 env——默认沿用你当前已激活的 env (CONDA_DEFAULT_ENV)，所以
# `conda activate wan22` 一次后所有 wan22 脚本都用它。想强制别的 env 就显式
# `CONDA_ENV=xxx`。DiffSynth-Studio 需要 python>=3.10 + torch>=2.1。
CONDA_ENV="${CONDA_ENV:-${CONDA_DEFAULT_ENV:-base}}"
if ! command -v conda >/dev/null 2>&1; then
    echo "ERROR: conda not found on PATH (need an activated env; or set CONDA_ENV)." >&2
    exit 1
fi
# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"

# Pin to a specific physical GPU (0-indexed) via GPU=N. It remaps
# CUDA_VISIBLE_DEVICES so cuda:0 inside the process == physical GPU N.
if [ -n "${GPU:-}" ]; then
    export CUDA_VISIBLE_DEVICES="$GPU"
fi

# --- DiffSynth-Studio: load models from the local store (no download) ---
# MODEL_DIR is the shared model root (same convention as hypir: ../../model).
# Set DIFFSYNTH_MODEL_BASE_PATH so ModelConfig looks under $MODEL_DIR/<org>/<model>/.
# Set DIFFSYNTH_SKIP_DOWNLOAD=True so it never tries to download (files are local).
MODEL_DIR="${MODEL_DIR:-$REPO_DIR/../../model}"
export DIFFSYNTH_MODEL_BASE_PATH="$MODEL_DIR"
export DIFFSYNTH_SKIP_DOWNLOAD="${DIFFSYNTH_SKIP_DOWNLOAD:-True}"
# Default download source = ModelScope (override with DIFFSYNTH_DOWNLOAD_SOURCE=HuggingFace).
export DIFFSYNTH_DOWNLOAD_SOURCE="${DIFFSYNTH_DOWNLOAD_SOURCE:-modelscope}"

# Official code path (cloned by 00_setup_env.sh / run_all.sh).
DIFFSYNTH_DIR="${DIFFSYNTH_DIR:-$REPO_DIR/../DiffSynth-Studio}"
export DIFFSYNTH_DIR MODEL_DIR

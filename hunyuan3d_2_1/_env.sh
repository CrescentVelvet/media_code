# _env.sh — shared setup: proxy + CA bundle + conda env activation + runtime env.
# Sourced by 00/01/02. Expects SCRIPT_DIR (this dir) to be set by the caller.
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
# Prefer a user-built bundle (see trellis-2/setup_ca_bundle.sh), then system.
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
# corporate proxy. If Xet still engages, also run: pip uninstall -y hf_xet
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"

# Fragmented CUDA allocations -> fewer OOMs (paint stage peaks ~21GB).
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

# CUDA Toolkit location — needed to compile hy3dpaint/custom_rasterizer and
# hy3dpaint/DifferentiableRenderer. Tested by upstream with cu124.
: "${CUDA_HOME:=/usr/local/cuda}"
export CUDA_HOME

# Activate the existing conda env. Pass "no-activate" as $1 to skip activation
# (used by 00 when CREATE_ENV=1 creates the env first).
CONDA_ENV="${CONDA_ENV:-hunyuan3d}"
if ! command -v conda >/dev/null 2>&1; then
    for _cb in "$HOME/miniconda3" "$HOME/anaconda3" "/opt/conda"; do
        if [ -f "$_cb/etc/profile.d/conda.sh" ]; then
            # shellcheck disable=SC1090
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
if [ "${1:-}" != "no-activate" ]; then
    conda activate "$CONDA_ENV" 2>/dev/null || true
fi

# Paths (all overridable; WSL users typically override in proxy.env, e.g.
# HY3D_DIR=~/repos/Hunyuan3D-2.1, MODEL_DIR=~/model).
MODEL_DIR="${MODEL_DIR:-$REPO_DIR/../../model}"
RESULTS_DIR="${RESULTS_DIR:-$REPO_DIR/../hunyuan3d_2_1_results}"
HY3D_DIR="${HY3D_DIR:-$REPO_DIR/../Hunyuan3D-2.1}"
export REPO_DIR MODEL_DIR RESULTS_DIR HY3D_DIR

# Pin to a specific physical GPU (0-indexed) via GPU=N. It remaps
# CUDA_VISIBLE_DEVICES so cuda:0 inside the process == physical GPU N.
if [ -n "${GPU:-}" ]; then
    export CUDA_VISIBLE_DEVICES="$GPU"
fi

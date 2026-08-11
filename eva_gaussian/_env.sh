# _env.sh — shared setup: proxy + CA bundle + conda env activation + GPU + paths.
# Sourced by 00/01/02/03/04/run_all. Expects SCRIPT_DIR (this dir) to be set by the caller.
#
# This pipeline uses a DEDICATED conda env (default `eva_gaussian`, CPython 3.10):
#   - EVA-Gaussian pins torch 2.5.0+cu118 (official). The feature-splatting CUDA
#     rasterizer (a modified diff-gaussian-rasterization) builds against it.
#   - We try local cu118 wheels at $MODEL_DIR first (company proxy blocks
#     download.pytorch.org); falls back to pip if no local wheels.
#   - Set CUDA_TAG=cu124 + TORCH_VERSION=2.6.0 to reuse the wan22_rotate/pi3_3dgs
#     local cp310 cu124 wheels instead (the rasterizer builds fine against 2.6 too).
REPO_DIR="$(dirname "$SCRIPT_DIR")"

# 1. proxy (gitignored proxy.env at repo root; see proxy.env.example).
if [ -f "$REPO_DIR/proxy.env" ]; then
    set -a; # shellcheck disable=SC1090
    source "$REPO_DIR/proxy.env"; set +a
fi

[ -n "${http_proxy:-}" ]  && export HTTP_PROXY="$http_proxy"
[ -n "${https_proxy:-}" ] && export HTTPS_PROXY="$https_proxy"

# 2. CA bundle (corporate proxy TLS interception).
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
    : "${CURL_CA_BUNDLE:=$CA_FILE}"
    export REQUESTS_CA_BUNDLE SSL_CERT_FILE GIT_SSL_CAINFO PIP_CERT CURL_CA_BUNDLE
fi

export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"

# 3. conda env activation (CPython 3.10; may not exist yet — 00 creates it).
CONDA_ENV="${CONDA_ENV:-eva_gaussian}"
export CONDA_ENV
if ! command -v conda >/dev/null 2>&1; then
    echo "ERROR: conda not found on PATH (need env '$CONDA_ENV')." >&2
    exit 1
fi
# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV" 2>/dev/null || true

# 4. GPU selection (0-indexed) via GPU=N.
if [ -n "${GPU:-}" ]; then
    export CUDA_VISIBLE_DEVICES="$GPU"
fi

# CUDA library paths (libcupti etc.).
for _cuda_lib in \
    "/usr/local/cuda/extras/CUPTI/lib64" \
    "/usr/local/cuda/lib64" \
    "$CONDA_PREFIX/lib"; do
    [ -d "$_cuda_lib" ] && export LD_LIBRARY_PATH="${_cuda_lib}:${LD_LIBRARY_PATH:-}"
done

# 5. paths (all ${VAR:-default} for CLI override).
# EVA-Gaussian official code (cloned to media_code's sibling).
EVA_DIR="${EVA_DIR:-$REPO_DIR/../EVA-Gaussian}"

# Shared model root (where local torch wheels live, shared by all algorithms).
MODEL_DIR="${MODEL_DIR:-$REPO_DIR/../../model}"

# Output root (experiments / checkpoints / tensorboard logs).
RESULTS_DIR="${RESULTS_DIR:-$REPO_DIR/../eva_gaussian_results}"

# Dataset root (THuman2.0 / THumansit in GPS-Gaussian format).
DATA_ROOT="${DATA_ROOT:-}"

# CUDA toolkit root for nvcc (must match torch's cu major: 11.8 for cu118, 12.4 for cu124).
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"

export REPO_DIR EVA_DIR MODEL_DIR RESULTS_DIR DATA_ROOT

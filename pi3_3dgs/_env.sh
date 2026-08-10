# _env.sh — shared setup: proxy + CA bundle + conda env activation + GPU + paths.
# Sourced by 00/01/02/03/run_all. Expects SCRIPT_DIR (this dir) to be set by the caller.
#
# This pipeline uses a DEDICATED conda env (default `pi3_3dgs`, CPython 3.10):
#   - Pi3 wants torch>=2.0; 2DGS's diff-surfel-rasterization builds fine against torch 2.6.
#   - We reuse the local cp310 torch 2.6.0+cu124 wheels (same as wan22_rotate) to avoid
#     re-downloading ~3GB of nvidia deps.
#   - The env is SEPARATE from wan22_rotate because 2DGS needs CUDA toolkit + compiled
#     rasterizer (nvcc) which wan22_rotate's env doesn't have.
REPO_DIR="$(dirname "$SCRIPT_DIR")"

# Optional proxy (gitignored proxy.env at repo root; see proxy.env.example).
if [ -f "$REPO_DIR/proxy.env" ]; then
    set -a; # shellcheck disable=SC1090
    source "$REPO_DIR/proxy.env"; set +a
fi

[ -n "${http_proxy:-}" ]  && export HTTP_PROXY="$http_proxy"
[ -n "${https_proxy:-}" ] && export HTTPS_PROXY="$https_proxy"

# --- Corporate proxy TLS interception workaround (pip/hf/git/curl) ---
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

# Activate the conda env (CPython 3.10; has Pi3 + 2DGS deps).
CONDA_ENV="${CONDA_ENV:-pi3_3dgs}"
export CONDA_ENV
if ! command -v conda >/dev/null 2>&1; then
    echo "ERROR: conda not found on PATH (need env '$CONDA_ENV')." >&2
    exit 1
fi
# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV" 2>/dev/null || true  # may not exist yet (00 creates it)

# Pin GPU (0-indexed) via GPU=N.
if [ -n "${GPU:-}" ]; then
    export CUDA_VISIBLE_DEVICES="$GPU"
fi

# CUDA library paths (libcupti.so.12 etc.).
for _cuda_lib in \
    "/usr/local/cuda/extras/CUPTI/lib64" \
    "/usr/local/cuda/lib64" \
    "$CONDA_PREFIX/lib"; do
    [ -d "$_cuda_lib" ] && export LD_LIBRARY_PATH="${_cuda_lib}:${LD_LIBRARY_PATH:-}"
done

# --- Paths ---
# Pi3 (ICLR 2026) — feed-forward pose estimation + dense point cloud.
PI3_DIR="${PI3_DIR:-$REPO_DIR/../Pi3}"
PI3_MODEL_DIR="${PI3_MODEL_DIR:-$REPO_DIR/../../model/Pi3}"
export PI3_CKPT="${PI3_CKPT:-$PI3_MODEL_DIR/model.safetensors}"

# 2D Gaussian Splatting (SIGGRAPH 2024) — surface-aligned radiance field.
GS2D_DIR="${GS2D_DIR:-$REPO_DIR/../2d-gaussian-splatting}"
# CUDA toolkit root for nvcc (must match torch's cu major, e.g. 12.4 for cu124).
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"

# Shared model root (where local torch wheels live, same as wan22_rotate).
WAN_MODEL_DIR="${WAN_MODEL_DIR:-$REPO_DIR/../../model}"

# Output dir (outside the repo, like wan22_rotate_results / vggt-omega/output).
RESULTS_DIR="${RESULTS_DIR:-$REPO_DIR/../pi3_3dgs_results}"

export REPO_DIR PI3_DIR PI3_MODEL_DIR GS2D_DIR WAN_MODEL_DIR RESULTS_DIR

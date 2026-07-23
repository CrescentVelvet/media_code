# _env.sh — shared setup: proxy + CA bundle + conda env activation + TRELLIS.2 runtime env.
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

# --- TRELLIS.2 runtime environment (must be set before importing cv2/torch) ---
# cv2 needs this to read the .exr HDRI env maps (assets/hdri/*.exr).
export OPENCV_IO_ENABLE_OPENEXR="${OPENCV_IO_ENABLE_OPENEXR:-1}"
# Fragmented CUDA allocations -> fewer OOMs with the 4B model.
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
# Attention backend: 'flash-attn' (default) or 'xformers' (e.g. NVIDIA V100,
# which can't run flash-attn). Only exported if the user sets it; leaving
# unset lets TRELLIS.2 use its default (flash-attn).
[ -n "${ATTN_BACKEND:-}" ] && export ATTN_BACKEND

# CUDA Toolkit location — needed to compile the CUDA extensions installed by
# the official setup.sh (flash-attn / nvdiffrast / nvdiffrec / cumesh /
# flexgemm). Recommended 12.4. Override if you have multiple CUDA versions.
: "${CUDA_HOME:=/usr/local/cuda}"
export CUDA_HOME

# Activate the existing conda env (torch already installed). Pass "no-activate"
# as $1 to skip activation (used by 00 when CREATE_ENV=1 lets setup.sh --new-env
# create the env). conda shell functions are always sourced either way.
CONDA_ENV="${CONDA_ENV:-trellis2}"
if ! command -v conda >/dev/null 2>&1; then
    echo "ERROR: conda not found on PATH (need env '$CONDA_ENV')." >&2
    exit 1
fi
# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"
if [ "${1:-}" != "no-activate" ]; then
    conda activate "$CONDA_ENV"
fi

# Pin to a specific physical GPU (0-indexed) via GPU=N. It remaps
# CUDA_VISIBLE_DEVICES so cuda:0 inside the process == physical GPU N.
# (example.py / app.py hardcode device='cuda' == cuda:0.) Default: first visible.
if [ -n "${GPU:-}" ]; then
    export CUDA_VISIBLE_DEVICES="$GPU"
fi

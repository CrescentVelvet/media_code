# _env.sh — shared setup: proxy + CA bundle + conda env activation.
# Sourced by 00/01/02/run_all. Expects SCRIPT_DIR (this dir) to be set by the caller.
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
# Prefer a user-built bundle (run setup_ca_bundle.sh once in the hypir/ dir,
# or any equivalent), then the system bundle.
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

# Activate the existing conda env (torch already installed; reuse to avoid
# re-downloading torch). Override the env name with CONDA_ENV.
#
# RetouchFormer pins torch==1.13.1 (cu117) + python==3.8 in its official
# requirements.txt. These pins CONFLICT with other algos in this repo
# (hunyuanvideo wants diffusers==0.35 / transformers==4.57, etc.). Use a
# DEDICATED env:
#   conda create -n retouchformer python=3.8 -y
#   pip install torch==1.13.1 torchvision==0.14.1  (cu117; A100/4090 OK)
#   # H100 (sm90) needs cu118+: use a newer torch, e.g.
#   # pip install torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cu121
#   CONDA_ENV=retouchformer INSTALL_DEPS=1 bash retouchformer/00_setup_env.sh
CONDA_ENV="${CONDA_ENV:-retouchformer}"
if ! command -v conda >/dev/null 2>&1; then
    echo "ERROR: conda not found on PATH (need env '$CONDA_ENV')." >&2
    exit 1
fi
# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"

# Pin to a specific physical GPU (0-indexed) via GPU=N. It remaps
# CUDA_VISIBLE_DEVICES so cuda:0 inside the process == physical GPU N.
# (The official img_retouching.py hardcodes CUDA_VISIBLE_DEVICES="0"; we let
#  the caller choose via GPU=, defaulting to the first visible device.)
if [ -n "${GPU:-}" ]; then
    export CUDA_VISIBLE_DEVICES="$GPU"
fi

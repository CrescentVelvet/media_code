# _env.sh — shared setup: proxy + CA bundle + conda env + GPU pinning.
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

# Disable HuggingFace Xet/CAS (Rust reqwest) download path — it doesn't honor
# REQUESTS_CA_BUNDLE / lowercase proxy and fails behind a TLS-intercepting proxy.
# Falls back to the legacy Python downloader (honors proxy + CA bundle).
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"

# Activate the existing conda env (torch already installed; reuse to avoid
# re-downloading). 不强制切 env——默认沿用你当前已激活的 env (CONDA_DEFAULT_ENV)，
# 所以 `conda activate wan_animate_2` 一次后所有脚本都用它。Wan-Animate-2 需
# python>=3.10 + torch>=2.7 (cu126)；建议专用 env：
#   conda create -n wan_animate_2 python=3.11 -y
CONDA_ENV="${CONDA_ENV:-${CONDA_DEFAULT_ENV:-base}}"
if ! command -v conda >/dev/null 2>&1; then
    echo "ERROR: conda not found on PATH (need an activated env; or set CONDA_ENV)." >&2
    exit 1
fi
# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"

# Pin to specific physical GPUs (0-indexed) via GPU=N[,M,...]. It remaps
# CUDA_VISIBLE_DEVICES so the process sees only those cards. Wan-Animate-2 is
# multi-GPU (sp_size/sharding_size in the YAML); leave GPU unset to use ALL
# visible cards, or set e.g. GPU=0,1 to restrict to a subset (sp_size auto =2).
if [ -n "${GPU:-}" ]; then
    export CUDA_VISIBLE_DEVICES="$GPU"
fi

# --- Paths ---
# OFFICIAL_DIR: the cloned Wan-Animate-2 repo (sibling of media_code).
# MODEL_DIR: shared model root (one level above <code-dir>); weights live under
#   $MODEL_DIR/Wan-AI/Wan2.2-Animate-2-14B/ (the HF/ModelScope "ckpts" content).
# CKPTS_DIR: where the model download lands; symlinked to $OFFICIAL_DIR/ckpts
#   so the official YAML's ../ckpts/... relative paths resolve unchanged.
# RESULTS_DIR: output videos.
OFFICIAL_DIR="${OFFICIAL_DIR:-$REPO_DIR/../Wan-Animate-2}"
MODEL_DIR="${MODEL_DIR:-$REPO_DIR/../../model}"
CKPTS_DIR="${CKPTS_DIR:-$MODEL_DIR/Wan-AI/Wan2.2-Animate-2-14B}"
RESULTS_DIR="${RESULTS_DIR:-$REPO_DIR/../wan_animate_2_results}"
export REPO_DIR OFFICIAL_DIR MODEL_DIR CKPTS_DIR RESULTS_DIR

# _env.sh — shared setup: proxy + CA bundle + conda env activation + headless GL.
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

# Activate the dedicated flux_human conda env. 00_setup_env.sh clones it from
# the shared 'doll' env (torch preinstalled) and installs the Flux1 + ControlNet
# + pyrender stack. We use a DEDICATED env because the diffusers/ControlNet
# versions would conflict with other algos' pins in 'doll'.
# Override with CONDA_ENV=xxx (or just `conda activate flux_human` once before running).
CONDA_ENV="${CONDA_ENV:-${CONDA_DEFAULT_ENV:-flux_human}}"
if ! command -v conda >/dev/null 2>&1; then
    echo "ERROR: conda not found on PATH (need env '$CONDA_ENV')." >&2
    exit 1
fi
# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV" 2>/dev/null || true  # env may not exist yet (00 creates it)

# Pin to a specific physical GPU (0-indexed) via GPU=N. It remaps
# CUDA_VISIBLE_DEVICES so cuda:0 inside the process == physical GPU N.
if [ -n "${GPU:-}" ]; then
    export CUDA_VISIBLE_DEVICES="$GPU"
fi

# --- Headless OpenGL for pyrender (03_render_depth renders depth maps) ---
# pyrender needs a GL backend. On a headless GPU server EGL is the most
# reliable (no X server). Override with PYOPENGL_PLATFORM=osmesa if EGL
# is unavailable (then also: apt install -y libosmesa6-dev).
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}"

# Paths (MODEL_DIR / RESULTS_DIR) are set per-script (like flux2/sam_3d_body),
# not here, so each step can default them to the right subdirectory.
export REPO_DIR

# _env.sh — shared setup: proxy + CA bundle + conda env activation + headless GL.
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
# re-downloading torch). 不强制切 env——默认沿用你当前已激活的 env
# (CONDA_DEFAULT_ENV)，所以 `conda activate sam_3d_body` 一次后所有脚本都用它，
# 不必再每条命令传 CONDA_ENV=sam_3d_body。想强制别的 env 就显式 `CONDA_ENV=xxx`。
CONDA_ENV="${CONDA_ENV:-${CONDA_DEFAULT_ENV:-sam_3d_body}}"
if ! command -v conda >/dev/null 2>&1; then
    echo "ERROR: conda not found on PATH (need an activated env; or set CONDA_ENV)." >&2
    exit 1
fi
# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"

# Pin to a specific physical GPU (0-indexed) via GPU=N. It remaps
# CUDA_VISIBLE_DEVICES so cuda:0 inside the process == physical GPU N.
# (demo.py uses cuda by default.) Default: first visible.
if [ -n "${GPU:-}" ]; then
    export CUDA_VISIBLE_DEVICES="$GPU"
fi

# --- Headless OpenGL for pyrender ---
# The SAM 3D Body renderer (sam_3d_body/visualization/renderer.py) uses pyrender,
# which needs a GL backend. On a headless GPU server EGL is the most reliable
# (no X server, no osmesa rebuild). Override with PYOPENGL_PLATFORM=osmesa if EGL
# is unavailable (then also: apt install libosmesa6-dev and rebuild PyOpenGL).
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}"

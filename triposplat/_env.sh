# _env.sh — shared setup: proxy + CA bundle + conda env activation.
# Sourced by 00/01/02. Expects SCRIPT_DIR (this dir) to be set by the caller.
REPO_DIR="$(dirname "$SCRIPT_DIR")"

# Optional proxy (gitignored proxy.env at repo root; see proxy.env.example).
if [ -f "$REPO_DIR/proxy.env" ]; then
    set -a; # shellcheck disable=SC1090
    source "$REPO_DIR/proxy.env"; set +a
fi

# --- Corporate proxy TLS interception workaround (pip/hf/git) ---
SYS_CA=/etc/ssl/certs/ca-certificates.crt
if [ -f "$SYS_CA" ]; then
    : "${REQUESTS_CA_BUNDLE:=$SYS_CA}"
    : "${SSL_CERT_FILE:=$SYS_CA}"
    : "${GIT_SSL_CAINFO:=$SYS_CA}"
    : "${PIP_CERT:=$SYS_CA}"
    export REQUESTS_CA_BUNDLE SSL_CERT_FILE GIT_SSL_CAINFO PIP_CERT
fi

# Activate the existing conda env (torch already installed; reuse to avoid
# re-downloading torch). Override the env name with CONDA_ENV.
CONDA_ENV="${CONDA_ENV:-doll}"
if ! command -v conda >/dev/null 2>&1; then
    echo "ERROR: conda not found on PATH (need env '$CONDA_ENV')." >&2
    exit 1
fi
# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"

# _env.sh — shared setup: proxy + CA bundle + GPU pinning + paths + conda helper.
# Sourced by 00/01/02/run_all. Expects SCRIPT_DIR (this dir) to be set by the caller.
# Does NOT activate any conda env — each step calls conda_activate <env> because
# step 01 needs the sam_3d_body env and step 02 needs the wan22 env (they have
# conflicting pins: detectron2 / networkx==3.2.1 vs diffsynth).
REPO_DIR="$(dirname "$SCRIPT_DIR")"

# Optional proxy (gitignored proxy.env at repo root; see proxy.env.example).
if [ -f "$REPO_DIR/proxy.env" ]; then
    set -a; # shellcheck disable=SC1090
    source "$REPO_DIR/proxy.env"; set +a
fi

[ -n "${http_proxy:-}" ]  && export HTTP_PROXY="$http_proxy"
[ -n "${https_proxy:-}" ] && export HTTPS_PROXY="$https_proxy"

# --- Corporate proxy TLS interception workaround (pip/hf/git) ---
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

export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"

# Pin GPU (0-indexed) via GPU=N.
if [ -n "${GPU:-}" ]; then
    export CUDA_VISIBLE_DEVICES="$GPU"
fi

# Headless OpenGL for pyrender (used by sam_3d_body renderer in step 01).
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}"

# --- Paths ---
# sam_3d_body (person detection + 3D pose + segmentation, step 01)
SAM3D_DIR="${SAM3D_DIR:-$REPO_DIR/../sam-3d-body}"
SAM3D_MODEL_DIR="${SAM3D_MODEL_DIR:-$REPO_DIR/../../model/sam-3d-body}"

# wan22 / DiffSynth-Studio (video generation, step 02)
DIFFSYNTH_DIR="${DIFFSYNTH_DIR:-$REPO_DIR/../DiffSynth-Studio}"
WAN_MODEL_DIR="${WAN_MODEL_DIR:-$REPO_DIR/../../model}"
export DIFFSYNTH_MODEL_BASE_PATH="$WAN_MODEL_DIR"
export DIFFSYNTH_SKIP_DOWNLOAD="${DIFFSYNTH_SKIP_DOWNLOAD:-True}"
export DIFFSYNTH_DOWNLOAD_SOURCE="${DIFFSYNTH_DOWNLOAD_SOURCE:-modelscope}"

# Output dir (outside the repo, like wan22_results / wan22_experiments).
RESULTS_DIR="${RESULTS_DIR:-$REPO_DIR/../wan22_rotate_results}"

export REPO_DIR SAM3D_DIR SAM3D_MODEL_DIR DIFFSYNTH_DIR WAN_MODEL_DIR RESULTS_DIR

# --- conda helper: each step activates its own env ---
# Step 01 (pick+segment) runs in sam_3d_body env; step 02 (video gen) in wan22 env.
conda_activate() {
    local env_name="$1"
    if ! command -v conda >/dev/null 2>&1; then
        echo "ERROR: conda not found on PATH (need env '$env_name')." >&2
        exit 1
    fi
    # shellcheck disable=SC1091
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate "$env_name"
}

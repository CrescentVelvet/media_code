# _env.sh — shared setup: proxy + CA bundle + conda env activation + GPU + paths.
# Sourced by 00/01/02/run_all. Expects SCRIPT_DIR (this dir) to be set by the caller.
# Uses a SINGLE conda env (default wan22_rotate, CPython 3.10 — NOT cloned from doll;
# the local torch/triton wheels are cp310 so doll's 3.11 won't fit) for both steps —
# sam_3d_body deps + diffsynth coexist (detectron2 installed --no-deps, networkx==3.2.1
# is stable enough for diffsynth). Override with CONDA_ENV=xxx.
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

# Activate the conda env (CPython 3.10; has both sam_3d_body + diffsynth deps).
CONDA_ENV="${CONDA_ENV:-wan22_rotate}"
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

# Headless OpenGL for pyrender (used by sam_3d_body renderer in step 01).
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}"

# CUDA library paths (libcupti.so.12 etc.)
for _cuda_lib in \
    "/usr/local/cuda/extras/CUPTI/lib64" \
    "/usr/local/cuda/lib64" \
    "$CONDA_PREFIX/lib"; do
    [ -d "$_cuda_lib" ] && export LD_LIBRARY_PATH="${_cuda_lib}:${LD_LIBRARY_PATH:-}"
done

# --- Paths ---
# sam_3d_body (person detection + 3D pose + segmentation, step 01)
SAM3D_DIR="${SAM3D_DIR:-$REPO_DIR/../sam-3d-body}"
SAM3D_MODEL_DIR="${SAM3D_MODEL_DIR:-$REPO_DIR/../../model/sam-3d-body}"

# SAM2 (person segmentation, step 01b simplified)
SAM2_DIR="${SAM2_DIR:-$REPO_DIR/../sam2}"
export SEGMENTOR_PATH="${SEGMENTOR_PATH:-$SAM2_DIR}"

# wan22 / DiffSynth-Studio (video generation, step 02)
DIFFSYNTH_DIR="${DIFFSYNTH_DIR:-$REPO_DIR/../DiffSynth-Studio-Human}"
WAN_MODEL_DIR="${WAN_MODEL_DIR:-$REPO_DIR/../../model}"
export DIFFSYNTH_MODEL_BASE_PATH="$WAN_MODEL_DIR"
export DIFFSYNTH_SKIP_DOWNLOAD="${DIFFSYNTH_SKIP_DOWNLOAD:-True}"
export DIFFSYNTH_DOWNLOAD_SOURCE="${DIFFSYNTH_DOWNLOAD_SOURCE:-modelscope}"

# Output dir (outside the repo, like wan22_results / wan22_experiments).
RESULTS_DIR="${RESULTS_DIR:-$REPO_DIR/../wan22_rotate_results}"

export REPO_DIR SAM3D_DIR SAM3D_MODEL_DIR SAM2_DIR DIFFSYNTH_DIR WAN_MODEL_DIR RESULTS_DIR

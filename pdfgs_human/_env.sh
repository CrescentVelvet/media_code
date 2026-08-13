# _env.sh — shared setup: proxy + CA bundle + conda env activation + GPU + paths.
# Sourced by 00/01/02/03. Expects SCRIPT_DIR (this dir) to be set by the caller.
#
# Uses a SINGLE conda env (default `pdfgs`, CPython 3.10) for ALL steps:
#   - step 01 segmentation (SAM2 / rembg)
#   - step 02 Pi3 pose + COLMAP export
#   - step 03 PDF-GS training (diff-gaussian-rasterization + DINOv3)
# PDF-GS pins torch 2.5.1+cu121 (official environment.yml); this is NOT the same
# as wan22_rotate's torch 2.6.0+cu124, so it gets its own env. Override with
# CONDA_ENV=xxx.
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

# Force offline mode at runtime — all models are pre-downloaded by 00_setup_env.sh
# (corporate proxy blocks huggingface.co). Prevents transformers/huggingface_hub
# from attempting network downloads (SSL errors) during training.
# DINOv3 (gated) is pre-downloaded by 00 into $HF_HOME/hub, then read offline here.
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"

# Activate the conda env (CPython 3.10; PDF-GS deps).
CONDA_ENV="${CONDA_ENV:-pdfgs}"
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

# CUDA library paths (libcupti.so.12 etc.)
for _cuda_lib in \
    "/usr/local/cuda/extras/CUPTI/lib64" \
    "/usr/local/cuda/lib64" \
    "$CONDA_PREFIX/lib"; do
    [ -d "$_cuda_lib" ] && export LD_LIBRARY_PATH="${_cuda_lib}:${LD_LIBRARY_PATH:-}"
done

# --- Paths ---
# Official code dirs (siblings of media_code, cloned by 00_setup_env.sh).
PDFGS_DIR="${PDFGS_DIR:-$REPO_DIR/../PDF-GS}"
PI3_DIR="${PI3_DIR:-$REPO_DIR/../Pi3}"
SAM2_DIR="${SAM2_DIR:-$REPO_DIR/../sam2}"

# Shared weight root (code-dir's parent, same as wan22_rotate / pi3_3dgs).
MODEL_DIR="${MODEL_DIR:-$REPO_DIR/../../model}"

# SAM2 checkpoint (person segmentation, step 01). sam2 repo layout:
#   $SAM2_DIR/checkpoints/sam2.1_hiera_large.pt
#   $SAM2_DIR/sam2/configs/sam2.1/sam2.1_hiera_large.yaml
export SAM2_CHECKPOINT="${SAM2_CHECKPOINT:-$SAM2_DIR/checkpoints/sam2.1_hiera_large.pt}"
export SAM2_CONFIG="${SAM2_CONFIG:-configs/sam2.1/sam2.1_hiera_large.yaml}"

# Pi3 checkpoint (feed-forward pose estimation, step 02).
export PI3_CKPT="${PI3_CKPT:-$MODEL_DIR/Pi3/model.safetensors}"

# DINOv3 (gated, step 03 — PDF-GS DINOv3FeatureExtractor loads via transformers
# from_pretrained("facebook/dinov3-vitb16-pretrain-lvd1689m"); pre-downloaded by 00
# into $HF_HOME/hub, read offline here).
export HF_HOME="${HF_HOME:-$MODEL_DIR/hf_home}"
# DINOv3 needs an HF token for the gated repo; only used by 00 at download time.
# At runtime (offline) no token is read.

# Output dir (outside the repo, like wan22_rotate_results / pi3_3dgs_results).
RESULTS_DIR="${RESULTS_DIR:-$REPO_DIR/../pdfgs_human_results}"

export REPO_DIR PDFGS_DIR PI3_DIR SAM2_DIR MODEL_DIR RESULTS_DIR

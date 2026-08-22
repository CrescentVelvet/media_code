# _env.sh — shared setup: proxy + CA bundle + conda env + GPU pinning + paths.
# Sourced by 00/01/02/03/04/05. Expects SCRIPT_DIR (this dir) to be set by the caller.
REPO_DIR="$(dirname "$SCRIPT_DIR")"

# 1. proxy (from gitignored proxy.env at repo root)
if [ -f "$REPO_DIR/proxy.env" ]; then
    set -a; # shellcheck disable=SC1090
    source "$REPO_DIR/proxy.env"; set +a
fi
[ -n "${http_proxy:-}" ]  && export HTTP_PROXY="$http_proxy"
[ -n "${https_proxy:-}" ] && export HTTPS_PROXY="$https_proxy"

# 2. CA bundle (corporate proxy TLS interception)
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

# 3. conda env activation
CONDA_ENV="${CONDA_ENV:-artifixer}"
export CONDA_ENV
if ! command -v conda >/dev/null 2>&1; then
    echo "ERROR: conda not found on PATH (need an activated env; or set CONDA_ENV)." >&2
    exit 1
fi
# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV" 2>/dev/null || true  # env may not exist yet (00 creates it)

# 4. GPU pinning
if [ -n "${GPU:-}" ]; then
    export CUDA_VISIBLE_DEVICES="$GPU"
fi

# 5. paths (all use ${VAR:-default} for override)
MODEL_DIR="${MODEL_DIR:-$REPO_DIR/../../model}"
RESULTS_DIR="${RESULTS_DIR:-$REPO_DIR/../artifixer_results}"
export REPO_DIR MODEL_DIR RESULTS_DIR

# 6. ArtiFixer official code path (cloned by 00_setup_env.sh)
ARTIFIXER_DIR="${ARTIFIXER_DIR:-$REPO_DIR/../ArtiFixer}"
export ARTIFIXER_DIR

# 7. Model selection: 1.3B (fits comfortably on single 80 GB A100 for all workflows).
#    Override with MODEL_VARIANT=14b for highest quality (needs 2× A100 or careful memory management).
MODEL_VARIANT="${MODEL_VARIANT:-1.3b}"
case "$MODEL_VARIANT" in
    1.3b)
        export ARTIFIXER_CHECKPOINT="${ARTIFIXER_CHECKPOINT:-$MODEL_DIR/artifixer/artifixer-1.3b.pt}"
        export WAN_MODEL_ID="${WAN_MODEL_ID:-$MODEL_DIR/Wan-AI/Wan2.1-T2V-1.3B-Diffusers}"
        export HF_MODEL_ID="${HF_MODEL_ID:-Wan-AI/Wan2.1-T2V-1.3B-Diffusers}"
        ;;
    14b)
        export ARTIFIXER_CHECKPOINT="${ARTIFIXER_CHECKPOINT:-$MODEL_DIR/artifixer/artifixer-14b.pt}"
        export WAN_MODEL_ID="${WAN_MODEL_ID:-$MODEL_DIR/Wan-AI/Wan2.1-T2V-14B-Diffusers}"
        export HF_MODEL_ID="${HF_MODEL_ID:-Wan-AI/Wan2.1-T2V-14B-Diffusers}"
        ;;
    *)
        echo "ERROR: MODEL_VARIANT must be 1.3b or 14b, got: $MODEL_VARIANT" >&2
        exit 1
        ;;
esac
export MODEL_VARIANT

# 8. Use local model path as --model_id (diffusers from_pretrained accepts local dirs).
#    HF_HUB_OFFLINE=1 during inference (no network). Set to 0 in download/data-prep scripts.
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
# HF cache under $MODEL_DIR so Qwen3-VL (hardcoded repo id in prepare_colmap) resolves locally.
export HF_HOME="${HF_HOME:-$MODEL_DIR/hf_cache}"

# 9. Captioning model (Qwen3-VL-30B-A3B is a MoE: ~60 GB bf16 load, 3B active).
#    The repo id is hardcoded in prepare_colmap_artifixer_inputs, so it must be in the HF cache.
#    We download it to $HF_HOME/hub/ via huggingface-cli (no --local-dir).
export CAPTIONING_MODEL_ID="${CAPTIONING_MODEL_ID:-Qwen/Qwen3-VL-30B-A3B-Instruct}"

# 10. MoGe (metric scale alignment during data prep)
export MOGE_MODEL_PATH="${MOGE_MODEL_PATH:-$MODEL_DIR/moge}"

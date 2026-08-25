# _env.sh — shared setup: proxy + CA bundle + conda env activation + GPU + paths.
# Sourced by 00/00a/01/02/04/08. Expects SCRIPT_DIR (this dir) to be set by the caller.
#
# OSEDiff: One-Step Effective Diffusion Network for Real-World Image Super-Resolution.
# Official repo cloned by 00/00a to $OSEDIFF_DIR; weights under $MODEL_DIR.
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

# Disable HF Xet/CAS (Rust reqwest) — fails behind TLS-intercepting corporate proxy.
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"

# Activate the conda env. 00a creates it from scratch; on servers reuse existing.
CONDA_ENV="${CONDA_ENV:-osediff}"
export CONDA_ENV
if ! command -v conda >/dev/null 2>&1; then
    # Fallback: try common conda locations (WSL/local dev without `conda init`).
    # On servers conda is already on PATH, so this block never triggers.
    for _cb in "$HOME/miniconda3" "$HOME/anaconda3" "/opt/conda"; do
        if [ -f "$_cb/etc/profile.d/conda.sh" ]; then
            # shellcheck disable=SC1091
            source "$_cb/etc/profile.d/conda.sh"
            break
        fi
    done
    unset _cb
fi
if ! command -v conda >/dev/null 2>&1; then
    echo "❌ ERROR: conda not found on PATH (need env '$CONDA_ENV')." >&2
    echo "       Install miniconda or run: source ~/miniconda3/etc/profile.d/conda.sh" >&2
    exit 1
fi
# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV" 2>/dev/null || true  # env may not exist yet (00/00a creates it)

# Pin GPU (0-indexed) via GPU=N.
if [ -n "${GPU:-}" ]; then
    export CUDA_VISIBLE_DEVICES="$GPU"
fi

# --- Paths ---
# OSEDiff official code (sibling of media_code; 00/00a clone it).
#   server: $REPO_DIR/../OSEDiff   |  WSL (proxy.env): ~/repos/OSEDiff
OSEDIFF_DIR="${OSEDIFF_DIR:-$REPO_DIR/../OSEDiff}"
OSEDIFF_REPO="${OSEDIFF_REPO:-https://github.com/cswry/OSEDiff.git}"

# Shared weight root (code-dir's parent, same as other algos).
#   server: $REPO_DIR/../../model/osediff  |  WSL: ~/model/osediff
MODEL_DIR="${MODEL_DIR:-$REPO_DIR/../../model/osediff}"

# Output (siblings of media_code, per AGENTS.md convention).
#   server: $REPO_DIR/../osediff_results  |  WSL: ~/output/osediff_results
RESULTS_DIR="${RESULTS_DIR:-$REPO_DIR/../osediff_results}"
# Training artifacts (checkpoints / tensorboard).
EXPERIMENTS_DIR="${EXPERIMENTS_DIR:-$REPO_DIR/../osediff_experiments}"

# --- Sub-model locations (overridable via env) ---
# SD2.1-Base (HF: Manojb/stable-diffusion-2-1-base), downloaded by 01.
SD21_BASE_DIR="${SD21_BASE_DIR:-$MODEL_DIR/sd21_base}"
# RAM (recognize-anything, ram_swin_large_14m.pth), downloaded by 01.
RAM_PATH="${RAM_PATH:-$MODEL_DIR/ram_swin_large_14m.pth}"
# DAPE (RAM fine-tuned for enhancement captions). repo ships a copy at
# preset/models/DAPE.pth; 01 copies it to $MODEL_DIR (or downloads full from gdrive).
DAPE_PATH="${DAPE_PATH:-$MODEL_DIR/DAPE.pth}"
# OSEDiff trained weights — shipped inside the repo (clone brings them).
OSEDIFF_PKL="${OSEDIFF_PKL:-$OSEDIFF_DIR/preset/models/osediff.pkl}"
OSEDIFF_FACE_PKL="${OSEDIFF_FACE_PKL:-$OSEDIFF_DIR/preset/models/osediff_face.pkl}"

export REPO_DIR OSEDIFF_DIR OSEDIFF_REPO MODEL_DIR RESULTS_DIR EXPERIMENTS_DIR \
       SD21_BASE_DIR RAM_PATH DAPE_PATH OSEDIFF_PKL OSEDIFF_FACE_PKL

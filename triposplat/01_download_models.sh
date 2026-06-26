#!/usr/bin/env bash
# 01_download_models.sh — pull TripoSplat weights from HuggingFace into the
# shared model store, then symlink the official repo's ckpts/ to it.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ALGO_DIR="$SCRIPT_DIR"
REPO_DIR="$(dirname "$ALGO_DIR")"

# Optional proxy (honored by huggingface_hub's HTTP client).
if [ -f "$REPO_DIR/proxy.env" ]; then
    set -a; # shellcheck disable=SC1090
    source "$REPO_DIR/proxy.env"; set +a
fi

# --- Corporate proxy TLS interception workaround (needed by `hf download`) ---
SYS_CA=/etc/ssl/certs/ca-certificates.crt
if [ -f "$SYS_CA" ]; then
    : "${REQUESTS_CA_BUNDLE:=$SYS_CA}"
    : "${SSL_CERT_FILE:=$SYS_CA}"
    export REQUESTS_CA_BUNDLE SSL_CERT_FILE
fi

# ---- Config ----------------------------------------------------------------
VENV_DIR="${VENV_DIR:-$ALGO_DIR/.venv}"
TRIPOSPLAT_DIR="${TRIPOSPLAT_DIR:-$REPO_DIR/../TripoSplat}"
MODEL_DIR="${MODEL_DIR:-$REPO_DIR/../model/TripoSplat}"
HF_REPO_ID="${HF_REPO_ID:-VAST-AI/TripoSplat}"
# ----------------------------------------------------------------------------

CKPTS_LINK="$TRIPOSPLAT_DIR/ckpts"

echo "=== [01] Downloading TripoSplat weights ==="
echo "  HF repo:  $HF_REPO_ID"
echo "  model:    $MODEL_DIR"
echo "  code:     $TRIPOSPLAT_DIR"

if [ ! -d "$TRIPOSPLAT_DIR" ]; then
    echo "ERROR: TripoSplat code dir not found at $TRIPOSPLAT_DIR." >&2
    echo "       Run run_all.sh first (it clones the official repo), or set TRIPOSPLAT_DIR." >&2
    exit 1
fi

# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"

mkdir -p "$MODEL_DIR"

# Public MIT repo — no token required. The snapshot covers every file
# referenced by run_example.py:
#   ckpts/diffusion_models/triposplat_fp16.safetensors
#   ckpts/vae/triposplat_vae_decoder_fp16.safetensors
#   ckpts/clip_vision/dino_v3_vit_h.safetensors
#   ckpts/vae/flux2-vae.safetensors
#   ckpts/background_removal/birefnet.safetensors
hf download "$HF_REPO_ID" --local-dir "$MODEL_DIR"

echo "--- downloaded files ---"
find "$MODEL_DIR" -maxdepth 2 -type f -printf '  %P\n' | sort

# Link ckpts -> shared model dir so the official run_example.py (which loads
# from ./ckpts/...) works without modifying any official code.
if [ -e "$CKPTS_LINK" ] && [ ! -L "$CKPTS_LINK" ]; then
    echo "WARNING: $CKPTS_LINK exists and is not a symlink; leaving it untouched." >&2
    echo "         Remove it if you want to use $MODEL_DIR." >&2
else
    [ -L "$CKPTS_LINK" ] && rm -f "$CKPTS_LINK"
    ln -s "$MODEL_DIR" "$CKPTS_LINK"
    echo "--- symlink: $CKPTS_LINK -> $MODEL_DIR"
fi

echo "=== [01] Done. Weights at: $MODEL_DIR (linked as $CKPTS_LINK) ==="

#!/usr/bin/env bash
# 01_download_models.sh — pull a FLUX.1 model snapshot from HuggingFace into
# the shared model store. FLUX.1 ships as a self-contained diffusers-format
# HF repo (transformer + VAE + T5/CLIP text encoders + scheduler), so there
# is NO separate official code repo to clone — the snapshot is everything.
#
# Default: black-forest-labs/FLUX.1-schnell  (Apache-2.0, PUBLIC, no token, 4-step)
# Higher-quality gated alternative (override HF_REPO_ID):
#   black-forest-labs/FLUX.1-dev   (gated; needs HF_TOKEN after accepting the
#                                  license on the model page; 28-step, better)
#
# ⚠️ FLUX.1-dev is GATED: `hf download` without a token returns 401 / "not
# found". You MUST 1) accept the license at
# https://huggingface.co/black-forest-labs/FLUX.1-dev and 2) pass HF_TOKEN.
# schnell needs neither.
#
# The whole snapshot is fetched (safetensors + configs + tokenizer assets).
# Falls back to an SSL-bypass downloader if the CDN MITM cert can't be verified.
# Use INCLUDE_PATTERNS to fetch only a subset (glob, comma-sep) — e.g. only the
# transformer fp8 weights: INCLUDE_PATTERNS="transformer/*,*.json,*.txt" .
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/_env.sh"

MODEL_DIR="${MODEL_DIR:-$REPO_DIR/../../model/FLUX1}"
HF_REPO_ID="${HF_REPO_ID:-black-forest-labs/FLUX.1-schnell}"

# Local snapshot dir = $MODEL_DIR/<repo-basename> (e.g. .../FLUX1/FLUX.1-schnell).
REPO_BASE="$(basename "$HF_REPO_ID")"
MODEL_PATH="${MODEL_PATH:-$MODEL_DIR/$REPO_BASE}"

echo "=== [01] Downloading FLUX.1 weights ==="
echo "  HF仓库ID:  $HF_REPO_ID"
echo "  模型路径:  $MODEL_PATH"

# Gated-repo guard for FLUX.1-dev (best-effort; doesn't block other gated repos).
case "$HF_REPO_ID" in
    *FLUX.1-dev*|*FLUX.1-[Dd]ev*)
        if [ -z "${HF_TOKEN:-}" ]; then
            echo "ERROR: $HF_REPO_ID is GATED and HF_TOKEN is not set." >&2
            echo "       1) accept the license at https://huggingface.co/$HF_REPO_ID" >&2
            echo "       2) create a read token at https://huggingface.co/settings/tokens" >&2
            echo "       3) HF_TOKEN=<token> bash $SCRIPT_DIR/01_download_models.sh" >&2
            echo "       (or use the public schnell default: unset HF_REPO_ID)" >&2
            exit 1
        fi
        ;;
esac

if ! python -c "import huggingface_hub" 2>/dev/null; then
    echo "ERROR: huggingface_hub not installed in env '$CONDA_ENV'. Install with: pip install huggingface_hub" >&2
    exit 1
fi

mkdir -p "$MODEL_PATH"

# --- download one HF repo with SSL-bypass fallback (mirrors qwen3vl 01) ---
# hf_get <repo_id> <local_dir> [include_patterns_csv] [token]
hf_get() {
    local repo="$1" dir="$2" includes="${3:-}" token="${4:-}"
    mkdir -p "$dir"
    local token_args=(); [ -n "$token" ] && token_args=(--token "$token")
    local include_args=()
    if [ -n "$includes" ]; then
        local parts=()
        IFS=',' read -ra parts <<< "$includes"
        for p in "${parts[@]}"; do include_args+=(--include "$p"); done
    fi
    if [ "${HF_DISABLE_SSL:-0}" = "1" ]; then
        echo "--- [$repo] downloading (SSL verification DISABLED via HF_DISABLE_SSL=1) ---"
        python "$SCRIPT_DIR/_hf_download.py" "$repo" "$dir" "$includes" "$token"
    else
        if ! hf download "$repo" "${token_args[@]}" "${include_args[@]}" --local-dir "$dir"; then
            echo "--- [$repo] hf download failed (likely SSL on CDN, or gated repo w/o token); retrying with SSL verification disabled ---"
            python "$SCRIPT_DIR/_hf_download.py" "$repo" "$dir" "$includes" "$token"
        fi
    fi
}

echo "--- downloading snapshot: $HF_REPO_ID ---"
hf_get "$HF_REPO_ID" "$MODEL_PATH" "${INCLUDE_PATTERNS:-}" "${HF_TOKEN:-}"

echo "--- downloaded files (depth 2) ---"
find "$MODEL_PATH" -maxdepth 2 -mindepth 1 \( -type f -o -type d \) -printf '  %P\n' 2>/dev/null | sort | head -60

# Sanity: a diffusers FluxPipeline needs model_index.json + the subfolders.
if [ -f "$MODEL_PATH/model_index.json" ]; then
    echo "--- model dir OK: $MODEL_PATH ---"
else
    echo "WARNING: $MODEL_PATH/model_index.json not found — the snapshot may be incomplete." >&2
    echo "         Check HF_REPO_ID / HF_TOKEN / network and rerun." >&2
fi

echo "=== [01] Done. Weights at: $MODEL_PATH ==="
echo "    Inference:  GPU=0 bash $SCRIPT_DIR/02_run_inference.sh"

#!/usr/bin/env bash
# 01_download_models.sh — pull a Qwen3-VL model snapshot from HuggingFace into
# the shared model store. Qwen3-VL ships as a self-contained diffusers-style
# HF repo (config + safetensors + processor assets), so there is NO separate
# official code repo to clone — the snapshot is everything.
#
# Default: Qwen/Qwen3-VL-7B-Instruct  (a dense 7B image-to-text VLM)
# Alternatives (override HF_REPO_ID):
#   Qwen/Qwen3-VL-30B-A3B-Instruct     (MoE 30B/3B-active; needs ~2x VRAM, use GPU=0,1)
#   Qwen/Qwen3-VL-30B-A3B-Thinking     (thinking variant; enable THINKING=1 at inference)
#   Qwen/Qwen3-VL-7B-Thinking          (7B thinking variant)
#
# The repo is PUBLIC (not gated) — no HF_TOKEN required. The whole snapshot is
# fetched (safetensors + configs + processor assets). Falls back to an
# SSL-bypass downloader if the CDN MITM cert can't be verified.
#
# Use INCLUDE_PATTERNS to fetch only a subset (glob, comma-sep) — e.g. only
# bf16 weights to save disk: INCLUDE_PATTERNS="*.json,*.txt,*.safetensors" .
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/_env.sh"

MODEL_DIR="${MODEL_DIR:-$REPO_DIR/../../model/Qwen3-VL}"
HF_REPO_ID="${HF_REPO_ID:-Qwen/Qwen3-VL-7B-Instruct}"

# Local snapshot dir = $MODEL_DIR/<repo-basename> (e.g. .../Qwen3-VL/Qwen3-VL-7B-Instruct).
REPO_BASE="$(basename "$HF_REPO_ID")"
MODEL_PATH="${MODEL_PATH:-$MODEL_DIR/$REPO_BASE}"

echo "=== [01] Downloading Qwen3-VL weights ==="
echo "  HF仓库ID:  $HF_REPO_ID"
echo "  模型路径:  $MODEL_PATH"

if ! python -c "import huggingface_hub" 2>/dev/null; then
    echo "ERROR: huggingface_hub not installed in env '$CONDA_ENV'. Install with: pip install huggingface_hub" >&2
    exit 1
fi

mkdir -p "$MODEL_PATH"

# --- download one HF repo with SSL-bypass fallback (mirrors hypir 01) ---
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
            echo "--- [$repo] hf download failed (likely SSL on CDN); retrying with SSL verification disabled ---"
            python "$SCRIPT_DIR/_hf_download.py" "$repo" "$dir" "$includes" "$token"
        fi
    fi
}

echo "--- downloading snapshot: $HF_REPO_ID ---"
hf_get "$HF_REPO_ID" "$MODEL_PATH" "${INCLUDE_PATTERNS:-}" "${HF_TOKEN:-}"

echo "--- downloaded files (depth 2) ---"
find "$MODEL_PATH" -maxdepth 2 -mindepth 1 \( -type f -o -type d \) -printf '  %P\n' 2>/dev/null | sort | head -60

# Sanity: the snapshot needs at least a config.json + a weights file to load.
if [ -f "$MODEL_PATH/config.json" ]; then
    echo "--- model dir OK: $MODEL_PATH ---"
else
    echo "WARNING: $MODEL_PATH/config.json not found — the snapshot may be incomplete." >&2
    echo "         Check HF_REPO_ID / network and rerun." >&2
fi

echo "=== [01] Done. Weights at: $MODEL_PATH ==="
echo "    Inference:  GPU=0 bash $SCRIPT_DIR/02_run_inference.sh"

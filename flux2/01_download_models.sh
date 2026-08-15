#!/usr/bin/env bash
# 01_download_models.sh — pull FLUX.2 diffusers-format snapshots from HuggingFace
# into the shared model store. FLUX.2 ships as a self-contained diffusers HF repo
# (transformer + VAE + text encoder [Mistral3 for dev / Qwen3 for klein] +
# scheduler), so there is NO separate official code repo to clone — the snapshot
# is everything; load it directly with Flux2Pipeline / Flux2KleinPipeline.
#
# Both FLUX.2-dev and FLUX.2-klein-9B are GATED (FLUX Non-Commercial License):
# `hf download` without a token returns 401 / "not found". You MUST
#   1) accept the license at https://huggingface.co/black-forest-labs/FLUX.2-dev
#      and https://huggingface.co/black-forest-labs/FLUX.2-klein-9B
#   2) pass HF_TOKEN (a read token from https://huggingface.co/settings/tokens)
#
# Which models? MODELS_TO_DOWNLOAD (space-separated, default "dev klein"):
#   MODELS_TO_DOWNLOAD="klein" bash ...   # just the 9B (fits consumer GPUs)
#   MODELS_TO_DOWNLOAD="dev klein" bash ...# both (dev needs H100-class or quant)
#
# Skip this script if you already downloaded the weights — just point MODEL_PATH
# at the snapshot dir in 02_run_inference.sh.
#
# Falls back to an SSL-bypass downloader if the CDN MITM cert can't be verified.
# Use INCLUDE_PATTERNS to fetch only a subset (glob, comma-sep).
set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/_env.sh"

MODEL_DIR="${MODEL_DIR:-$REPO_DIR/../../model/FLUX2}"
MODELS_TO_DOWNLOAD="${MODELS_TO_DOWNLOAD:-dev klein}"

# repo id + local basename per model keyword.
case_kw() {
    case "$1" in
        dev)   echo "black-forest-labs/FLUX.2-dev FLUX.2-dev" ;;
        klein) echo "black-forest-labs/FLUX.2-klein-9B FLUX.2-klein-9B" ;;
        *) echo ""; return 1 ;;
    esac
}

echo "=== [01] Downloading FLUX.2 weights ==="
echo "  模型根目录: $MODEL_DIR"
echo "  下载目标:   $MODELS_TO_DOWNLOAD"
echo "  HF_TOKEN:   ${HF_TOKEN:+已设置}${HF_TOKEN:-❌ 未设置 (gated 仓需要!)}"

# Gated-repo guard: BOTH repos are gated.
if [ -z "${HF_TOKEN:-}" ]; then
    echo "❌ ERROR: FLUX.2 repos are GATED and HF_TOKEN is not set." >&2
    echo "       1) accept the license at https://huggingface.co/black-forest-labs/FLUX.2-dev" >&2
    echo "          and https://huggingface.co/black-forest-labs/FLUX.2-klein-9B" >&2
    echo "       2) create a read token at https://huggingface.co/settings/tokens" >&2
    echo "       3) HF_TOKEN=<token> MODELS_TO_DOWNLOAD=\"klein\" bash $SCRIPT_DIR/01_download_models.sh" >&2
    exit 1
fi

if ! python -c "import huggingface_hub" 2>/dev/null; then
    echo "❌ ERROR: huggingface_hub not installed in env '$CONDA_ENV'. Install with: pip install huggingface_hub" >&2
    exit 1
fi

mkdir -p "$MODEL_DIR"

# --- download one HF repo with SSL-bypass fallback (mirrors flux1 01) ---
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

for kw in $MODELS_TO_DOWNLOAD; do
    pair="$(case_kw "$kw")" || { echo "⚠️ unknown model keyword '$kw' (use: dev | klein)" >&2; continue; }
    repo_id="${pair%% *}"
    base="${pair##* }"
    dest="$MODEL_DIR/$base"
    echo "--- downloading snapshot: $repo_id -> $dest ---"
    hf_get "$repo_id" "$dest" "${INCLUDE_PATTERNS:-}" "${HF_TOKEN:-}"

    if [ -f "$dest/model_index.json" ]; then
        echo "✅ model dir OK: $dest"
    else
        echo "⚠️ WARNING: $dest/model_index.json not found — snapshot may be incomplete." >&2
        echo "         Check HF_TOKEN / network and rerun." >&2
    fi
done

echo "--- downloaded files (depth 1) ---"
find "$MODEL_DIR" -maxdepth 2 -mindepth 1 \( -type f -o -type d \) -printf '  %P\n' 2>/dev/null | sort | head -60

echo "=== [01] Done. Weights under: $MODEL_DIR ==="
echo "    Inference (klein, fast):  GPU=0 MODEL_TYPE=klein MODEL_PATH=$MODEL_DIR/FLUX.2-klein-9B bash $SCRIPT_DIR/02_run_inference.sh"
echo "    Inference (dev, quality): GPU=0 MODEL_TYPE=dev   MODEL_PATH=$MODEL_DIR/FLUX.2-dev       bash $SCRIPT_DIR/02_run_inference.sh"

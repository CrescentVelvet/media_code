#!/usr/bin/env bash
# 01_download_models.sh — pull MiniMax-H3 weights (HuggingFace snapshot) into
# the shared model store. The HF repo `MiniMaxAI/MiniMax-H3` hosts the original
# checkpoint (FL2VA/, Ref2VA/) that SGLang serves.
#
# By default only FL2VA is fetched (T2VA / I2VA / L2VA / FL2VA tasks).
# Set DOWNLOAD_REF2VA=1 to also fetch Ref2VA (reference-to-video tasks).
#
# NOTE: MiniMax-H3 is released under the MiniMax H3 Community License. The HF
# repo MAY be gated — if you get 401 / "repository not found", you need to:
#   1) open https://huggingface.co/MiniMaxAI/MiniMax-H3 and accept the license;
#   2) create a read token at https://huggingface.co/settings/tokens;
#   3) HF_TOKEN=<token> bash minimax_h3/01_download_models.sh
# (the script forwards HF_TOKEN to `hf download` and the SSL-bypass downloader).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/_env.sh"

MINIMAX_H3_DIR="${MINIMAX_H3_DIR:-$REPO_DIR/../MiniMax-H3}"   # GitHub code (reference)
MODEL_DIR="${MODEL_DIR:-$REPO_DIR/../../model/MiniMax-H3}"    # HF weight snapshot (SGLang serves this)

HF_REPO="${HF_REPO:-MiniMaxAI/MiniMax-H3}"
MODEL_VARIANT_DEFAULT="${MODEL_VARIANT_DEFAULT:-fl2va}"

# Build the include list. Top-level model_index.json is the public entry SGLang
# reads; FL2VA/* is the FL2VA task family (default); Ref2VA/* is opt-in.
INCLUDES="model_index.json,FL2VA/*"
if [ "${DOWNLOAD_REF2VA:-0}" = "1" ]; then
    INCLUDES="$INCLUDES,Ref2VA/*"
fi

echo "=== [01] Downloading MiniMax-H3 weights ==="
echo "  HF repo:      $HF_REPO"
echo "  model dir:    $MODEL_DIR  (SGLang --model-path points here)"
echo "  code dir:     $MINIMAX_H3_DIR  (GitHub reference repo)"
echo "  includes:     $INCLUDES"
if [ -n "${HF_TOKEN:-}" ]; then
    echo "  HF_TOKEN:     *** (set)"
else
    echo "  HF_TOKEN:     (unset — OK if repo is public; needed if gated)"
fi

if ! python -c "import huggingface_hub" 2>/dev/null; then
    echo "ERROR: huggingface_hub not installed in env '$CONDA_ENV'. Install with: pip install huggingface_hub" >&2
    exit 1
fi

mkdir -p "$MODEL_DIR"

# --- download the HF snapshot with SSL-bypass fallback (mirrors hypir 01) ---
# hf_get <repo_id> <local_dir> <include_patterns_csv> [token]
hf_get() {
    local repo="$1" dir="$2" includes="$3" token="${4:-}"
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

echo "--- [1/1] snapshot: $HF_REPO -> $MODEL_DIR ---"
hf_get "$HF_REPO" "$MODEL_DIR" "$INCLUDES" "${HF_TOKEN:-}"

echo "--- downloaded structure ---"
echo "  model dir top-level:"; find "$MODEL_DIR" -maxdepth 1 -mindepth 1 -printf '    %P\n' 2>/dev/null | sort
echo "  FL2VA/:"; find "$MODEL_DIR/FL2VA" -maxdepth 1 -mindepth 1 -printf '    %P\n' 2>/dev/null 2>/dev/null | sort || echo "    (missing)"
if [ "${DOWNLOAD_REF2VA:-0}" = "1" ]; then
    echo "  Ref2VA/:"; find "$MODEL_DIR/Ref2VA" -maxdepth 1 -mindepth 1 -printf '    %P\n' 2>/dev/null 2>/dev/null | sort || echo "    (missing)"
fi

# Sanity-check the entry + the FL2VA task family SGLang loads by default.
ok=1
[ -f "$MODEL_DIR/model_index.json" ] || { echo "WARNING: $MODEL_DIR/model_index.json missing" >&2; ok=0; }
[ -d "$MODEL_DIR/FL2VA" ] || { echo "WARNING: $MODEL_DIR/FL2VA/ missing (FL2VA not downloaded?)" >&2; ok=0; }
if [ "$ok" = "1" ]; then
    echo "--- weight snapshot OK: $MODEL_DIR ---"
    echo "    serve with:  MODEL_PATH=$MODEL_DIR bash minimax_h3/02_serve.sh"
fi

echo "=== [01] Done. Weights at: $MODEL_DIR ==="

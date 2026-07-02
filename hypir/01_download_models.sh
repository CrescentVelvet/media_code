#!/usr/bin/env bash
# 01_download_models.sh — pull HYPIR weights into the shared model store:
#   1. stabilityai/stable-diffusion-2-1-base  -> $MODEL_DIR/sd2_base  (base diffusers model)
#   2. lxq007/HYPIR                            -> $MODEL_DIR           (HYPIR_sd2.pth LoRA)
#
# The base SD2 model is fetched with --include for only the subfolders the code
# actually loads (scheduler/tokenizer/text_encoder/unet/vae); the LoRA repo is
# a single .pth. Both fall back to an SSL-bypass downloader if the CDN MITM
# cert can't be verified.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/_env.sh"

HYPIR_DIR="${HYPIR_DIR:-$REPO_DIR/../HYPIR}"
MODEL_DIR="${MODEL_DIR:-$REPO_DIR/../../model/HYPIR}"

HF_BASE_REPO="${HF_BASE_REPO:-stabilityai/stable-diffusion-2-1-base}"
HF_LORA_REPO="${HF_LORA_REPO:-lxq007/HYPIR}"
LORA_FILE="${LORA_FILE:-HYPIR_sd2.pth}"

BASE_MODEL_DIR="${BASE_MODEL_DIR:-$MODEL_DIR/sd2_base}"
WEIGHT_PATH="${WEIGHT_PATH:-$MODEL_DIR/$LORA_FILE}"

# Only these subfolders are referenced by HYPIR (DDPMScheduler/CLIPTextModel/
# CLIPTokenizer/UNet2DConditionModel/AutoencoderKL all use subfolder=...).
BASE_INCLUDES="model_index.json,scheduler/*,tokenizer/*,text_encoder/*,unet/*,vae/*"

echo "=== [01] Downloading HYPIR weights ==="
echo "  model dir:  $MODEL_DIR"
echo "  code dir:   $HYPIR_DIR"
echo "  base model: $HF_BASE_REPO -> $BASE_MODEL_DIR"
echo "  lora:       $HF_LORA_REPO -> $WEIGHT_PATH"

if [ ! -d "$HYPIR_DIR" ]; then
    echo "ERROR: HYPIR code dir not found at $HYPIR_DIR." >&2
    echo "       Run run_all.sh first (it clones the official repo), or set HYPIR_DIR." >&2
    exit 1
fi

if ! python -c "import huggingface_hub" 2>/dev/null; then
    echo "ERROR: huggingface_hub not installed in env '$CONDA_ENV'. Install with: pip install huggingface_hub" >&2
    exit 1
fi

mkdir -p "$MODEL_DIR" "$BASE_MODEL_DIR"

# --- download one HF repo with SSL-bypass fallback (mirrors hunyuanvideo 01) ---
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
        python "$SCRIPT_DIR/_hf_download.py" "$repo" "$dir" "$includes"
    else
        if ! hf download "$repo" "${token_args[@]}" "${include_args[@]}" --local-dir "$dir"; then
            echo "--- [$repo] hf download failed (likely SSL on CDN); retrying with SSL verification disabled ---"
            python "$SCRIPT_DIR/_hf_download.py" "$repo" "$dir" "$includes"
        fi
    fi
}

# 1. Base SD2 diffusers model (only the needed subfolders).
if [ "${SKIP_BASE:-0}" != "1" ]; then
    echo "--- [1/2] base model: $HF_BASE_REPO ---"
    hf_get "$HF_BASE_REPO" "$BASE_MODEL_DIR" "$BASE_INCLUDES"
else
    echo "--- [1/2] SKIPPED base model (SKIP_BASE=1) ---"
fi

# 2. HYPIR LoRA weights (single .pth file; whole repo is fine).
if [ "${SKIP_LORA:-0}" != "1" ]; then
    echo "--- [2/2] lora weights: $HF_LORA_REPO ---"
    hf_get "$HF_LORA_REPO" "$MODEL_DIR"
else
    echo "--- [2/2] SKIPPED lora weights (SKIP_LORA=1) ---"
fi

echo "--- downloaded files ---"
echo "  base model:"; find "$BASE_MODEL_DIR" -maxdepth 1 -mindepth 1 -printf '    %P\n' 2>/dev/null | sort
echo "  lora dir:";   find "$MODEL_DIR" -maxdepth 1 -type f -printf '    %P\n' 2>/dev/null | sort

if [ -f "$WEIGHT_PATH" ]; then
    echo "--- lora weight OK: $WEIGHT_PATH ($(du -h "$WEIGHT_PATH" | cut -f1)) ---"
else
    echo "WARNING: expected lora weight not found at $WEIGHT_PATH." >&2
    echo "         Check the HF repo file name (LORA_FILE=$LORA_FILE) and rerun." >&2
fi

# Optional: fill weight_path in the gradio config so `python app.py` works as-is.
# This edits the cloned official config in place (the only TODO in sd2_gradio.yaml).
GRADIO_CFG="$HYPIR_DIR/configs/sd2_gradio.yaml"
if [ -f "$GRADIO_CFG" ] && grep -q 'weight_path: TODO' "$GRADIO_CFG" 2>/dev/null; then
    sed -i "s|weight_path: TODO|weight_path: $WEIGHT_PATH|" "$GRADIO_CFG"
    echo "--- filled weight_path in $GRADIO_CFG -> $WEIGHT_PATH"
fi

echo "=== [01] Done. Weights at: $MODEL_DIR ==="
echo "    base model: $BASE_MODEL_DIR"
echo "    lora:       $WEIGHT_PATH"

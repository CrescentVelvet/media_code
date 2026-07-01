#!/usr/bin/env bash
# 01_download_models.sh — pull HunyuanVideo-1.5 weights into the shared model
# store, then symlink the official repo's ckpts/ to it.
#
# Pulls (all optional via SKIP_* env vars):
#   1. tencent/HunyuanVideo-1.5          -> $MODEL_DIR           (DiT + VAE + SR + all transformer variants)
#   2. Qwen/Qwen2.5-VL-7B-Instruct       -> $MODEL_DIR/text_encoder/llm   (MLLM text encoder)
#   3. google/byt5-small                 -> $MODEL_DIR/text_encoder/byt5-small  (byT5 tokenizer)
#   4. AI-ModelScope/Glyph-SDXL-v2       -> $MODEL_DIR/text_encoder/Glyph-SDXL-v2 (byT5 weights; via modelscope)
#   5. black-forest-labs/FLUX.1-Redux-dev -> $MODEL_DIR/vision_encoder/siglip (gated; needs HF_TOKEN — i2v only)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/_env.sh"

HYVIDEO_DIR="${HYVIDEO_DIR:-$REPO_DIR/../HunyuanVideo-1.5}"
MODEL_DIR="${MODEL_DIR:-$REPO_DIR/../../model/HunyuanVideo-1.5}"
CKPTS_LINK="$HYVIDEO_DIR/ckpts"

HF_REPO_ID="${HF_REPO_ID:-tencent/HunyuanVideo-1.5}"
LLM_REPO_ID="${LLM_REPO_ID:-Qwen/Qwen2.5-VL-7B-Instruct}"
BYT5_REPO_ID="${BYT5_REPO_ID:-google/byt5-small}"
GLYPH_MS_ID="${GLYPH_MS_ID:-AI-ModelScope/Glyph-SDXL-v2}"
SIGLIP_REPO_ID="${SIGLIP_REPO_ID:-black-forest-labs/FLUX.1-Redux-dev}"

echo "=== [01] Downloading HunyuanVideo-1.5 weights ==="
echo "  model dir: $MODEL_DIR"
echo "  code dir:  $HYVIDEO_DIR"
echo "  HF repo:   $HF_REPO_ID"

if [ ! -d "$HYVIDEO_DIR" ]; then
    echo "ERROR: HunyuanVideo-1.5 code dir not found at $HYVIDEO_DIR." >&2
    echo "       Run run_all.sh first (it clones the official repo), or set HYVIDEO_DIR." >&2
    exit 1
fi

if ! python -c "import huggingface_hub" 2>/dev/null; then
    echo "ERROR: huggingface_hub not installed in env '$CONDA_ENV'. Install with: pip install huggingface_hub" >&2
    exit 1
fi

mkdir -p "$MODEL_DIR"

# --- download one HF repo with SSL-bypass fallback (mirrors triposplat 01) ---
hf_get() {  # hf_get <repo_id> <local_dir> [token]
    local repo="$1" dir="$2" token="${3:-}"
    mkdir -p "$dir"
    local token_args=()
    [ -n "$token" ] && token_args=(--token "$token")
    if [ "${HF_DISABLE_SSL:-0}" = "1" ]; then
        echo "--- [$repo] downloading (SSL verification DISABLED via HF_DISABLE_SSL=1) ---"
        python "$SCRIPT_DIR/_hf_download.py" "$repo" "$dir"
    else
        if ! hf download "$repo" "${token_args[@]}" --local-dir "$dir"; then
            echo "--- [$repo] hf download failed (likely SSL on CDN); retrying with SSL verification disabled ---"
            python "$SCRIPT_DIR/_hf_download.py" "$repo" "$dir"
        fi
    fi
}

# 1. Main DiT/VAE/SR checkpoint tree (one repo covers every transformer variant).
if [ "${SKIP_MAIN:-0}" != "1" ]; then
    echo "--- [1/5] main checkpoint: $HF_REPO_ID ---"
    hf_get "$HF_REPO_ID" "$MODEL_DIR"
fi

# 2. MLLM text encoder (Qwen2.5-VL-7B-Instruct).
if [ "${SKIP_LLM:-0}" != "1" ]; then
    echo "--- [2/5] text encoder (MLLM): $LLM_REPO_ID ---"
    hf_get "$LLM_REPO_ID" "$MODEL_DIR/text_encoder/llm"
fi

# 3. byT5 tokenizer (google/byt5-small).
if [ "${SKIP_BYT5:-0}" != "1" ]; then
    echo "--- [3/5] byT5 tokenizer: $BYT5_REPO_ID ---"
    hf_get "$BYT5_REPO_ID" "$MODEL_DIR/text_encoder/byt5-small"
fi

# 4. byT5 weights via modelscope (Glyph-SDXL-v2). modelscope honors its own
#    proxy/SSL settings; if it fails, download manually from the modelscope URL.
if [ "${SKIP_GLYPH:-0}" != "1" ]; then
    echo "--- [4/5] byT5 weights (modelscope): $GLYPH_MS_ID ---"
    if ! python -c "import modelscope" 2>/dev/null; then
        echo "    [!] modelscope not installed — installing it first ..."
        pip install --trusted-host pypi.org --trusted-host files.pythonhosted.org \
            --timeout 600 --retries 10 modelscope || true
    fi
    if python -c "import modelscope" 2>/dev/null; then
        mkdir -p "$MODEL_DIR/text_encoder/Glyph-SDXL-v2"
        if ! modelscope download --model "$GLYPH_MS_ID" --local_dir "$MODEL_DIR/text_encoder/Glyph-SDXL-v2"; then
            echo "    [!] modelscope download failed (proxy/SSL). Manual fallback:" >&2
            echo "        open https://modelscope.cn/models/$GLYPH_MS_ID/files" >&2
            echo "        download checkpoints/byt5_model.pt (and assets/) into:" >&2
            echo "        $MODEL_DIR/text_encoder/Glyph-SDXL-v2/" >&2
        fi
    else
        echo "    [!] modelscope still unavailable — download manually from" >&2
        echo "        https://modelscope.cn/models/$GLYPH_MS_ID/files" >&2
        echo "        into: $MODEL_DIR/text_encoder/Glyph-SDXL-v2/" >&2
    fi
fi

# 5. Vision encoder (siglip, from gated FLUX.1-Redux-dev) — only needed for I2V.
#    Requires HF_TOKEN after you request access at the model page.
if [ "${SKIP_VISION_ENCODER:-0}" != "1" ]; then
    if [ -n "${HF_TOKEN:-}" ]; then
        echo "--- [5/5] vision encoder (gated): $SIGLIP_REPO_ID ---"
        hf_get "$SIGLIP_REPO_ID" "$MODEL_DIR/vision_encoder/siglip" "$HF_TOKEN"
    else
        echo "--- [5/5] SKIPPED vision encoder (no HF_TOKEN). ---"
        echo "    I2V needs black-forest-labs/FLUX.1-Redux-dev (gated):" >&2
        echo "      1) request access at https://huggingface.co/black-forest-labs/FLUX.1-Redux-dev" >&2
        echo "      2) HF_TOKEN=<your_token> bash $SCRIPT_DIR/01_download_models.sh" >&2
        echo "    T2V works without it." >&2
    fi
else
    echo "--- [5/5] SKIPPED vision encoder (SKIP_VISION_ENCODER=1). ---"
fi

echo "--- downloaded files (depth 2) ---"
find "$MODEL_DIR" -maxdepth 2 -mindepth 1 \( -type f -o -type d \) -printf '  %P\n' | sort | head -60

# Link ckpts -> shared model dir so the official generate.py / train.py (which
# load from ./ckpts/...) work without modifying any official code.
if [ -e "$CKPTS_LINK" ] && [ ! -L "$CKPTS_LINK" ]; then
    echo "WARNING: $CKPTS_LINK exists and is not a symlink; leaving it untouched." >&2
    echo "         Remove it if you want to use $MODEL_DIR." >&2
else
    [ -L "$CKPTS_LINK" ] && rm -f "$CKPTS_LINK"
    ln -s "$MODEL_DIR" "$CKPTS_LINK"
    echo "--- symlink: $CKPTS_LINK -> $MODEL_DIR"
fi

echo "=== [01] Done. Weights at: $MODEL_DIR (linked as $CKPTS_LINK) ==="

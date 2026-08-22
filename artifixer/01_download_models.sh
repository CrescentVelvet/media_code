#!/usr/bin/env bash
# 01_download_models.sh — download ArtiFixer checkpoint + Wan2.1 base model + MoGe + Qwen3-VL.
#
# Downloads to $MODEL_DIR (shared model root). All paths are local so inference
# never hits the network (HF_HUB_OFFLINE=1 in _env.sh).
#
# Required for inference:
#   - nvidia/ArtiFixer: artifixer-1.3b.pt (or -14b.pt)
#   - Wan-AI/Wan2.1-T2V-1.3B-Diffusers (transformer + VAE + scheduler + tokenizer + text_encoder)
#
# Required for data prep (02_prepare_data.sh):
#   - Qwen/Qwen3-VL-30B-A3B-Instruct (captioning, ~60 GB bf16 MoE)
#   - MoGe v2 weights (metric scale alignment)
set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/_env.sh"

# Unset HF_HUB_OFFLINE for downloads
export HF_HUB_OFFLINE=0

echo "🚀 [01] Download ArtiFixer weights"
echo "  📁 model dir:  $MODEL_DIR"
echo "  🤖 variant:    $MODEL_VARIANT"
echo "  🏋️ checkpoint: $ARTIFIXER_CHECKPOINT"
echo "  🤖 base model: $WAN_MODEL_ID  ($HF_MODEL_ID)"

PIP_FLAGS=(--trusted-host pypi.org --trusted-host pypi.python.org \
    --trusted-host files.pythonhosted.org --timeout 600 --retries 10)

# helper: download a HF repo to a local dir (idempotent)
hf_download() {
    local repo_id="$1"
    local local_dir="$2"
    local extra="${3:-}"
    if [ -d "$local_dir" ] && [ -n "$(ls -A "$local_dir" 2>/dev/null)" ]; then
        echo "  ⏭️  already downloaded: $repo_id -> $local_dir"
        return 0
    fi
    echo "  📦 downloading: $repo_id -> $local_dir"
    mkdir -p "$local_dir"
    # shellcheck disable=SC2086
    python -m huggingface_hub.commands.huggingface_cli download \
        "$repo_id" $extra --local-dir "$local_dir"
}

# helper: download a single file from a HF repo
hf_download_file() {
    local repo_id="$1"
    local filename="$2"
    local local_dir="$3"
    local target="$local_dir/$filename"
    if [ -f "$target" ]; then
        echo "  ⏭️  already exists: $target"
        return 0
    fi
    echo "  📦 downloading: $repo_id/$filename -> $target"
    mkdir -p "$local_dir"
    python -c "
from huggingface_hub import hf_hub_download
hf_hub_download('$repo_id', '$filename', local_dir='$local_dir')
"
}

# --- 1. ArtiFixer checkpoint ---
if [ "$MODEL_VARIANT" = "1.3b" ]; then
    hf_download_file "nvidia/ArtiFixer" "artifixer-1.3b.pt" "$MODEL_DIR/artifixer"
else
    hf_download_file "nvidia/ArtiFixer" "artifixer-14b.pt" "$MODEL_DIR/artifixer"
fi

# --- 2. Wan2.1 base model (diffusers format: transformer/ vae/ scheduler/ tokenizer/ text_encoder/) ---
hf_download "$HF_MODEL_ID" "$WAN_MODEL_ID"

# --- 3. MoGe v2 weights (for metric scale alignment in data prep) ---
if [ ! -d "$MOGE_MODEL_PATH" ] || [ -z "$(ls -A "$MOGE_MODEL_PATH" 2>/dev/null)" ]; then
    echo "📦 downloading MoGe v2 weights..."
    # MoGe weights are hosted on HuggingFace: microsoft/MoGe
    hf_download "microsoft/MoGe" "$MOGE_MODEL_PATH"
    # MoGe library looks for weights in MOGE_MODEL_PATH or downloads automatically
else
    echo "  ⏭️  MoGe weights already present: $MOGE_MODEL_PATH"
fi

# --- 4. Qwen3-VL captioning model (only needed for data prep; ~60 GB) ---
# Downloaded to HF cache (not --local-dir) because prepare_colmap_artifixer_inputs
# hardcodes the repo id "Qwen/Qwen3-VL-30B-A3B-Instruct" and from_pretrained checks the cache.
if [ "${SKIP_CAPTION_MODEL:-0}" = "1" ]; then
    echo "⏭️  skipping Qwen3-VL (SKIP_CAPTION_MODEL=1)"
else
    echo "📦 downloading Qwen3-VL to HF cache ($HF_HOME)..."
    # Check if already in cache
    QWEN_CACHE_DIR="$HF_HOME/hub/models--Qwen--Qwen3-VL-30B-A3B-Instruct"
    if [ -d "$QWEN_CACHE_DIR" ] && [ -n "$(ls -A "$QWEN_CACHE_DIR" 2>/dev/null)" ]; then
        echo "  ⏭️  already in HF cache: $QWEN_CACHE_DIR"
    else
        python -m huggingface_hub.commands.huggingface_cli download "$CAPTIONING_MODEL_ID"
    fi
fi

# --- 5. default_negative_prompt.pt (in ArtiFixer repo root, not a download) ---
if [ ! -f "$ARTIFIXER_DIR/default_negative_prompt.pt" ]; then
    echo "⚠️  default_negative_prompt.pt not found in $ARTIFIXER_DIR"
    echo "   (should be in the repo root after git clone)"
fi

# --- summary ---
echo ""
echo "✅ [01] Download summary:"
echo "  🏋️ checkpoint:  $ARTIFIXER_CHECKPOINT  $([ -f "$ARTIFIXER_CHECKPOINT" ] && echo '(present)' || echo '(MISSING)')"
echo "  🤖 base model:   $WAN_MODEL_ID  $([ -d "$WAN_MODEL_ID" ] && echo '(present)' || echo '(MISSING)')"
echo "  📐 MoGe:        $MOGE_MODEL_PATH  $([ -d "$MOGE_MODEL_PATH" ] && echo '(present)' || echo '(MISSING)')"
QWEN_CACHE_DIR="$HF_HOME/hub/models--Qwen--Qwen3-VL-30B-A3B-Instruct"
echo "  📝 Qwen3-VL:    HF cache $QWEN_CACHE_DIR  $([ -d "$QWEN_CACHE_DIR" ] && echo '(present)' || echo '(MISSING/skipped)')"
echo ""
echo "🎉 [01] Done."
echo "    Next: bash $SCRIPT_DIR/02_prepare_data.sh  (prepare scene data)"

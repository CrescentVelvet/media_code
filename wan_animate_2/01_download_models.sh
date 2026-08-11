#!/usr/bin/env bash
# 01_download_models.sh — download Wan2.2-Animate-2-14B weights to $MODEL_DIR and
# symlink $OFFICIAL_DIR/ckpts -> $CKPTS_DIR so the official YAML's ../ckpts/...
# relative paths (t5 / vae / clip / transformer) resolve unchanged.
#
# The HF/ModelScope repo Wan-AI/Wan2.2-Animate-2-14B contains the "ckpts/" tree:
#   wan_animate_2/wan_animate_2_bf16.safetensors               (base transformer)
#   wan_animate_2/wan_animate_2_bf16_distillation.safetensors  (distillation)
#   videomodel/Wan-AI/models_t5_umt5-xxl-enc-bf16.pth           (UMT5-XXL)
#   videomodel/Wan-AI/umt5-xxl/                                 (T5 tokenizer)
#   videomodel/Wan-AI/vae.pth                                   (VAE)
#   videomodel/Wan-AI/models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth  (CLIP)
#   videomodel/Wan-AI/xlm-roberta-large/                        (CLIP tokenizer)
set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/_env.sh"

MODEL_REPO="${MODEL_REPO:-Wan-AI/Wan2.2-Animate-2-14B}"
DOWNLOAD_SOURCE="${DOWNLOAD_SOURCE:-modelscope}"   # modelscope | HuggingFace

# Key files expected (relative to $CKPTS_DIR, matching the YAML ../ckpts/ layout).
EXPECTED=(
    "wan_animate_2/wan_animate_2_bf16.safetensors"
    "wan_animate_2/wan_animate_2_bf16_distillation.safetensors"
    "videomodel/Wan-AI/models_t5_umt5-xxl-enc-bf16.pth"
    "videomodel/Wan-AI/umt5-xxl"
    "videomodel/Wan-AI/vae.pth"
    "videomodel/Wan-AI/models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth"
    "videomodel/Wan-AI/xlm-roberta-large"
)

echo "=== [01] Download Wan-Animate-2 weights ==="
echo "  source:     $DOWNLOAD_SOURCE"
echo "  model repo: $MODEL_REPO"
echo "  -> $CKPTS_DIR  (symlinked to $OFFICIAL_DIR/ckpts)"

# Check if already downloaded (all key files present).
_all_present=1
for f in "${EXPECTED[@]}"; do
    [ -e "$CKPTS_DIR/$f" ] || _all_present=0
done

if [ "$_all_present" = "1" ]; then
    echo "⏭️  all key weights already present in $CKPTS_DIR — skipping download"
else
    mkdir -p "$CKPTS_DIR"
    if [ "$DOWNLOAD_SOURCE" = "HuggingFace" ]; then
        echo "📦 huggingface-cli download $MODEL_REPO --local-dir $CKPTS_DIR"
        huggingface-cli download "$MODEL_REPO" --local-dir "$CKPTS_DIR"
    else
        echo "📦 modelscope download --model $MODEL_REPO --local_dir $CKPTS_DIR"
        modelscope download --model "$MODEL_REPO" --local_dir "$CKPTS_DIR"
    fi
fi

# Symlink $OFFICIAL_DIR/ckpts -> $CKPTS_DIR so YAML ../ckpts/... resolves.
mkdir -p "$OFFICIAL_DIR"
if [ -L "$OFFICIAL_DIR/ckpts" ]; then
    echo "--- refreshing symlink $OFFICIAL_DIR/ckpts -> $CKPTS_DIR"
    ln -sfn "$CKPTS_DIR" "$OFFICIAL_DIR/ckpts"
elif [ -e "$OFFICIAL_DIR/ckpts" ]; then
    echo "⚠️ $OFFICIAL_DIR/ckpts already exists (not a symlink; left as-is)"
else
    echo "✅ creating symlink $OFFICIAL_DIR/ckpts -> $CKPTS_DIR"
    ln -sfn "$CKPTS_DIR" "$OFFICIAL_DIR/ckpts"
fi

# Verify.
echo "🔍 verify key weights"
_all_ok=1
for f in "${EXPECTED[@]}"; do
    if [ -e "$CKPTS_DIR/$f" ]; then
        size=$(du -h "$CKPTS_DIR/$f" 2>/dev/null | cut -f1 || echo "?")
        echo "  ✅ $(basename "$f")  ($size)"
    else
        echo "  ❌ MISSING: $f"
        _all_ok=0
    fi
done

if [ "$_all_ok" = "1" ]; then
    echo "=== [01] All weights present. Ready for inference. ==="
    echo "    Next: bash $SCRIPT_DIR/02_run_inference.sh"
else
    echo "" >&2
    echo "⚠️ some weights missing in $CKPTS_DIR." >&2
    echo "  Re-run: DOWNLOAD_SOURCE=modelscope bash $SCRIPT_DIR/01_download_models.sh" >&2
    echo "  or:    DOWNLOAD_SOURCE=HuggingFace bash $SCRIPT_DIR/01_download_models.sh" >&2
fi

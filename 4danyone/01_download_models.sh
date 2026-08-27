#!/usr/bin/env bash
# 01_download_models.sh — install 4DAnyone model weights.
#
# huggingface.co / hf-mirror.com are unreachable on this network, so weights
# come from (in priority order):
#   1. D:\wheel\4danyone_ms\   — manual downloads per download_urls.txt (fast)
#   2. modelscope.cn mirror   — auto-fetch any missing file (~230 KB/s, slow)
# Plus: BiRefNet foreground model (D:\wheel\birefnet\),
#        SMPL-X body model (D:\wheel\models_smplx_v1_1.zip, separately licensed).
#
# After files are in $MODEL_DIR, the official download_model.py is called to
# build the GVHMR compatibility symlinks (it detects files present → skips HF).
#
# Usage:
#   bash 4danyone/01_download_models.sh
#   SMPLX_ARCHIVE=/mnt/d/wheel/models_smplx_v1_1.zip bash 4danyone/01_download_models.sh
set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/_env.sh"

# Local manual-download locations (D:\wheel in Windows).
WHEELS_MS_DIR="${WHEELS_MS_DIR:-/mnt/d/wheel/4danyone_ms}"
BIREFNET_LOCAL="${BIREFNET_LOCAL:-/mnt/d/wheel/birefnet}"
SMPLX_ARCHIVE="${SMPLX_ARCHIVE:-/mnt/d/wheel/models_smplx_v1_1.zip}"
MS_BASE="https://modelscope.cn/models/AntResearch/4DAnyone/resolve/master"

# Model files (from fdanyone/assets.py MODEL_FILES).
MODEL_FILES=(
    "4danyone/model.safetensors"
    "4danyone/smplx_to_goliath70.pt"
    "4danyone/Wan2.2_VAE.pth"
    "4danyone/models_t5_umt5-xxl-enc-bf16.pth"
    "4danyone/umt5-xxl/special_tokens_map.json"
    "4danyone/umt5-xxl/spiece.model"
    "4danyone/umt5-xxl/tokenizer.json"
    "4danyone/umt5-xxl/tokenizer_config.json"
    "gvhmr/gvhmr_siga24_release.ckpt"
    "gvhmr/epoch=10-step=25000.ckpt"
    "gvhmr/vitpose-h-multi-coco.pth"
    "gvhmr/yolov8x.pt"
    "perceptual/imagenet-vgg-verydeep-19-conv.safetensors"
)
BIREFNET_FILES=("BiRefNet_config.py" "birefnet.py" "config.json" "model.safetensors")

echo "🚀 [01] Install 4DAnyone models"
echo "  🏋️ model_dir:     $MODEL_DIR"
echo "  📁 local cache:   $WHEELS_MS_DIR"
echo "  🔗 GVHMR:         $GVHMR_DIR"
echo "  🌐 modelscope:    $MS_BASE"
echo ""

if [ ! -f "$GVHMR_DIR/hmr4d/__init__.py" ]; then
    echo "❌ ERROR: GVHMR submodule missing at $GVHMR_DIR (run 00_setup_env first)" >&2
    exit 1
fi
mkdir -p "$MODEL_DIR"

# ---------------------------------------------------------------------------
# 1. Copy 4DAnyone model files from D:\wheel\4danyone_ms\ -> $MODEL_DIR.
# ---------------------------------------------------------------------------
if [ "$WHEELS_MS_DIR" = "$MODEL_DIR" ]; then
    echo "⏭️  MODEL_DIR == WHEELS_MS_DIR ($MODEL_DIR): using models in place (no copy)"
elif [ -d "$WHEELS_MS_DIR" ]; then
    _n=$(find "$WHEELS_MS_DIR" -type f 2>/dev/null | wc -l)
    echo "📦 copying $_n files from $WHEELS_MS_DIR -> $MODEL_DIR"
    cp -rn "$WHEELS_MS_DIR/." "$MODEL_DIR/" 2>/dev/null || \
        cp -r "$WHEELS_MS_DIR/." "$MODEL_DIR/"
else
    echo "⚠️ $WHEELS_MS_DIR not found — no local model cache."
    echo "   Pre-fetch weights per $SCRIPT_DIR/download_urls.txt (modelscope links)."
fi

# ---------------------------------------------------------------------------
# 2. Fetch any missing file from modelscope (~230 KB/s; resume with -C -).
# ---------------------------------------------------------------------------
_missing=()
for f in "${MODEL_FILES[@]}"; do
    if [ ! -f "$MODEL_DIR/$f" ]; then
        _missing+=("$f")
    fi
done
if [ ${#_missing[@]} -gt 0 ]; then
    echo ""
    echo "📦 ${#_missing[@]} file(s) missing — fetching from modelscope (slow, ~230 KB/s):"
    for f in "${_missing[@]}"; do
        echo "  ⬇️ $f"
        mkdir -p "$MODEL_DIR/$(dirname "$f")"
        if curl -L --fail --max-time 7200 -C - "$MS_BASE/$f" -o "$MODEL_DIR/$f"; then
            echo "    ✅ done"
        else
            echo "    ❌ failed — download manually from $MS_BASE/$f -> D:\\wheel\\4danyone_ms\\$f" >&2
        fi
    done
else
    echo "✅ all 4DAnyone model files present in $MODEL_DIR"
fi

# ---------------------------------------------------------------------------
# 3. BiRefNet foreground model from D:\wheel\birefnet\ -> $MODEL_DIR/birefnet/.
#    modelscope path for BiRefNet is unconfirmed; user must pre-fetch it.
# ---------------------------------------------------------------------------
echo ""
_biref_ok=true
if [ -d "$BIREFNET_LOCAL" ]; then
    mkdir -p "$MODEL_DIR/birefnet"
    cp -rn "$BIREFNET_LOCAL/." "$MODEL_DIR/birefnet/" 2>/dev/null || \
        cp -r "$BIREFNET_LOCAL/." "$MODEL_DIR/birefnet/"
    echo "📦 copied BiRefNet from $BIREFNET_LOCAL -> $MODEL_DIR/birefnet/"
fi
for bf in "${BIREFNET_FILES[@]}"; do
    if [ ! -f "$MODEL_DIR/birefnet/$bf" ]; then
        _biref_ok=false
        echo "  [MISS] birefnet/$bf"
    fi
done
if [ "$_biref_ok" = false ]; then
    echo "⚠️ BiRefNet incomplete. Pre-fetch from modelscope (search 'BiRefNet') or HF:" >&2
    echo "   https://modelscope.cn/models/ZhengPeng7/BiRefNet/files" >&2
    echo "   -> save the 4 files to D:\\wheel\\birefnet\\ then rerun" >&2
fi

# ---------------------------------------------------------------------------
# 4. Build GVHMR compatibility symlinks via official download_model.py.
#    With files present it only links (skips HF download). If files are
#    missing it will error — fix the missing ones above and rerun.
# ---------------------------------------------------------------------------
echo ""
echo "🔗 building GVHMR compatibility links..."
( cd "$FDANYONE_DIR" && python scripts/download_model.py \
    --model_dir "$MODEL_DIR" --gvhmr_root "$GVHMR_DIR" ) || \
    echo "⚠️ download_model.py failed — check missing files above" >&2

# ---------------------------------------------------------------------------
# 5. SMPL-X (separately licensed) from the user-provided zip.
# ---------------------------------------------------------------------------
echo ""
if [ -f "$SMPLX_PATH" ]; then
    echo "⏭️  SMPL-X already installed: $SMPLX_PATH"
elif [ -f "$SMPLX_ARCHIVE" ]; then
    echo "🏋️ installing SMPL-X from $SMPLX_ARCHIVE..."
    ( cd "$FDANYONE_DIR" && python scripts/download_smplx.py \
        --archive_path "$SMPLX_ARCHIVE" \
        --model_dir "$MODEL_DIR" --gvhmr_root "$GVHMR_DIR" ) || \
        echo "❌ SMPL-X install failed" >&2
else
    echo "⚠️ SMPL-X not installed."
    echo "   Register + accept license: https://smpl-x.is.tue.mpg.de/"
    echo "   Download models_smplx_v1_1.zip -> D:\\wheel\\"
    echo "   Then: SMPLX_ARCHIVE=/mnt/d/wheel/models_smplx_v1_1.zip bash $0"
fi

# ---------------------------------------------------------------------------
# 6. Summary.
# ---------------------------------------------------------------------------
echo ""
echo "--- [verify] key files ---"
for f in "4danyone/model.safetensors" "4danyone/Wan2.2_VAE.pth" \
         "4danyone/models_t5_umt5-xxl-enc-bf16.pth" "gvhmr/gvhmr_siga24_release.ckpt" \
         "birefnet/model.safetensors"; do
    if [ -f "$MODEL_DIR/$f" ]; then echo "  ✅ $f"; else echo "  [MISS] $f"; fi
done
[ -f "$SMPLX_PATH" ] && echo "  ✅ SMPL-X" || echo "  [MISS] SMPL-X"

echo ""
echo "🎉 [01] Done."
echo "  ⚠️ Inference needs ~43 GiB VRAM — run step 02 on a >=48 GiB GPU on the server."

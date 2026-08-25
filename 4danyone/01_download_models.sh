#!/usr/bin/env bash
# 01_download_models.sh — download the 4DAnyone model checkpoints from
# Hugging Face and install the separately-licensed SMPL-X body model.
#
# Downloads (from HF repo `AntResearch/4DAnyone`, public, not gated):
#   4danyone/model.safetensors            # main 4DAnyone checkpoint
#   4danyone/smplx_to_goliath70.pt        # SMPL-X -> MHR70 regressor
#   4danyone/Wan2.2_VAE.pth              # Wan2.2 VAE
#   4danyone/models_t5_umt5-xxl-enc-bf16.pth  # T5/UMT5-XXL text encoder (~5 GB)
#   4danyone/umt5-xxl/*                   # tokenizer (4 files)
#   gvhmr/*.ckpt / .pth / .pt            # GVHMR + HMR2 + ViTPose + YOLOv8x
#   perceptual/imagenet-vgg-verydeep-19-conv.safetensors
# Plus BiRefNet foreground model (from `ZhengPeng7/BiRefNet`).
# Then creates GVHMR compatibility symlinks (download_model.py does this).
#
# ⚠️  SMPL-X is NOT on Hugging Face — it needs a free account + license
#    acceptance at https://smpl-x.is.tue.mpg.de/. Download
#    models_smplx_v1_1.zip manually, then pass its path via SMPLX_ARCHIVE.
#
# Usage:
#   bash 4danyone/01_download_models.sh
#   SMPLX_ARCHIVE=/path/to/models_smplx_v1_1.zip bash 4danyone/01_download_models.sh
#   EXAMPLE=1 bash 4danyone/01_download_models.sh   # also fetch example clips
#
# If huggingface.co is unreachable, set in proxy.env:
#   export HF_ENDPOINT="https://hf-mirror.com"
set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/_env.sh"

echo "🚀 [01] Download 4DAnyone models"
echo "  🤖 repo:    $FDANYONE_DIR"
echo "  🏋️ model_dir: $MODEL_DIR"
echo "  🔗 GVHMR:   $GVHMR_DIR"
echo "  🌐 HF:      ${HF_ENDPOINT:-https://huggingface.co}"
echo ""

# Pre-check: GVHMR submodule must be initialized (download_model.py needs it
# to create the compatibility symlinks under $GVHMR_DIR/inputs/checkpoints/).
if [ ! -f "$GVHMR_DIR/hmr4d/__init__.py" ]; then
    echo "❌ ERROR: GVHMR submodule not initialized at $GVHMR_DIR" >&2
    echo "       Run first: INSTALL_DEPS=1 bash $SCRIPT_DIR/00_setup_env.sh" >&2
    exit 1
fi

mkdir -p "$MODEL_DIR"

# ---------------------------------------------------------------------------
# 1. 4DAnyone model checkpoints (HF snapshot) + BiRefNet + GVHMR links.
#    Delegates to the official scripts/download_model.py, pointing its
#    --model_dir / --gvhmr_root at our absolute paths (it defaults to the
#    relative "models" / "third_party/GVHMR" inside the repo).
# ---------------------------------------------------------------------------
echo "📦 downloading 4DAnyone checkpoints from HF (AntResearch/4DAnyone)..."
echo "  (snapshot_download with resume; large files — umt5-xxl enc ~5 GB)"
( cd "$FDANYONE_DIR" && python scripts/download_model.py \
    --model_dir "$MODEL_DIR" \
    --gvhmr_root "$GVHMR_DIR" )
if [ $? -ne 0 ]; then
    echo "❌ download_model.py failed" >&2
    echo "  Check network / HF_ENDPOINT. If partial, rerun — it resumes." >&2
    exit 1
fi
echo "✅ 4DAnyone checkpoints downloaded to $MODEL_DIR"

# Verify the key files landed where expected.
echo ""
echo "--- [verify] key model files ---"
for f in \
    "4danyone/model.safetensors" \
    "4danyone/Wan2.2_VAE.pth" \
    "4danyone/models_t5_umt5-xxl-enc-bf16.pth" \
    "gvhmr/gvhmr_siga24_release.ckpt" \
    "birefnet/model.safetensors"; do
    if [ -f "$MODEL_DIR/$f" ]; then
        echo "  ✅ $f"
    else
        echo "  [MISS] $f"
    fi
done

# ---------------------------------------------------------------------------
# 2. SMPL-X body model (separately licensed, NOT on HF).
#    User must download models_smplx_v1_1.zip from https://smpl-x.is.tue.mpg.de/
#    (register + accept license), then pass the zip via SMPLX_ARCHIVE.
#    download_smplx.py --archive_path installs it non-interactively.
# ---------------------------------------------------------------------------
echo ""
if [ -f "$SMPLX_PATH" ]; then
    echo "⏭️  SMPL-X already installed: $SMPLX_PATH"
else
    if [ -n "${SMPLX_ARCHIVE:-}" ] && [ -f "$SMPLX_ARCHIVE" ]; then
        echo "🏋️ installing SMPL-X from $SMPLX_ARCHIVE..."
        ( cd "$FDANYONE_DIR" && python scripts/download_smplx.py \
            --archive_path "$SMPLX_ARCHIVE" \
            --model_dir "$MODEL_DIR" \
            --gvhmr_root "$GVHMR_DIR" )
        if [ $? -ne 0 ] || [ ! -f "$SMPLX_PATH" ]; then
            echo "❌ SMPL-X install failed" >&2
            exit 1
        fi
        echo "✅ SMPL-X installed: $SMPLX_PATH"
    else
        echo "⚠️  SMPL-X is NOT installed (separately licensed, not on HF)."
        echo "   1. Register + accept license: https://smpl-x.is.tue.mpg.de/"
        echo "   2. Download models_smplx_v1_1.zip"
        echo "   3. Rerun: SMPLX_ARCHIVE=/path/to/models_smplx_v1_1.zip bash $0"
        echo "   (inference will fail without SMPL-X)"
    fi
fi

# ---------------------------------------------------------------------------
# 3. Optional: bundled example clips (8 pexels videos, for first test).
# ---------------------------------------------------------------------------
if [ "${EXAMPLE:-0}" = "1" ]; then
    echo ""
    echo "📦 downloading bundled example clips (EXAMPLE=1)..."
    _data_dir="$FDANYONE_DIR/data"
    mkdir -p "$_data_dir"
    ( cd "$FDANYONE_DIR" && python scripts/download_example.py --data_dir "$_data_dir" )
    if [ $? -ne 0 ]; then
        echo "  ⚠️ example download failed (non-fatal)" >&2
    else
        echo "✅ examples in $_data_dir/source/pexels/"
    fi
fi

echo ""
echo "🎉 [01] Done. Next:"
echo "  GPU=0 VIDEO_PATH=\$FDANYONE_DIR/data/source/pexels/2785536-uhd_2160_3840_25fps.mp4 \\"
echo "    RESULTS_DIR=\$RESULTS_DIR bash $SCRIPT_DIR/02_run_inference.sh"
echo "  ⚠️  Needs ~43 GiB VRAM — run on a >=48 GiB GPU on the server."

#!/usr/bin/env bash
# 01_segment_all.sh — segment the person in EVERY image of an orbit shoot onto a
# white background, producing the multi-view set PDF-GS needs.
#
# Unlike wan22_rotate step 01 (picks ONE best front-facing image), this keeps ALL
# views: PDF-GS triangulates the body from multiple viewpoints and uses their
# cross-view DINOv3-feature consistency to flag micro-motion (breathing / hair /
# clothing) pixels as distractors and filter them out. Fewer views = less to
# triangulate against = weaker distractor filtering.
#
# Segmentor: SAM2 automatic mask generation (largest salient mask = person) primary,
# rembg fallback. No SAM 3D Body / detectron2 / GATED weights — fully self-contained
# in the `pdfgs` env.
#
# Output: $SEGMENTED_DIR/<rel>.png (white-bg person, preserves input rel subpath).
#         This folder is the INPUT for step 02 (Pi3 + COLMAP).
#
# Prerequisites:
#   - INSTALL_DEPS=1 bash pdfgs_human/00_setup_env.sh (first time)
#
# Env (all optional, defaults shown):
#   INPUT_DIR=             # orbit shoot folder (with image/ subdir or raw images)
#   SEGMENTED_DIR=         # output (default: $RESULTS_DIR/segmented_frames)
#   SEGMENTOR=auto         # auto (SAM2 then rembg) | sam2 | rembg
#   WHITE_BG=1             # 1=white bg (matches PDF-GS --white_background); 0=black
#   MIN_MASK_FRAC=0.02     # reject masks < 2% of image (wrong object)
#   DEVICE=cuda            # SAM2 device; cpu works but slow
#   JPG_QUALITY=95         # only used if output ext is .jpg (we save .png by default)
set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/_env.sh"

INPUT_DIR="${INPUT_DIR:-}"
if [ -z "$INPUT_DIR" ]; then
    echo "❌ ERROR: INPUT_DIR not set (orbit shoot folder, with image/ subdir or raw images)." >&2
    echo "   e.g. INPUT_DIR=../Reconstruction/dataset/B003_Human_Data_w_pose/test_task_xxx \\" >&2
    echo "        bash pdfgs_human/01_segment_all.sh" >&2
    exit 1
fi
if [ ! -d "$INPUT_DIR" ]; then
    echo "❌ ERROR: INPUT_DIR not found: $INPUT_DIR" >&2
    exit 1
fi

SEGMENTED_DIR="${SEGMENTED_DIR:-$RESULTS_DIR/segmented_frames}"
SEGMENTOR="${SEGMENTOR:-auto}"
WHITE_BG="${WHITE_BG:-1}"
MIN_MASK_FRAC="${MIN_MASK_FRAC:-0.02}"
DEVICE="${DEVICE:-cuda}"
JPG_QUALITY="${JPG_QUALITY:-95}"

# Sanity: SAM2 OR rembg must be importable (00 should have installed at least one).
if ! python -c "from sam2 import build_sam2" 2>/dev/null && \
   ! python -c "import rembg" 2>/dev/null; then
    echo "❌ ERROR: neither sam2 nor rembg importable in env '$CONDA_ENV'." >&2
    echo "       Run: INSTALL_DEPS=1 bash $SCRIPT_DIR/00_setup_env.sh" >&2
    exit 1
fi

mkdir -p "$SEGMENTED_DIR"
export INPUT_DIR SEGMENTED_DIR SEGMENTOR WHITE_BG MIN_MASK_FRAC DEVICE JPG_QUALITY

echo "🚀 [01] segment all images (white-bg person, multi-view set for PDF-GS)"
echo "  📂 input:       $INPUT_DIR"
echo "  💾 output:       $SEGMENTED_DIR"
echo "  ✂️ segmentor:    $SEGMENTOR  (SAM2 primary, rembg fallback)"
echo "  🎨 white_bg:     $WHITE_BG  min_mask: ${MIN_MASK_FRAC}"
echo "  🎮 device:       $DEVICE"
echo ""

python "$SCRIPT_DIR/segment_all.py"
if [ $? -ne 0 ]; then
    echo "❌ FAILED. Segmentation did not complete." >&2
    exit 1
fi

echo ""
echo "🎉 [01] Done. Next:"
echo "  INPUT=$SEGMENTED_DIR bash $SCRIPT_DIR/02_pi3_colmap.sh"

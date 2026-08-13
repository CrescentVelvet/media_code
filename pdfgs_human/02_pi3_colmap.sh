#!/usr/bin/env bash
# 02_pi3_colmap.sh — Pi3 (π³, ICLR 2026) feed-forward pose estimation + dense
# point cloud, exported to COLMAP text format for PDF-GS training.
#
# One forward pass over the segmented frames -> per-view camera_poses (c2w,
# OpenCV) + dense cloud -> COLMAP source/ that PDF-GS's train.py -s reads:
#   source/images/<frame>.png        # the white-bg person images (copied)
#   source/sparse/0/cameras.txt      # PINHOLE intrinsics (fx=fy=max(W,H))
#   source/sparse/0/images.txt       # image_id, w2c (qw qx qy qz tx ty tz), name
#   source/sparse/0/points3D.txt     # sampled dense cloud w/ RGB (3DGS init)
#
# Why Pi3 (not COLMAP SfM): real orbit photos of a person may not SfM-match
# well (low-texture skin, white bg, repeated clothing patterns). Pi3 is a single
# feed-forward model — robust pose + dense cloud in one pass, no feature matching.
# Its poses + points come from the same model so are mutually consistent; PDF-GS
# only fixes poses+intrinsics and optimizes gaussians, so even approximate
# intrinsics produce a plausible reconstruction.
#
# Reuses pi3_3dgs/pi3_recon.py (shared helper) — WITHOUT --no_colmap, so the
# COLMAP text export runs (step 04 of wan22_rotate uses --no_colmap = pose only).
#
# Prerequisites:
#   - INSTALL_DEPS=1 bash pdfgs_human/00_setup_env.sh (first time; clones Pi3 + Dnlds ckpt)
#
# Env (all optional, defaults shown):
#   INPUT=                 # image folder (default: $RESULTS_DIR/segmented_frames)
#   OUTPUT_NAME=orbit      # base name (affects output dirs)
#   RESULTS_DIR=           # output root
#   PI3_DIR=               # Pi3 official repo (default: ../Pi3)
#   PI3_CKPT=              # Pi3 checkpoint (default: $MODEL_DIR/Pi3/model.safetensors)
#   PI3_OUTPUT_DIR=        # Pi3 output (default: $RESULTS_DIR/<name>/pi3)
#   SOURCE_DIR=            # COLMAP scene (default: $PI3_OUTPUT_DIR/source)
#   FRAME_FPS=10           # (image-folder input: ignored; copies all up to FRAME_MAX)
#   FRAME_MAX=60            # max frames (Pi3 VRAM scales ~linearly with N)
#   CONF_THRES=0.1         # sigmoid-conf threshold for filtering init points
#   DEVICE=cuda
#   SKIP_PI3=0              # 1 = reuse existing source/ (skip Pi3)
set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/_env.sh"

OUTPUT_NAME="${OUTPUT_NAME:-orbit}"
INPUT="${INPUT:-$RESULTS_DIR/segmented_frames}"
PI3_OUTPUT_DIR="${PI3_OUTPUT_DIR:-$RESULTS_DIR/$OUTPUT_NAME/pi3}"
SOURCE_DIR="${SOURCE_DIR:-$PI3_OUTPUT_DIR/source}"
FRAME_FPS="${FRAME_FPS:-10}"
FRAME_MAX="${FRAME_MAX:-60}"
CONF_THRES="${CONF_THRES:-0.1}"
DEVICE="${DEVICE:-cuda}"

echo "🚀 [02] Pi3 pose + COLMAP export"
echo "  🤖 Pi3 ckpt:     $PI3_CKPT"
echo "  📂 input:        $INPUT"
echo "  💾 Pi3 output:   $PI3_OUTPUT_DIR"
echo "  💾 COLMAP src:   $SOURCE_DIR"
echo "  📐 frame_max:    $FRAME_MAX  conf_thres: $CONF_THRES"
echo "  🎮 device:        $DEVICE"
echo ""

# ── 0. Sanity checks ──────────────────────────────────────────────────────
if [ ! -d "$INPUT" ]; then
    echo "❌ ERROR: input not found: $INPUT" >&2
    echo "       Run step 01 first: INPUT_DIR=... bash $SCRIPT_DIR/01_segment_all.sh" >&2
    exit 1
fi

# Pi3 repo (00 clones it; auto-clone here as a fallback, like wan22_rotate step 05a).
PI3_KEYFILE="$PI3_DIR/pi3/models/pi3.py"
if [ ! -f "$PI3_KEYFILE" ]; then
    if [ -d "$PI3_DIR" ]; then
        echo "❌ ERROR: $PI3_DIR exists but is incomplete (missing $PI3_KEYFILE)." >&2
        echo "       Please remove it: rm -rf $PI3_DIR" >&2
        exit 1
    fi
    echo "📦 Pi3 repo not found — cloning..."
    mkdir -p "$(dirname "$PI3_DIR")"
    git clone https://github.com/yyfz/Pi3.git "$PI3_DIR" || \
        git -c http.sslVerify=false clone https://github.com/yyfz/Pi3.git "$PI3_DIR"
fi
if [ ! -f "$PI3_CKPT" ]; then
    echo "❌ ERROR: Pi3 ckpt not found at $PI3_CKPT" >&2
    echo "       Download from https://huggingface.co/yyfz233/Pi3/resolve/main/model.safetensors" >&2
    echo "       Or re-run: INSTALL_DEPS=1 bash $SCRIPT_DIR/00_setup_env.sh" >&2
    exit 1
fi

# pi3_recon.py (shared helper in the sibling pi3_3dgs/ folder)
PI3_RECON_PY="$REPO_DIR/pi3_3dgs/pi3_recon.py"
if [ ! -f "$PI3_RECON_PY" ]; then
    echo "❌ ERROR: $PI3_RECON_PY not found." >&2
    echo "       pi3_3dgs/ must exist alongside pdfgs_human/ in media_code/." >&2
    exit 1
fi

# ── Pi3 inference + COLMAP export ──────────────────────────────────────────
if [ "${SKIP_PI3:-0}" = "1" ]; then
    echo "⏭️ skip Pi3 (SKIP_PI3=1, reusing existing source/)"
    if [ ! -d "$SOURCE_DIR/images" ]; then
        echo "❌ ERROR: $SOURCE_DIR/images not found — cannot skip Pi3 without existing COLMAP scene." >&2
        echo "       Run step 02 without SKIP_PI3 first." >&2
        exit 1
    fi
else
    echo "🔍 Pi3 inference + COLMAP export (~10-60s)"
    echo "  ✂️ COLMAP export: ENABLED (cameras/images/points3D.txt for PDF-GS)"
    echo ""
    python "$PI3_RECON_PY" \
        --input "$INPUT" \
        --output_dir "$PI3_OUTPUT_DIR" \
        --ckpt "$PI3_CKPT" \
        --pi3_dir "$PI3_DIR" \
        --device "$DEVICE" \
        --frame_fps "$FRAME_FPS" \
        --frame_max "$FRAME_MAX" \
        --conf_thres "$CONF_THRES"
    if [ $? -ne 0 ]; then
        echo "❌ FAILED. Pi3 inference did not complete." >&2
        exit 1
    fi
    echo "✅ Pi3 + COLMAP done"
    echo "  📁 source: $SOURCE_DIR/{images, sparse/0/}"
    echo ""
fi

if [ ! -d "$SOURCE_DIR/images" ] || [ ! -d "$SOURCE_DIR/sparse/0" ]; then
    echo "❌ ERROR: COLMAP scene not ready: $SOURCE_DIR" >&2
    echo "       Expected: $SOURCE_DIR/images/ + $SOURCE_DIR/sparse/0/*.txt" >&2
    exit 1
fi

echo ""
echo "🎉 [02] Done. Next:"
echo "  bash $SCRIPT_DIR/03_train_pdfgs.sh"

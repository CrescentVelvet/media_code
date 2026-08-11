#!/usr/bin/env bash
# 04_pi3_pose.sh — split video into JPG frames + run Pi3 pose estimation
# (NO 3D reconstruction — no COLMAP export, no 2DGS training).
#
# This step reuses the wan22_rotate conda env (which already has all Pi3 deps:
# torch 2.6.0+cu124, numpy 1.26.4, cv2, safetensors). plyfile is NOT needed
# (pi3_recon.py uses its own ASCII PLY writer).
# It does NOT need the pi3_3dgs env (which is only required for the 2DGS CUDA
# rasterizer extensions). It calls pi3_3dgs/pi3_recon.py with --no_colmap to
# skip the COLMAP text format export (that's only needed when feeding 2DGS).
#
# Workflow:
#   1. Ensure JPG frames exist under $RESULTS_DIR/<video_name>/image/ (run
#      step 03 if missing — the "split video into JPG frames" part).
#   2. Run Pi3 inference on those frames -> predictions.npz + dense_cloud.ply
#      + poses.json (human-readable c2w matrices).
#
# Output structure (under $RESULTS_DIR/<video_name>/pi3/):
#   $RESULTS_DIR/rotate_360/
#     image/                 # JPG frames (from step 03, or auto-extracted here)
#       00000.jpg, ...
#     pi3/                   # NEW: Pi3 outputs (step 04)
#       frames/              #   frames Pi3 actually used (copy of image/, or
#                            #     extracted from video if step 03 was skipped)
#       predictions.npz      #   raw Pi3 outputs (points, camera_poses, conf, ...)
#       dense_cloud.ply      #   confidence-filtered dense point cloud
#       poses.json           #   human-readable: frame_names + c2w matrices
#
# Env (all optional):
#   VIDEO_PATH=             # default $RESULTS_DIR/${OUTPUT_NAME:-rotate_360}.mp4
#   INPUT=                  # override the Pi3 input directly (video or image folder)
#                            #   default: $RESULTS_DIR/<video_name>/image/ if exists,
#                            #   else $VIDEO_PATH (will extract frames internally)
#   OUTPUT_DIR=              # default $RESULTS_DIR/<video_name>/pi3
#   FRAME_FPS=10             # frame sampling fps when extracting from video
#   FRAME_MAX=60             # max frames (avoid OOM; Pi3 scales ~linearly)
#   CONF_THRES=0.1           # sigmoid-conf threshold for dense cloud filter
#   PI3_DIR=                 # default ../../Pi3 (auto-cloned if missing)
#   PI3_CKPT=                # default $PI3_DIR/../model/Pi3/model.safetensors
#   DEVICE=cuda              # or cpu (slow)
set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/_env.sh"

OUTPUT_NAME="${OUTPUT_NAME:-rotate_360}"
VIDEO_PATH="${VIDEO_PATH:-$RESULTS_DIR/${OUTPUT_NAME}.mp4}"
# Default OUTPUT_DIR: $RESULTS_DIR/<video_name>/pi3 (organized under the video's folder)
OUTPUT_DIR="${OUTPUT_DIR:-$RESULTS_DIR/${OUTPUT_NAME}/pi3}"

# Pi3 repo + checkpoint (same defaults as pi3_3dgs/_env.sh — we don't source that
# file because it would activate the pi3_3dgs env, which we don't need here).
REPO_ROOT="$(dirname "$REPO_DIR")"
PI3_DIR="${PI3_DIR:-$REPO_ROOT/Pi3}"
PI3_CKPT="${PI3_CKPT:-$REPO_ROOT/model/Pi3/model.safetensors}"

echo "🚀 [04] Pi3 pose estimation (no 3D reconstruction)"
echo "  🤖 Pi3 ckpt:  $PI3_CKPT"
echo "  💾 output:    $OUTPUT_DIR"
echo ""

# ── 0. Sanity: Pi3 repo + ckpt ────────────────────────────────────────────
# Check for the key import file (pi3/models/pi3.py) rather than .git —
# handles partial/failed clones where the dir exists but is incomplete.
PI3_KEYFILE="$PI3_DIR/pi3/models/pi3.py"
if [ ! -f "$PI3_KEYFILE" ]; then
    if [ -d "$PI3_DIR" ]; then
        echo "❌ ERROR: $PI3_DIR exists but is incomplete (missing $PI3_KEYFILE)." >&2
        echo "       This is likely a broken/partial clone. Please manually remove it:" >&2
        echo "         rm -rf $PI3_DIR" >&2
        echo "       Then re-run this script to clone fresh." >&2
        exit 1
    fi
    echo "📦 Pi3 repo not found at $PI3_DIR — cloning..."
    mkdir -p "$(dirname "$PI3_DIR")"
    git clone https://github.com/yyfz/Pi3.git "$PI3_DIR" || \
        git -c http.sslVerify=false clone https://github.com/yyfz/Pi3.git "$PI3_DIR"
fi

if [ ! -f "$PI3_CKPT" ]; then
    echo "❌ ERROR: Pi3 checkpoint not found at $PI3_CKPT" >&2
    echo "       Download from https://huggingface.co/yyfz233/Pi3/resolve/main/model.safetensors" >&2
    echo "       Place at: $PI3_CKPT" >&2
    echo "       Or run: INSTALL_DEPS=1 bash pi3_3dgs/00_setup_env.sh" >&2
    exit 1
fi

# ── 1. Determine INPUT (image folder from step 03, or video) ──────────────
IMAGE_DIR="$RESULTS_DIR/${OUTPUT_NAME}/image"
if [ -z "${INPUT:-}" ]; then
    if [ -d "$IMAGE_DIR" ] && [ -n "$(ls -A "$IMAGE_DIR" 2>/dev/null)" ]; then
        # Step 03 already ran — reuse its JPGs.
        INPUT="$IMAGE_DIR"
        echo "📁 input: $INPUT (reusing step 03's JPGs)"
    elif [ -f "$VIDEO_PATH" ]; then
        # No JPGs yet — use the video; pi3_recon.py will extract frames internally.
        # But first, run step 03 to produce the canonical image/ folder (the user
        # asked for "split video into JPG frames" as part of this step).
        INPUT="$IMAGE_DIR"
        echo "📁 input: $INPUT (will run step 03 to extract JPGs first)"
        echo ""
        bash "$SCRIPT_DIR/03_extract_frames.sh"
        if [ $? -ne 0 ]; then
            echo "❌ [04] step 03 failed; cannot proceed without frames." >&2
            exit 1
        fi
        echo ""
    else
        echo "❌ ERROR: no input found. Set INPUT= (video or image folder) or VIDEO_PATH=" >&2
        echo "       Default VIDEO_PATH=$VIDEO_PATH also not found." >&2
        exit 1
    fi
else
    echo "📁 input: $INPUT (from env)"
fi

# ── 2. Verify deps are importable in the wan22_rotate env ─────────────────
# Pi3 deps: torch, numpy, cv2, safetensors (all in wan22_rotate env).
# plyfile is NOT needed — pi3_recon.py uses its own ASCII PLY writer.
if ! python - <<'PY'
import sys
for mod in ["torch", "numpy", "cv2", "safetensors"]:
    try:
        __import__(mod)
    except ImportError:
        print(f"missing: {mod}", file=sys.stderr)
        sys.exit(1)
print("Pi3 deps OK")
PY
then
    echo "❌ ERROR: missing Pi3 deps in env '$CONDA_ENV'." >&2
    echo "       Run: INSTALL_DEPS=1 bash $SCRIPT_DIR/00_setup_env.sh" >&2
    exit 1
fi

# ── 3. Run Pi3 inference via pi3_3dgs/pi3_recon.py --no_colmap ─────────────
# We call pi3_3dgs/pi3_recon.py directly (not via pi3_3dgs/01_pi3_recon.sh,
# which would source pi3_3dgs/_env.sh and activate the pi3_3dgs env — we want
# to stay in wan22_rotate env). The --no_colmap flag skips the COLMAP text
# format export + open3d voxel downsample (not needed without 2DGS).
PI3_RECON_PY="$REPO_DIR/pi3_3dgs/pi3_recon.py"
if [ ! -f "$PI3_RECON_PY" ]; then
    echo "❌ ERROR: $PI3_RECON_PY not found." >&2
    echo "       The pi3_3dgs/ folder must exist alongside wan22_rotate/." >&2
    exit 1
fi

mkdir -p "$OUTPUT_DIR"

echo ""
echo "🔍 [04] running Pi3 inference (this may take ~10-60s depending on N)"
echo "  📂 input:      $INPUT"
echo "  💾 output_dir: $OUTPUT_DIR"
echo "  📐 frame_fps:  ${FRAME_FPS:-10}  frame_max: ${FRAME_MAX:-60}  conf_thres: ${CONF_THRES:-0.1}"
echo "  ⏭️  COLMAP export: SKIPPED (--no_colmap; no 3D reconstruction)"
echo ""

python "$PI3_RECON_PY" \
    --input "$INPUT" \
    --output_dir "$OUTPUT_DIR" \
    --ckpt "$PI3_CKPT" \
    --pi3_dir "$PI3_DIR" \
    --device "${DEVICE:-cuda}" \
    --frame_fps "${FRAME_FPS:-10}" \
    --frame_max "${FRAME_MAX:-60}" \
    --conf_thres "${CONF_THRES:-0.1}" \
    --no_colmap

if [ $? -ne 0 ]; then
    echo "❌ [04] FAILED. Pi3 inference did not complete." >&2
    exit 1
fi

echo ""
echo "🎉 [04] Done. Pi3 pose estimation complete (no 3D reconstruction)."
echo "  📊 predictions.npz: $OUTPUT_DIR/predictions.npz  (raw Pi3 outputs)"
echo "  🌐 dense_cloud.ply: $OUTPUT_DIR/dense_cloud.ply  (inspect in MeshLab/SuperSplat)"
echo "  📝 poses.json:     $OUTPUT_DIR/poses.json       (human-readable c2w matrices)"
echo ""
echo "  To proceed with 3D reconstruction (2DGS), run:"
echo "    GPU=0 INPUT=$RESULTS_DIR/${OUTPUT_NAME} bash pi3_3dgs/run_all.sh"

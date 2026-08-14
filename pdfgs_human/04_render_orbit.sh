#!/usr/bin/env bash
# 04_render_orbit.sh — render a turntable / orbit video of the reconstructed
# person from the PDF-GS gaussians (step 03 output).
#
# 03_train_pdfgs.sh only renders the INPUT training views (reconstruction-vs-GT);
# this generates a NOVEL orbit camera path around the person and renders an mp4 —
# the standard showcase output for a human 3D-Gaussian reconstruction.
#
# Approach: load final-phase gaussians (point_cloud.ply) -> centroid=orbit center,
# PCA=person vertical axis (height), horizontal half-extent=orbit radius; FoV/size
# from COLMAP cameras.txt; per-angle MiniCam + render() -> mp4 + frames/. Runs
# fully in the pdfgs env (needs diff_gaussian_rasterization, same as 03).
#
# Prerequisites:
#   - step 03 done (final-phase gaussians exist)
#
# Env (all optional, defaults shown):
#   OUTPUT_NAME=orbit          # base name (must match step 02/03)
#   RESULTS_DIR=               # output root
#   SOURCE_DIR=                # COLMAP scene (default: $RESULTS_DIR/<name>/pi3/source)
#   GAUSSIAN_DIR=             # gaussians root (default: $RESULTS_DIR/<name>/model_pdfgs)
#   PHASE=4                    # which phase to render (final = NUM_PHASES from step 03)
#   ITER=                      # iteration to load (default: auto-detect max under phase_<PHASE>/point_cloud/)
#   ORBIT_FRAMES=120           # frames in the orbit video
#   ORBIT_TURNS=1.0            # full turns (1.0 = 360°, 2.0 = 720°)
#   ORBIT_RADIUS_MULT=3.5      # orbit radius = MULT × person horizontal half-extent (tune framing)
#   ORBIT_HEIGHT=0.0           # camera height offset along person up axis, × person half-height
#   UP_AXIS=                   # ""(auto via PCA) | x | y | z  (force the person's up axis)
#   WHITE_BG=1                 # 1=white bg (matches segmented input); 0=black
#   FPS=30                     # output mp4 fps
#   RES=                       # cap max edge (e.g. 1280); "" = camera native size
#   SH_DEGREE=3                # SH degree PDF-GS trained with (ModelParams default = 3)
#   DEVICE=cuda
set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/_env.sh"

OUTPUT_NAME="${OUTPUT_NAME:-orbit}"
SOURCE_DIR="${SOURCE_DIR:-$RESULTS_DIR/$OUTPUT_NAME/pi3/source}"
# ⚠️ _env.sh exports MODEL_DIR as the WEIGHT root; use GAUSSIAN_DIR for gaussians
#    output to avoid the name collision (same bug as step 03).
GAUSSIAN_DIR="${GAUSSIAN_DIR:-$RESULTS_DIR/$OUTPUT_NAME/model_pdfgs}"
PHASE="${PHASE:-4}"
ITER="${ITER:-}"
ORBIT_FRAMES="${ORBIT_FRAMES:-120}"
ORBIT_TURNS="${ORBIT_TURNS:-1.0}"
ORBIT_RADIUS_MULT="${ORBIT_RADIUS_MULT:-3.5}"
ORBIT_HEIGHT="${ORBIT_HEIGHT:-0.0}"
UP_AXIS="${UP_AXIS:-}"
WHITE_BG="${WHITE_BG:-1}"
FPS="${FPS:-30}"
RES="${RES:-}"
SH_DEGREE="${SH_DEGREE:-3}"
DEVICE="${DEVICE:-cuda}"

export OUTPUT_NAME SOURCE_DIR GAUSSIAN_DIR PHASE ITER ORBIT_FRAMES ORBIT_TURNS
export ORBIT_RADIUS_MULT ORBIT_HEIGHT UP_AXIS WHITE_BG FPS RES SH_DEGREE DEVICE

echo "🚀 [04] render orbit/turntable video from PDF-GS gaussians"
echo "  🏋️ gaussians:   $GAUSSIAN_DIR/phase_$PHASE/point_cloud/iteration_*/point_cloud.ply"
echo "  📂 source:      $SOURCE_DIR"
echo "  🎬 frames:      $ORBIT_FRAMES  turns: $ORBIT_TURNS  fps: $FPS"
echo "  📐 radius_mult: $ORBIT_RADIUS_MULT  height: $ORBIT_HEIGHT  up: ${UP_AXIS:-auto}"
echo "  🎨 white_bg:    $WHITE_BG  res: ${RES:-native}  sh: $SH_DEGREE"
if [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
    echo "  🎮 GPU:         physical $CUDA_VISIBLE_DEVICES"
else
    echo "  🎮 GPU:         default cuda:0  [set GPU=N to pin]"
fi
echo ""

# ── Sanity checks ─────────────────────────────────────────────────────────
if [ ! -f "$PDFGS_DIR/render.py" ]; then
    echo "❌ ERROR: PDF-GS repo not found at $PDFGS_DIR (no render.py)." >&2
    echo "       Run: INSTALL_DEPS=1 bash $SCRIPT_DIR/00_setup_env.sh" >&2
    exit 1
fi
if [ ! -d "$SOURCE_DIR/sparse/0" ]; then
    echo "❌ ERROR: COLMAP scene not ready: $SOURCE_DIR" >&2
    echo "       Run step 02 first: bash $SCRIPT_DIR/02_pi3_colmap.sh" >&2
    exit 1
fi
_ply_dir="$GAUSSIAN_DIR/phase_$PHASE/point_cloud"
if [ ! -d "$_ply_dir" ]; then
    echo "❌ ERROR: final-phase gaussians not found under $_ply_dir." >&2
    echo "       Run step 03 first: bash $SCRIPT_DIR/03_train_pdfgs.sh" >&2
    exit 1
fi
if ! python -c "import diff_gaussian_rasterization" 2>/dev/null; then
    echo "❌ ERROR: diff_gaussian_rasterization not importable (PDF-GS CUDA ext)." >&2
    echo "       Run: INSTALL_DEPS=1 bash $SCRIPT_DIR/00_setup_env.sh (needs nvcc)" >&2
    exit 1
fi

# Run inside the PDF-GS repo for relative imports (scene/, gaussian_renderer/,
# arguments/, utils/) — same as 03.
( cd "$PDFGS_DIR" && python "$SCRIPT_DIR/render_orbit.py" )
if [ $? -ne 0 ]; then
    echo "❌ FAILED. Orbit render did not complete." >&2
    exit 1
fi

echo ""
echo "🎉 [04] Done. Next (mesh):"
echo "  v1 has NO mesh (PDF-GS ships no extract_mesh). For a mesh run"
echo "  wan22_rotate step 05/05a/05b, or add a TSDF-on-depth step (future)."

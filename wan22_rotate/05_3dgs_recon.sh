#!/usr/bin/env bash
# 05_3dgs_recon.sh — 3D Gaussian Splatting reconstruction (Pi3+COLMAP → 2DGS train → render+mesh)
#
# Runs ENTIRELY in the wan22_rotate env (same as steps 01-04). No separate
# pi3_3dgs env needed. The 2DGS CUDA extensions (diff-surfel-rasterization +
# simple-knn) are compiled into the wan22_rotate env by 00_setup_env.sh
# (INSTALL_2DGS=1) — reuses the same gxx_linux-64=12 already installed for
# detectron2, plus the system CUDA toolkit (nvcc).
#
# Pipeline (3 sub-steps):
#   5a) Pi3 inference + COLMAP export (pi3_recon.py WITHOUT --no_colmap)
#       — same as step 04 but WITH cameras/images/points3D.txt export
#   5b) 2DGS training (train.py: gaussians init from COLMAP points → optimize)
#   5c) Render + TSDF mesh (render.py: render cameras + extract mesh)
#
# This is equivalent to pi3_3dgs/run_all.sh but uses wan22_rotate env.
# The pi3_recon.py script is shared (lives in pi3_3dgs/).
#
# Prerequisites:
#   - INSTALL_DEPS=1 INSTALL_2DGS=1 bash wan22_rotate/00_setup_env.sh (first time)
#   - Pi3 repo + checkpoint (same as step 04)
#
# Env (all optional, defaults shown):
#   INPUT=                 # video or image folder (default: $RESULTS_DIR/rotate_360.mp4)
#   OUTPUT_NAME=rotate_360  # video base name (affects default INPUT + output dirs)
#   RESULTS_DIR=            # output root (default: ../wan22_rotate_results)
#   WHITE_BG=1              # 1=white bg training (wan22_rotate input is white-bg)
#   UNBOUNDED=1             # 1=unbounded TSDF (person in white void)
#   MESH_RES=2048           # TSDF voxel resolution
#   ITERATIONS=30000        # 2DGS training steps (7000=quick demo)
#   FRAME_FPS=10            # video frame sampling fps (Pi3)
#   FRAME_MAX=60            # max frames (Pi3 VRAM scales ~linearly)
#   CONF_THRES=0.1          # Pi3 confidence threshold
#   SKIP_PI3=0              # 1=skip 5a (reuse existing COLMAP source/)
#   SKIP_TRAIN=0            # 1=skip 5b (reuse existing model/)
#   SKIP_RENDER=0           # 1=skip 5c
#   DEVICE=cuda
set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/_env.sh"

# ── Paths ─────────────────────────────────────────────────────────────────
REPO_ROOT="$(dirname "$REPO_DIR")"
PI3_DIR="${PI3_DIR:-$REPO_ROOT/Pi3}"
PI3_CKPT="${PI3_CKPT:-$REPO_ROOT/model/Pi3/model.safetensors}"
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"

OUTPUT_NAME="${OUTPUT_NAME:-rotate_360}"
INPUT="${INPUT:-$RESULTS_DIR/${OUTPUT_NAME}.mp4}"
# Pi3 + COLMAP output (same dir as step 04, but with source/ added)
PI3_OUTPUT_DIR="${PI3_OUTPUT_DIR:-$RESULTS_DIR/${OUTPUT_NAME}/pi3}"
SOURCE_DIR="${SOURCE_DIR:-$PI3_OUTPUT_DIR/source}"
# 2DGS training output
MODEL_DIR="${MODEL_DIR:-$RESULTS_DIR/${OUTPUT_NAME}/model_2dgs}"

# ── Params ────────────────────────────────────────────────────────────────
WHITE_BG="${WHITE_BG:-1}"
UNBOUNDED="${UNBOUNDED:-1}"
MESH_RES="${MESH_RES:-2048}"
ITERATIONS="${ITERATIONS:-30000}"
FRAME_FPS="${FRAME_FPS:-10}"
FRAME_MAX="${FRAME_MAX:-60}"
CONF_THRES="${CONF_THRES:-0.1}"
DEVICE="${DEVICE:-cuda}"

echo "🚀 [05] 3DGS reconstruction (Pi3+COLMAP → 2DGS train → render+mesh)"
echo "  🤖 Pi3 ckpt:    $PI3_CKPT"
echo "  📂 input:       $INPUT"
echo "  💾 Pi3 output:  $PI3_OUTPUT_DIR"
echo "  💾 2DGS model:  $MODEL_DIR"
echo "  📐 mesh_res:    $MESH_RES  unbounded: $UNBOUNDED  white_bg: $WHITE_BG"
echo "  🏋️ iterations:  $ITERATIONS"
echo ""

# ── 0. Sanity checks ──────────────────────────────────────────────────────
# Pi3 repo + ckpt
PI3_KEYFILE="$PI3_DIR/pi3/models/pi3.py"
if [ ! -f "$PI3_KEYFILE" ]; then
    if [ -d "$PI3_DIR" ]; then
        echo "❌ ERROR: $PI3_DIR exists but is incomplete (missing $PI3_KEYFILE)." >&2
        echo "       Please manually remove it: rm -rf $PI3_DIR" >&2
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
    exit 1
fi

# 2DGS repo + CUDA extensions
if [ ! -f "$GS2D_DIR/train.py" ]; then
    echo "❌ ERROR: 2DGS repo not found at $GS2D_DIR" >&2
    echo "       Run: INSTALL_DEPS=1 INSTALL_2DGS=1 bash $SCRIPT_DIR/00_setup_env.sh" >&2
    exit 1
fi
if ! python -c "import simple_knn, diff_surfel_rasterization" 2>/dev/null; then
    echo "❌ ERROR: 2DGS CUDA extensions not importable (simple_knn / diff_surfel_rasterization)." >&2
    echo "       Run: INSTALL_DEPS=1 INSTALL_2DGS=1 bash $SCRIPT_DIR/00_setup_env.sh" >&2
    exit 1
fi

# pi3_recon.py (shared with pi3_3dgs)
PI3_RECON_PY="$REPO_DIR/pi3_3dgs/pi3_recon.py"
if [ ! -f "$PI3_RECON_PY" ]; then
    echo "❌ ERROR: $PI3_RECON_PY not found (pi3_3dgs/ must exist alongside wan22_rotate/)." >&2
    exit 1
fi

# ── 5a) Pi3 inference + COLMAP export ─────────────────────────────────────
if [ "${SKIP_PI3:-0}" = "1" ]; then
    echo "⏭️ [skip 5a] SKIP_PI3=1 (reusing existing source/)"
    if [ ! -d "$SOURCE_DIR/images" ]; then
        echo "❌ ERROR: $SOURCE_DIR/images not found — cannot skip 5a without existing COLMAP scene." >&2
        exit 1
    fi
else
    echo "🔍 [5a] Pi3 inference + COLMAP export (~10-60s)"
    echo "  📂 input:      $INPUT"
    echo "  💾 output_dir:  $PI3_OUTPUT_DIR"
    echo "  📐 frame_fps:  $FRAME_FPS  frame_max: $FRAME_MAX  conf_thres: $CONF_THRES"
    echo "  ✂️ COLMAP export: ENABLED (cameras/images/points3D.txt for 2DGS)"
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
        echo "❌ [5a] FAILED. Pi3 inference did not complete." >&2
        exit 1
    fi
    echo "✅ [5a] Pi3 + COLMAP done"
    echo "  📁 source: $SOURCE_DIR/{images, sparse/0/}"
    echo ""
fi

# ── 5b) 2DGS training ─────────────────────────────────────────────────────
if [ "${SKIP_TRAIN:-0}" = "1" ]; then
    echo "⏭️ [skip 5b] SKIP_TRAIN=1 (reusing existing model_2dgs/)"
else
    if [ ! -d "$SOURCE_DIR/images" ] || [ ! -d "$SOURCE_DIR/sparse/0" ]; then
        echo "❌ ERROR: source dir not ready: $SOURCE_DIR" >&2
        echo "       Expected: $SOURCE_DIR/images/ + $SOURCE_DIR/sparse/0/*.txt" >&2
        exit 1
    fi
    mkdir -p "$MODEL_DIR"
    echo "🏋️ [5b] 2DGS training ($ITERATIONS iterations)"
    echo "  📂 source:  $SOURCE_DIR"
    echo "  💾 model:   $MODEL_DIR"
    echo "  🎨 white_bg: $WHITE_BG"
    echo ""
    TRAIN_FLAGS=(
        -s "$SOURCE_DIR"
        -m "$MODEL_DIR"
        --iterations "$ITERATIONS"
        --port 0
    )
    [ "$WHITE_BG" = "1" ] && TRAIN_FLAGS+=(--white_background)
    if [ -n "${EXTRA_ARGS:-}" ]; then
        # shellcheck disable=SC2206
        TRAIN_FLAGS+=($EXTRA_ARGS)
    fi
    # Run inside 2DGS repo for relative imports (scene/, gaussian_renderer/, arguments/)
    ( cd "$GS2D_DIR" && python train.py "${TRAIN_FLAGS[@]}" )
    if [ $? -ne 0 ]; then
        echo "❌ [5b] FAILED. 2DGS training did not complete." >&2
        exit 1
    fi
    echo "✅ [5b] 2DGS training done"
    echo "  📁 gaussians: $MODEL_DIR/point_cloud/iteration_$ITERATIONS/point_cloud.ply"
    echo ""
fi

# ── 5c) Render + mesh ────────────────────────────────────────────────────
if [ "${SKIP_RENDER:-0}" = "1" ]; then
    echo "⏭️ [skip 5c] SKIP_RENDER=1"
else
    if [ ! -f "$MODEL_DIR/cfg_args" ]; then
        echo "❌ ERROR: trained model not found at $MODEL_DIR (no cfg_args)." >&2
        echo "       Run 5b first or: SKIP_PI3=1 SKIP_TRAIN=0 bash $0" >&2
        exit 1
    fi
    echo "🎬 [5c] render + mesh (TSDF fusion)"
    echo "  💾 model:     $MODEL_DIR"
    echo "  📐 mesh_res:  $MESH_RES  unbounded: $UNBOUNDED"
    echo ""
    RENDER_FLAGS=(
        -s "$SOURCE_DIR"
        -m "$MODEL_DIR"
        --iteration -1
        --depth_ratio 0
        --mesh_res "$MESH_RES"
        --skip_train
    )
    [ "$UNBOUNDED" = "1" ] && RENDER_FLAGS+=(--unbounded)
    if [ -n "${EXTRA_ARGS:-}" ]; then
        # shellcheck disable=SC2206
        RENDER_FLAGS+=($EXTRA_ARGS)
    fi
    ( cd "$GS2D_DIR" && python render.py "${RENDER_FLAGS[@]}" )
    if [ $? -ne 0 ]; then
        echo "❌ [5c] FAILED. Render did not complete." >&2
        exit 1
    fi
    # Find the actual iteration dir
    ITER_STR="$(ls -d "$MODEL_DIR/point_cloud/iteration_"* 2>/dev/null \
        | sort -V | tail -1 | sed 's/.*iteration_//')"
    echo "✅ [5c] render + mesh done"
    echo "  🖼️  renders:  $MODEL_DIR/test/ours_$ITER_STR/renders/*.png"
    echo "  🌐 mesh:     $MODEL_DIR/test/ours_$ITER_STR/mesh.ply"
    echo ""
fi

echo "🎉 [05] Done. 3DGS reconstruction complete."
echo "  📊 Pi3 + COLMAP:  $PI3_OUTPUT_DIR/{predictions.npz, source/}"
echo "  🏋️ 2DGS model:    $MODEL_DIR/point_cloud/iteration_*/point_cloud.ply"
echo "  🌐 Mesh:          $MODEL_DIR/test/ours_*/mesh.ply"
echo ""
echo "  Inspect mesh: meshlab $MODEL_DIR/test/ours_*/mesh.ply"
echo "  Renders:      $MODEL_DIR/test/ours_*/renders/*.png (2DGS 原生渲染)"

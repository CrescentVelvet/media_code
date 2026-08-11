#!/usr/bin/env bash
# 05a_3dgs_recon.sh — Gaussian Opacity Fields (GOF) reconstruction
# (Pi3+COLMAP → GOF train → Marching Tetrahedra mesh)
#
# 与 05_3dgs_recon.sh (2DGS) 并列的替代方案。GOF 用 Marching Tetrahedra 提网格
# (非 TSDF fusion), 网格质量在多数 benchmark 上超过 2DGS; 同时继承 Mip-Splatting
# 的抗锯齿滤波, 外观质量也优于原版 3DGS。
#
# 全程在 wan22_rotate env 里跑 (与 01-04 同 env)。首次需 INSTALL_GOF=1 编 3 个
# CUDA/C++ 扩展 (diff-gaussian-rasterization + simple-knn + tetra-triangulation),
# 复用 2DGS 已装的 gxx_linux-64=12 + 系统 CUDA toolkit (nvcc)。
#
# 与 05 (2DGS) 的区别:
#   - 用 diff-gaussian-rasterization (3DGS 光栅化器) 而非 diff-surfel-rasterization
#   - 用 GOF 的 extract_mesh.py (Marching Tetrahedra) 而非 2DGS 的 render.py (TSDF)
#   - 输出网格: model_gof/test/ours_<N>/fusion/mesh_binary_search_7.ply
#   - 共用 Pi3+COLMAP 的 source/ (若 05 已跑过, 设 SKIP_PI3=1 复用)
#
# Prerequisites:
#   - INSTALL_DEPS=1 INSTALL_GOF=1 bash wan22_rotate/00_setup_env.sh (first time)
#   - Pi3 repo + checkpoint (same as step 04/05)
#
# Env (all optional, defaults shown):
#   INPUT=                 # video or image folder (default: $RESULTS_DIR/rotate_360.mp4)
#   OUTPUT_NAME=rotate_360  # video base name (affects default INPUT + output dirs)
#   RESULTS_DIR=            # output root (default: ../wan22_rotate_results)
#   WHITE_BG=1              # 1=white bg training (wan22_rotate input is white-bg)
#   FILTER_MESH=1           # 1=filter mesh by gaussian size (recommended)
#   TEXTURE_MESH=1          # 1=extract vertex colors (colored mesh)
#   NEAR=0.02               # near plane for tetrahedra generation
#   FAR=1000000             # far plane (1e6 = unbounded, good for white-bg person)
#   ITERATIONS=30000        # GOF training steps (7000=quick demo)
#   FRAME_FPS=10            # video frame sampling fps (Pi3)
#   FRAME_MAX=60            # max frames (Pi3 VRAM scales ~linearly)
#   CONF_THRES=0.1          # Pi3 confidence threshold
#   SKIP_PI3=0              # 1=skip Pi3+COLMAP (reuse existing source/)
#   SKIP_TRAIN=0            # 1=skip GOF training (reuse existing model_gof/)
#   SKIP_MESH=0             # 1=skip mesh extraction
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
# Pi3 + COLMAP output (SAME as step 05 — can share source/ between 2DGS and GOF)
PI3_OUTPUT_DIR="${PI3_OUTPUT_DIR:-$RESULTS_DIR/${OUTPUT_NAME}/pi3}"
SOURCE_DIR="${SOURCE_DIR:-$PI3_OUTPUT_DIR/source}"
# GOF training output (SEPARATE from 2DGS's model/ to avoid clashing)
MODEL_DIR="${MODEL_DIR:-$RESULTS_DIR/${OUTPUT_NAME}/model_gof}"

# ── Params ────────────────────────────────────────────────────────────────
WHITE_BG="${WHITE_BG:-1}"
FILTER_MESH="${FILTER_MESH:-1}"
TEXTURE_MESH="${TEXTURE_MESH:-1}"
NEAR="${NEAR:-0.02}"
# 1e6 = unbounded (GOF default; person in white void)
FAR="${FAR:-1000000}"
ITERATIONS="${ITERATIONS:-30000}"
FRAME_FPS="${FRAME_FPS:-10}"
FRAME_MAX="${FRAME_MAX:-60}"
CONF_THRES="${CONF_THRES:-0.1}"
DEVICE="${DEVICE:-cuda}"

echo "🚀 [05a] GOF reconstruction (Pi3+COLMAP → GOF train → Marching Tetrahedra mesh)"
echo "  🤖 Pi3 ckpt:     $PI3_CKPT"
echo "  📂 input:        $INPUT"
echo "  💾 Pi3 output:   $PI3_OUTPUT_DIR"
echo "  💾 GOF model:    $MODEL_DIR"
echo "  📐 near/far:     $NEAR / $FAR  filter: $FILTER_MESH  texture: $TEXTURE_MESH"
echo "  🎨 white_bg:    $WHITE_BG"
echo "  🏋️ iterations:   $ITERATIONS"
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

# GOF repo + CUDA extensions
if [ ! -f "$GOF_DIR/train.py" ]; then
    echo "❌ ERROR: GOF repo not found at $GOF_DIR" >&2
    echo "       Run: INSTALL_DEPS=1 INSTALL_GOF=1 bash $SCRIPT_DIR/00_setup_env.sh" >&2
    exit 1
fi
if ! python -c "import simple_knn, diff_gaussian_rasterization" 2>/dev/null; then
    echo "❌ ERROR: GOF CUDA extensions not importable (simple_knn / diff_gaussian_rasterization)." >&2
    echo "       Run: INSTALL_DEPS=1 INSTALL_GOF=1 bash $SCRIPT_DIR/00_setup_env.sh" >&2
    exit 1
fi
# tetra-triangulation (needed for extract_mesh.py / Marching Tetrahedra)
if ! python -c "from tetranerf.utils.extension import cpp" 2>/dev/null; then
    echo "⚠️ WARNING: tetra-triangulation not importable — mesh extraction will fail." >&2
    echo "       Re-run: INSTALL_DEPS=1 INSTALL_GOF=1 bash $SCRIPT_DIR/00_setup_env.sh" >&2
    echo "       (if build fails on OptiX, see 00_setup_env.sh error message for manual fix)" >&2
fi

# pi3_recon.py (shared with pi3_3dgs)
PI3_RECON_PY="$REPO_DIR/pi3_3dgs/pi3_recon.py"
if [ ! -f "$PI3_RECON_PY" ]; then
    echo "❌ ERROR: $PI3_RECON_PY not found (pi3_3dgs/ must exist alongside wan22_rotate/)." >&2
    exit 1
fi

# ── Pi3 inference + COLMAP export ─────────────────────────────────────────
if [ "${SKIP_PI3:-0}" = "1" ]; then
    echo "⏭️ skip Pi3 (SKIP_PI3=1, reusing existing source/)"
    if [ ! -d "$SOURCE_DIR/images" ]; then
        echo "❌ ERROR: $SOURCE_DIR/images not found — cannot skip Pi3 without existing COLMAP scene." >&2
        echo "       Run step 05 or 05a without SKIP_PI3 first, or run step 04 then set SOURCE_DIR." >&2
        exit 1
    fi
else
    echo "🔍 Pi3 inference + COLMAP export (~10-60s)"
    echo "  📂 input:      $INPUT"
    echo "  💾 output_dir:  $PI3_OUTPUT_DIR"
    echo "  📐 frame_fps:  $FRAME_FPS  frame_max: $FRAME_MAX  conf_thres: $CONF_THRES"
    echo "  ✂️ COLMAP export: ENABLED (cameras/images/points3D.txt for GOF)"
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

# ── GOF training ───────────────────────────────────────────────────────────
if [ "${SKIP_TRAIN:-0}" = "1" ]; then
    echo "⏭️ skip GOF training (SKIP_TRAIN=1, reusing existing model_gof/)"
else
    if [ ! -d "$SOURCE_DIR/images" ] || [ ! -d "$SOURCE_DIR/sparse/0" ]; then
        echo "❌ ERROR: source dir not ready: $SOURCE_DIR" >&2
        echo "       Expected: $SOURCE_DIR/images/ + $SOURCE_DIR/sparse/0/*.txt" >&2
        exit 1
    fi
    mkdir -p "$MODEL_DIR"
    echo "🏋️ GOF training ($ITERATIONS iterations)"
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
    # Run inside GOF repo for relative imports (scene/, gaussian_renderer/, arguments/)
    ( cd "$GOF_DIR" && python train.py "${TRAIN_FLAGS[@]}" )
    if [ $? -ne 0 ]; then
        echo "❌ FAILED. GOF training did not complete." >&2
        exit 1
    fi
    echo "✅ GOF training done"
    echo "  📁 gaussians: $MODEL_DIR/point_cloud/iteration_$ITERATIONS/point_cloud.ply"
    echo ""
fi

# ── GOF mesh extraction (Marching Tetrahedra) ─────────────────────────────
if [ "${SKIP_MESH:-0}" = "1" ]; then
    echo "⏭️ skip mesh extraction (SKIP_MESH=1)"
else
    if [ ! -f "$MODEL_DIR/cfg_args" ]; then
        echo "❌ ERROR: trained model not found at $MODEL_DIR (no cfg_args)." >&2
        echo "       Run training first or: SKIP_PI3=1 SKIP_TRAIN=0 bash $0" >&2
        exit 1
    fi
    echo "🎬 GOF mesh extraction (Marching Tetrahedra + binary search)"
    echo "  💾 model:     $MODEL_DIR"
    echo "  📐 near/far:  $NEAR / $FAR"
    echo "  ✂️ filter:    $FILTER_MESH  texture: $TEXTURE_MESH"
    echo ""
    MESH_FLAGS=(
        -m "$MODEL_DIR"
        --iteration "$ITERATIONS"
        --near "$NEAR"
        --far "$FAR"
    )
    [ "$FILTER_MESH" = "1" ] && MESH_FLAGS+=(--filter_mesh)
    [ "$TEXTURE_MESH" = "1" ] && MESH_FLAGS+=(--texture_mesh)
    if [ -n "${MESH_EXTRA_ARGS:-}" ]; then
        # shellcheck disable=SC2206
        MESH_FLAGS+=($MESH_EXTRA_ARGS)
    fi
    ( cd "$GOF_DIR" && python extract_mesh.py "${MESH_FLAGS[@]}" )
    if [ $? -ne 0 ]; then
        echo "❌ FAILED. GOF mesh extraction did not complete." >&2
        echo "       Common cause: tetra-triangulation not built. Re-run INSTALL_GOF=1." >&2
        exit 1
    fi
    echo "✅ GOF mesh extraction done"
    echo "  🌐 mesh:  $MODEL_DIR/test/ours_$ITERATIONS/fusion/mesh_binary_search_7.ply"
    echo ""
fi

echo "🎉 [05a] Done. GOF reconstruction complete."
echo "  📊 Pi3 + COLMAP:  $PI3_OUTPUT_DIR/{predictions.npz, source/}"
echo "  🏋️ GOF model:     $MODEL_DIR/point_cloud/iteration_*/point_cloud.ply"
echo "  🌐 Mesh:           $MODEL_DIR/test/ours_*/fusion/mesh_binary_search_7.ply"
echo ""
echo "  Inspect mesh: meshlab $MODEL_DIR/test/ours_*/fusion/mesh_binary_search_7.ply"
echo "  Or web:       https://playcanvas.com/supersplat/editor (drag .ply)"
echo ""
echo "  💡 GOF vs 2DGS: GOF 用 Marching Tetrahedra 提网格 (非 TSDF),"
echo "     网格质量在多数 benchmark 上超过 2DGS。对比: 05_3dgs_recon.sh (2DGS) 的"
echo "     model/test/ours_*/mesh.ply"

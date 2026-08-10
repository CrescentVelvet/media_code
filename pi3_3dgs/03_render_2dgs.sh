#!/usr/bin/env bash
# 03_render_2dgs.sh — render the trained 2DGS scene + extract mesh.
#
# 2DGS's render.py (same source as vanilla 3DGS) renders the train/test cameras
# and saves them to <model>/test/ours_<iter>/ and <model>/train/ours_<iter>/.
# When --skip_train --skip_test --mesh_res <N> is set, it instead runs TSDF
# fusion to extract a mesh (the 2DGS headline feature — geometrically accurate
# surfaces). Use --unbounded for the unbounded TSDF variant (handles scenes
# extending to infinity, e.g. a person in a white-bg void).
#
# Env (all optional, defaults shown):
#   MODEL_DIR=             # trained-model dir (env: $RESULTS_DIR/model)
#   SOURCE_DIR=            # COLMAP source dir (env: $RESULTS_DIR/source)
#   ITERATION=-1           # -1 = latest saved point_cloud; else use that step
#   MESH_RES=1024          # TSDF voxel resolution (mesh quality vs memory)
#   UNBOUNDED=0            # 1 = unbounded TSDF (recommended for orbit-around-person
#                            #     in white-bg void; 0 = bounded TSDF)
#   DEPTH_RATIO=0          # 0 = mean depth (default; works for most), 1 = median depth
#   SKIP_TRAIN=1          # 1 = skip rendering train cameras (faster; renders test only)
#   SKIP_TEST=0           # 1 = skip rendering test cameras (then mesh only)
#   EXTRA_ARGS=""          # forwarded to render.py (e.g. --voxel_size 0.01)
set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/_env.sh"

MODEL_DIR="${MODEL_DIR:-$RESULTS_DIR/model}"
SOURCE_DIR="${SOURCE_DIR:-$RESULTS_DIR/source}"
ITERATION="${ITERATION:--1}"
MESH_RES="${MESH_RES:-1024}"
UNBOUNDED="${UNBOUNDED:-0}"
DEPTH_RATIO="${DEPTH_RATIO:-0}"
SKIP_TRAIN="${SKIP_TRAIN:-1}"
SKIP_TEST="${SKIP_TEST:-0}"

if [ ! -d "$MODEL_DIR" ] || [ ! -f "$MODEL_DIR/cfg_args" ]; then
    echo "ERROR: trained model not found at $MODEL_DIR (no cfg_args)" >&2
    echo "       Run 02_train_2dgs.sh first:  bash $SCRIPT_DIR/02_train_2dgs.sh" >&2
    exit 1
fi
if [ ! -f "$GS2D_DIR/render.py" ]; then
    echo "ERROR: 2DGS render.py not found at $GS2D_DIR/render.py" >&2
    echo "       Run INSTALL_DEPS=1 bash $SCRIPT_DIR/00_setup_env.sh first" >&2
    exit 1
fi
python -c "import simple_knn, diff_surfel_rasterization" 2>/dev/null || {
    echo "ERROR: 2DGS CUDA extensions not importable." >&2
    echo "       Run: BUILD_CUDA=1 bash $SCRIPT_DIR/00_setup_env.sh" >&2
    exit 1
}

RENDER_FLAGS=(
    -s "$SOURCE_DIR"
    -m "$MODEL_DIR"
    --iteration "$ITERATION"
    --depth_ratio "$DEPTH_RATIO"
    --mesh_res "$MESH_RES"
)
[ "$UNBOUNDED" = "1" ]  && RENDER_FLAGS+=(--unbounded)
[ "$SKIP_TRAIN" = "1" ] && RENDER_FLAGS+=(--skip_train)
[ "$SKIP_TEST"  = "1" ] && RENDER_FLAGS+=(--skip_test)
if [ -n "${EXTRA_ARGS:-}" ]; then
    # shellcheck disable=SC2206
    RENDER_FLAGS+=($EXTRA_ARGS)
fi

echo "=== [03] Render 2DGS (renders + mesh) ==="
echo "  model:        $MODEL_DIR  (iter=$ITERATION)"
echo "  source:       $SOURCE_DIR"
echo "  mesh_res:     $MESH_RES  unbounded: $UNBOUNDED  depth_ratio: $DEPTH_RATIO"
echo "  skip_train:   $SKIP_TRAIN  skip_test: $SKIP_TEST"
echo "  cmd:          python render.py ${RENDER_FLAGS[*]}"
echo ""

# Run inside the 2DGS repo (same as 02) so its relative imports resolve.
( cd "$GS2D_DIR" && python render.py "${RENDER_FLAGS[@]}" )

echo ""
# Figure out the actual iteration rendered (so we can point at the right dir).
ITER_STR="$ITERATION"
if [ "$ITERATION" = "-1" ]; then
    # -1 means latest; pick the highest iteration_* dir under point_cloud/.
    ITER_STR="$(ls -d "$MODEL_DIR/point_cloud/iteration_"* 2>/dev/null \
        | sort -V | tail -1 | sed 's/.*iteration_//')"
fi

echo "=== [03] Done. ==="
echo "    Gaussians:  $MODEL_DIR/point_cloud/iteration_$ITER_STR/point_cloud.ply"
echo "    Test renders: $MODEL_DIR/test/ours_$ITER_STR/renders/*.png"
echo "    Mesh:        $MODEL_DIR/test/$ITER_STR/mesh.ply   (or under ours_$ITER_STR/mesh.ply)"
echo ""
echo "    Tip: To inspect the .ply / mesh in a GUI:"
echo "      - SuperSplat (web, no install): https://playcanvas.com/supersplat/editor"
echo "      - MeshLab:    meshlab $MODEL_DIR/test/.../mesh.ply"
echo "      - SIBR viewer (pre-built Windows): https://github.com/RongLiu-Leo/Gaussian-Splatting-Monitor"

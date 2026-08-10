#!/usr/bin/env bash
# run_all.sh — one-click: clone Pi3 + 2DGS -> verify env (deps + CUDA exts) ->
# run Pi3 (pose + dense cloud -> COLMAP format) -> train 2DGS -> render + mesh.
#
# The full pipeline takes an input video (or folder of images) and produces a
# trained 2DGS scene (point_cloud.ply + renders + mesh). The default target is
# the wan22_rotate pipeline's rotate_360.mp4 — a 360° orbit around a person
# (white-bg segmented), which Pi3 can pose-estimate well and 2DGS can fit into
# a clean surface mesh.
#
# Set INPUT to your video (or image folder). Other env vars are documented in
# the per-step README sections (FRAME_FPS, ITERATIONS, MESH_RES, ...).
#
# Example (continue from wan22_rotate):
#   GPU=0 INPUT=../wan22_rotate_results/rotate_360.mp4 \
#     bash pi3_3dgs/run_all.sh
#
# Example (your own orbit video or image folder):
#   GPU=0 INPUT=/path/to/orbit.mp4 ITERATIONS=30000 bash pi3_3dgs/run_all.sh
set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"

# Optional proxy (must be set before git clone in 00).
if [ -f "$REPO_DIR/proxy.env" ]; then
    set -a; # shellcheck disable=SC1090
    source "$REPO_DIR/proxy.env"; set +a
fi

# Corporate proxy TLS workaround for git clone.
SYS_CA=/etc/ssl/certs/ca-certificates.crt
if [ -f "$SYS_CA" ]; then
    : "${GIT_SSL_CAINFO:=$SYS_CA}"
    export GIT_SSL_CAINFO
fi

# Defaults: take wan22_rotate's rotate_360.mp4 as input if not specified.
DEFAULT_INPUT="$REPO_DIR/../wan22_rotate_results/rotate_360.mp4"
INPUT="${INPUT:-$DEFAULT_INPUT}"

echo "=== [run_all] Pi3 + 2D Gaussian Splatting one-click pipeline ==="
echo "  conda env: ${CONDA_ENV:-pi3_3dgs}"
echo "  input:     $INPUT"
echo "  results:   ${RESULTS_DIR:-$REPO_DIR/../pi3_3dgs_results}"
echo ""

# 0. Setup: clone repos, install deps, build CUDA exts (first run takes ~10-30min).
#    Idempotent — re-running with deps present just verifies.
echo "=== [0/3] setup env (clone Pi3+2DGS, install deps, build CUDA) ==="
INSTALL_DEPS="${INSTALL_DEPS:-1}" BUILD_CUDA="${BUILD_CUDA:-1}" \
    bash "$SCRIPT_DIR/00_setup_env.sh" || {
        echo "ERROR during 00_setup_env.sh; see log above." >&2
        exit 1
    }

# 1. Pi3 reconstruction (extract frames -> Pi3 inference -> COLMAP text format).
echo ""
echo "=== [1/3] Pi3 reconstruction -> COLMAP ==="
bash "$SCRIPT_DIR/01_pi3_recon.sh" || {
    echo "ERROR during 01_pi3_recon.sh; see log above." >&2
    exit 1
}

# 2. Train 2D Gaussian Splatting on the COLMAP scene.
echo ""
echo "=== [2/3] train 2DGS ==="
bash "$SCRIPT_DIR/02_train_2dgs.sh" || {
    echo "ERROR during 02_train_2dgs.sh; see log above." >&2
    exit 1
}

# 3. Render + extract mesh.
echo ""
echo "=== [3/3] render + mesh ==="
bash "$SCRIPT_DIR/03_render_2dgs.sh" || {
    echo "ERROR during 03_render_2dgs.sh; see log above." >&2
    exit 1
}

echo ""
echo "=== [run_all] All steps finished. ==="
RESULTS_DIR="${RESULTS_DIR:-$REPO_DIR/../pi3_3dgs_results}"
echo "    Input:           $INPUT"
echo "    Pi3 predictions: $RESULTS_DIR/predictions.npz"
echo "    Dense cloud:      $RESULTS_DIR/dense_cloud.ply"
echo "    COLMAP source:    $RESULTS_DIR/source/  (sparse/0/*.txt + images/)"
echo "    Trained 2DGS:     $RESULTS_DIR/model/point_cloud/iteration_*/point_cloud.ply"
echo "    Renders + mesh:   $RESULTS_DIR/model/test/ours_*/"
echo ""
echo "    Variants:"
echo "      Shorter demo train:  ITERATIONS=7000 bash $SCRIPT_DIR/run_all.sh"
echo "      Higher-res mesh:     MESH_RES=2048 bash $SCRIPT_DIR/03_render_2dgs.sh"
echo "      Unbounded mesh:      UNBOUNDED=1 bash $SCRIPT_DIR/03_render_2dgs.sh"
echo "      White-bg input:      WHITE_BG=1 bash $SCRIPT_DIR/02_train_2dgs.sh  # retrain"
echo "      More frames:         FRAME_MAX=120 FRAME_FPS=5 bash $SCRIPT_DIR/run_all.sh"

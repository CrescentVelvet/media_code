#!/usr/bin/env bash
# run_all.sh — one-click: clone official repo (recursive, w/ submodules) ->
# verify env (+deps +CUDA exts) -> download D-NeRF dataset -> train one scene ->
# render test split + compute PSNR/SSIM/LPIPS.
#
# Deformable-3D-Gaussians is a TRAINING-based dynamic-3DGS method: there are
# no pretrained weights, each scene is fit from scratch. So run_all trains a
# short run on one D-NeRF scene (default `hook`, 7000 iters — a few minutes)
# then renders + scores it. Bump ITERATIONS=40000 to reproduce the paper's
# numbers; the checkpoint lands in $DG_DIR/output/<scene>/.
#
# Uses the existing conda env (torch preinstalled); set INSTALL_DEPS=1 once to
# install requirements.txt, and BUILD_CUDA=1 once to build the two CUDA
# extensions (depth-diff-gaussian-rasterization + simple-knn).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"

# Optional proxy (must be set before git clone).
if [ -f "$REPO_DIR/proxy.env" ]; then
    set -a; # shellcheck disable=SC1090
    source "$REPO_DIR/proxy.env"; set +a
fi

# --- Corporate proxy TLS interception workaround (git clone over HTTPS) ---
SYS_CA=/etc/ssl/certs/ca-certificates.crt
if [ -f "$SYS_CA" ]; then
    : "${GIT_SSL_CAINFO:=$SYS_CA}"
    export GIT_SSL_CAINFO
fi

DG_DIR="${DG_DIR:-$REPO_DIR/../Deformable-3D-Gaussians}"
DG_REPO="${DG_REPO:-https://github.com/ingra14m/Deformable-3D-Gaussians.git}"
MODEL_DIR="${MODEL_DIR:-$REPO_DIR/../../model/deformable-3d-gaussians}"
DNERF_DIR="${DNERF_DIR:-$MODEL_DIR/data/D-NeRF}"

# Which scene to train + render. Default `hook` (a standard fast D-NeRF scene).
# Use SCENE=lego for the paper's adjusted Lego (val-as-test).
SCENE="${SCENE:-hook}"
# Short train for the one-click demo; set ITERATIONS=40000 to reproduce paper.
ITERATIONS="${ITERATIONS:-7000}"
IS_BLENDER="${IS_BLENDER:-1}"   # D-NeRF is synthetic Blender; set 0 for NeRF-DS/HyperNeRF
DG_OUTPUT_ROOT="${DG_OUTPUT_ROOT:-$DG_DIR/output}"
MODEL_PATH="${MODEL_PATH:-$DG_OUTPUT_ROOT/$SCENE}"
# -s path passed to train.py. Defaults to the D-NeRF scene; for NeRF-DS/HyperNeRF
# set DNERF_DIR=/path/to/<dataset> SCENE=<scene> (and SKIP_DATA=1, since 01 only
# fetches D-NeRF). Override outright with SOURCE_PATH.
SOURCE_PATH="${SOURCE_PATH:-$DNERF_DIR/$SCENE}"
SKIP_DATA="${SKIP_DATA:-0}"     # 1 = skip the D-NeRF download (you bring your own data)

echo "=== [run_all] Deformable-3D-Gaussians one-click pipeline (conda env: ${CONDA_ENV:-deformable_gaussians}) ==="

# 0. Clone the official repo (recursive -> fetches the two CUDA submodules).
#    simple-knn lives on gitlab.inria.fr; depth-diff-gaussian-rasterization on
#    github (branch filter-norm). A recursive clone over HTTPS can fail behind
#    a TLS-intercepting proxy -> retry with sslVerify=false.
if [ ! -d "$DG_DIR" ]; then
    mkdir -p "$(dirname "$DG_DIR")"
    echo "--- cloning official repo (recursive) -> $DG_DIR ---"
    git clone --recursive "$DG_REPO" "$DG_DIR" || \
        git -c http.sslVerify=false clone --recursive "$DG_REPO" "$DG_DIR"
else
    echo "--- official repo already present: $DG_DIR ---"
fi

# Ensure submodules are checked out (a plain clone or a failed recursive fetch
# leaves submodules/ empty, which breaks the BUILD_CUDA step).
if [ ! -f "$DG_DIR/submodules/simple-knn/setup.py" ] || \
   [ ! -f "$DG_DIR/submodules/depth-diff-gaussian-rasterization/setup.py" ]; then
    echo "--- ensuring submodules are initialized ---"
    ( cd "$DG_DIR" && git submodule update --init --recursive ) || \
        ( cd "$DG_DIR" && git -c http.sslVerify=false submodule update --init --recursive )
fi

export DG_DIR MODEL_DIR DNERF_DIR

# First-time setup. Deformable-GS pins torch==1.13.1+cu116 (python 3.7) that
# conflict with other algos — use a dedicated env (CONDA_ENV=deformable_gaussians).
if [ "${INSTALL_DEPS:-1}" = "1" ]; then
    INSTALL_DEPS=1 bash "$SCRIPT_DIR/00_setup_env.sh"
else
    bash "$SCRIPT_DIR/00_setup_env.sh"
fi
# Build the CUDA rasterizer + simple-knn on the first run_all (needed for train + render).
if [ "${BUILD_CUDA:-1}" = "1" ]; then
    BUILD_CUDA=1 bash "$SCRIPT_DIR/00_setup_env.sh"
fi

# Download the D-NeRF dataset (adjusted, Deformable-GS release asset). Skip
# when you bring your own data (NeRF-DS/HyperNeRF, placed under DNERF_DIR).
if [ "$SKIP_DATA" != "1" ]; then
    bash "$SCRIPT_DIR/01_download_models.sh"
else
    echo "--- SKIP_DATA=1 -> skipping D-NeRF download (using $SOURCE_PATH) ---"
fi

# Pick a scene: if the requested -s path isn't present, fall back to the first
# available D-NeRF scene under DNERF_DIR (only meaningful for the D-NeRF flow).
if [ ! -d "$SOURCE_PATH" ]; then
    FIRST_SCENE="$(find "$DNERF_DIR" -maxdepth 2 -name transforms_train.json -printf '%h\n' 2>/dev/null \
        | head -1 | xargs basename 2>/dev/null || true)"
    if [ -n "$FIRST_SCENE" ]; then
        echo "--- requested '$SCENE' not found; using '$FIRST_SCENE' ---"
        SCENE="$FIRST_SCENE"
        SOURCE_PATH="$DNERF_DIR/$SCENE"
        MODEL_PATH="$DG_OUTPUT_ROOT/$SCENE"
    else
        echo "ERROR: scene data not found: $SOURCE_PATH" >&2
        echo "       (for D-NeRF run 01 first; for other datasets set DNERF_DIR + SCENE)" >&2
        exit 1
    fi
fi

# Train the scene. D-NeRF (Blender) needs --is_blender; NeRF-DS/HyperNeRF use
# --iterations 20000 and omit --is_blender. --eval splits train/test so metrics
# can score the held-out test cameras.
TRAIN_FLAGS=(-s "$SOURCE_PATH" -m "$MODEL_PATH" --eval --iterations "$ITERATIONS")
if [ "$IS_BLENDER" = "1" ]; then TRAIN_FLAGS+=(--is_blender); fi
if [ "${IS_6DOF:-0}" = "1" ]; then TRAIN_FLAGS+=(--is_6dof); fi
echo "--- train.py ${TRAIN_FLAGS[*]} ---"
( cd "$DG_DIR" && python train.py "${TRAIN_FLAGS[@]}" )

# Render the test split + compute PSNR/SSIM/LPIPS.
MODEL_PATH="$MODEL_PATH" SCENE="$SCENE" bash "$SCRIPT_DIR/02_run_inference.sh"

echo "=== [run_all] All steps finished. ==="
echo "    Trained scene:  $MODEL_PATH  (iteration $ITERATIONS)"
echo "    Renders:        $MODEL_PATH/test/ours_$ITERATIONS/renders/"
echo "    Metrics:        $MODEL_PATH/test/results.json"
echo "    To reproduce paper numbers: ITERATIONS=40000 bash $SCRIPT_DIR/run_all.sh"
echo "    To render a different mode (time-interp video):"
echo "      MODEL_PATH=$MODEL_PATH MODE=time bash $SCRIPT_DIR/02_run_inference.sh"
echo "    To train another scene:"
echo "      SCENE=lego bash $SCRIPT_DIR/run_all.sh"

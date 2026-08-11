#!/usr/bin/env bash
# run_all.sh — one-click: clone EVA-Gaussian -> verify env -> prepare data ->
# pretrain depthnet (stage 1) -> train full EVA-Gaussian (stage 2).
#
# EVA-Gaussian: 3D Gaussian-Based Real-time Human Novel View Synthesis Under
# Diverse Camera Settings (Hu et al., arXiv:2410.01425). Feed-forward stereo
# depth estimation -> Gaussian parameter prediction -> differentiable splatting
# -> feature refinement, all in one network trained end-to-end.
#
# The full pipeline needs ~25 GB GPU memory at batch_size=1 (28 GB GPU minimum).
#
# Example:
#   GPU=0 DATA_ROOT=/path/to/thuman2.0_rendered bash eva_gaussian/run_all.sh
#
# With anchor loss (needs landmark.json; run 04_gen_landmarks.sh first):
#   GPU=0 ANCHOR=1 DATA_ROOT=/path/to/dataset bash eva_gaussian/run_all.sh
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

DATA_ROOT="${DATA_ROOT:?❌ DATA_ROOT env var required (dataset in GPS-Gaussian format)}"

echo "=== [run_all] EVA-Gaussian one-click training pipeline ==="
echo "  conda env:  ${CONDA_ENV:-eva_gaussian}"
echo "  DATA_ROOT:  $DATA_ROOT"
echo "  ANCHOR:     ${ANCHOR:-0}"
echo "  results:    ${RESULTS_DIR:-$REPO_DIR/../eva_gaussian_results}"
echo ""

# 0. Setup: clone repo, install deps, build CUDA rasterizer (first run ~10-30min).
echo "=== [0/3] setup env (clone EVA-Gaussian, install deps, build CUDA) ==="
INSTALL_DEPS="${INSTALL_DEPS:-1}" BUILD_CUDA="${BUILD_CUDA:-1}" \
    bash "$SCRIPT_DIR/00_setup_env.sh" || {
        echo "❌ during 00_setup_env.sh; see log above." >&2
        exit 1
    }

# 1. Prepare data: check dataset structure + generate configs.
echo ""
echo "=== [1/3] prepare data + generate configs ==="
bash "$SCRIPT_DIR/01_prepare_data.sh" || {
    echo "❌ during 01_prepare_data.sh; see log above." >&2
    exit 1
}

# 2. Stage 1: pretrain depthnet.
echo ""
echo "=== [2/3] pretrain depthnet (stage 1) ==="
bash "$SCRIPT_DIR/02_pretrain_depth.sh" || {
    echo "❌ during 02_pretrain_depth.sh; see log above." >&2
    exit 1
}

# 3. Stage 2: train full EVA-Gaussian.
echo ""
echo "=== [3/3] train EVA-Gaussian (stage 2) ==="
bash "$SCRIPT_DIR/03_train.sh" || {
    echo "❌ during 03_train.sh; see log above." >&2
    exit 1
}

echo ""
echo "=== [run_all] All steps finished. ==="
RESULTS_DIR="${RESULTS_DIR:-$REPO_DIR/../eva_gaussian_results}"
echo "    Experiments:  $RESULTS_DIR/experiments/"
echo "    TensorBoard:  tensorboard --logdir $RESULTS_DIR/experiments"
echo ""
echo "    Variants:"
echo "      Anchor loss:      GPU=0 ANCHOR=1 DATA_ROOT=... bash $SCRIPT_DIR/run_all.sh"
echo "      Fewer steps:      NUM_STEPS=50000 GPU=0 DATA_ROOT=... bash $SCRIPT_DIR/run_all.sh"
echo "      Resume stage 2:   RESUME_CKPT=<ckpt> STAGE1_CKPT=<s1> DATA_ROOT=... bash $SCRIPT_DIR/03_train.sh"

#!/usr/bin/env bash
# run_all.sh — one-click: clone official repo -> verify env (+deps +CUDA exts) ->
# weights (+SMPL-X check) -> inference. Inference needs YOUR prepared dataset,
# so run_all stops before 02 unless DATA_DIR is set (prints the command to run).
# Uses the existing conda env (torch preinstalled); set INSTALL_DEPS=1 once to
# install requirements.txt, and BUILD_CUDA=1 once to build the two CUDA
# extensions (diff-gaussian-rasterization + StyleUNet) the avatar net needs.
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

F3G_DIR="${F3G_DIR:-$REPO_DIR/../F3G-avatar}"
F3G_REPO="${F3G_REPO:-https://github.com/wjmenu/F3G-avatar.git}"

echo "=== [run_all] F3G-Avatar one-click pipeline (conda env: ${CONDA_ENV:-f3g_avatar}) ==="

# 0. Clone the official repo if it is not present yet.
if [ ! -d "$F3G_DIR" ]; then
    mkdir -p "$(dirname "$F3G_DIR")"
    echo "--- cloning official repo -> $F3G_DIR ---"
    git clone "$F3G_REPO" "$F3G_DIR" || \
        git -c http.sslVerify=false clone "$F3G_REPO" "$F3G_DIR"
else
    echo "--- official repo already present: $F3G_DIR ---"
fi

export F3G_DIR

# First-time setup. F3G pins torch==2.1.0+cu121 / pytorch3d / torch-scatter /
# triton that conflict with other algos — use a dedicated env (CONDA_ENV=f3g_avatar).
# INSTALL_DEPS=1 installs requirements.txt; BUILD_CUDA=1 builds the CUDA exts.
if [ "${INSTALL_DEPS:-0}" = "1" ]; then
    INSTALL_DEPS=1 bash "$SCRIPT_DIR/00_setup_env.sh"
else
    bash "$SCRIPT_DIR/00_setup_env.sh"
fi
# Build CUDA extensions on the first run_all (needed for inference).
if [ "${BUILD_CUDA:-1}" = "1" ]; then
    BUILD_CUDA=1 bash "$SCRIPT_DIR/00_setup_env.sh"
fi
bash "$SCRIPT_DIR/01_download_models.sh"

# Inference needs a prepared dataset (DATA_DIR) + checkpoint + SMPL-X. The
# released checkpoint is "coming soon" on HF — if absent, 01 prints train-from-
# scratch instructions. Only run 02 when the caller has data ready.
if [ -n "${DATA_DIR:-}" ] && [ -f "${PREV_CKPT:-$REPO_DIR/../../model/F3G-avatar/avatarrex_zzr/net.pt}" ]; then
    bash "$SCRIPT_DIR/02_run_inference.sh"
else
    cat <<EOF

=== [run_all] Setup finished. To render, prepare data + a checkpoint then: ===

  # 1. Prepare your multiview capture (face crops + MHR template + pose maps);
  #    see README "Data Preparation". Point DATA_DIR at the prepared folder.
  # 2. Render (free-view turntable from the training capture):
  GPU=0 DATA_DIR=/path/to/your_subject \\
    bash $SCRIPT_DIR/02_run_inference.sh

  # Or animate an external AMASS pose sequence:
  GPU=0 DATA_DIR=/path/to/your_subject POSE_DATA=/path/to/poses.npz \\
    VIEW_SETTING=free bash $SCRIPT_DIR/02_run_inference.sh

  # Train your own checkpoint first (if the HF release isn't up yet):
  cd "$F3G_DIR" && python main_avatar.py -c configs/avatarrex_zzr/avatar.yaml -m train
  # then:  PREV_CKPT=$F3G_DIR/results/avatarrex_zzr/avatar/batch_700000 \\
  #        DATA_DIR=/path/to/your_subject bash $SCRIPT_DIR/02_run_inference.sh
EOF
fi

echo "=== [run_all] All steps finished. ==="

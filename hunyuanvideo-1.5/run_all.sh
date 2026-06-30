#!/usr/bin/env bash
# run_all.sh — one-click: clone official repo -> verify env -> weights ->
# inference. Training (03_train.sh) is separate because it needs a dataset.
# Uses the existing conda env (torch preinstalled); set INSTALL_DEPS=1 once to
# install the official requirements.txt (upgrades torch to >=2.6).
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

HYVIDEO_DIR="${HYVIDEO_DIR:-$REPO_DIR/../HunyuanVideo-1.5}"
HYVIDEO_REPO="${HYVIDEO_REPO:-https://github.com/Tencent-Hunyuan/HunyuanVideo-1.5.git}"

echo "=== [run_all] HunyuanVideo-1.5 one-click pipeline (conda env: ${CONDA_ENV:-doll}) ==="

# 0. Clone the official repo if it is not present yet.
if [ ! -d "$HYVIDEO_DIR" ]; then
    mkdir -p "$(dirname "$HYVIDEO_DIR")"
    echo "--- cloning official repo -> $HYVIDEO_DIR ---"
    git clone "$HYVIDEO_REPO" "$HYVIDEO_DIR" || \
        git -c http.sslVerify=false clone "$HYVIDEO_REPO" "$HYVIDEO_DIR"
else
    echo "--- official repo already present: $HYVIDEO_DIR ---"
fi

export HYVIDEO_DIR

# First-time setup: install requirements into the env (upgrades torch to >=2.6).
# On later runs set INSTALL_DEPS=0 (or omit) to skip the pip step.
if [ "${INSTALL_DEPS:-1}" = "1" ]; then
    INSTALL_DEPS=1 bash "$SCRIPT_DIR/00_setup_env.sh"
else
    bash "$SCRIPT_DIR/00_setup_env.sh"
fi
bash "$SCRIPT_DIR/01_download_models.sh"
bash "$SCRIPT_DIR/02_run_inference.sh"

echo "=== [run_all] All steps finished. ==="
echo "    To train on your own data:  DATA_DIR=/path/to/dataset N_TRAIN_GPU=8 bash $SCRIPT_DIR/03_train.sh"

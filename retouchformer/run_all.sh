#!/usr/bin/env bash
# run_all.sh — one-click: clone official repo -> verify env -> weights ->
# inference (on the bundled datasets/test). There is no training/eval script
# here (only inference); see the official train.py / eval.py for those.
# Uses the existing conda env (torch preinstalled); set INSTALL_DEPS=1 once to
# install the inference-only deps (requirements_inference.txt).
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

RETOUCH_DIR="${RETOUCH_DIR:-$REPO_DIR/../RetouchFormer}"
RETOUCH_REPO="${RETOUCH_REPO:-https://github.com/Davidcoach/RetouchFormer_AAAI_24.git}"

echo "=== [run_all] RetouchFormer one-click pipeline (conda env: ${CONDA_ENV:-retouchformer}) ==="

# 0. Clone the official repo if it is not present yet.
if [ ! -d "$RETOUCH_DIR" ]; then
    mkdir -p "$(dirname "$RETOUCH_DIR")"
    echo "--- cloning official repo -> $RETOUCH_DIR ---"
    git -c http.sslVerify=false clone "$RETOUCH_REPO" "$RETOUCH_DIR" || \
        git clone "$RETOUCH_REPO" "$RETOUCH_DIR"
else
    echo "--- official repo already present: $RETOUCH_DIR ---"
fi

export RETOUCH_DIR

# 1. Env: install deps on first run; skip on later runs by omitting INSTALL_DEPS.
if [ "${INSTALL_DEPS:-1}" = "1" ]; then
    INSTALL_DEPS=1 bash "$SCRIPT_DIR/00_setup_env.sh"
else
    bash "$SCRIPT_DIR/00_setup_env.sh"
fi

# 2. Weights (Baidu manual step — script prints instructions if missing).
bash "$SCRIPT_DIR/01_download_models.sh"

# 3. Inference on the bundled test folder (if any images present).
if [ -d "$RETOUCH_DIR/datasets/test" ] && \
   [ -n "$(find "$RETOUCH_DIR/datasets/test" -type f \( -iname '*.png' -o -iname '*.jpg' -o -iname '*.jpeg' \) -print -quit 2>/dev/null)" ]; then
    bash "$SCRIPT_DIR/02_run_inference.sh"
else
    echo "--- no images in $RETOUCH_DIR/datasets/test; skipping inference ---"
    echo "    Point at your own folder:"
    echo "      GPU=0 INPUT_DIR=/path/to/faces bash $SCRIPT_DIR/02_run_inference.sh"
fi

echo "=== [run_all] All steps finished. ==="
echo "    Inference on your own data:"
echo "      GPU=0 INPUT_DIR=/path/to/faces bash $SCRIPT_DIR/02_run_inference.sh"

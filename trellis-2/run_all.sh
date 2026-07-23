#!/usr/bin/env bash
# run_all.sh — one-click: clone official repo -> verify env / install deps ->
# download weights -> run inference. Uses an existing conda env (torch
# preinstalled) by default; set CREATE_ENV=1 to create the `trellis2` env fresh.
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

TRELLIS_DIR="${TRELLIS_DIR:-$REPO_DIR/../TRELLIS.2}"
TRELLIS_REPO="${TRELLIS_REPO:-https://github.com/microsoft/TRELLIS.2.git}"

echo "=== [run_all] TRELLIS.2 one-click pipeline (conda env: ${CONDA_ENV:-trellis2}) ==="

# 0. Clone the official repo (--recursive for the o-voxel submodule) if absent.
if [ ! -d "$TRELLIS_DIR" ]; then
    mkdir -p "$(dirname "$TRELLIS_DIR")"
    echo "--- cloning official repo -> $TRELLIS_DIR ---"
    git clone --recursive "$TRELLIS_REPO" "$TRELLIS_DIR" || \
        git -c http.sslVerify=false clone --recursive "$TRELLIS_REPO" "$TRELLIS_DIR"
else
    echo "--- official repo already present: $TRELLIS_DIR ---"
fi

export TRELLIS_DIR

bash "$SCRIPT_DIR/00_setup_env.sh"
bash "$SCRIPT_DIR/01_download_models.sh"
bash "$SCRIPT_DIR/02_run_inference.sh"

echo "=== [run_all] All steps finished. ==="

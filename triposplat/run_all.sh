#!/usr/bin/env bash
# run_all.sh — one-click: clone official repo -> verify env -> weights -> inference.
# Uses the existing conda env (torch preinstalled); no venv, no torch download.
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

TRIPOSPLAT_DIR="${TRIPOSPLAT_DIR:-$REPO_DIR/../TripoSplat}"
TRIPOSPLAT_REPO="${TRIPOSPLAT_REPO:-https://github.com/VAST-AI-Research/TripoSplat.git}"

echo "=== [run_all] TripoSplat one-click pipeline (conda env: ${CONDA_ENV:-doll}) ==="

# 0. Clone the official repo if it is not present yet.
if [ ! -d "$TRIPOSPLAT_DIR" ]; then
    mkdir -p "$(dirname "$TRIPOSPLAT_DIR")"
    echo "--- cloning official repo -> $TRIPOSPLAT_DIR ---"
    git clone "$TRIPOSPLAT_REPO" "$TRIPOSPLAT_DIR" || \
        git -c http.sslVerify=false clone "$TRIPOSPLAT_REPO" "$TRIPOSPLAT_DIR"
else
    echo "--- official repo already present: $TRIPOSPLAT_DIR ---"
fi

export TRIPOSPLAT_DIR

bash "$SCRIPT_DIR/00_setup_env.sh"
bash "$SCRIPT_DIR/01_download_models.sh"
bash "$SCRIPT_DIR/02_run_inference.sh"

echo "=== [run_all] All steps finished. ==="

#!/usr/bin/env bash
# run_all.sh — one-click: clone official repo -> verify env -> weights -> inference.
# Uses the existing conda env (torch preinstalled); no venv, no torch download.
# NOTE: unlike TripoSplat, the VGGT-Omega HF repo is GATED — before running,
# request access (https://huggingface.co/facebook/VGGT-Omega) and put
# `export HF_TOKEN=hf_xxx` in proxy.env. run_all.sh will stop at 01 if the
# token is missing; run 01 again after adding it.
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

VGGT_DIR="${VGGT_DIR:-$REPO_DIR/../vggt-omega}"
VGGT_REPO="${VGGT_REPO:-https://github.com/facebookresearch/vggt-omega.git}"

echo "=== [run_all] VGGT-Omega one-click pipeline (conda env: ${CONDA_ENV:-doll}) ==="

# 0. Clone the official repo if it is not present yet.
if [ ! -d "$VGGT_DIR" ]; then
    mkdir -p "$(dirname "$VGGT_DIR")"
    echo "--- cloning official repo -> $VGGT_DIR ---"
    git clone "$VGGT_REPO" "$VGGT_DIR" || \
        git -c http.sslVerify=false clone "$VGGT_REPO" "$VGGT_DIR"
else
    echo "--- official repo already present: $VGGT_DIR ---"
fi

export VGGT_DIR

bash "$SCRIPT_DIR/00_setup_env.sh"
bash "$SCRIPT_DIR/01_download_models.sh"
bash "$SCRIPT_DIR/02_run_inference.sh"

echo "=== [run_all] All steps finished. (Gated repo: if 01 failed on HF_TOKEN, add it to proxy.env and rerun 01.) ==="

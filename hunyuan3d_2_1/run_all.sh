#!/usr/bin/env bash
# run_all.sh — one-click: clone official repo -> setup env -> download weights
# -> run inference (only if input images are given).
set -o pipefail

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

HY3D_DIR="${HUNYUAN3D_DIR:-${HY3D_DIR:-$REPO_DIR/../Hunyuan3D-2.1}}"
HY3D_REPO="${HY3D_REPO:-https://github.com/Tencent-Hunyuan/Hunyuan3D-2.1.git}"

echo "=== [run_all] Hunyuan3D-2.1 one-click pipeline (conda env: ${CONDA_ENV:-hunyuan3d}) ==="

# 0. Clone the official repo if absent.
if [ ! -d "$HY3D_DIR" ]; then
    mkdir -p "$(dirname "$HY3D_DIR")"
    echo "--- cloning official repo -> $HY3D_DIR ---"
    git clone "$HY3D_REPO" "$HY3D_DIR" || \
        git -c http.sslVerify=false clone "$HY3D_REPO" "$HY3D_DIR" || exit 1
else
    echo "--- official repo already present: $HY3D_DIR ---"
fi

export HY3D_DIR

bash "$SCRIPT_DIR/00_setup_env.sh" || exit 1
bash "$SCRIPT_DIR/01_download_models.sh" || exit 1

if [ -n "${IMAGE:-}" ] || [ -n "${FRONT:-}" ] || [ -n "${BACK:-}" ] || \
   [ -n "${LEFT:-}" ] || [ -n "${RIGHT:-}" ]; then
    bash "$SCRIPT_DIR/02_run_inference.sh" || exit 1
else
    echo "--- no input given (IMAGE= or FRONT=/...), skipping inference ---"
fi

echo "=== [run_all] All steps finished. ==="

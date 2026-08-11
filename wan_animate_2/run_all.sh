#!/usr/bin/env bash
# run_all.sh — one-click: clone Wan-Animate-2 -> download weights -> inference
# (official demo1: reference.png + template.mp4 + default cat prompt).
# Uses the existing conda env (torch preinstalled); set INSTALL_DEPS=1 once to
# install the wan-animate-2 package + deps.
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

echo "=== [run_all] Wan-Animate-2 one-click pipeline (conda env: ${CONDA_ENV:-wan_animate_2}) ==="

# 0. Clone + env setup.
if [ "${INSTALL_DEPS:-1}" = "1" ]; then
    INSTALL_DEPS=1 bash "$SCRIPT_DIR/00_setup_env.sh"
else
    bash "$SCRIPT_DIR/00_setup_env.sh"
fi

# 1. Download weights + symlink ckpts.
bash "$SCRIPT_DIR/01_download_models.sh"

# 2. Inference on the official demo1 (base variant).
bash "$SCRIPT_DIR/02_run_inference.sh"

echo "=== [run_all] All steps finished. ==="
echo "    Result: ../wan_animate_2_results/animate.mp4"
echo "    Customize:"
echo "      GPU=0,1 MODEL_VARIANT=distillation PROMPT='...' \\"
echo "        REFER_IMAGE=/path/to/your.png REFER_VIDEO=/path/to/driver.mp4 \\"
echo "        bash $SCRIPT_DIR/02_run_inference.sh"

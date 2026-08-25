#!/usr/bin/env bash
# run_all.sh — one-click pipeline: env setup -> download models -> inference.
#
# Defaults to WSL (00a_setup_env.sh). On a server, replace 00a with 00:
#   sed 's/00a_setup_env/00_setup_env/' osediff/run_all.sh | bash
set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"

# Optional proxy (must be set before git clone).
if [ -f "$REPO_DIR/proxy.env" ]; then
    set -a; # shellcheck disable=SC1090
    source "$REPO_DIR/proxy.env"; set +a
fi

# Corporate proxy TLS interception workaround (git clone over HTTPS).
SYS_CA=/etc/ssl/certs/ca-certificates.crt
if [ -f "$SYS_CA" ]; then
    : "${GIT_SSL_CAINFO:=$SYS_CA}"
    export GIT_SSL_CAINFO
fi

echo "=== [run_all] OSEDiff one-click pipeline (conda env: ${CONDA_ENV:-osediff}) ==="
echo "  (WSL: 00a_setup_env.sh | server: edit this script to use 00_setup_env.sh)"
echo ""

# 0. env setup (WSL: creates env + clones repo + installs deps).
if [ "${INSTALL_DEPS:-1}" = "1" ]; then
    INSTALL_DEPS=1 bash "$SCRIPT_DIR/00a_setup_env.sh"
else
    bash "$SCRIPT_DIR/00a_setup_env.sh"
fi

# 1. download SD2.1-Base + RAM (stage DAPE from repo).
bash "$SCRIPT_DIR/01_download_models.sh"

# 2. inference on the repo's preset test images.
bash "$SCRIPT_DIR/02_run_inference.sh"

echo ""
echo "=== [run_all] All steps finished. ==="
echo "  Super-resolved images: $RESULTS_DIR/output"
echo ""
echo "  To train on your own data:"
echo "    # 1. build a dataset txt (one absolute image path per line):"
echo "    find /path/to/images -type f \\( -name '*.png' -o -name '*.jpg' \\) > lsdir.txt"
echo "    # 2. train (single card, ~24GB VRAM):"
echo "    GPU=0 DATASET_TXT=$PWD/lsdir.txt bash $SCRIPT_DIR/04_train_lora.sh"
echo "  To move results to Windows D: (WSL only):"
echo "    bash $SCRIPT_DIR/08_move_output.sh"

#!/usr/bin/env bash
# run_all.sh — one-click: verify env -> download weights -> inference.
# Qwen3-VL has NO official code repo to clone (it's a self-contained HF model
# used via transformers), so run_all is just 00 -> 01 -> 02 (on any sample
# images you place under ../Qwen3-VL/examples/images, or set IMAGE_DIR).
# Uses the existing conda env (torch preinstalled); set INSTALL_DEPS=1 once to
# install transformers / qwen-vl-utils / accelerate.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"

echo "=== [run_all] Qwen3-VL one-click image-to-text pipeline (conda env: ${CONDA_ENV:-doll}) ==="

if [ "${INSTALL_DEPS:-1}" = "1" ]; then
    INSTALL_DEPS=1 bash "$SCRIPT_DIR/00_setup_env.sh"
else
    bash "$SCRIPT_DIR/00_setup_env.sh"
fi
bash "$SCRIPT_DIR/01_download_models.sh"
bash "$SCRIPT_DIR/02_run_inference.sh"

echo "=== [run_all] All steps finished. ==="
echo "    Generated text is in: ../Qwen3-VL/results/<image_dir>/result/*.txt"
echo "    Re-run with your own images:  IMAGE_DIR=/path/to/images GPU=0 bash $SCRIPT_DIR/02_run_inference.sh"

#!/usr/bin/env bash
# run_all.sh — one-click: verify env -> download weights -> inference.
# FLUX.1 has NO official code repo to clone (it's a self-contained diffusers
# HF model), so run_all is just 00 -> 01 -> 02 (generates one sample image from
# the default PROMPT; set PROMPT or PROMPTS_FILE for your own).
# Uses the existing conda env (torch preinstalled); set INSTALL_DEPS=1 once to
# install diffusers / transformers / accelerate.
#
# Default model is the public FLUX.1-schnell (no token). To use the gated
# FLUX.1-dev instead: HF_REPO_ID=black-forest-labs/FLUX.1-dev HF_TOKEN=<tok> bash ...
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"

echo "=== [run_all] FLUX.1 one-click text-to-image pipeline (conda env: ${CONDA_ENV:-doll}) ==="

if [ "${INSTALL_DEPS:-1}" = "1" ]; then
    INSTALL_DEPS=1 bash "$SCRIPT_DIR/00_setup_env.sh"
else
    bash "$SCRIPT_DIR/00_setup_env.sh"
fi
bash "$SCRIPT_DIR/01_download_models.sh"
bash "$SCRIPT_DIR/02_run_inference.sh"

echo "=== [run_all] All steps finished. ==="
echo "    Generated images are in: ../FLUX1/results/prompt/result/*.png"
echo "    Re-run with your own prompts:"
echo "      GPU=0 PROMPT=\"a robot painting a sunset\" bash $SCRIPT_DIR/02_run_inference.sh"
echo "      GPU=0 PROMPTS_FILE=/path/to/prompts.txt bash $SCRIPT_DIR/02_run_inference.sh"

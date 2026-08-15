#!/usr/bin/env bash
# run_all.sh — one-click: verify env -> download weights -> inference.
# FLUX.2 has NO official code repo to clone (self-contained diffusers HF model),
# so run_all is just 00 -> 01 -> 02 (generates one sample T2I image from the
# default PROMPT; set PROMPT or PROMPTS_FILE for your own).
# Uses a dedicated conda env cloned from 'doll' (torch reused); set
# INSTALL_DEPS=1 once to upgrade diffusers to 0.39 (Flux2 pipelines).
#
# Default model is FLUX.2-klein-9B (fast, distilled 4-step, fits consumer GPUs).
# To use FLUX.2-dev (max quality, needs H100-class or quant): MODEL_TYPE=dev ...
set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "=== [run_all] FLUX.2 one-click text-to-image pipeline (conda env: ${CONDA_ENV:-flux2}) ==="

if [ "${INSTALL_DEPS:-1}" = "1" ]; then
    INSTALL_DEPS=1 bash "$SCRIPT_DIR/00_setup_env.sh"
else
    bash "$SCRIPT_DIR/00_setup_env.sh"
fi
bash "$SCRIPT_DIR/01_download_models.sh"
bash "$SCRIPT_DIR/02_run_inference.sh"

echo "=== [run_all] All steps finished. ==="
echo "    Generated images: ../FLUX2/results/prompt/result/*.png"
echo "    Re-run with your own prompts:"
echo "      GPU=0 MODEL_TYPE=klein PROMPT=\"a robot painting a sunset\" bash $SCRIPT_DIR/02_run_inference.sh"
echo "    Image editing (single/multi reference):"
echo "      GPU=0 MODEL_TYPE=klein INPUT_IMAGES=\"cat.jpg,dog.jpg\" PROMPT=\"...\" bash $SCRIPT_DIR/02_run_inference.sh"
echo "    Max quality (dev; needs H100-class or QUANT=4bit on consumer GPU):"
echo "      GPU=0 MODEL_TYPE=dev PROMPT=\"...\" bash $SCRIPT_DIR/02_run_inference.sh"

#!/usr/bin/env bash
# 00_setup_env.sh — verify that sam_3d_body + wan22 (DiffSynth-Studio) are
# ready: code dirs exist, conda envs importable, model weights present.
# Does NOT install anything. For first-time setup, run the respective scripts:
#   sam_3d_body: INSTALL_DEPS=1 HF_TOKEN=hf_xxx bash sam_3d_body/00_setup_env.sh
#                HF_TOKEN=hf_xxx bash sam_3d_body/01_download_models.sh
#   wan22:       INSTALL_DEPS=1 bash wan22/00_setup_env.sh
#                bash wan22/01_verify_models.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/_env.sh"

SAM3D_ENV="${SAM3D_ENV:-sam_3d_body}"
WAN_ENV="${WAN_ENV:-wan22}"
HF_REPO_ID="${HF_REPO_ID:-facebook/sam-3d-body-dinov3}"
CKPT_DIR="$SAM3D_MODEL_DIR/$(basename "$HF_REPO_ID")"

echo "=== [00] Verify prerequisites for wan22_rotate ==="
echo "  sam_3d_body env: $SAM3D_ENV"
echo "  wan22 env:       $WAN_ENV"
echo ""

# --- 1. sam_3d_body code ---
echo "--- [1/5] SAM 3D Body code: $SAM3D_DIR ---"
if [ ! -d "$SAM3D_DIR" ]; then
    echo "  [MISS] Run: INSTALL_DEPS=1 HF_TOKEN=hf_xxx bash $REPO_DIR/sam_3d_body/run_all.sh" >&2
    exit 1
fi
echo "  [OK]"

# --- 2. sam_3d_body weights ---
echo "--- [2/5] SAM 3D Body weights: $CKPT_DIR ---"
_ok_sam=1
if [ ! -f "$CKPT_DIR/model.ckpt" ]; then
    echo "  [MISS] model.ckpt — Run: HF_TOKEN=hf_xxx bash $REPO_DIR/sam_3d_body/01_download_models.sh" >&2
    _ok_sam=0
fi
if [ ! -f "$CKPT_DIR/assets/mhr_model.pt" ]; then
    echo "  [MISS] assets/mhr_model.pt" >&2
    _ok_sam=0
fi
if [ "$_ok_sam" = "1" ]; then echo "  [OK]"; fi

# --- 3. DiffSynth-Studio code ---
echo "--- [3/5] DiffSynth-Studio code: $DIFFSYNTH_DIR ---"
if [ ! -d "$DIFFSYNTH_DIR" ]; then
    echo "  [MISS] Run: INSTALL_DEPS=1 bash $REPO_DIR/wan22/00_setup_env.sh" >&2
    exit 1
fi
echo "  [OK]"

# --- 4. Wan2.2 weights ---
echo "--- [4/5] Wan2.2-TI2V-5B weights: $WAN_MODEL_DIR ---"
bash "$REPO_DIR/wan22/01_verify_models.sh" 2>/dev/null || \
    echo "  (some weights may be missing — run wan22/01_verify_models.sh for details)"

# --- 5. conda envs ---
echo "--- [5/5] conda envs ---"
conda_activate "$SAM3D_ENV"
if python -c "import torch, cv2" 2>/dev/null; then
    echo "  [OK] $SAM3D_ENV: torch + cv2"
else
    echo "  [MISS] $SAM3D_ENV: Run INSTALL_DEPS=1 bash $REPO_DIR/sam_3d_body/00_setup_env.sh" >&2
fi

conda_activate "$WAN_ENV"
if python -c "import diffsynth" 2>/dev/null; then
    echo "  [OK] $WAN_ENV: diffsynth"
else
    echo "  [MISS] $WAN_ENV: Run INSTALL_DEPS=1 bash $REPO_DIR/wan22/00_setup_env.sh" >&2
fi

echo ""
echo "=== [00] Done. Prerequisites checked. ==="
echo "    Next: INPUT_DIR=/path/to/subject_folder WEIGHT_PATH=/path/to/lora.safetensors bash $SCRIPT_DIR/run_all.sh"

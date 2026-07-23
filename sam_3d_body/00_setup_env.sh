#!/usr/bin/env bash
# 00_setup_env.sh — activate conda env (torch preinstalled) & verify torch+CUDA.
# Set INSTALL_DEPS=1 to install the official dependencies from INSTALL.md.
#
# Official env per INSTALL.md: conda create -n sam_3d_body python=3.11 -y, then
# install PyTorch (CUDA build) + the pip list + detectron2 + (optional) MoGe /
# SAM3. Use a DEDICATED env — these pins (networkx==3.2.1, a detectron2 git pin)
# can clash with other algos in this repo.
#
# Optional component flags (only matter when INSTALL_DEPS=1):
#   DETECTRON2=1 (default) — install detectron2 @a1ce2f9 (needed by the default
#                            ViTDet human detector; --no-deps to avoid pin clash)
#   MOGE=1       (default) — install microsoft/MoGe (needed by the default moge2
#                            FOV estimator)
#   SAM3=0       (default) — install facebookresearch/sam3 (only needed if you
#                            pass --detector_name sam3 / --segmentor_name sam3)
#   SKIP_CORE=0            — skip the big core pip list (already installed it)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/_env.sh"

SAM3D_DIR="${SAM3D_DIR:-$REPO_DIR/../sam-3d-body}"

echo "=== [00] Verify torch in conda env '$CONDA_ENV' ==="
python - <<'PY'
import torch
print(f"torch: {torch.__version__}  cuda: {torch.version.cuda}  available: {torch.cuda.is_available()}")
if not torch.cuda.is_available():
    raise SystemExit("ERROR: torch.cuda not available in this env — install a CUDA-enabled torch or check GPU visibility.")
PY

if [ "${INSTALL_DEPS:-0}" = "1" ]; then
    PIP_FLAGS=(--trusted-host pypi.org --trusted-host pypi.python.org \
        --trusted-host files.pythonhosted.org --timeout 600 --retries 10)

    # --- 1. Core pip list (verbatim from INSTALL.md step 3) ---
    if [ "${SKIP_CORE:-0}" != "1" ]; then
        echo "--- installing core dependencies (INSTALL.md step 3) ---"
        pip install "${PIP_FLAGS[@]}" \
            pytorch-lightning pyrender opencv-python yacs scikit-image einops \
            timm dill pandas rich hydra-core hydra-submitit-launcher \
            hydra-colorlog pyrootutils webdataset chump "networkx==3.2.1" roma \
            joblib seaborn wandb appdirs appnope ffmpeg cython jsonlines pytest \
            xtcocotools loguru optree fvcore black pycocotools tensorboard \
            huggingface_hub
    else
        echo "--- SKIP_CORE=1 -> skipping core pip list (assume already installed) ---"
    fi

    # --- 2. Detectron2 (default ViTDet human detector needs it) ---
    if [ "${DETECTRON2:-1}" = "1" ]; then
        echo "--- installing detectron2 @a1ce2f9 (--no-build-isolation --no-deps) ---"
        # git+https uses git clone under the hood; _env.sh sets GIT_SSL_CAINFO.
        # If SSL still fails, run once: git -c http.sslVerify=false clone ...
        pip install "${PIP_FLAGS[@]}" \
            'git+https://github.com/facebookresearch/detectron2.git@a1ce2f9' \
            --no-build-isolation --no-deps
    else
        echo "--- DETECTRON2=0 -> skipping detectron2 (ViTDet detector disabled) ---"
    fi

    # --- 3. MoGe (default moge2 FOV estimator needs it) ---
    if [ "${MOGE:-1}" = "1" ]; then
        echo "--- installing MoGe (FOV estimator; microsoft/MoGe) ---"
        pip install "${PIP_FLAGS[@]}" 'git+https://github.com/microsoft/MoGe.git'
    else
        echo "--- MOGE=0 -> skipping MoGe (moge2 FOV estimator disabled) ---"
    fi

    # --- 4. SAM3 (optional; only for detector_name/segmentor_name = sam3) ---
    if [ "${SAM3:-0}" = "1" ]; then
        echo "--- installing SAM3 (minimal inference install) ---"
        SAM3_REPO="${SAM3_REPO:-https://github.com/facebookresearch/sam3.git}"
        SAM3_DIR_LOCAL="${SAM3_DIR_LOCAL:-$REPO_DIR/../sam3}"
        if [ ! -d "$SAM3_DIR_LOCAL" ]; then
            git clone "$SAM3_REPO" "$SAM3_DIR_LOCAL" || \
                git -c http.sslVerify=false clone "$SAM3_REPO" "$SAM3_DIR_LOCAL"
        fi
        # shellcheck disable=SC1091
        ( cd "$SAM3_DIR_LOCAL" && pip install "${PIP_FLAGS[@]}" -e . && \
          pip install "${PIP_FLAGS[@]}" decord psutil )
    else
        echo "--- SAM3=0 -> skipping SAM3 (set SAM3=1 if you need sam3 detector/segmentor) ---"
    fi

    echo "--- deps installed. Tip: verify with 'python -c \"import detectron2,moge,pyrootutils\"' ---"
fi

echo "=== [00] Done. Env '$CONDA_ENV' ready. (Missing a dep? INSTALL_DEPS=1 bash this, or pip install it) ==="

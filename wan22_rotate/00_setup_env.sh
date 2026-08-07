#!/usr/bin/env bash
# 00_setup_env.sh — create the wan22_rotate conda env (cloned from doll), install
# both sam_3d_body + diffsynth deps into it, and verify everything is ready.
#
# First time:
#   INSTALL_DEPS=1 bash wan22_rotate/00_setup_env.sh
# After that (verify only):
#   bash wan22_rotate/00_setup_env.sh
#
# The env needs: sam_3d_body code + weights, DiffSynth-Studio code + Wan2.2 weights.
# For weights, run the respective download scripts first:
#   sam_3d_body: HF_TOKEN=hf_xxx bash sam_3d_body/01_download_models.sh
#   wan22:       bash wan22/01_verify_models.sh
set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/_env.sh"

# _env.sh tolerated a missing env; create it now if needed.
if ! conda env list 2>/dev/null | grep -qw "$CONDA_ENV"; then
    echo "--- conda env '$CONDA_ENV' not found; cloning from doll ---"
    if conda env list 2>/dev/null | grep -qw "doll"; then
        conda create -n "$CONDA_ENV" --clone doll -y
        conda activate "$CONDA_ENV"
    else
        echo "ERROR: neither '$CONDA_ENV' nor 'doll' exists." >&2
        echo "       Create one first: conda create -n doll python=3.11 -y && conda activate doll" >&2
        echo "                         && pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124" >&2
        exit 1
    fi
fi

HF_REPO_ID="${HF_REPO_ID:-facebook/sam-3d-body-dinov3}"
CKPT_DIR="$SAM3D_MODEL_DIR/$(basename "$HF_REPO_ID")"

echo "=== [00] Verify prerequisites for wan22_rotate ==="
echo "  conda env:  $CONDA_ENV  (python $(python --version 2>&1 | cut -d' ' -f2))"
echo "  sam_3d_body:  $SAM3D_DIR"
echo "  diffsynth:    $DIFFSYNTH_DIR"
echo ""

# --- 0. Install deps (first time) ---
if [ "${INSTALL_DEPS:-0}" = "1" ]; then
    PIP_FLAGS=(--trusted-host pypi.org --trusted-host pypi.python.org \
        --trusted-host files.pythonhosted.org --timeout 600 --retries 10)
    _proxy="${https_proxy:-${http_proxy:-}}"
    if [ -n "$_proxy" ]; then
        PIP_FLAGS+=(--proxy "$_proxy")
        echo "--- using proxy: $_proxy ---"
    else
        echo "WARNING: no proxy set (http_proxy/https_proxy); pip may fail if no direct internet." >&2
        echo "         Create proxy.env at repo root: see proxy.env.example" >&2
    fi

    # 0a. Install PyTorch cu124 + nvidia deps from local wheels (download.pytorch.org blocked 403)
    echo "--- installing PyTorch cu124 + nvidia libs ---"
    LOCAL_WHEELS=()
    for w in \
        "$WAN_MODEL_DIR/torch-2.6.0+cu124-cp310-cp310-linux_x86_64.whl" \
        "$WAN_MODEL_DIR/torchvision-0.21.0+cu124-cp310-cp310-linux_x86_64.whl" \
        "$WAN_MODEL_DIR/nvidia_cudnn_cu12-9.1.0.70-py3-none-manylinux2014_x86_64.whl" \
        "$WAN_MODEL_DIR/nvidia_cublas_cu12-12.4.5.8-py3-none-manylinux2014_x86_64.whl" \
        "$WAN_MODEL_DIR/nvidia_cuda_cupti_cu12-"*.whl \
        ; do
        [ -f "$w" ] && LOCAL_WHEELS+=("$w")
    done
    if [ ${#LOCAL_WHEELS[@]} -ge 2 ]; then
        echo "  installing from local wheels: ${LOCAL_WHEELS[*]}"
        pip install --force-reinstall --no-deps "${LOCAL_WHEELS[@]}"
    else
        echo "  local wheels not found in $WAN_MODEL_DIR, falling back to PyPI"
        pip install "${PIP_FLAGS[@]}" --force-reinstall --no-deps torch torchvision
    fi
    echo "  torch.version.cuda = $(python -c 'import torch; print(torch.version.cuda)')"

    # 0a2. nvidia-cuda-cupti-cu12 (libcupti.so.12, torch profiling needs it)
    echo "--- installing nvidia-cuda-cupti-cu12 ---"
    pip install "${PIP_FLAGS[@]}" nvidia-cuda-cupti-cu12

    # 0b. Install gcc 12 into the conda env (system gcc too old for CUDA 12.4)
    echo "--- installing gcc 12 into conda env (for detectron2 compilation) ---"
    conda install -y -c conda-forge gxx_linux-64=12 --no-update-deps
    export CC=$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-gcc
    export CXX=$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-g++
    export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"

    # 0c. DiffSynth-Studio-Human (fresh clone, editable install)
    if [ -d "$DIFFSYNTH_DIR" ]; then
        echo "--- installing diffsynth (editable) from $DIFFSYNTH_DIR ---"
        pip install "${PIP_FLAGS[@]}" -e "$DIFFSYNTH_DIR"
    else
        echo "--- cloning DiffSynth-Studio -> $DIFFSYNTH_DIR ---"
        mkdir -p "$(dirname "$DIFFSYNTH_DIR")"
        git clone https://github.com/modelscope/DiffSynth-Studio.git "$DIFFSYNTH_DIR" || \
            git -c http.sslVerify=false clone https://github.com/modelscope/DiffSynth-Studio.git "$DIFFSYNTH_DIR"
        pip install "${PIP_FLAGS[@]}" -e "$DIFFSYNTH_DIR"
    fi

    # 0b. sam_3d_body core deps (from INSTALL.md, minus torch which doll already has)
    echo "--- installing sam_3d_body core deps ---"
    pip install "${PIP_FLAGS[@]}" \
        pytorch-lightning pyrender opencv-python yacs scikit-image einops \
        timm dill pandas rich hydra-core hydra-submitit-launcher \
        hydra-colorlog pyrootutils webdataset chump "networkx==3.2.1" roma \
        joblib seaborn wandb appdirs appnope ffmpeg cython jsonlines pytest \
        xtcocotools loguru optree fvcore black pycocotools tensorboard \
        huggingface_hub

    # 0c. detectron2 (@a1ce2f9, --no-deps to avoid pin clash)
    echo "--- installing detectron2 @a1ce2f9 (--no-build-isolation --no-deps) ---"
    pip install "${PIP_FLAGS[@]}" \
        'git+https://github.com/facebookresearch/detectron2.git@a1ce2f9' \
        --no-build-isolation --no-deps

    # 0d. MoGe (FOV estimator, only needed by full 01, not 01b)
    echo "--- installing MoGe (microsoft/MoGe) ---"
    pip install "${PIP_FLAGS[@]}" 'git+https://github.com/microsoft/MoGe.git'

    # 0e. SAM2 (person segmentation, needed by 01b simplified)
    echo "--- setting up SAM2 ---"
    if [ ! -d "$SAM2_DIR" ]; then
        echo "--- cloning SAM2 -> $SAM2_DIR ---"
        mkdir -p "$(dirname "$SAM2_DIR")"
        git clone https://github.com/facebookresearch/sam2.git "$SAM2_DIR" || \
            git -c http.sslVerify=false clone https://github.com/facebookresearch/sam2.git "$SAM2_DIR"
    fi
    pip install "${PIP_FLAGS[@]}" -e "$SAM2_DIR"
    CKPT_DIR="$SAM2_DIR/checkpoints"
    mkdir -p "$CKPT_DIR"
    if [ ! -f "$CKPT_DIR/sam2.1_hiera_large.pt" ]; then
        echo "--- downloading SAM2 checkpoint (sam2.1_hiera_large) ---"
        wget --no-check-certificate -q -O "$CKPT_DIR/sam2.1_hiera_large.pt" \
            "https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2.1_hiera_large.pt" || \
            echo "WARNING: SAM2 checkpoint download failed (dl.fbaipublicfiles.com may be blocked)." >&2
            echo "         Download manually from https://github.com/facebookresearch/sam2#segment-anything-2-checkpoints" >&2
            echo "         Place at: $CKPT_DIR/sam2.1_hiera_large.pt" >&2
    fi

    echo "--- deps installed ---"
fi

# --- 1. sam_3d_body code ---
echo "--- [1/4] SAM 3D Body code: $SAM3D_DIR ---"
if [ ! -d "$SAM3D_DIR" ]; then
    echo "  [MISS] Run: INSTALL_DEPS=1 bash $REPO_DIR/sam_3d_body/run_all.sh" >&2
    exit 1
fi
echo "  [OK]"

# --- 2. sam_3d_body weights ---
echo "--- [2/4] SAM 3D Body weights: $CKPT_DIR ---"
_ok_sam=1
if [ ! -f "$CKPT_DIR/model.ckpt" ]; then
    echo "  [MISS] model.ckpt — Run: HF_TOKEN=hf_xxx bash $REPO_DIR/sam_3d_body/01_download_models.sh" >&2
    _ok_sam=0
fi
if [ ! -f "$CKPT_DIR/assets/mhr_model.pt" ]; then
    echo "  [MISS] assets/mhr_model.pt" >&2
    _ok_sam=0
fi
[ "$_ok_sam" = "1" ] && echo "  [OK]"

# --- 3. DiffSynth-Studio code ---
echo "--- [3/4] DiffSynth-Studio code: $DIFFSYNTH_DIR ---"
if [ ! -d "$DIFFSYNTH_DIR" ]; then
    echo "  [MISS] Run: INSTALL_DEPS=1 bash $0" >&2
    exit 1
fi
echo "  [OK]"

# --- 4. Wan2.2 weights + imports ---
echo "--- [4/4] Wan2.2-TI2V-5B weights: $WAN_MODEL_DIR ---"
bash "$REPO_DIR/wan22/01_verify_models.sh" 2>/dev/null || \
    echo "  (some weights may be missing — run wan22/01_verify_models.sh for details)"

# --- 5. verify imports ---
echo "--- verify imports ---"
if python -c "import diffsynth; print('  [OK] diffsynth')" 2>/dev/null; then :; else
    echo "  [MISS] diffsynth — Run: INSTALL_DEPS=1 bash $0" >&2
fi
if python -c "import sam_3d_body, cv2, detectron2; print('  [OK] sam_3d_body + cv2 + detectron2')" 2>/dev/null; then :; else
    echo "  [MISS] sam_3d_body/cv2/detectron2 — Run: INSTALL_DEPS=1 bash $0" >&2
fi

echo ""
echo "=== [00] Done. Env '$CONDA_ENV' ready. ==="
echo "    Next: INPUT_DIR=/path/to/subject_folder WEIGHT_PATH=/path/to/lora.safetensors bash $SCRIPT_DIR/run_all.sh"

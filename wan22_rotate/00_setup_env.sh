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
# NB: 本 env 必须 python=3.10 (CPython) —— 本地 torch/triton wheel 是 cp310
# (torch-2.6.0+cu124-cp310-...、triton-3.2.0-cp310-...)，3.11 装不上。
# 不能 `conda create --clone doll`（doll 是 3.11）。直接建 3.10。
if ! conda env list 2>/dev/null | grep -qw "$CONDA_ENV"; then
    echo "--- conda env '$CONDA_ENV' not found; creating python=3.10 (CPython) ---"
    conda create -n "$CONDA_ENV" python=3.10 -y
    conda activate "$CONDA_ENV"
    # 防御：极少数情况下 conda solver 会塞 GraalPy 当 python 实现，那 cp310 wheel
    # 和 numpy 全废。建完立刻校验是 CPython。
    impl="$(python -c 'import platform,sys; print(platform.python_implementation())')"
    if [ "$impl" != "CPython" ]; then
        echo "ERROR: env '$CONDA_ENV' 的 python 实现是 $impl（应为 CPython）。" >&2
        echo "       通常是 conda-forge 把 graalpy 当 python 塞了。删掉重建：" >&2
        echo "         conda env remove -n $CONDA_ENV" >&2
        echo "         conda create -n $CONDA_ENV python=3.10 -y --override-channels -c defaults" >&2
        exit 1
    fi
    echo "  [OK] python=$(python --version 2>&1 | cut -d' ' -f2) ($impl)"
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

    # 0a. Uninstall torchaudio + install PyTorch cu124 + all nvidia deps
    echo "--- uninstalling torchaudio (version conflict) ---"
    pip uninstall -y torchaudio 2>/dev/null || true

    echo "--- installing PyTorch cu124 local wheels ---"
    LOCAL_WHEELS=()
    for w in \
        "$WAN_MODEL_DIR/nvidia_cublas_cu12-12.4.5.8-py3-none-manylinux2014_x86_64.whl" \
        "$WAN_MODEL_DIR/nvidia_cudnn_cu12-9.1.0.70-py3-none-manylinux2014_x86_64.whl" \
        "$WAN_MODEL_DIR/nvidia_cufft_cu12-11.2.1.3-py3-none-manylinux2014_x86_64.whl" \
        "$WAN_MODEL_DIR/triton-3.2.0-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl" \
        "$WAN_MODEL_DIR/nvidia_cusparse_cu12-12.3.1.170-py3-none-manylinux2014_x86_64.whl" \
        "$WAN_MODEL_DIR/nvidia_cusparselt_cu12-0.6.2-py3-none-manylinux2014_x86_64.whl" \
        "$WAN_MODEL_DIR/nvidia_nccl_cu12-2.21.5-py3-none-manylinux2014_x86_64.whl" \
        "$WAN_MODEL_DIR/torch-2.6.0+cu124-cp310-cp310-linux_x86_64.whl" \
        "$WAN_MODEL_DIR/torchvision-0.21.0+cu124-cp310-cp310-linux_x86_64.whl" \
        ; do
        [ -f "$w" ] && LOCAL_WHEELS+=("$w")
    done
    if [ ${#LOCAL_WHEELS[@]} -ge 1 ]; then
        echo "  installing local wheels: ${LOCAL_WHEELS[*]}"
        pip install --force-reinstall --no-deps "${LOCAL_WHEELS[@]}"
    fi

    echo "--- installing nvidia deps + sympy + triton (torch 2.6.0+cu124 pins) ---"
    pip install "${PIP_FLAGS[@]}" \
        nvidia-cuda-nvrtc-cu12==12.4.127 \
        nvidia-cuda-runtime-cu12==12.4.127 \
        nvidia-cuda-cupti-cu12==12.4.127 \
        nvidia-cudnn-cu12==9.1.0.70 \
        nvidia-cublas-cu12==12.4.5.8 \
        nvidia-cufft-cu12==11.2.1.3 \
        nvidia-curand-cu12==10.3.5.147 \
        nvidia-cusolver-cu12==11.6.1.9 \
        nvidia-cusparse-cu12==12.3.1.170 \
        nvidia-cusparselt-cu12==0.6.2 \
        nvidia-nccl-cu12==2.21.5 \
        nvidia-nvtx-cu12==12.4.127 \
        nvidia-nvjitlink-cu12==12.4.127 \
        sympy==1.13.1 \
        triton==3.2.0

    # numpy 1.26.4 是 sam_3d_body + diffsynth + detectron2 共同接受的版本；新 env 默认
    # 装的较新 numpy 会和 detectron2 ABI 冲突，故显式钉 1.26.4。gxx 装完还会再校验一次。
    echo "--- installing numpy==1.26.4 (pinned for detectron2/diffsynth) ---"
    pip install "${PIP_FLAGS[@]}" --force-reinstall --no-deps numpy==1.26.4

    echo "  torch.version.cuda = $(python -c 'import torch; print(torch.version.cuda)')"

    # 0b. Install gcc 12 into the conda env (system gcc too old for CUDA 12.4).
    # ⚠️ 显式 pin python=3.10 防止 conda-forge 把 python 换成 GraalPy：
    #    --no-update-deps 不够，conda 仍会把 python 槽位换成 GraalPy（满足依赖）。
    #    加 python=3.10 强制保持 CPython，GraalPy 进不来。
    echo "--- installing gcc 12 into conda env (for detectron2 compilation) ---"
    conda install -y -c conda-forge gxx_linux-64=12 python=3.10
    # 校验：gxx 没把 python 掉包成 GraalPy
    impl2="$(python -c 'import platform; print(platform.python_implementation())' 2>/dev/null || echo unknown)"
    if [ "$impl2" != "CPython" ]; then
        echo "WARNING: gxx install 把 python 掉包成 '$impl2'，自动修复..." >&2
        conda install -y -c defaults python=3.10 --force-reinstall
        impl2="$(python -c 'import platform; print(platform.python_implementation())' 2>/dev/null || echo unknown)"
        if [ "$impl2" != "CPython" ]; then
            echo "ERROR: 无法恢复 CPython，请手动重建 env" >&2
            echo "  conda env remove -n $CONDA_ENV && conda create -n $CONDA_ENV python=3.10 -y" >&2
            exit 1
        fi
        echo "  [OK] 恢复为 CPython，重装 numpy/torch"
        pip install "${PIP_FLAGS[@]}" --force-reinstall --no-deps numpy==1.26.4
        for w in "$WAN_MODEL_DIR"/torch-*.whl "$WAN_MODEL_DIR"/torchvision-*.whl; do
            [ -f "$w" ] && pip install --force-reinstall --no-deps "$w"
        done
    fi
    export CC=$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-gcc
    export CXX=$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-g++
    export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
    # gxx 装完 numpy 可能被碰坏；import 失败就重装一次
    if ! python -c "import numpy" 2>/dev/null; then
        echo "--- numpy import 失败，重装 numpy==1.26.4 ---"
        pip install "${PIP_FLAGS[@]}" --force-reinstall --no-deps numpy==1.26.4
    fi

    # 0c. DiffSynth-Studio-Human (fresh clone, editable install)
    if [ ! -d "$DIFFSYNTH_DIR/setup.py" ] && [ ! -d "$DIFFSYNTH_DIR/pyproject.toml" ]; then
        echo "--- cloning DiffSynth-Studio -> $DIFFSYNTH_DIR ---"
        mkdir -p "$(dirname "$DIFFSYNTH_DIR")"
        LD_LIBRARY_PATH= git clone https://github.com/modelscope/DiffSynth-Studio.git "$DIFFSYNTH_DIR" || \
            LD_LIBRARY_PATH= git -c http.sslVerify=false clone https://github.com/modelscope/DiffSynth-Studio.git "$DIFFSYNTH_DIR"
    fi
    pip install "${PIP_FLAGS[@]}" -e "$DIFFSYNTH_DIR"

    # 0d. sam_3d_body core deps (from INSTALL.md, minus torch which env already has)
    echo "--- installing sam_3d_body core deps ---"
    pip install "${PIP_FLAGS[@]}" \
        pytorch-lightning pyrender opencv-python yacs scikit-image einops \
        timm dill pandas rich hydra-core hydra-submitit-launcher \
        hydra-colorlog pyrootutils webdataset chump "networkx==3.2.1" roma \
        joblib seaborn wandb appdirs appnope ffmpeg cython jsonlines pytest \
        xtcocotools loguru optree fvcore black pycocotools tensorboard \
        huggingface_hub plyfile

    # 0e. detectron2 (@a1ce2f9, --no-deps to avoid pin clash)
    # 手动 clone 后本地 install——LD_LIBRARY_PATH= 防 conda libffi 和系统 libp11-kit 冲突
    echo "--- installing detectron2 @a1ce2f9 ---"
    DETECTRON2_DIR="${DETECTRON2_DIR:-$REPO_DIR/../detectron2}"
    if [ ! -d "$DETECTRON2_DIR/.git" ]; then
        LD_LIBRARY_PATH= git clone https://github.com/facebookresearch/detectron2.git "$DETECTRON2_DIR" || \
            LD_LIBRARY_PATH= git -c http.sslVerify=false clone https://github.com/facebookresearch/detectron2.git "$DETECTRON2_DIR"
    fi
    (cd "$DETECTRON2_DIR" && git checkout a1ce2f9)
    pip install "${PIP_FLAGS[@]}" --no-build-isolation --no-deps -e "$DETECTRON2_DIR"

    # 预下载 ViTDet 权重到 $WAN_MODEL_DIR/ViTDet——DETECTOR_PATH 默认指向这里,
    # detectron2 用 os.path.join(path, ...) 加载, 不走 urllib (SSL 被代理拦)
    VITDET_DIR="$WAN_MODEL_DIR/ViTDet"
    mkdir -p "$VITDET_DIR"
    if [ ! -f "$VITDET_DIR/model_final_f05665.pkl" ]; then
        echo "--- downloading ViTDet weights -> $VITDET_DIR ---"
        wget --no-check-certificate -q -O "$VITDET_DIR/model_final_f05665.pkl" \
            "https://dl.fbaipublicfiles.com/detectron2/ViTDet/COCO/cascade_mask_rcnn_vitdet_h/f328730692/model_final_f05665.pkl" || \
            echo "WARNING: ViTDet download failed; 手动放到 $VITDET_DIR/model_final_f05665.pkl" >&2
    fi

    # 0f. MoGe (FOV estimator, only needed by full 01, not 01b)
    echo "--- installing MoGe ---"
    MOGE_DIR="${MOGE_DIR:-$REPO_DIR/../MoGe}"
    if [ ! -d "$MOGE_DIR/.git" ]; then
        LD_LIBRARY_PATH= git clone https://github.com/microsoft/MoGe.git "$MOGE_DIR" || \
            LD_LIBRARY_PATH= git -c http.sslVerify=false clone https://github.com/microsoft/MoGe.git "$MOGE_DIR"
    fi
    pip install "${PIP_FLAGS[@]}" -e "$MOGE_DIR"

    # 0g. SAM2 (person segmentation, needed by 01b simplified)
    echo "--- setting up SAM2 ---"
    if [ ! -d "$SAM2_DIR/.git" ]; then
        rm -rf "$SAM2_DIR"
        echo "--- cloning SAM2 -> $SAM2_DIR ---"
        mkdir -p "$(dirname "$SAM2_DIR")"
        LD_LIBRARY_PATH= git clone https://github.com/facebookresearch/sam2.git "$SAM2_DIR" || \
            LD_LIBRARY_PATH= git -c http.sslVerify=false clone https://github.com/facebookresearch/sam2.git "$SAM2_DIR"
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

    # 0h. 2D Gaussian Splatting (step 05 — 3DGS reconstruction, same env)
    #     Gated by INSTALL_2DGS=1. Clones 2DGS repo + submodules, installs deps,
    #     builds two CUDA extensions (simple-knn + diff-surfel-rasterization).
    #     Needs CUDA toolkit (nvcc) at CUDA_HOME — same as detectron2 needs gxx.
    if [ "${INSTALL_2DGS:-0}" = "1" ]; then
        echo "--- setting up 2D Gaussian Splatting (step 05) ---"
        if [ ! -d "$GS2D_DIR/.git" ]; then
            echo "  cloning 2DGS -> $GS2D_DIR"
            mkdir -p "$(dirname "$GS2D_DIR")"
            LD_LIBRARY_PATH= git clone --recursive \
                https://github.com/hbb1/2d-gaussian-splatting.git "$GS2D_DIR" || \
                LD_LIBRARY_PATH= git -c http.sslVerify=false clone --recursive \
                https://github.com/hbb1/2d-gaussian-splatting.git "$GS2D_DIR"
        fi
        # Ensure submodules (simple-knn on gitlab.inria.fr, diff-surfel-rasterization on github).
        # git submodule update may fail if gitlab.inria.fr is blocked by proxy;
        # fall back to cloning each submodule directly.
        if [ ! -f "$GS2D_DIR/submodules/simple-knn/setup.py" ] || \
           [ ! -f "$GS2D_DIR/submodules/diff-surfel-rasterization/setup.py" ]; then
            echo "  ensuring 2DGS submodules"
            ( cd "$GS2D_DIR" && git submodule update --init --recursive ) || \
                ( cd "$GS2D_DIR" && git -c http.sslVerify=false submodule update --init --recursive ) || true
            # Fallback: clone submodules individually if submodule update failed
            SUBMOD_DIR="$GS2D_DIR/submodules"
            if [ ! -f "$SUBMOD_DIR/simple-knn/setup.py" ]; then
                echo "  cloning simple-knn directly (gitlab.inria.fr may be blocked)"
                rm -rf "$SUBMOD_DIR/simple-knn"
                git clone https://gitlab.inria.fr/bkerbl/simple-knn.git "$SUBMOD_DIR/simple-knn" || \
                    git -c http.sslVerify=false clone https://gitlab.inria.fr/bkerbl/simple-knn.git "$SUBMOD_DIR/simple-knn" || \
                    git clone https://github.com/bkerbl/simple-knn.git "$SUBMOD_DIR/simple-knn" || \
                    echo "  WARNING: simple-knn clone failed — try manually:" >&2
            fi
            if [ ! -f "$SUBMOD_DIR/diff-surfel-rasterization/setup.py" ]; then
                echo "  cloning diff-surfel-rasterization directly"
                rm -rf "$SUBMOD_DIR/diff-surfel-rasterization"
                git clone https://github.com/hbb1/diff-surfel-rasterization.git "$SUBMOD_DIR/diff-surfel-rasterization" || \
                    git -c http.sslVerify=false clone https://github.com/hbb1/diff-surfel-rasterization.git "$SUBMOD_DIR/diff-surfel-rasterization"
            fi
            # Final check
            if [ ! -f "$SUBMOD_DIR/simple-knn/setup.py" ] || \
               [ ! -f "$SUBMOD_DIR/diff-surfel-rasterization/setup.py" ]; then
                echo "  ❌ 2DGS submodules not ready. Manual clone:" >&2
                echo "     cd $SUBMOD_DIR" >&2
                echo "     git clone https://gitlab.inria.fr/bkerbl/simple-knn.git simple-knn" >&2
                echo "     git clone https://github.com/hbb1/diff-surfel-rasterization.git diff-surfel-rasterization" >&2
            fi
        fi
        # 2DGS Python deps (torch/numpy already installed, just the extras)
        echo "  installing 2DGS Python deps"
        pip install "${PIP_FLAGS[@]}" \
            open3d==0.18.0 mediapy==1.1.2 lpips==0.1.4 \
            scikit-image==0.21.0 tqdm==4.66.2 trimesh==4.3.2 \
            plyfile "setuptools<70"
        # Build CUDA extensions (same gxx as detectron2, needs nvcc from CUDA toolkit)
        export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
        if [ -x "$CUDA_HOME/bin/nvcc" ]; then
            echo "  nvcc: $($CUDA_HOME/bin/nvcc --version | tail -1 | xargs)"
            echo "  building CUDA ext: simple-knn"
            pip install "${PIP_FLAGS[@]}" --no-build-isolation \
                "$GS2D_DIR/submodules/simple-knn"
            echo "  building CUDA ext: diff-surfel-rasterization"
            pip install "${PIP_FLAGS[@]}" --no-build-isolation \
                "$GS2D_DIR/submodules/diff-surfel-rasterization"
            python -c "import simple_knn, diff_surfel_rasterization; print('  [OK] 2DGS CUDA exts')" || \
                echo "  WARNING: CUDA exts built but not importable" >&2
        else
            echo "  WARNING: nvcc not found at $CUDA_HOME/bin/nvcc — 2DGS CUDA exts NOT built." >&2
            echo "           Install CUDA toolkit 12.4 and re-run: INSTALL_2DGS=1 bash $0" >&2
        fi
    fi

    # 最后钉 numpy + setuptools——前面装的依赖会把 numpy 升到 2.x（detectron2 不兼容），
    # setuptools>=70 去掉了 pkg_resources（detectron2 model_zoo 要用）
    echo "--- pinning numpy==1.26.4 + setuptools==69.5.1 ---"
    pip install "${PIP_FLAGS[@]}" --force-reinstall --no-deps numpy==1.26.4 "setuptools<70"
    echo "  numpy=$(python -c 'import numpy; print(numpy.__version__)')  setuptools=$(python -c 'import setuptools; print(setuptools.__version__)')"

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

# --- 5b. verify 2DGS (step 05, only if INSTALL_2DGS was used) ---
if [ "${INSTALL_2DGS:-0}" = "1" ] || python -c "import diff_surfel_rasterization" 2>/dev/null; then
    echo "--- [5b] verify 2DGS (step 05) ---"
    if [ ! -d "$GS2D_DIR" ]; then
        echo "  [MISS] 2DGS repo — Run: INSTALL_DEPS=1 INSTALL_2DGS=1 bash $0" >&2
    elif ! python -c "import simple_knn, diff_surfel_rasterization" 2>/dev/null; then
        echo "  [MISS] 2DGS CUDA exts — Run: INSTALL_DEPS=1 INSTALL_2DGS=1 bash $0" >&2
    else
        echo "  [OK] 2DGS repo + CUDA exts"
    fi
fi

echo ""
echo "=== [00] Done. Env '$CONDA_ENV' ready. ==="
echo "    Next: INPUT_DIR=/path/to/subject_folder WEIGHT_PATH=/path/to/lora.safetensors bash $SCRIPT_DIR/run_all.sh"

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

    # 0h. MediaPipe (face mesh for step 01c — lightweight front-image pick on CPU).
    #     Only needed by 01c; 01/01b don't use it.
    #     mediapipe >= 1.0 removed the legacy 'solutions' API — it uses the Tasks
    #     API which needs a .task model file (face_landmarker.task, ~1MB).
    #     mediapipe < 1.0 uses legacy 'solutions.face_mesh' (no model file needed).
    #     Both are supported by pick_and_segment_mediapipe.py.
    #     mediapipe also pulls opencv-contrib-python (conflicts with opencv-python)
    #     — force-reinstall opencv-python afterward.
    echo "--- installing MediaPipe (step 01c) ---"
    pip install "${PIP_FLAGS[@]}" mediapipe || \
        echo "WARNING: mediapipe install failed; step 01c needs it (pip install mediapipe)" >&2
    pip install "${PIP_FLAGS[@]}" --force-reinstall --no-deps opencv-python

    # Download face_landmarker.task model (only needed for mediapipe >= 1.0;
    # legacy < 1.0 ignores it). Small file (~1MB), from Google storage.
    MP_MODEL_DIR="$WAN_MODEL_DIR/mediapipe"
    mkdir -p "$MP_MODEL_DIR"
    if [ ! -f "$MP_MODEL_DIR/face_landmarker.task" ]; then
        echo "--- downloading MediaPipe face_landmarker.task (for mediapipe >= 1.0 Tasks API) ---"
        wget --no-check-certificate -q -O "$MP_MODEL_DIR/face_landmarker.task" \
            "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task" || \
            echo "WARNING: face_landmarker.task download failed (Google storage may be blocked)." >&2
            echo "         Manual: wget -O $MP_MODEL_DIR/face_landmarker.task" >&2
            echo "           https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task" >&2
    fi

    # 0i. 2D Gaussian Splatting (step 05 — 3DGS reconstruction, same env)
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
        # git submodule update may fail if gitlab.inria.fr is blocked/slow;
        # fall back to cloning each submodule directly (LD_LIBRARY_PATH= fixes
        # conda libffi vs system libp11-kit conflict that makes git hang).
        if [ ! -f "$GS2D_DIR/submodules/simple-knn/setup.py" ] || \
           [ ! -f "$GS2D_DIR/submodules/diff-surfel-rasterization/setup.py" ]; then
            echo "  ensuring 2DGS submodules"
            ( cd "$GS2D_DIR" && LD_LIBRARY_PATH= git submodule update --init --recursive ) || \
                ( cd "$GS2D_DIR" && LD_LIBRARY_PATH= git -c http.sslVerify=false submodule update --init --recursive ) || true
            # Fallback: clone submodules individually
            SUBMOD_DIR="$GS2D_DIR/submodules"
            if [ ! -f "$SUBMOD_DIR/simple-knn/setup.py" ]; then
                echo "  cloning simple-knn (gitlab.inria.fr, may be slow)"
                rm -rf "$SUBMOD_DIR/simple-knn"
                LD_LIBRARY_PATH= git clone https://gitlab.inria.fr/bkerbl/simple-knn.git "$SUBMOD_DIR/simple-knn" || \
                    LD_LIBRARY_PATH= git -c http.sslVerify=false clone https://gitlab.inria.fr/bkerbl/simple-knn.git "$SUBMOD_DIR/simple-knn" || {
                        # Last resort: download zip (single HTTP request, faster than git clone)
                        echo "  git clone failed, trying zip download..."
                        _zip="/tmp/simple-knn-main.zip"
                        curl -k -L --connect-timeout 30 --max-time 300 -o "$_zip" \
                            "https://gitlab.inria.fr/bkerbl/simple-knn/-/archive/main/simple-knn-main.zip" && \
                            unzip -o "$_zip" -d "$SUBMOD_DIR" && \
                            mv "$SUBMOD_DIR/simple-knn-main" "$SUBMOD_DIR/simple-knn" || \
                            echo "  ❌ simple-knn download failed — manual:" >&2
                    }
            fi
            if [ ! -f "$SUBMOD_DIR/diff-surfel-rasterization/setup.py" ]; then
                echo "  cloning diff-surfel-rasterization"
                rm -rf "$SUBMOD_DIR/diff-surfel-rasterization"
                LD_LIBRARY_PATH= git clone https://github.com/hbb1/diff-surfel-rasterization.git "$SUBMOD_DIR/diff-surfel-rasterization" || \
                    LD_LIBRARY_PATH= git -c http.sslVerify=false clone https://github.com/hbb1/diff-surfel-rasterization.git "$SUBMOD_DIR/diff-surfel-rasterization"
            fi
            # GLM (header-only math lib, also a 2DGS submodule — diff-surfel-rasterization needs it)
            GLM_DIR="$GS2D_DIR/third_party/glm"
            if [ ! -f "$GLM_DIR/glm/glm.hpp" ]; then
                echo "  cloning GLM (diff-surfel-rasterization dependency)"
                rm -rf "$GLM_DIR"
                LD_LIBRARY_PATH= git clone https://github.com/g-truc/glm.git "$GLM_DIR" || \
                    LD_LIBRARY_PATH= git -c http.sslVerify=false clone https://github.com/g-truc/glm.git "$GLM_DIR"
            fi
            # Final check
            if [ ! -f "$SUBMOD_DIR/simple-knn/setup.py" ] || \
               [ ! -f "$SUBMOD_DIR/diff-surfel-rasterization/setup.py" ]; then
                echo "  ❌ 2DGS submodules not ready. Manual:" >&2
                echo "     cd $SUBMOD_DIR" >&2
                echo "     curl -kL -o sk.zip https://gitlab.inria.fr/bkerbl/simple-knn/-/archive/main/simple-knn-main.zip && unzip sk.zip && mv simple-knn-main simple-knn" >&2
                echo "     git clone https://github.com/hbb1/diff-surfel-rasterization.git diff-surfel-rasterization" >&2
            fi
        fi
        # 2DGS Python deps (torch/numpy already installed, just the extras)
        # matplotlib<3.10: 2DGS colormap() uses fig.canvas.tostring_rgb() (removed in 3.10)
        echo "  installing 2DGS Python deps"
        pip install "${PIP_FLAGS[@]}" \
            open3d==0.18.0 mediapy==1.1.2 lpips==0.1.4 \
            scikit-image==0.21.0 tqdm==4.66.2 trimesh==4.3.2 \
            plyfile "setuptools<70" "matplotlib<3.10"
        # Build CUDA extensions (same gxx as detectron2, needs nvcc from CUDA toolkit).
        # Auto-detect CUDA 12.x: /usr/local/cuda might point to 11.8, find cuda-12.4.
        export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
        if [ -x "$CUDA_HOME/bin/nvcc" ]; then
            _nvcc_ver="$($CUDA_HOME/bin/nvcc --version | grep -oP 'release \K[0-9]+\.[0-9]+' || echo '')"
            _torch_ver="$(python -c 'import torch; print(torch.version.cuda)' 2>/dev/null || echo '')"
            if [ -n "$_nvcc_ver" ] && [ -n "$_torch_ver" ]; then
                _nvcc_major="${_nvcc_ver%%.*}"
                _torch_major="${_torch_ver%%.*}"
                if [ "$_nvcc_major" != "$_torch_major" ]; then
                    echo "  ⚠️ CUDA_HOME=$CUDA_HOME is $_nvcc_ver but torch is $_torch_ver"
                    for _d in /usr/local/cuda-${_torch_major}*; do
                        if [ -x "$_d/bin/nvcc" ]; then
                            export CUDA_HOME="$_d"
                            echo "  → switched CUDA_HOME to $CUDA_HOME"
                            break
                        fi
                    done
                fi
            fi
            export PATH="$CUDA_HOME/bin:$PATH"
            echo "  nvcc: $($CUDA_HOME/bin/nvcc --version | tail -1 | xargs)"
            echo "  building CUDA ext: simple-knn"
            pip install "${PIP_FLAGS[@]}" --no-build-isolation --no-deps --no-index \
                "$GS2D_DIR/submodules/simple-knn"
            echo "  building CUDA ext: diff-surfel-rasterization"
            pip install "${PIP_FLAGS[@]}" --no-build-isolation --no-deps --no-index \
                "$GS2D_DIR/submodules/diff-surfel-rasterization"
            python -c "import simple_knn, diff_surfel_rasterization; print('  [OK] 2DGS CUDA exts')" || \
                echo "  WARNING: CUDA exts built but not importable" >&2
        else
            echo "  WARNING: nvcc not found at $CUDA_HOME/bin/nvcc — 2DGS CUDA exts NOT built." >&2
            echo "           Install CUDA toolkit 12.4 and re-run: INSTALL_2DGS=1 bash $0" >&2
        fi
    fi

    # 0j. Gaussian Opacity Fields (step 05a — GOF 重建, 与 05 (2DGS) 并列)
    #     GOF 用 Marching Tetrahedra 提网格 (非 TSDF), 网格质量超 2DGS。
    #     3 个扩展:
    #       - diff-gaussian-rasterization (3DGS 光栅化器, 不同于 2DGS 的 diff-surfel-rasterization)
    #       - simple-knn (与 2DGS 同源, 若已装则复用)
    #       - tetra-triangulation (C++ 扩展, 需 cmake/gmp/cgal; 来自 tetra-nerf)
    #     GOF 自己是 3DGS + Mip-Splatting 的扩展 (anti-aliasing + normal/distortion 正则)。
    if [ "${INSTALL_GOF:-0}" = "1" ]; then
        echo "--- setting up Gaussian Opacity Fields (step 05a GOF) ---"
        if [ ! -d "$GOF_DIR/.git" ]; then
            echo "  cloning GOF -> $GOF_DIR"
            mkdir -p "$(dirname "$GOF_DIR")"
            LD_LIBRARY_PATH= git clone --recursive \
                https://github.com/autonomousvision/gaussian-opacity-fields.git "$GOF_DIR" || \
                LD_LIBRARY_PATH= git -c http.sslVerify=false clone --recursive \
                https://github.com/autonomousvision/gaussian-opacity-fields.git "$GOF_DIR"
        fi
        # Ensure submodules (diff-gaussian-rasterization on github, simple-knn on
        # gitlab.inria.fr, tetra-triangulation on github). Fall back to manual clone.
        if [ ! -f "$GOF_DIR/submodules/diff-gaussian-rasterization/setup.py" ] || \
           [ ! -f "$GOF_DIR/submodules/simple-knn/setup.py" ] || \
           [ ! -f "$GOF_DIR/submodules/tetra-triangulation/CMakeLists.txt" ]; then
            echo "  ensuring GOF submodules"
            ( cd "$GOF_DIR" && LD_LIBRARY_PATH= git submodule update --init --recursive ) || \
                ( cd "$GOF_DIR" && LD_LIBRARY_PATH= git -c http.sslVerify=false submodule update --init --recursive ) || true
            # Fallback: clone submodules individually
            SUBMOD_DIR="$GOF_DIR/submodules"
            # diff-gaussian-rasterization (3DGS 光栅化器, 不同于 2DGS 的 diff-surfel-rasterization)
            if [ ! -f "$SUBMOD_DIR/diff-gaussian-rasterization/setup.py" ]; then
                echo "  cloning diff-gaussian-rasterization"
                rm -rf "$SUBMOD_DIR/diff-gaussian-rasterization"
                LD_LIBRARY_PATH= git clone https://github.com/graphdeco-inria/diff-gaussian-rasterization.git "$SUBMOD_DIR/diff-gaussian-rasterization" || \
                    LD_LIBRARY_PATH= git -c http.sslVerify=false clone https://github.com/graphdeco-inria/diff-gaussian-rasterization.git "$SUBMOD_DIR/diff-gaussian-rasterization"
            fi
            # simple-knn (与 2DGS 同源, gitlab.inria.fr 可能被封)
            if [ ! -f "$SUBMOD_DIR/simple-knn/setup.py" ]; then
                echo "  cloning simple-knn (gitlab.inria.fr, may be slow)"
                rm -rf "$SUBMOD_DIR/simple-knn"
                LD_LIBRARY_PATH= git clone https://gitlab.inria.fr/bkerbl/simple-knn.git "$SUBMOD_DIR/simple-knn" || \
                    LD_LIBRARY_PATH= git -c http.sslVerify=false clone https://gitlab.inria.fr/bkerbl/simple-knn.git "$SUBMOD_DIR/simple-knn" || {
                        # Last resort: zip download (single HTTP request, faster than git clone)
                        echo "  git clone failed, trying zip download..."
                        _zip="/tmp/simple-knn-main.zip"
                        curl -k -L --connect-timeout 30 --max-time 300 -o "$_zip" \
                            "https://gitlab.inria.fr/bkerbl/simple-knn/-/archive/main/simple-knn-main.zip" && \
                            unzip -o "$_zip" -d "$SUBMOD_DIR" && \
                            mv "$SUBMOD_DIR/simple-knn-main" "$SUBMOD_DIR/simple-knn" || \
                            echo "  ❌ simple-knn download failed — see 2DGS setup fallback" >&2
                    }
            fi
            # tetra-triangulation (C++ 扩展, 来自 tetra-nerf 的 triangulation 部分)
            if [ ! -f "$SUBMOD_DIR/tetra-triangulation/CMakeLists.txt" ]; then
                echo "  cloning tetra-triangulation (from tetra-nerf repo)"
                rm -rf "$SUBMOD_DIR/tetra-triangulation"
                LD_LIBRARY_PATH= git clone https://github.com/jkulhanek/tetra-nerf.git "$SUBMOD_DIR/tetra-triangulation" || \
                    LD_LIBRARY_PATH= git -c http.sslVerify=false clone https://github.com/jkulhanek/tetra-nerf.git "$SUBMOD_DIR/tetra-triangulation"
            fi
            # GLM (header-only math lib, diff-gaussian-rasterization 需要)
            GLM_DIR_GOF="$GOF_DIR/third_party/glm"
            if [ ! -f "$GLM_DIR_GOF/glm/glm.hpp" ]; then
                echo "  cloning GLM (diff-gaussian-rasterization dependency)"
                rm -rf "$GLM_DIR_GOF"
                LD_LIBRARY_PATH= git clone https://github.com/g-truc/glm.git "$GLM_DIR_GOF" || \
                    LD_LIBRARY_PATH= git -c http.sslVerify=false clone https://github.com/g-truc/glm.git "$GLM_DIR_GOF"
            fi
            # Final check
            if [ ! -f "$SUBMOD_DIR/diff-gaussian-rasterization/setup.py" ] || \
               [ ! -f "$SUBMOD_DIR/tetra-triangulation/CMakeLists.txt" ]; then
                echo "  ❌ GOF submodules not ready. Manual:" >&2
                echo "     cd $SUBMOD_DIR" >&2
                echo "     git clone https://github.com/graphdeco-inria/diff-gaussian-rasterization.git diff-gaussian-rasterization" >&2
                echo "     git clone https://github.com/jkulhanek/tetra-nerf.git tetra-triangulation" >&2
            fi
        fi
        # Install cmake + gmp + cgal (for tetra-triangulation C++ build)
        # ⚠️ 显式 pin python=3.10 防 GraalPy 掉包 (与 gxx 同理, 见上方注释)
        echo "  installing cmake + gmp + cgal (for tetra-triangulation)"
        conda install -y -c conda-forge --no-update-deps cmake gmp cgal python=3.10
        # 校验 python 没被掉包成 GraalPy
        impl3="$(python -c 'import platform; print(platform.python_implementation())' 2>/dev/null || echo unknown)"
        if [ "$impl3" != "CPython" ]; then
            echo "  ⚠️ cmake/gmp/cgal install 把 python 掉包成 '$impl3'，自动修复..." >&2
            conda install -y -c defaults python=3.10 --force-reinstall
            pip install "${PIP_FLAGS[@]}" --force-reinstall --no-deps numpy==1.26.4
            for w in "$WAN_MODEL_DIR"/torch-*.whl "$WAN_MODEL_DIR"/torchvision-*.whl; do
                [ -f "$w" ] && pip install --force-reinstall --no-deps "$w"
            done
        fi
        # GOF Python deps (torch/numpy already installed; open3d/lpips/trimesh/plyfile
        # already installed if INSTALL_2DGS was run; just the GOF-specific extras)
        echo "  installing GOF Python deps (ninja, GPUtil)"
        pip install "${PIP_FLAGS[@]}" ninja GPUtil || true
        # Auto-detect CUDA 12.x (与 2DGS 同, /usr/local/cuda 可能是 11.8)
        export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
        if [ -x "$CUDA_HOME/bin/nvcc" ]; then
            _nvcc_ver="$($CUDA_HOME/bin/nvcc --version | grep -oP 'release \K[0-9]+\.[0-9]+' || echo '')"
            _torch_ver="$(python -c 'import torch; print(torch.version.cuda)' 2>/dev/null || echo '')"
            if [ -n "$_nvcc_ver" ] && [ -n "$_torch_ver" ]; then
                _nvcc_major="${_nvcc_ver%%.*}"
                _torch_major="${_torch_ver%%.*}"
                if [ "$_nvcc_major" != "$_torch_major" ]; then
                    echo "  ⚠️ CUDA_HOME=$CUDA_HOME is $_nvcc_ver but torch is $_torch_ver"
                    for _d in /usr/local/cuda-${_torch_major}*; do
                        if [ -x "$_d/bin/nvcc" ]; then
                            export CUDA_HOME="$_d"
                            echo "  → switched CUDA_HOME to $CUDA_HOME"
                            break
                        fi
                    done
                fi
            fi
            export PATH="$CUDA_HOME/bin:$PATH"
            export CC=$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-gcc
            export CXX=$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-g++
            echo "  nvcc: $($CUDA_HOME/bin/nvcc --version | tail -1 | xargs)"
            # Build tetra-triangulation (C++ extension for Delaunay tetrahedralization,
            # used by GOF's extract_mesh.py for Marching Tetrahedra).
            # Needs: cmake + gmp + cgal (just installed) + torch (already in env).
            # Note: tetra-triangulation is a submodule from tetra-nerf repo. The full
            # tetra-nerf CMakeLists.txt requires OptiX (NVIDIA ray-tracing lib). GOF's
            # submodule MAY be a stripped-down version without OptiX dep — try build,
            # if it fails on OptiX, see error message below for manual fix.
            TETRA_DIR="$GOF_DIR/submodules/tetra-triangulation"
            if [ -f "$TETRA_DIR/CMakeLists.txt" ]; then
                echo "  building tetra-triangulation (cmake + make)"
                ( cd "$TETRA_DIR" && \
                  cmake . -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH="$CONDA_PREFIX" && \
                  make -j ) || {
                    echo "  ⚠️ tetra-triangulation build failed." >&2
                    echo "     Common cause: OptiX not found (full tetra-nerf CMakeLists.txt requires it)." >&2
                    echo "     If GOF's submodule is the full tetra-nerf (not stripped):" >&2
                    echo "       1) Download OptiX 7.6 from NVIDIA: https://developer.nvidia.com/designworks/optix/downloads/legacy" >&2
                    echo "       2) export OPTIX_PATH=/path/to/optix" >&2
                    echo "       3) Re-run: INSTALL_GOF=1 bash $0" >&2
                }
                # pip install the Python wrapper (--no-deps: skip nerfstudio, trimesh is already installed)
                pip install "${PIP_FLAGS[@]}" --no-build-isolation --no-deps -e "$TETRA_DIR" || \
                    echo "  ⚠️ tetra-triangulation pip install failed (C++ ext may still be usable if make succeeded)" >&2
            else
                echo "  ⚠️ tetra-triangulation submodule not found at $TETRA_DIR" >&2
                echo "     GOF mesh extraction (extract_mesh.py) will not work without it." >&2
            fi
            # Build diff-gaussian-rasterization (3DGS CUDA rasterizer, GOF's train.py needs it)
            echo "  building CUDA ext: diff-gaussian-rasterization"
            pip install "${PIP_FLAGS[@]}" --no-build-isolation --no-deps --no-index \
                "$GOF_DIR/submodules/diff-gaussian-rasterization" || \
                echo "  ⚠️ diff-gaussian-rasterization build failed" >&2
            # Build simple-knn if not already installed (from 2DGS setup)
            if ! python -c "import simple_knn" 2>/dev/null; then
                echo "  building CUDA ext: simple-knn (not yet installed)"
                pip install "${PIP_FLAGS[@]}" --no-build-isolation --no-deps --no-index \
                    "$GOF_DIR/submodules/simple-knn" || \
                    echo "  ⚠️ simple-knn build failed" >&2
            else
                echo "  [OK] simple-knn already installed (from 2DGS setup)"
            fi
            # Verify imports
            python -c "import diff_gaussian_rasterization; print('  [OK] diff_gaussian_rasterization')" || \
                echo "  ⚠️ diff_gaussian_rasterization not importable" >&2
            python -c "import simple_knn; print('  [OK] simple_knn')" || \
                echo "  ⚠️ simple_knn not importable" >&2
            python -c "from tetranerf.utils.extension import cpp; print('  [OK] tetra-triangulation')" 2>/dev/null || \
                echo "  ⚠️ tetra-triangulation (tetranerf.utils.extension.cpp) not importable — mesh extraction won't work" >&2
        else
            echo "  ⚠️ nvcc not found at $CUDA_HOME/bin/nvcc — GOF CUDA exts NOT built." >&2
            echo "           Install CUDA toolkit 12.4 and re-run: INSTALL_GOF=1 bash $0" >&2
        fi
    fi

    # 0k. LHM (step 05b — feed-forward 单图人体高斯重建, ICCV 2025)。
    #     与 05 (2DGS) / 05a (GOF) 并列, 但范式不同: 单张正面图 → 前馈网络 → 可动画人体高斯 + 网格。
    #     ⚠️ 用独立 conda env `lhm` (torch 2.3.0, numpy 1.23.0), 与 wan22_rotate 的
    #        torch 2.6.0 / numpy 1.26.4 不兼容——必须在独立 env 里装。
    #     流程: 建 lhm env → 装 torch 2.3.0+cu121 + xformers + requirements → 装 LHM 改版
    #     sam2 + ashawkey diff-gaussian-rasterization + simple-knn + pytorch3d →
    #     clone LHM 仓 → 软链 pretrained_models → 下 prior_model + LHM 权重。
    #     结束后切回 wan22_rotate env (本块下面的 numpy 钉版本 + verify 都在 wan22_rotate)。
    if [ "${INSTALL_LHM:-0}" = "1" ]; then
        echo "--- setting up LHM (step 05b, separate env 'lhm') ---"
        LHM_ENV="lhm"
        # 1) 建 lhm env (CPython 3.10, 匹配 LHM 的 cp310 轮子 / 源码编译)
        if ! conda env list 2>/dev/null | grep -qw "$LHM_ENV"; then
            echo "  creating conda env '$LHM_ENV' (python=3.10)"
            conda create -n "$LHM_ENV" python=3.10 -y
        fi
        # 切到 lhm env 做安装 (CONDA_ENV 仍 = wan22_rotate, 块尾切回)
        conda activate "$LHM_ENV"
        impl_lhm="$(python -c 'import platform; print(platform.python_implementation())' 2>/dev/null || echo unknown)"
        if [ "$impl_lhm" != "CPython" ]; then
            echo "  ⚠️ lhm env python 实现是 $impl_lhm (应为 CPython), 修复..." >&2
            conda install -y -c defaults python=3.10 --force-reinstall
        fi
        echo "  lhm env: python=$(python --version 2>&1 | cut -d' ' -f2)"

        # 2) torch 2.3.0+cu121 (PyPI 默认即 cu121 轮子, 不走 download.pytorch.org——代理封 403)
        #    本仓 torch 2.6 轮子是 cp310 但 LHM 钉 torch 2.3.0, 不能复用。
        echo "  installing torch 2.3.0 + torchvision 0.18.0 + torchaudio 2.3.0 (cu121, from PyPI)"
        pip install "${PIP_FLAGS[@]}" torch==2.3.0 torchvision==0.18.0 torchaudio==2.3.0 || \
            echo "  ⚠️ torch 2.3.0 install failed (PyPI/代理?), 手动: pip install torch==2.3.0 ..." >&2
        # xformers (LHM INSTALL.md step 2; PyPI 的 0.0.26.post1 匹配 torch 2.3.0+cu121)
        echo "  installing xformers==0.0.26.post1"
        pip install "${PIP_FLAGS[@]}" xformers==0.0.26.post1 || \
            echo "  ⚠️ xformers install failed; LHM 部分算子可能回退到 torch attention" >&2

        # 3) LHM requirements.txt 依赖 (torch/numpy 已装, 这里装其余)
        #    ⚠️ LHM 钉 numpy==1.23.0, 与 wan22_rotate 的 1.26.4 不同——这就是要独立 env 的原因。
        echo "  installing LHM requirements.txt deps"
        pip install "${PIP_FLAGS[@]}" \
            einops roma accelerate smplx chumpy decord==0.6.0 \
            diffusers==0.32.0 dna==0.0.1 imageio==2.34.1 imageio-ffmpeg \
            jaxtyping==0.2.38 kiui==0.2.14 kornia==0.7.2 loguru==0.7.3 \
            lpips==0.1.4 matplotlib==3.5.3 megfile==4.1.0.post2 \
            numpy==1.23.0 omegaconf==2.3.0 open3d==0.19.0 opencv-python \
            opencv-python-headless Pillow==10.4.0 plyfile pygltflib==1.16.2 \
            pyrender==0.1.45 PyYAML==6.0.1 rembg==2.0.63 Requests==2.32.3 \
            scipy spaces setuptools==74.0.0 \
            taming_transformers_rom1504==0.0.6 timm==1.0.15 tqdm==4.66.4 \
            transformers==4.41.2 trimesh==4.4.9 typeguard==2.13.3 xatlas==0.0.9 || \
            echo "  ⚠️ 部分 requirements 安装失败 (上面有 WARNING), 继续装其余" >&2

        # 4) basicsr 从源码装 (requirements 的 basicsr==1.4.2 和 torchvision 冲突, 走源码)
        echo "  installing basicsr from source (XPixelGroup/BasicSR)"
        pip uninstall -y basicsr 2>/dev/null || true
        pip install "${PIP_FLAGS[@]}" "git+https://github.com/XPixelGroup/BasicSR.git" || \
            LD_LIBRARY_PATH= pip install "${PIP_FLAGS[@]}" \
                "git+https://github.com/XPixelGroup/BasicSR.git" || \
            echo "  ⚠️ basicsr 源码安装失败; app/animation 可能受影响" >&2

        # 5) LHM 改版 sam2 (hitsz-zuoqi/sam2, 非官方 facebookresearch/sam2)。
        #    手动 clone 再 pip install -e (LD_LIBRARY_PATH= 防 conda libffi 冲突让 git 卡住)。
        SAM2_LHM_DIR="${SAM2_LHM_DIR:-$REPO_DIR/../sam2_lhm}"
        if [ ! -d "$SAM2_LHM_DIR/.git" ]; then
            echo "  cloning LHM-modified sam2 -> $SAM2_LHM_DIR"
            mkdir -p "$(dirname "$SAM2_LHM_DIR")"
            LD_LIBRARY_PATH= git clone https://github.com/hitsz-zuoqi/sam2.git "$SAM2_LHM_DIR" || \
                LD_LIBRARY_PATH= git -c http.sslVerify=false clone https://github.com/hitsz-zuoqi/sam2.git "$SAM2_LHM_DIR"
        fi
        pip install "${PIP_FLAGS[@]}" -e "$SAM2_LHM_DIR" || \
            echo "  ⚠️ LHM sam2 install failed; LHM 会回退到 rembg 抠背景 (质量略降)" >&2

        # 6) 装 gxx 12 (编 CUDA 扩展要的; python=3.10 pin 防 GraalPy 掉包, --no-update-deps)
        echo "  installing gxx_linux-64=12 into lhm env (for CUDA ext compile)"
        conda install -y -c conda-forge --no-update-deps gxx_linux-64=12 python=3.10 || \
            echo "  ⚠️ gxx install failed; 用系统 gcc 编 CUDA 扩展可能失败" >&2
        impl_lhm2="$(python -c 'import platform; print(platform.python_implementation())' 2>/dev/null || echo unknown)"
        if [ "$impl_lhm2" != "CPython" ]; then
            echo "  ⚠️ gxx 把 lhm python 掉包成 $impl_lhm2, 修复..." >&2
            conda install -y -c defaults python=3.10 --force-reinstall
            pip install "${PIP_FLAGS[@]}" --force-reinstall --no-deps numpy==1.23.0
        fi
        export CC=$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-gcc
        export CXX=$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-g++

        # 7) CUDA toolkit 路径 (auto-detect cuda-12.x; /usr/local/cuda 可能是 11.8)
        export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
        if [ -x "$CUDA_HOME/bin/nvcc" ]; then
            _nvcc_ver="$($CUDA_HOME/bin/nvcc --version | grep -oP 'release \K[0-9]+\.[0-9]+' || echo '')"
            echo "  nvcc: ${_nvcc_ver:-unknown} at $CUDA_HOME"
            export PATH="$CUDA_HOME/bin:$PATH"
        else
            echo "  ⚠️ nvcc not found at $CUDA_HOME/bin/nvcc — LHM CUDA exts NOT built." >&2
        fi

        # 8) diff-gaussian-rasterization (ashawkey 改版, 不同于 GOF 的 graphdeco-inria 版)。
        #    LHM INSTALL.md step 5: 用 ashawkey/diff-gaussian-rasterization (含修改)。
        DGR_LHM_DIR="${DGR_LHM_DIR:-$REPO_DIR/../diff-gaussian-rasterization_lhm}"
        if [ ! -d "$DGR_LHM_DIR/.git" ]; then
            echo "  cloning ashawkey diff-gaussian-rasterization -> $DGR_LHM_DIR"
            mkdir -p "$(dirname "$DGR_LHM_DIR")"
            LD_LIBRARY_PATH= git clone --recursive https://github.com/ashawkey/diff-gaussian-rasterization.git "$DGR_LHM_DIR" || \
                LD_LIBRARY_PATH= git -c http.sslVerify=false clone --recursive \
                https://github.com/ashawkey/diff-gaussian-rasterization.git "$DGR_LHM_DIR"
        fi
        if [ -f "$DGR_LHM_DIR/setup.py" ] || [ -f "$DGR_LHM_DIR/pyproject.toml" ]; then
            echo "  building diff-gaussian-rasterization (LHM 改版)"
            pip install "${PIP_FLAGS[@]}" --no-build-isolation --no-deps "$DGR_LHM_DIR" || \
                echo "  ⚠️ diff-gaussian-rasterization build failed (nvcc/gcc 版本?)" >&2
        fi
        # simple-knn (camenduru 版, LHM INSTALL.md 指定)
        SK_LHM_DIR="${SK_LHM_DIR:-$REPO_DIR/../simple-knn_lhm}"
        if [ ! -d "$SK_LHM_DIR/.git" ]; then
            echo "  cloning simple-knn (camenduru) -> $SK_LHM_DIR"
            mkdir -p "$(dirname "$SK_LHM_DIR")"
            LD_LIBRARY_PATH= git clone https://github.com/camenduru/simple-knn.git "$SK_LHM_DIR" || \
                LD_LIBRARY_PATH= git -c http.sslVerify=false clone https://github.com/camenduru/simple-knn.git "$SK_LHM_DIR"
        fi
        if [ -f "$SK_LHM_DIR/setup.py" ] || [ -f "$SK_LHM_DIR/pyproject.toml" ]; then
            echo "  building simple-knn"
            pip install "${PIP_FLAGS[@]}" --no-build-isolation --no-deps "$SK_LHM_DIR" || \
                echo "  ⚠️ simple-knn build failed" >&2
        fi

        # 9) pytorch3d (INSTALL.md step 6; mesh 导出 + 某些几何算子用)。
        #    先装 fvcore + iopath (依赖), 再试预编译轮子, 失败则源码编译。
        echo "  installing pytorch3d deps (fvcore, iopath)"
        pip install "${PIP_FLAGS[@]}" fvcore iopath || true
        echo "  installing pytorch3d (try wheel, fallback to source)"
        if ! pip install "${PIP_FLAGS[@]}" --no-build-isolation "git+https://github.com/facebookresearch/pytorch3d.git@v0.7.6" 2>/dev/null; then
            LD_LIBRARY_PATH= pip install "${PIP_FLAGS[@]}" --no-build-isolation \
                "git+https://github.com/facebookresearch/pytorch3d.git@v0.7.6" || \
                echo "  ⚠️ pytorch3d build failed (mesh 导出可能受影响); " \
                     "手动: 见 https://github.com/facebookresearch/pytorch3d/blob/main/INSTALL.md" >&2
        fi

        # 10) clone LHM 仓 + 软链 pretrained_models -> $LHM_MODEL_DIR (权重统一放 model/)
        if [ ! -d "$LHM_DIR/.git" ]; then
            echo "  cloning LHM -> $LHM_DIR"
            mkdir -p "$(dirname "$LHM_DIR")"
            LD_LIBRARY_PATH= git clone https://github.com/aigc3d/LHM.git "$LHM_DIR" || \
                LD_LIBRARY_PATH= git -c http.sslVerify=false clone https://github.com/aigc3d/LHM.git "$LHM_DIR"
        fi
        mkdir -p "$LHM_MODEL_DIR"
        # LHM 代码用相对 ./pretrained_models/ 读权重; 软链到共享 model/LHM (不复制)
        if [ ! -e "$LHM_DIR/pretrained_models" ]; then
            ln -sfn "$LHM_MODEL_DIR" "$LHM_DIR/pretrained_models"
        elif [ -L "$LHM_DIR/pretrained_models" ]; then
            ln -sfn "$LHM_MODEL_DIR" "$LHM_DIR/pretrained_models"
        else
            echo "  ⚠️ $LHM_DIR/pretrained_models 已存在且非软链, 不覆盖 (权重可能已在仓内)" >&2
        fi

        # 11) 下 prior_model (SMPL-X / sapiens / sam2 / gagatracker / dense_sample_points 等)。
        #     OSS bucket (virutalbuy-public.oss-...), 一般不被代理封; 解压到 $LHM_MODEL_DIR。
        OSS_BASE="https://virutalbuy-public.oss-cn-hangzhou.aliyuncs.com/share/aigc3d/data/LHM"
        if [ ! -d "$LHM_MODEL_DIR/human_model_files" ] && \
           [ ! -f "$LHM_MODEL_DIR/.prior_extracted" ]; then
            echo "  downloading LHM prior models (LHM_prior_model.tar, ~?GB from OSS)"
            _tar="$LHM_MODEL_DIR/LHM_prior_model.tar"
            wget --no-check-certificate -c -O "$_tar" "$OSS_BASE/LHM_prior_model.tar" || \
                echo "  ⚠️ prior_model 下载失败 (OSS 被封?); 手动: wget -O $_tar $OSS_BASE/LHM_prior_model.tar" >&2
            if [ -f "$_tar" ]; then
                echo "  extracting LHM_prior_model.tar -> $LHM_MODEL_DIR"
                tar -xf "$_tar" -C "$LHM_MODEL_DIR" && touch "$LHM_MODEL_DIR/.prior_extracted"
                # tar 解压出 ./pretrained_models/* ; 移到 $LHM_MODEL_DIR 根 (软链指向这里)
                if [ -d "$LHM_MODEL_DIR/pretrained_models" ]; then
                    cp -rn "$LHM_MODEL_DIR/pretrained_models/." "$LHM_MODEL_DIR/" 2>/dev/null || true
                fi
            fi
        else
            echo "  [OK] LHM prior models already present at $LHM_MODEL_DIR"
        fi

        # 12) 下 LHM 主权重 (默认 LHM-500M-HF; HF repo 3DAIGC/LHM-500M-HF)。
        #     ⚠️ _env.sh 设了 HF_HUB_OFFLINE=1 防联网报错; 下载时临时关掉。
        #     HF 被代理封则试 ModelScope (Damo_XR_Lab/LHM-500M-HF), 都不行给手动指令。
        LHM_MODEL_NAME="${LHM_MODEL_NAME:-LHM-500M-HF}"
        LHM_HF_DIR="$LHM_MODEL_DIR/huggingface"
        mkdir -p "$LHM_HF_DIR"
        echo "  downloading LHM model '$LHM_MODEL_NAME' (HF 3DAIGC/$LHM_MODEL_NAME)"
        if [ ! -d "$LHM_HF_DIR/models--3DAIGC--$(echo "$LHM_MODEL_NAME" | sed 's/-/_/g')" ]; then
            HF_HUB_OFFLINE=0 TRANSFORMERS_OFFLINE=0 python -c "
from huggingface_hub import snapshot_download
snapshot_download(repo_id='3DAIGC/${LHM_MODEL_NAME}', cache_dir='${LHM_HF_DIR}')
print('  [OK] HF download done')
" 2>&1 || {
                echo "  ⚠️ HF 下载失败 (代理封 huggingface.co?), 试 ModelScope..." >&2
                pip install "${PIP_FLAGS[@]}" modelscope 2>/dev/null || true
                python -c "
from modelscope import snapshot_download
snapshot_download(model_id='Damo_XR_Lab/${LHM_MODEL_NAME}', cache_dir='${LHM_HF_DIR}')
print('  [OK] ModelScope download done')
" 2>&1 || {
                    echo "  ⚠️ ModelScope 也失败。手动下载:" >&2
                    echo "    HF:      https://huggingface.co/3DAIGC/${LHM_MODEL_NAME}" >&2
                    echo "    ModelScope: https://modelscope.cn/models/Damo_XR_Lab/${LHM_MODEL_NAME}" >&2
                    echo "    放到 $LHM_HF_DIR/ (snapshot_download 的 cache 结构)" >&2
                }
            }
        else
            echo "  [OK] LHM model '$LHM_MODEL_NAME' already present"
        fi

        # 13) (可选) 下 motion 示例 (animation 用; mesh 导出不需要)。
        if [ "${LHM_DOWNLOAD_MOTION:-0}" = "1" ]; then
            echo "  downloading LHM motion examples (motion_video.tar)"
            if [ ! -d "$LHM_DIR/train_data/motion_video" ]; then
                _mtar="$LHM_DIR/train_data/motion_video.tar"
                mkdir -p "$(dirname "$_mtar")"
                wget --no-check-certificate -c -O "$_mtar" "$OSS_BASE/motion_video.tar" || \
                    echo "  ⚠️ motion_video 下载失败 (可选, 仅 animation 用)" >&2
                [ -f "$_mtar" ] && tar -xf "$_mtar" -C "$LHM_DIR/train_data/" 2>/dev/null || true
            fi
        fi

        # 14) (可选) 下 video2motion 权重 (从 rotate_360.mp4 提取 SMPL-X 动作用)。
        if [ "${LHM_DOWNLOAD_POSE:-0}" = "1" ]; then
            echo "  downloading yolov8x + vitpose (video2motion 用, 可选)"
            _pe="$LHM_MODEL_DIR/human_model_files/pose_estimate"
            mkdir -p "$_pe"
            [ -f "$_pe/yolov8x.pt" ] || \
                wget --no-check-certificate -c -O "$_pe/yolov8x.pt" \
                "$OSS_BASE/yolov8x.pt" || echo "  ⚠️ yolov8x 下载失败" >&2
            [ -f "$_pe/vitpose-h-wholebody.pth" ] || \
                wget --no-check-certificate -c -O "$_pe/vitpose-h-wholebody.pth" \
                "https://virutalbuy-public.oss-cn-hangzhou.aliyuncs.com/share/aigc3d/data/LHM/vitpose-h-wholebody.pth" \
                || echo "  ⚠️ vitpose 下载失败" >&2
            # video2motion 还需 mmcv==1.3.9 + ultralytics + ViTPose
            pip install "${PIP_FLAGS[@]}" mmcv==1.3.9 ultralytics || \
                echo "  ⚠️ mmcv/ultralytics 装失败 (video2motion 用)" >&2
            VITPOSE_DIR="${VITPOSE_DIR:-$REPO_DIR/../ViTPose_lhm}"
            if [ ! -d "$VITPOSE_DIR/.git" ]; then
                LD_LIBRARY_PATH= git clone https://github.com/ViTAE-Transformer/ViTPose.git "$VITPOSE_DIR" || \
                    LD_LIBRARY_PATH= git -c http.sslVerify=false clone https://github.com/ViTAE-Transformer/ViTPose.git "$VITPOSE_DIR"
            fi
            pip install "${PIP_FLAGS[@]}" -v -e "$VITPOSE_DIR/third-party/ViTPose" 2>/dev/null || \
                echo "  ⚠️ ViTPose 安装失败 (mmcv 版本冲突常见)" >&2
        fi

        # 15) verify LHM imports + 回到 wan22_rotate env
        echo "  --- verify LHM imports ---"
        python -c "import torch; print('  torch', torch.__version__, 'cuda', torch.version.cuda)" 2>/dev/null
        python -c "import numpy; print('  numpy', numpy.__version__)" 2>/dev/null
        python -c "import diff_gaussian_rasterization; print('  [OK] diff_gaussian_rasterization')" 2>/dev/null || \
            echo "  [MISS] diff_gaussian_rasterization" >&2
        python -c "import simple_knn; print('  [OK] simple_knn')" 2>/dev/null || \
            echo "  [MISS] simple_knn" >&2
        python -c "import pytorch3d; print('  [OK] pytorch3d')" 2>/dev/null || \
            echo "  [MISS] pytorch3d (mesh 导出可能受影响)" >&2
        python -c "from sam2 import build_sam2; print('  [OK] sam2 (LHM 改版)')" 2>/dev/null || \
            echo "  [MISS] sam2 (LHM 会回退到 rembg)" >&2
        # 切回 wan22_rotate (下面的 numpy 钉版本 + verify 都在 wan22_rotate env)
        conda activate "$CONDA_ENV"
        echo "  restored env: $CONDA_ENV"
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

# --- 5a. verify MediaPipe (step 01c, optional) ---
echo "--- [5a] verify MediaPipe (step 01c) ---"
if python -c "import mediapipe; print('  [OK] mediapipe', mediapipe.__version__)" 2>/dev/null; then :; else
    echo "  [MISS] mediapipe — Run: INSTALL_DEPS=1 bash $0  (only needed by step 01c)" >&2
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

# --- 5c. verify GOF (step 05a, only if INSTALL_GOF was used) ---
if [ "${INSTALL_GOF:-0}" = "1" ] || python -c "import diff_gaussian_rasterization" 2>/dev/null; then
    echo "--- [5c] verify GOF (step 05a) ---"
    _ok_gof=1
    if [ ! -d "$GOF_DIR" ]; then
        echo "  [MISS] GOF repo — Run: INSTALL_DEPS=1 INSTALL_GOF=1 bash $0" >&2
        _ok_gof=0
    elif ! python -c "import diff_gaussian_rasterization, simple_knn" 2>/dev/null; then
        echo "  [MISS] GOF CUDA exts (diff_gaussian_rasterization / simple_knn)" >&2
        echo "         Run: INSTALL_DEPS=1 INSTALL_GOF=1 bash $0" >&2
        _ok_gof=0
    fi
    # tetra-triangulation (optional but needed for mesh extraction via extract_mesh.py)
    if [ "$_ok_gof" = "1" ]; then
        if python -c "from tetranerf.utils.extension import cpp" 2>/dev/null; then
            echo "  [OK] GOF repo + CUDA exts + tetra-triangulation"
        else
            echo "  [OK] GOF repo + CUDA exts (tetra-triangulation NOT built — mesh extraction won't work)" >&2
            echo "      Re-run: INSTALL_DEPS=1 INSTALL_GOF=1 bash $0" >&2
        fi
    fi
fi

# --- 5d. verify LHM (step 05b, 独立 lhm env; 用 conda run 不切 env) ---
if [ "${INSTALL_LHM:-0}" = "1" ] || conda run -n lhm python -c "import diff_gaussian_rasterization" 2>/dev/null; then
    echo "--- [5d] verify LHM (step 05b, env 'lhm') ---"
    if [ ! -d "$LHM_DIR" ]; then
        echo "  [MISS] LHM repo — Run: INSTALL_DEPS=1 INSTALL_LHM=1 bash $0" >&2
    elif ! conda env list 2>/dev/null | grep -qw "lhm"; then
        echo "  [MISS] lhm conda env — Run: INSTALL_DEPS=1 INSTALL_LHM=1 bash $0" >&2
    else
        _ok_lhm=1
        conda run -n lhm python -c "import torch,numpy; print('  torch',torch.__version__,'numpy',numpy.__version__)" 2>/dev/null || _ok_lhm=0
        conda run -n lhm python -c "import diff_gaussian_rasterization, simple_knn; print('  [OK] LHM CUDA exts')" 2>/dev/null || {
            echo "  [MISS] LHM CUDA exts (diff_gaussian_rasterization / simple_knn)" >&2
            echo "         Run: INSTALL_DEPS=1 INSTALL_LHM=1 bash $0" >&2
            _ok_lhm=0
        }
        conda run -n lhm python -c "import pytorch3d; print('  [OK] pytorch3d')" 2>/dev/null || \
            echo "  [MISS] pytorch3d (mesh 导出可能受影响)" >&2
        conda run -n lhm python -c "from sam2 import build_sam2; print('  [OK] sam2 (LHM 改版)')" 2>/dev/null || \
            echo "  [MISS] sam2 (LHM 会回退到 rembg 抠背景, 质量略降)" >&2
        # 权重: prior_model (human_model_files) + LHM 主模型 (huggingface/)
        if [ ! -d "$LHM_MODEL_DIR/human_model_files" ]; then
            echo "  [MISS] LHM prior models (human_model_files/) at $LHM_MODEL_DIR" >&2
            echo "         Re-run: INSTALL_DEPS=1 INSTALL_LHM=1 bash $0  (or手动下 LHM_prior_model.tar)" >&2
            _ok_lhm=0
        fi
        if [ ! -d "$LHM_MODEL_DIR/huggingface" ] || \
           [ -z "$(ls -d "$LHM_MODEL_DIR/huggingface/models--3DAIGC--"* 2>/dev/null)" ]; then
            echo "  [MISS] LHM model weights at $LHM_MODEL_DIR/huggingface/" >&2
            echo "         Re-run: INSTALL_DEPS=1 INSTALL_LHM=1 bash $0  (or手动下 3DAIGC/LHM-500M-HF)" >&2
            _ok_lhm=0
        fi
        [ "$_ok_lhm" = "1" ] && echo "  [OK] LHM env + repo + weights ready"
    fi
fi

echo ""
echo "=== [00] Done. Env '$CONDA_ENV' ready. ==="
echo "    Next: INPUT_DIR=/path/to/subject_folder WEIGHT_PATH=/path/to/lora.safetensors bash $SCRIPT_DIR/run_all.sh"

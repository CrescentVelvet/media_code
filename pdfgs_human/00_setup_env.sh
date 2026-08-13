#!/usr/bin/env bash
# 00_setup_env.sh — create the `pdfgs` conda env, install PDF-GS + SAM2 + Pi3 deps,
# build the two CUDA extensions (diff-gaussian-rasterization + simple-knn), and
# download the DINOv3 + Pi3 weights. Self-contained: this one folder does NOT
# depend on wan22_rotate / sam_3d_body.
#
# First time:
#   INSTALL_DEPS=1 bash pdfgs_human/00_setup_env.sh
# Verify only (after install):
#   bash pdfgs_human/00_setup_env.sh
#
# ⚠️ DINOv3 (`facebook/dinov3-vitb16-pretrain-lvd1689m`) is GATED on HuggingFace.
#    You must first "Request access" on the model page, then pass HF_TOKEN:
#      HF_TOKEN=hf_xxx INSTALL_DEPS=1 bash pdfgs_human/00_setup_env.sh
#    Without the token, DINOv3 download is skipped (with a warning) and step 03
#    will fail at runtime — re-run with HF_TOKEN once access is granted.
set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/_env.sh"

# conda 不读 REQUESTS_CA_BUNDLE / SSL_CERT_FILE —— 公司代理 TLS 拦截会让 conda
# (create + install) 报 SSL 错。彻底关闭 conda 的 SSL 验证（不指向任何 CA bundle，
# 因为代理根 CA 未必在 bundle 里；false = 不验证任何证书，最稳）。
# condarc 优先级: env 级 > user 级 > system 级——env 级会覆盖 user 级，故三层都设。
# ⚠️ 必须在 `conda create` 之前调，否则建 env 时下 python 就 SSL 失败。
_conda_disable_ssl() {
    conda config --system --set ssl_verify false 2>/dev/null || true
    conda config         --set ssl_verify false 2>/dev/null || true
    conda config --env   --set ssl_verify false 2>/dev/null || true
    echo "--- conda ssl_verify = false (彻底关闭, system+user+env levels) ---"
}
_conda_disable_ssl

# _env.sh tolerated a missing env; create it now if needed.
# NB: this env MUST be python=3.10 CPython — local torch/triton wheels are cp310
# (and PDF-GS environment.yml pins python=3.10). conda-forge may slip GraalPy
# into the python slot; we verify CPython immediately after create + after every
# conda install.
if ! conda env list 2>/dev/null | grep -qw "$CONDA_ENV"; then
    echo "--- conda env '$CONDA_ENV' not found; creating python=3.10 (CPython) ---"
    conda create -n "$CONDA_ENV" python=3.10 -y
    conda activate "$CONDA_ENV"
    _conda_disable_ssl  # re-set env-level for the freshly activated pdfgs env
    impl="$(python -c 'import platform; print(platform.python_implementation())')"
    if [ "$impl" != "CPython" ]; then
        echo "ERROR: env '$CONDA_ENV' python 实现是 $impl（应为 CPython）。" >&2
        echo "       conda-forge 把 graalpy 当 python 塞了。删掉重建：" >&2
        echo "         conda env remove -n $CONDA_ENV" >&2
        echo "         conda create -n $CONDA_ENV python=3.10 -y --override-channels -c defaults" >&2
        exit 1
    fi
    echo "  [OK] python=$(python --version 2>&1 | cut -d ' ' -f2) ($impl)"
fi

echo "=== [00] Verify prerequisites for pdfgs_human ==="
echo "  conda env: $CONDA_ENV  (python $(python --version 2>&1 | cut -d ' ' -f2))"
echo "  PDF-GS:    $PDFGS_DIR"
echo "  Pi3:       $PI3_DIR"
echo "  SAM2:      $SAM2_DIR"
echo "  weights:   $MODEL_DIR"
echo ""

# Helper: verify CPython after each conda install; auto-recover if GraalPy slipped in.
verify_cpython() {
    local _impl
    _impl="$(python -c 'import platform; print(platform.python_implementation())' 2>/dev/null || echo unknown)"
    if [ "$_impl" != "CPython" ]; then
        echo "  ⚠️ python 被掉包成 '$_impl'，修复..." >&2
        conda install -y -c defaults python=3.10 --force-reinstall
        # numpy/torch may have been clobbered; re-pin
        pip install --force-reinstall --no-deps numpy==1.26.4 2>/dev/null || true
        _impl="$(python -c 'import platform; print(platform.python_implementation())' 2>/dev/null || echo unknown)"
        if [ "$_impl" != "CPython" ]; then
            echo "ERROR: 无法恢复 CPython，请手动重建 env" >&2
            echo "  conda env remove -n $CONDA_ENV && conda create -n $CONDA_ENV python=3.10 -y" >&2
            exit 1
        fi
        echo "  [OK] 恢复 CPython"
    fi
}

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
    _conda_disable_ssl  # env-level may need refresh after any env activation

    # 0a. PyTorch 2.5.1 + cu121 (PDF-GS environment.yml pin).
    #     PyPI default wheels are cu121 — do NOT use download.pytorch.org (代理封 403).
    echo "--- installing PyTorch 2.5.1 + cu121 (from PyPI default index) ---"
    pip install "${PIP_FLAGS[@]}" \
        torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 || \
        echo "  ⚠️ torch 2.5.1 install failed (PyPI/代理?). 手动: pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1" >&2
    echo "  torch.version.cuda = $(python -c 'import torch; print(torch.version.cuda)' 2>/dev/null)"

    # 0b. gcc 12 into the conda env (system gcc too old for CUDA 12.x rasterizer build).
    #     ⚠️ pin python=3.10 防 GraalPy 掉包 (see AGENTS.md §6 conda pitfalls).
    echo "--- installing gcc 12 into conda env (for CUDA ext compilation) ---"
    conda install -y -c conda-forge --no-update-deps gxx_linux-64=12 python=3.10 || \
        echo "  ⚠️ gxx install failed; 用系统 gcc 编 CUDA 扩展可能失败" >&2
    verify_cpython
    export CC=$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-gcc
    export CXX=$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-g++
    # numpy may have been touched; pin to a 3DGS-compatible 1.26.4 (PDF-GS works with 1.26.x)
    if ! python -c "import numpy" 2>/dev/null; then
        pip install "${PIP_FLAGS[@]}" --force-reinstall --no-deps numpy==1.26.4
    fi

    # 0c. Clone PDF-GS (+ submodules: diff-gaussian-rasterization, simple-knn).
    if [ ! -d "$PDFGS_DIR/.git" ]; then
        echo "--- cloning PDF-GS -> $PDFGS_DIR ---"
        mkdir -p "$(dirname "$PDFGS_DIR")"
        LD_LIBRARY_PATH= git clone --recursive https://github.com/kangrnin/PDF-GS.git "$PDFGS_DIR" || \
            LD_LIBRARY_PATH= git -c http.sslVerify=false clone --recursive \
                https://github.com/kangrnin/PDF-GS.git "$PDFGS_DIR"
    fi
    # Ensure submodules (diff-gaussian-rasterization on github, simple-knn on gitlab.inria.fr).
    # git submodule update may fail if gitlab.inria.fr is blocked; fall back to cloning each.
    if [ ! -f "$PDFGS_DIR/submodules/diff-gaussian-rasterization/setup.py" ] || \
       [ ! -f "$PDFGS_DIR/submodules/simple-knn/setup.py" ]; then
        echo "  ensuring PDF-GS submodules"
        ( cd "$PDFGS_DIR" && LD_LIBRARY_PATH= git submodule update --init --recursive ) || \
            ( cd "$PDFGS_DIR" && LD_LIBRARY_PATH= git -c http.sslVerify=false submodule update --init --recursive ) || true
        SUBMOD_DIR="$PDFGS_DIR/submodules"
        # diff-gaussian-rasterization (3DGS 原版光栅化器, graphdeco-inria)
        if [ ! -f "$SUBMOD_DIR/diff-gaussian-rasterization/setup.py" ]; then
            echo "  cloning diff-gaussian-rasterization"
            rm -rf "$SUBMOD_DIR/diff-gaussian-rasterization"
            LD_LIBRARY_PATH= git clone https://github.com/graphdeco-inria/diff-gaussian-rasterization.git "$SUBMOD_DIR/diff-gaussian-rasterization" || \
                LD_LIBRARY_PATH= git -c http.sslVerify=false clone https://github.com/graphdeco-inria/diff-gaussian-rasterization.git "$SUBMOD_DIR/diff-gaussian-rasterization"
        fi
        # simple-knn (gitlab.inria.fr, may be blocked/slow → zip fallback)
        if [ ! -f "$SUBMOD_DIR/simple-knn/setup.py" ]; then
            echo "  cloning simple-knn (gitlab.inria.fr, may be slow)"
            rm -rf "$SUBMOD_DIR/simple-knn"
            LD_LIBRARY_PATH= git clone https://gitlab.inria.fr/bkerbl/simple-knn.git "$SUBMOD_DIR/simple-knn" || \
                LD_LIBRARY_PATH= git -c http.sslVerify=false clone https://gitlab.inria.fr/bkerbl/simple-knn.git "$SUBMOD_DIR/simple-knn" || {
                    echo "  git clone failed, trying zip download..."
                    _zip="/tmp/simple-knn-main.zip"
                    curl -k -L --connect-timeout 30 --max-time 300 -o "$_zip" \
                        "https://gitlab.inria.fr/bkerbl/simple-knn/-/archive/main/simple-knn-main.zip" && \
                        unzip -o "$_zip" -d "$SUBMOD_DIR" && \
                        mv "$SUBMOD_DIR/simple-knn-main" "$SUBMOD_DIR/simple-knn" || \
                        echo "  ❌ simple-knn download failed — manual:" >&2
                    echo "     cd $SUBMOD_DIR" >&2
                    echo "     curl -kL -o sk.zip https://gitlab.inria.fr/bkerbl/simple-knn/-/archive/main/simple-knn-main.zip" >&2
                    echo "     unzip sk.zip && mv simple-knn-main simple-knn" >&2
                }
        fi
        # GLM (header-only math lib, diff-gaussian-rasterization needs it)
        GLM_DIR="$PDFGS_DIR/third_party/glm"
        if [ ! -f "$GLM_DIR/glm/glm.hpp" ]; then
            echo "  cloning GLM (diff-gaussian-rasterization dependency)"
            rm -rf "$GLM_DIR"
            LD_LIBRARY_PATH= git clone https://github.com/g-truc/glm.git "$GLM_DIR" || \
                LD_LIBRARY_PATH= git -c http.sslVerify=false clone https://github.com/g-truc/glm.git "$GLM_DIR"
        fi
    fi

    # 0d. PDF-GS Python deps (torch/numpy already installed). Per environment.yml:
    #     transformers>=4.56 (DINOv3), torchmetrics==1.2.0 (metrics.py), mediapy,
    #     opencv_python, scipy, joblib, plyfile, tqdm, matplotlib. Plus lpips
    #     (3DGS metrics.py needs it) + huggingface_hub (snapshot_download).
    echo "--- installing PDF-GS Python deps ---"
    pip install "${PIP_FLAGS[@]}" \
        "transformers>=4.56" torchmetrics==1.2.0 mediapy==1.1.2 \
        opencv-python scipy joblib plyfile tqdm "matplotlib<3.10" \
        lpips==0.1.4 huggingface_hub || \
        echo "  ⚠️ 部分 PDF-GS 依赖装失败 (上面有 WARNING), 继续装其余" >&2

    # 0e. SAM2 (person segmentation, step 01). Clone official sam2 + checkpoint.
    if [ ! -d "$SAM2_DIR/.git" ]; then
        echo "--- cloning SAM2 -> $SAM2_DIR ---"
        rm -rf "$SAM2_DIR"
        mkdir -p "$(dirname "$SAM2_DIR")"
        LD_LIBRARY_PATH= git clone https://github.com/facebookresearch/sam2.git "$SAM2_DIR" || \
            LD_LIBRARY_PATH= git -c http.sslVerify=false clone https://github.com/facebookresearch/sam2.git "$SAM2_DIR"
    fi
    pip install "${PIP_FLAGS[@]}" -e "$SAM2_DIR" || \
        echo "  ⚠️ SAM2 install failed; step 01 会回退到 rembg (质量略降, 不致命)" >&2
    SAM2_CKPT_DIR="$SAM2_DIR/checkpoints"
    mkdir -p "$SAM2_CKPT_DIR"
    if [ ! -f "$SAM2_CKPT_DIR/sam2.1_hiera_large.pt" ]; then
        echo "--- downloading SAM2 checkpoint (sam2.1_hiera_large) ---"
        wget --no-check-certificate -q -O "$SAM2_CKPT_DIR/sam2.1_hiera_large.pt" \
            "https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2.1_hiera_large.pt" || \
            echo "  ⚠️ SAM2 checkpoint download failed (dl.fbaipublicfiles.com may be blocked)." >&2
        echo "         Manual: https://github.com/facebookresearch/sam2#segment-anything-2-checkpoints" >&2
        echo "         Place at: $SAM2_CKPT_DIR/sam2.1_hiera_large.pt" >&2
    fi

    # 0f. rembg (fallback segmentor when SAM2 unavailable / fails to load).
    #     Pure-pip background remover, no CUDA compile. segment_all.py tries SAM2
    #     first, falls back to rembg per image.
    echo "--- installing rembg (fallback segmentor) ---"
    pip install "${PIP_FLAGS[@]}" rembg onnxruntime==1.18.1 || \
        echo "  ⚠️ rembg install failed; step 01 仅靠 SAM2 (若 SAM2 也没装好则分割失败)" >&2

    # 0g. Pi3 (feed-forward pose estimation, step 02). Clone official repo.
    if [ ! -d "$PI3_DIR/.git" ]; then
        echo "--- cloning Pi3 -> $PI3_DIR ---"
        mkdir -p "$(dirname "$PI3_DIR")"
        LD_LIBRARY_PATH= git clone https://github.com/yyfz/Pi3.git "$PI3_DIR" || \
            LD_LIBRARY_PATH= git -c http.sslVerify=false clone https://github.com/yyfz/Pi3.git "$PI3_DIR"
    fi
    # Pi3 deps (torch/cv2/safetensors/plyfile already in env). Just in case:
    pip install "${PIP_FLAGS[@]}" safetensors plyfile 2>/dev/null || true

    # 0h. Build the two CUDA extensions (diff-gaussian-rasterization + simple-knn).
    #     Auto-detect CUDA 12.x: /usr/local/cuda may point to 11.8, find cuda-12.x
    #     matching torch's CUDA major (12). Same logic as wan22_rotate/00.
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
        echo "  building CUDA ext: diff-gaussian-rasterization"
        pip install "${PIP_FLAGS[@]}" --no-build-isolation --no-deps --no-index \
            "$PDFGS_DIR/submodules/diff-gaussian-rasterization" || \
            echo "  ⚠️ diff-gaussian-rasterization build failed (nvcc/gcc 版本不匹配? GLM 缺失?)" >&2
        echo "  building CUDA ext: simple-knn"
        pip install "${PIP_FLAGS[@]}" --no-build-isolation --no-deps --no-index \
            "$PDFGS_DIR/submodules/simple-knn" || \
            echo "  ⚠️ simple-knn build failed" >&2
        python -c "import diff_gaussian_rasterization, simple_knn; print('  [OK] PDF-GS CUDA exts')" || \
            echo "  ⚠️ CUDA exts built but not importable" >&2
    else
        echo "  ⚠️ nvcc not found at $CUDA_HOME/bin/nvcc — PDF-GS CUDA exts NOT built." >&2
        echo "           Install CUDA toolkit 12.x and re-run: INSTALL_DEPS=1 bash $0" >&2
    fi

    # 0i. Pi3 checkpoint (公开, 免 token, ~1GB).
    mkdir -p "$MODEL_DIR/Pi3"
    if [ ! -f "$MODEL_DIR/Pi3/model.safetensors" ]; then
        echo "--- downloading Pi3 checkpoint -> $MODEL_DIR/Pi3 ---"
        wget --no-check-certificate -c -O "$MODEL_DIR/Pi3/model.safetensors" \
            "https://huggingface.co/yyfz233/Pi3/resolve/main/model.safetensors" || \
            echo "  ⚠️ Pi3 checkpoint download failed (HF blocked?); manual:" >&2
        echo "    wget -O $MODEL_DIR/Pi3/model.safetensors https://huggingface.co/yyfz233/Pi3/resolve/main/model.safetensors" >&2
    fi

    # 0j. DINOv3 checkpoint (GATED, step 03 needs it). facebook/dinov3-vitb16-pretrain-lvd1689m.
    #     Must "Request access" on the HF model page first, then pass HF_TOKEN.
    #     Pre-downloaded into $HF_HOME/hub so _env.sh's HF_HUB_OFFLINE=1 can read it at train time.
    echo "--- downloading DINOv3 (gated: facebook/dinov3-vitb16-pretrain-lvd1689m) ---"
    mkdir -p "$HF_HOME"
    if [ -z "${HF_TOKEN:-}" ]; then
        echo "  ⚠️ HF_TOKEN not set — DINOv3 is GATED." >&2
        echo "     1) Request access: https://huggingface.co/facebook/dinov3-vitb16-pretrain-lvd1689m" >&2
        echo "     2) Re-run: HF_TOKEN=hf_xxx INSTALL_DEPS=1 bash $0" >&2
        echo "     (skipping DINOv3 download; step 03 will fail at runtime until this is done)" >&2
    else
        # Temporarily disable offline so snapshot_download can reach HF.
        if HF_HUB_OFFLINE=0 TRANSFORMERS_OFFLINE=0 HF_TOKEN="$HF_TOKEN" python -c "
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id='facebook/dinov3-vitb16-pretrain-lvd1689m',
    cache_dir='${HF_HOME}/hub',
    token='${HF_TOKEN}',
)
print('  [OK] DINOv3 downloaded to ${HF_HOME}/hub')
" 2>&1; then
            : # downloaded
        else
            echo "  ⚠️ DINOv3 download failed (HF blocked / token invalid?). Manual:" >&2
            echo "    from huggingface_hub import snapshot_download" >&2
            echo "    snapshot_download(repo_id='facebook/dinov3-vitb16-pretrain-lvd1689m', cache_dir='$HF_HOME/hub', token=...)" >&2
        fi
    fi

    # 最后钉 numpy==1.26.4 (PDF-GS 与 3DGS rasterizer 兼容; 前面装依赖可能升级到 2.x)
    echo "--- pinning numpy==1.26.4 ---"
    pip install "${PIP_FLAGS[@]}" --force-reinstall --no-deps numpy==1.26.4
    echo "  numpy=$(python -c 'import numpy; print(numpy.__version__)' 2>/dev/null)"

    echo "--- deps installed ---"
fi

# --- 1. PDF-GS code ---
echo "--- [1/5] PDF-GS code: $PDFGS_DIR ---"
if [ ! -f "$PDFGS_DIR/train.py" ]; then
    echo "  [MISS] Run: INSTALL_DEPS=1 bash $0" >&2
    exit 1
fi
echo "  [OK]"

# --- 2. CUDA extensions importable ---
echo "--- [2/5] CUDA extensions (diff-gaussian-rasterization, simple-knn) ---"
if python -c "import diff_gaussian_rasterization, simple_knn; print('  [OK]')" 2>/dev/null; then :; else
    echo "  [MISS] Run: INSTALL_DEPS=1 bash $0 (needs CUDA toolkit / nvcc)" >&2
fi

# --- 3. SAM2 (segmentation, step 01) ---
echo "--- [3/5] SAM2: $SAM2_DIR ---"
if python -c "from sam2 import build_sam2; print('  [OK] sam2')" 2>/dev/null; then :; else
    echo "  [MISS] sam2 — step 01 will fall back to rembg (or fail if rembg also missing)" >&2
fi
if [ ! -f "$SAM2_CHECKPOINT" ]; then
    echo "  [MISS] SAM2 checkpoint: $SAM2_CHECKPOINT" >&2
fi

# --- 4. Pi3 (pose estimation, step 02) ---
echo "--- [4/5] Pi3: $PI3_DIR ---"
if [ ! -d "$PI3_DIR" ]; then
    echo "  [MISS] Run: INSTALL_DEPS=1 bash $0" >&2
fi
if [ ! -f "$PI3_CKPT" ]; then
    echo "  [MISS] Pi3 ckpt: $PI3_CKPT" >&2
    echo "         wget -O $PI3_CKPT https://huggingface.co/yyfz233/Pi3/resolve/main/model.safetensors" >&2
else
    echo "  [OK] Pi3 ckpt"
fi

# --- 5. DINOv3 (PDF-GS DINOv3FeatureExtractor, step 03) ---
echo "--- [5/5] DINOv3 (gated, HF cache): $HF_HOME/hub ---"
if [ -d "$HF_HOME/hub" ] && [ -n "$(ls -A "$HF_HOME/hub" 2>/dev/null)" ]; then
    echo "  [OK] HF cache populated (verify DINOv3 by running step 03 once)"
else
    echo "  [MISS] DINOv3 not downloaded — HF_TOKEN=hf_xxx INSTALL_DEPS=1 bash $0" >&2
    echo "         (first Request access: https://huggingface.co/facebook/dinov3-vitb16-pretrain-lvd1689m)" >&2
fi

echo ""
echo "=== [00] Done. Next:"
echo "  INPUT_DIR=../<your_orbit_shoot> bash pdfgs_human/01_segment_all.sh"

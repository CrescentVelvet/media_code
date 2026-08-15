#!/usr/bin/env bash
# 00_setup_env.sh — clone VGGT-Omega + gaussian-splatting + HYPIR repos, verify torch,
# install deps, compile CUDA extensions (diff-gaussian-rasterization, simple-knn).
# Creates `vggt_human` conda env by cloning `doll` (has torch>=2.3); installs HYPIR
# deps (diffusers/transformers/peft) + mediapipe on top.
set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/_env.sh"

# Create vggt_human env by copying doll directory (avoids conda's network-dependent
# clone which fails on corporate proxy SSL — conda create --clone tries to re-download
# packages like libnsl even though doll already has them; --offline fails on .partial
# cache residue. cp -a + shebang fix is equivalent and fully offline).
_doll_prefix="$(conda info --base 2>/dev/null)/envs/doll"
_new_prefix="$(conda info --base 2>/dev/null)/envs/$CONDA_ENV"

if [ -d "$_new_prefix/conda-meta" ] && [ -f "$_new_prefix/conda-meta/history" ]; then
    echo "--- $CONDA_ENV env already exists at $_new_prefix ---"
else
    if [ ! -d "$_doll_prefix" ]; then
        echo "❌ ERROR: 'doll' env not found at $_doll_prefix" >&2
        echo "       Create it first: bash vggt-omega/00_setup_env.sh" >&2
        exit 1
    fi
    # Clean up any broken/partial env from a previous failed conda clone
    rm -rf "$_new_prefix"
    echo "--- cloning doll -> $CONDA_ENV via cp -a (no network) ---"
    cp -a "$_doll_prefix" "$_new_prefix"
    echo "  ✅ copied doll -> $CONDA_ENV"
fi

# Activate the env
# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV" 2>/dev/null || true

echo "=== [00] Setup vggt_human env '$CONDA_ENV' ==="
echo "  VGGT-Omega:  $VGGT_DIR"
echo "  3DGS:        $GS_DIR"
echo "  weights:     $MODEL_DIR"
echo "  output:      $RESULTS_DIR"
echo ""

# ── 1. Clone VGGT-Omega official repo ──────────────────────────────────────
if [ ! -d "$VGGT_DIR" ]; then
    echo "--- cloning VGGT-Omega -> $VGGT_DIR ---"
    mkdir -p "$(dirname "$VGGT_DIR")"
    git clone "$VGGT_REPO" "$VGGT_DIR" || \
        git -c http.sslVerify=false clone "$VGGT_REPO" "$VGGT_DIR"
else
    echo "--- VGGT-Omega already present: $VGGT_DIR ---"
fi

# ── 2. Clone gaussian-splatting (original 3DGS) ────────────────────────────
if [ ! -d "$GS_DIR/.git" ]; then
    echo "--- cloning gaussian-splatting -> $GS_DIR ---"
    mkdir -p "$(dirname "$GS_DIR")"
    LD_LIBRARY_PATH= git clone "$GS_REPO" "$GS_DIR" || \
        LD_LIBRARY_PATH= git -c http.sslVerify=false clone "$GS_REPO" "$GS_DIR"
fi

# Ensure submodules (diff-gaussian-rasterization on GitHub, simple-knn on gitlab.inria.fr).
SUBMOD_DIR="$GS_DIR/submodules"
if [ ! -f "$SUBMOD_DIR/diff-gaussian-rasterization/setup.py" ] || \
   [ ! -f "$SUBMOD_DIR/simple-knn/setup.py" ]; then
    echo "  ensuring gaussian-splatting submodules"
    # Replace gitlab.inria.fr submodule URLs with GitHub mirrors (corporate proxy blocks gitlab.inria.fr)
    cd "$GS_DIR"
    if [ -f .gitmodules ]; then
        git config -f .gitmodules submodule.submodules/simple-knn.url https://github.com/yindaheng98/simple-knn.git
        git config -f .gitmodules submodule.third_party/glm.url https://github.com/g-truc/glm.git || true
        git config -f .gitmodules submodule.submodules/SIBR_viewers.url https://github.com/graphdeco-inria/SIBR_viewers.git 2>/dev/null || true
        git submodule sync
    fi
    cd - >/dev/null
    ( cd "$GS_DIR" && LD_LIBRARY_PATH= git submodule update --init --recursive ) || \
        ( cd "$GS_DIR" && LD_LIBRARY_PATH= git -c http.sslVerify=false submodule update --init --recursive ) || true

    # diff-gaussian-rasterization (original 3DGS rasterizer, main branch — no antialiasing)
    if [ ! -f "$SUBMOD_DIR/diff-gaussian-rasterization/setup.py" ]; then
        echo "  cloning diff-gaussian-rasterization"
        rm -rf "$SUBMOD_DIR/diff-gaussian-rasterization"
        LD_LIBRARY_PATH= git clone https://github.com/graphdeco-inria/diff-gaussian-rasterization.git \
            "$SUBMOD_DIR/diff-gaussian-rasterization" || \
            LD_LIBRARY_PATH= git -c http.sslVerify=false clone \
                https://github.com/graphdeco-inria/diff-gaussian-rasterization.git \
                "$SUBMOD_DIR/diff-gaussian-rasterization"
    fi

    # simple-knn (gitlab.inria.fr may be blocked; fall back to GitHub mirrors)
    if [ ! -f "$SUBMOD_DIR/simple-knn/setup.py" ]; then
        echo "  cloning simple-knn (gitlab.inria.fr may be blocked; GitHub mirrors as fallback)"
        rm -rf "$SUBMOD_DIR/simple-knn"
        _sk_done=false
        for _sk_url in \
            "https://gitlab.inria.fr/bkerbl/simple-knn.git" \
            "https://github.com/yindaheng98/simple-knn.git" \
            "https://github.com/jteng2127/simple-knn.git"; do
            echo "    trying: $_sk_url"
            if LD_LIBRARY_PATH= git clone "$_sk_url" "$SUBMOD_DIR/simple-knn" 2>/dev/null || \
               LD_LIBRARY_PATH= git -c http.sslVerify=false clone "$_sk_url" "$SUBMOD_DIR/simple-knn" 2>/dev/null; then
                _sk_done=true
                echo "    ✅ cloned from $_sk_url"
                break
            fi
            rm -rf "$SUBMOD_DIR/simple-knn"
        done
        if [ "$_sk_done" != "true" ]; then
            echo "  ❌ simple-knn clone failed (all mirrors) — manual:" >&2
            echo "     cd $SUBMOD_DIR && git clone https://github.com/yindaheng98/simple-knn.git simple-knn" >&2
        fi
    fi

    # GLM (header-only math lib, diff-gaussian-rasterization needs it)
    GLM_DIR="$GS_DIR/third_party/glm"
    if [ ! -f "$GLM_DIR/glm/glm.hpp" ]; then
        echo "  cloning GLM (diff-gaussian-rasterization dependency)"
        rm -rf "$GLM_DIR"
        LD_LIBRARY_PATH= git clone https://github.com/g-truc/glm.git "$GLM_DIR" || \
            LD_LIBRARY_PATH= git -c http.sslVerify=false clone https://github.com/g-truc/glm.git "$GLM_DIR"
    fi
fi

# ── 3. Install deps (first time) ────────────────────────────────────────────
# Always ensure mediapipe is installed (lightweight, needed for step 01/06).
# Full deps (3DGS exts, HYPIR, etc.) require INSTALL_DEPS=1.
if ! python -c "import mediapipe" 2>/dev/null; then
    echo "--- installing mediapipe (needed for step 01/06) ---"
    _PIP_FLAGS=(--trusted-host pypi.org --trusted-host pypi.python.org \
        --trusted-host files.pythonhosted.org --timeout 600 --retries 10)
    pip install "${_PIP_FLAGS[@]}" mediapipe || \
        echo "  ⚠️ mediapipe install failed — run: pip install mediapipe" >&2
fi

# ── 4. Verify torch + CUDA ──────────────────────────────────────────────────
python - <<'PY'
import torch
print(f"torch: {torch.__version__}  cuda: {torch.version.cuda}  available: {torch.cuda.is_available()}")
if not torch.cuda.is_available():
    raise SystemExit("ERROR: torch.cuda not available in env '$CONDA_ENV'.")
PY

# ── 5. Full deps install (first time) ───────────────────────────────────────
if [ "${INSTALL_DEPS:-0}" = "1" ]; then
    PIP_FLAGS=(--trusted-host pypi.org --trusted-host pypi.python.org \
        --trusted-host files.pythonhosted.org --timeout 600 --retries 10)

    # VGGT-Omega runtime deps (torch NOT reinstalled)
    echo "--- installing VGGT-Omega runtime deps ---"
    pip install "${PIP_FLAGS[@]}" "numpy<2" Pillow einops safetensors opencv-python \
        huggingface_hub

    # 3DGS runtime deps + mediapipe (face detection, step 01/06)
    echo "--- installing 3DGS + mediapipe runtime deps ---"
    pip install "${PIP_FLAGS[@]}" plyfile tqdm torchmetrics lpips \
        scipy trimesh matplotlib mediapipe

    # HYPIR deps (face enhancement, step 01/06; diffusers/transformers/peft)
    echo "--- installing HYPIR deps (diffusers, transformers, peft) ---"
    pip install "${PIP_FLAGS[@]}" "diffusers==0.32.2" "transformers==4.49.0" "peft==0.14.0" \
        omegaconf kornia accelerate

    # gcc 12 for CUDA ext compilation (pin python=3.10 to prevent GraalPy swap).
    # --offline: packages already in ~/miniconda3/pkgs/ cache; avoids fetching
    # repodata.json which fails on corporate proxy SSL.
    echo "--- installing gcc 12 (for CUDA ext compilation) ---"
    if ! conda install -y -c conda-forge --offline --no-update-deps gxx_linux-64=12 python=3.10; then
        echo "  ⚠️ --no-update-deps blocked; retrying without it (python=3.10 pinned)" >&2
        conda install -y -c conda-forge --offline gxx_linux-64=12 python=3.10 || \
            echo "  ⚠️ gxx install failed; system gcc will be used (may fail)" >&2
    fi

    # Pin numpy (conda may have touched it)
    pip install "${PIP_FLAGS[@]}" --force-reinstall --no-deps numpy==1.26.4

    # ── Compile CUDA extensions ──────────────────────────────────────────────
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

        # Resolve CC/CXX to conda gcc12 (matches CUDA 12.x)
        _gcc="$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-gcc"
        _gpp="$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-g++"
        if [ -x "$_gcc" ] && [ -x "$_gpp" ]; then
            export CC="$_gcc"; export CXX="$_gpp"
            echo "  CC/CXX = conda gcc12: $_gcc"
        else
            echo "  ⚠️ conda gcc not found; trying system gcc/g++" >&2
            command -v gcc >/dev/null 2>&1 && export CC="$(command -v gcc)" || true
            command -v g++ >/dev/null 2>&1 && export CXX="$(command -v g++)" || true
        fi
        echo "  nvcc: $($CUDA_HOME/bin/nvcc --version | tail -1 | xargs)"

        _DGR="$SUBMOD_DIR/diff-gaussian-rasterization"
        if [ -f "$_DGR/setup.py" ] || [ -f "$_DGR/pyproject.toml" ]; then
            echo "  building CUDA ext: diff-gaussian-rasterization (main branch, original 3DGS)"
            rm -rf "$_DGR/build" "$_DGR/dist" "$_DGR"/*.egg-info
            # GLM symlink: diff-gaussian-rasterization's third_party/glm may be empty
            if [ ! -f "$_DGR/third_party/glm/glm/glm.hpp" ] && [ -f "$GS_DIR/third_party/glm/glm/glm.hpp" ]; then
                rm -rf "$_DGR/third_party/glm"
                ln -sf "$GS_DIR/third_party/glm" "$_DGR/third_party/glm"
            fi
            pip install "${PIP_FLAGS[@]}" --no-build-isolation --no-deps "$_DGR" || \
                echo "  ⚠️ diff-gaussian-rasterization build failed" >&2
        else
            echo "  ⚠️ skip diff-gaussian-rasterization: setup.py not found" >&2
        fi

        _SK="$SUBMOD_DIR/simple-knn"
        if [ -f "$_SK/setup.py" ]; then
            echo "  building CUDA ext: simple-knn"
            pip install "${PIP_FLAGS[@]}" --no-build-isolation --no-deps "$_SK" || \
                echo "  ⚠️ simple-knn build failed" >&2
        fi

        python -c "import diff_gaussian_rasterization, simple_knn; print('  [OK] CUDA exts importable')" || \
            echo "  ⚠️ CUDA exts not importable" >&2
    else
        echo "  ⚠️ nvcc not found at $CUDA_HOME/bin/nvcc — CUDA exts NOT built." >&2
        echo "         Install CUDA toolkit and re-run: INSTALL_DEPS=1 bash $0" >&2
    fi

    # ── HYPIR repo + SD2 base model (face enhancement, step 01/06) ────────
    if [ ! -d "$HYPIR_DIR/.git" ]; then
        echo "--- cloning HYPIR -> $HYPIR_DIR ---"
        mkdir -p "$(dirname "$HYPIR_DIR")"
        LD_LIBRARY_PATH= git clone https://github.com/XPixelGroup/HYPIR.git "$HYPIR_DIR" || \
            LD_LIBRARY_PATH= git -c http.sslVerify=false clone https://github.com/XPixelGroup/HYPIR.git "$HYPIR_DIR"
    fi
    if [ ! -d "$HYPIR_BASE_MODEL" ]; then
        echo "--- downloading SD2 base model -> $HYPIR_BASE_MODEL (large, ~5GB) ---"
        mkdir -p "$HYPIR_MODEL_DIR"
        LD_LIBRARY_PATH= git clone https://huggingface.co/stabilityai/stable-diffusion-2-base "$HYPIR_BASE_MODEL" || \
            LD_LIBRARY_PATH= git -c http.sslVerify=false clone https://huggingface.co/stabilityai/stable-diffusion-2-base "$HYPIR_BASE_MODEL" || \
            echo "  ⚠️ SD2 base model download failed. Manual:" >&2
        echo "    git clone https://huggingface.co/stabilityai/stable-diffusion-2-base $HYPIR_BASE_MODEL" >&2
    fi

    # ── Denoiser models (optional, step 04) ────────────────────────────────
    # Set INSTALL_DENOISER=1 to clone + download weights for DiffBIR / SwinIR.
    if [ "${INSTALL_DENOISER:-0}" = "1" ]; then
        echo "--- installing denoiser models (INSTALL_DENOISER=1) ---"

        # DiffBIR (generative diffusion prior, higher quality, slower)
        if [ ! -d "$DIFFBIR_DIR/.git" ]; then
            echo "  cloning DiffBIR -> $DIFFBIR_DIR"
            mkdir -p "$(dirname "$DIFFBIR_DIR")"
            LD_LIBRARY_PATH= git clone "$DIFFBIR_REPO" "$DIFFBIR_DIR" || \
                LD_LIBRARY_PATH= git -c http.sslVerify=false clone "$DIFFBIR_REPO" "$DIFFBIR_DIR"
        fi
        mkdir -p "$(dirname "$DIFFBIR_CKPT")"
        if [ ! -s "$DIFFBIR_CKPT" ]; then
            echo "  downloading DiffBIR checkpoint..."
            wget --no-check-certificate -q -O "$DIFFBIR_CKPT" \
                "https://huggingface.co/zhonghuiyi/ckpt/resolve/main/cldm.pth" || true
            if [ ! -s "$DIFFBIR_CKPT" ]; then
                echo "  ⚠️ DiffBIR checkpoint download failed (empty/missing). Manual:" >&2
                echo "    wget -O $DIFFBIR_CKPT <download_url>" >&2
            fi
        fi
        # DiffBIR Python deps (diffusers, omegaconf, einops)
        pip install "${PIP_FLAGS[@]}" diffusers omegaconf einops kornia || \
            echo "  ⚠️ DiffBIR deps install failed (some packages may be missing)" >&2

        # SwinIR (single-forward, faster, good for real-image denoising)
        if [ ! -d "$SWINIR_DIR/.git" ]; then
            echo "  cloning SwinIR -> $SWINIR_DIR"
            mkdir -p "$(dirname "$SWINIR_DIR")"
            LD_LIBRARY_PATH= git clone "$SWINIR_REPO" "$SWINIR_DIR" || \
                LD_LIBRARY_PATH= git -c http.sslVerify=false clone "$SWINIR_REPO" "$SWINIR_DIR"
        fi
        mkdir -p "$(dirname "$SWINIR_CKPT")"
        if [ ! -s "$SWINIR_CKPT" ]; then
            echo "  downloading SwinIR denoising checkpoint..."
            wget --no-check-certificate -q -O "$SWINIR_CKPT" \
                "https://github.com/JingyunLiang/SwinIR/releases/download/v0.0/001_classicalSR_DIV2K_s128w8_SwinIR-M_x4.pth" || true
            if [ ! -s "$SWINIR_CKPT" ]; then
                echo "  ⚠️ SwinIR checkpoint download failed (empty/missing). Manual:" >&2
                echo "    See: https://github.com/JingyunLiang/SwinIR#model-zoo" >&2
            fi
        fi
    fi
fi

# ── 5. Verify ───────────────────────────────────────────────────────────────
echo ""
echo "--- verification ---"
[ -d "$VGGT_DIR/vggt_omega" ] && echo "  [OK] VGGT-Omega code" || echo "  [MISS] VGGT-Omega code: $VGGT_DIR"
[ -f "$GS_DIR/train.py" ] && echo "  [OK] 3DGS train.py" || echo "  [MISS] 3DGS: $GS_DIR"
python -c "import diff_gaussian_rasterization, simple_knn; print('  [OK] CUDA exts')" 2>/dev/null || \
    echo "  [MISS] CUDA exts (run INSTALL_DEPS=1)"

# Denoiser models (optional, step 04)
[ -d "$DIFFBIR_DIR/.git" ] && echo "  [OK] DiffBIR code" || \
    echo "  [---] DiffBIR (INSTALL_DENOISER=1; needed for DENOISER=diffbir)"
[ -f "$DIFFBIR_CKPT" ] && echo "  [OK] DiffBIR ckpt" || true
[ -d "$SWINIR_DIR/.git" ] && echo "  [OK] SwinIR code" || \
    echo "  [---] SwinIR (INSTALL_DENOISER=1; needed for DENOISER=swinir)"
[ -f "$SWINIR_CKPT" ] && echo "  [OK] SwinIR ckpt" || true

# HYPIR (face enhancement, step 01/06)
[ -d "$HYPIR_DIR/.git" ] && echo "  [OK] HYPIR code" || echo "  [MISS] HYPIR code (INSTALL_DEPS=1)"
[ -d "$HYPIR_BASE_MODEL" ] && echo "  [OK] SD2 base model" || echo "  [MISS] SD2 base model (INSTALL_DEPS=1)"
python -c "import mediapipe; print('  [OK] mediapipe')" 2>/dev/null || echo "  [MISS] mediapipe (INSTALL_DEPS=1)"

echo ""
echo "=== [00] Done. Next:"
echo "  GPU=0 INPUT_DIR=../<images> RESULTS_DIR=../../output/vggt_human_results bash vggt_human/01_face_enhance.sh"

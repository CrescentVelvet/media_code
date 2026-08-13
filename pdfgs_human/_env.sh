# _env.sh — shared setup: proxy + CA bundle + conda env activation + GPU + paths.
# Sourced by 00/01/02/03. Expects SCRIPT_DIR (this dir) to be set by the caller.
#
# Uses a SINGLE conda env (default `pdfgs`, CPython 3.10) for ALL steps:
#   - step 01 segmentation (SAM2 / rembg)
#   - step 02 Pi3 pose + COLMAP export
#   - step 03 PDF-GS training (diff-gaussian-rasterization + DINOv3)
# PDF-GS pins torch 2.5.1+cu121 (official environment.yml); this is NOT the same
# as wan22_rotate's torch 2.6.0+cu124, so it gets its own env. Override with
# CONDA_ENV=xxx.
REPO_DIR="$(dirname "$SCRIPT_DIR")"

# Optional proxy (gitignored proxy.env at repo root; see proxy.env.example).
if [ -f "$REPO_DIR/proxy.env" ]; then
    set -a; # shellcheck disable=SC1090
    source "$REPO_DIR/proxy.env"; set +a
fi

[ -n "${http_proxy:-}" ]  && export HTTP_PROXY="$http_proxy"
[ -n "${https_proxy:-}" ] && export HTTPS_PROXY="$https_proxy"

# --- Corporate proxy TLS interception workaround (pip/hf/git/conda) ---
# SSL_VERIFY switch (default true). The corporate proxy does TLS MITM and its root
# CA is NOT in the system/user CA bundle. true → point tools at the bundle (only
# works if the bundle actually contains the proxy CA). false → UNSET all CA-bundle
# env vars (so _env.sh doesn't re-export them — cmdline `unset` is futile because
# _env.sh re-exports on every source) + PYTHONHTTPSVERIFY=0; 00_setup_env.sh also
# injects sitecustomize.py (ssl._create_unverified_context) so Python (hf/requests/
# urllib) skips cert verification entirely.
# Use: SSL_VERIFY=false CONDA_SSL_VERIFY=false INSTALL_DEPS=1 bash pdfgs_human/00_setup_env.sh
export SSL_VERIFY="${SSL_VERIFY:-true}"
if [ "$SSL_VERIFY" = "false" ]; then
    unset REQUESTS_CA_BUNDLE SSL_CERT_FILE GIT_SSL_CAINFO PIP_CERT CURL_CA_BUNDLE
    export PYTHONHTTPSVERIFY=0
else
    # -s = exists AND non-empty. An EMPTY bundle pointed at a tool = "trust no CAs"
    # → SSL fails. Prefer non-empty user bundle, else non-empty system bundle.
    SYS_CA=/etc/ssl/certs/ca-certificates.crt
    USER_CA="$HOME/.ca-bundle.crt"
    if [ -s "$USER_CA" ]; then CA_FILE="$USER_CA"
    elif [ -s "$SYS_CA" ]; then CA_FILE="$SYS_CA"
    else CA_FILE=""; fi
    if [ -n "$CA_FILE" ]; then
        : "${REQUESTS_CA_BUNDLE:=$CA_FILE}"
        : "${SSL_CERT_FILE:=$CA_FILE}"
        : "${GIT_SSL_CAINFO:=$CA_FILE}"
        : "${PIP_CERT:=$CA_FILE}"
        : "${CURL_CA_BUNDLE:=$CA_FILE}"
        export REQUESTS_CA_BUNDLE SSL_CERT_FILE GIT_SSL_CAINFO PIP_CERT CURL_CA_BUNDLE
    fi
fi

export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"

# Force offline mode at runtime — all models are pre-downloaded by 00_setup_env.sh
# (corporate proxy blocks huggingface.co). Prevents transformers/huggingface_hub
# from attempting network downloads (SSL errors) during training.
# DINOv3 (gated) is pre-downloaded by 00 into $HF_HOME/hub, then read offline here.
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"

# Activate the conda env (CPython 3.10; PDF-GS deps).
CONDA_ENV="${CONDA_ENV:-pdfgs}"
export CONDA_ENV
if ! command -v conda >/dev/null 2>&1; then
    echo "ERROR: conda not found on PATH (need env '$CONDA_ENV')." >&2
    exit 1
fi
# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"

# In non-interactive bash, `conda activate <env>` often silently fails (shell
# hooks not installed) and `|| true` swallows the error → CONDA_PREFIX stays at
# base (/root/miniconda3, Python 3.9) → pip installs land in base, CC points at
# a non-existent gcc, nvcc build breaks. _conda_activate tries the real
# activate, verifies CONDA_PREFIX actually changed, and falls back to manually
# exporting PATH/LD_LIBRARY_PATH if it didn't.
_conda_activate() {
    local _env="$1" _base _expected
    _base="$(conda info --base 2>/dev/null)"
    _expected="$_base/envs/$_env"
    conda activate "$_env" 2>/dev/null || true
    if [ "$CONDA_PREFIX" = "$_expected" ]; then
        return 0
    fi
    if [ -d "$_expected" ]; then
        echo "⚠️ 'conda activate $_env' failed (non-interactive shell?); manually exporting PATH/LD_LIBRARY_PATH -> $_expected" >&2
        export CONDA_PREFIX="$_expected"
        export CONDA_DEFAULT_ENV="$_env"
        export PATH="$_expected/bin:$PATH"
        export LD_LIBRARY_PATH="$_expected/lib:${LD_LIBRARY_PATH:-}"
        return 0
    fi
    echo "⚠️ conda env '$_env' not found at $_expected (00_setup_env.sh will create it)" >&2
}
_conda_activate "$CONDA_ENV"

# Pin GPU (0-indexed) via GPU=N.
if [ -n "${GPU:-}" ]; then
    export CUDA_VISIBLE_DEVICES="$GPU"
fi

# CUDA library paths (libcupti.so.12 etc.)
for _cuda_lib in \
    "/usr/local/cuda/extras/CUPTI/lib64" \
    "/usr/local/cuda/lib64" \
    "$CONDA_PREFIX/lib"; do
    [ -d "$_cuda_lib" ] && export LD_LIBRARY_PATH="${_cuda_lib}:${LD_LIBRARY_PATH:-}"
done

# --- Paths ---
# Official code dirs (siblings of media_code, cloned by 00_setup_env.sh).
PDFGS_DIR="${PDFGS_DIR:-$REPO_DIR/../PDF-GS}"
PI3_DIR="${PI3_DIR:-$REPO_DIR/../Pi3}"
SAM2_DIR="${SAM2_DIR:-$REPO_DIR/../sam2}"

# Shared weight root (code-dir's parent, same as wan22_rotate / pi3_3dgs).
MODEL_DIR="${MODEL_DIR:-$REPO_DIR/../../model}"

# SAM2 checkpoint (person segmentation, step 01). sam2 repo layout:
#   $SAM2_DIR/checkpoints/sam2.1_hiera_large.pt
#   $SAM2_DIR/sam2/configs/sam2.1/sam2.1_hiera_large.yaml
export SAM2_CHECKPOINT="${SAM2_CHECKPOINT:-$SAM2_DIR/checkpoints/sam2.1_hiera_large.pt}"
export SAM2_CONFIG="${SAM2_CONFIG:-configs/sam2.1/sam2.1_hiera_large.yaml}"

# Pi3 checkpoint (feed-forward pose estimation, step 02).
export PI3_CKPT="${PI3_CKPT:-$MODEL_DIR/Pi3/model.safetensors}"

# rembg (fallback segmentor in step 01, when SAM2 unavailable). rembg loads its
# u2net model from $U2NET_HOME (default ~/.u2net, auto-downloads if missing — the
# corporate proxy blocks the model host). Point it at the local $MODEL_DIR/u2net so
# a pre-downloaded u2net.onnx is used without network. MODEL_CHECKSUM_DISABLED
# skips the checksum check (a manually-downloaded u2net.onnx may differ from rembg's
# pinned version, which would otherwise trigger a re-download).
export U2NET_HOME="${U2NET_HOME:-$MODEL_DIR/u2net}"
export MODEL_CHECKSUM_DISABLED="${MODEL_CHECKSUM_DISABLED:-1}"

# DINOv3 (step 03 — PDF-GS DINOv3FeatureExtractor loads via transformers
# from_pretrained(repo). PDF-GS hardcodes facebook/dinov3-vitb16-pretrain-lvd1689m
# (ViT-B/16, GATED). We prefer a LOCAL dir if present — e.g. dinov3-vitl16-pretrain-lvd1689m
# (ViT-L/16, larger but architecture-compatible: patch=16 + 4 register tokens, just
# 1024-dim features vs 768; runs fine, avoids the gated download). 00 patches train.py
# to honor DINOV3_REPO; override DINOV3_REPO=... to force a path or HF repo id.
export HF_HOME="${HF_HOME:-$MODEL_DIR/hf_home}"
if [ -z "${DINOV3_REPO:-}" ]; then
    if [ -d "$MODEL_DIR/dinov3-vitl16-pretrain-lvd1689m" ]; then
        DINOV3_REPO="$MODEL_DIR/dinov3-vitl16-pretrain-lvd1689m"
    else
        DINOV3_REPO="facebook/dinov3-vitb16-pretrain-lvd1689m"
    fi
fi
export DINOV3_REPO
# HF_TOKEN only needed at download time, and only when DINOV3_REPO is the gated HF
# repo id (not a local dir). At runtime (offline) a local dir loads without token.

# Output dir (outside the repo, like wan22_rotate_results / pi3_3dgs_results).
RESULTS_DIR="${RESULTS_DIR:-$REPO_DIR/../pdfgs_human_results}"

export REPO_DIR PDFGS_DIR PI3_DIR SAM2_DIR MODEL_DIR RESULTS_DIR

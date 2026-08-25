# _env.sh — shared setup: proxy + CA bundle + conda env activation + GPU + paths.
# Sourced by 00/01/02/run_all. Expects SCRIPT_DIR (this dir) to be set by caller.
#
# 4DAnyone (ant-research) turns a monocular video into multi-view videos for
# downstream 4DGS reconstruction. It uses a DEDICATED conda env (default
# `4danyone`, CPython 3.11) with torch 2.8 (PyPI build ships cu12x by default,
# no download.pytorch.org needed).
#
# ⚠️  Inference needs ~43 GiB peak VRAM (6-view minimum, see
#    docs/inference_performance.md). RTX 3090 (24 GiB) is NOT enough; run on a
#    >=48 GiB card (A6000/A100/H100/H200) on the server.
REPO_DIR="$(dirname "$SCRIPT_DIR")"

# 1. proxy (gitignored proxy.env at repo root; see proxy.env.example)
if [ -f "$REPO_DIR/proxy.env" ]; then
    set -a; # shellcheck disable=SC1090
    source "$REPO_DIR/proxy.env"; set +a
fi
[ -n "${http_proxy:-}" ]  && export HTTP_PROXY="$http_proxy"
[ -n "${https_proxy:-}" ] && export HTTPS_PROXY="$https_proxy"

# 2. CA bundle (corporate proxy TLS interception)
SYS_CA=/etc/ssl/certs/ca-certificates.crt
USER_CA="$HOME/.ca-bundle.crt"
if [ -f "$USER_CA" ]; then CA_FILE="$USER_CA"
elif [ -f "$SYS_CA" ]; then CA_FILE="$SYS_CA"
else CA_FILE=""; fi
if [ -n "$CA_FILE" ]; then
    : "${REQUESTS_CA_BUNDLE:=$CA_FILE}"
    : "${SSL_CERT_FILE:=$CA_FILE}"
    : "${GIT_SSL_CAINFO:=$CA_FILE}"
    : "${PIP_CERT:=$CA_FILE}"
    : "${CURL_CA_BUNDLE:=$CA_FILE}"
    export REQUESTS_CA_BUNDLE SSL_CERT_FILE GIT_SSL_CAINFO PIP_CERT CURL_CA_BUNDLE
fi
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"

# 3. conda env activation
CONDA_ENV="${CONDA_ENV:-4danyone}"
export CONDA_ENV
# Fallback: conda not on PATH -> look in common install locations (WSL without
# `conda init`). On the server conda is already on PATH so this block is a no-op.
if ! command -v conda >/dev/null 2>&1; then
    for _cb in "$HOME/miniconda3" "$HOME/anaconda3" "/opt/conda"; do
        if [ -f "$_cb/etc/profile.d/conda.sh" ]; then
            # shellcheck disable=SC1091
            source "$_cb/etc/profile.d/conda.sh"
            break
        fi
    done
    unset _cb
fi
if ! command -v conda >/dev/null 2>&1; then
    echo "❌ ERROR: conda not found on PATH (need env '$CONDA_ENV')." >&2
    echo "       Install miniconda or run: source ~/miniconda3/etc/profile.d/conda.sh" >&2
    exit 1
fi
# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV" 2>/dev/null || true  # env may not exist yet (00 creates it)

# 4. GPU selection (0-indexed via GPU=N)
if [ -n "${GPU:-}" ]; then
    export CUDA_VISIBLE_DEVICES="$GPU"
fi

# 5. paths (all use ${VAR:-default} so external overrides work)
#    Official repo:  $REPO_DIR/../4DAnyone  (sibling of media_code)
#    Model root:     $REPO_DIR/../../model/4danyone  (shared model root, one level up)
#    Output:         $REPO_DIR/../4danyone_results  (sibling of media_code)
export FDANYONE_DIR="${FDANYONE_DIR:-$REPO_DIR/../4DAnyone}"
export GVHMR_DIR="${GVHMR_DIR:-$FDANYONE_DIR/third_party/GVHMR}"
export MODEL_DIR="${MODEL_DIR:-$REPO_DIR/../../model/4danyone}"
export RESULTS_DIR="${RESULTS_DIR:-$REPO_DIR/../4danyone_results}"

# SMPL-X is separately licensed; user must install it manually (see 01 script).
# The body model lives under $MODEL_DIR/body_models/smplx/SMPLX_NEUTRAL.npz.
export SMPLX_PATH="${SMPLX_PATH:-$MODEL_DIR/body_models/smplx/SMPLX_NEUTRAL.npz}"

export REPO_DIR

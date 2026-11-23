# Shared environment for Apple Depth Pro 2 (Python >= 3.11).
REPO_DIR="$(dirname "$SCRIPT_DIR")"
if [ -f "$REPO_DIR/proxy.env" ]; then
    set -a; source "$REPO_DIR/proxy.env"; set +a
fi
[ -n "${http_proxy:-}" ] && export HTTP_PROXY="$http_proxy"
[ -n "${https_proxy:-}" ] && export HTTPS_PROXY="$https_proxy"
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
    export REQUESTS_CA_BUNDLE SSL_CERT_FILE GIT_SSL_CAINFO PIP_CERT
fi
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"
CONDA_ENV="${CONDA_ENV:-sharp}"
export CONDA_ENV
if ! command -v conda >/dev/null 2>&1; then
    for _cb in "$HOME/miniconda3" "$HOME/anaconda3" "/opt/conda"; do
        if [ -f "$_cb/etc/profile.d/conda.sh" ]; then
            source "$_cb/etc/profile.d/conda.sh"
            break
        fi
    done
    unset _cb
fi
if ! command -v conda >/dev/null 2>&1; then
    echo "ERROR: conda not found on PATH (need env '$CONDA_ENV')." >&2
    exit 1
fi
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV" 2>/dev/null || true
if [ -n "${GPU:-}" ]; then export CUDA_VISIBLE_DEVICES="$GPU"; fi
DEPTH_PRO_2_DIR="${DEPTH_PRO_2_DIR:-$REPO_DIR/../ml-depth-pro-2}"
DEPTH_PRO_2_REPO="${DEPTH_PRO_2_REPO:-https://github.com/apple-aiml-research/ml-depth-pro-2.git}"
MODEL_DIR="${MODEL_DIR:-$REPO_DIR/../../model/depth_pro_2}"
RESULTS_DIR="${RESULTS_DIR:-$REPO_DIR/../depth_pro_2_results}"
CKPT_PATH="${CKPT_PATH:-$MODEL_DIR/depthpro2-h-eivtm5w7m5.pt}"
FOV_CKPT_PATH="${FOV_CKPT_PATH:-$MODEL_DIR/depthpro2-fov-ew55zg4kq7.pt}"
export REPO_DIR DEPTH_PRO_2_DIR DEPTH_PRO_2_REPO MODEL_DIR RESULTS_DIR CKPT_PATH FOV_CKPT_PATH

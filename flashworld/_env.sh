# _env.sh — shared setup: proxy + CA bundle + conda env + GPU + paths.
# Sourced by 00a/01/02/08. Expects SCRIPT_DIR (this dir) to be set by the caller.
#
# FlashWorld (Tencent, ICLR 2026 Oral) — 单图/文本 → 3DGS，3D-oriented 多视图扩散 + 跨模式蒸馏。
# 官方仓只发推理代码（训练代码需填表申请）；权重：
#   - FlashWorld model.ckpt（~19.5 GB，HF imlixinyang/FlashWorld）
#   - Wan2.2-TI2V-5B-Diffusers 基座（~34.2 GB，vae/tokenizer/text_encoder/transformer/scheduler）
REPO_DIR="$(dirname "$SCRIPT_DIR")"

# Optional proxy (gitignored proxy.env at repo root).
if [ -f "$REPO_DIR/proxy.env" ]; then
    set -a; # shellcheck disable=SC1090
    source "$REPO_DIR/proxy.env"; set +a
fi

[ -n "${http_proxy:-}" ]  && export HTTP_PROXY="$http_proxy"
[ -n "${https_proxy:-}" ] && export HTTPS_PROXY="$https_proxy"

# --- Corporate proxy TLS interception workaround (pip/hf/git) ---
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

# Activate the flashworld conda env (00a creates it: python=3.11 + torch 2.6.0+cu124).
CONDA_ENV="${CONDA_ENV:-flashworld}"
export CONDA_ENV
if ! command -v conda >/dev/null 2>&1; then
    # Fallback: try common conda locations (WSL without `conda init`).
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
    echo "ERROR: conda not found on PATH (need env '$CONDA_ENV')." >&2
    echo "       Install miniconda or run: source ~/miniconda3/etc/profile.d/conda.sh" >&2
    exit 1
fi
# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV" 2>/dev/null || true  # env may not exist yet (00a creates it)

# Pin GPU (0-indexed) via GPU=N.
if [ -n "${GPU:-}" ]; then
    export CUDA_VISIBLE_DEVICES="$GPU"
fi

# --- CUDA toolkit for gsplat native build ---
# gsplat 1.5.2 (pinned commit) is built FROM SOURCE here: CUDAExtension needs
# nvcc + g++ at INSTALL time (not JIT like sharp's 1.5.3 wheel).
# 00a installs cuda-nvcc + gxx into the env (no sudo) -> point CUDA_HOME at prefix.
if [ -n "${CONDA_PREFIX:-}" ] && [ -x "$CONDA_PREFIX/bin/nvcc" ]; then
    export CUDA_HOME="${CUDA_HOME:-$CONDA_PREFIX}"
elif [ -d /usr/local/cuda ]; then
    export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
fi
[ -n "${CUDA_HOME:-}" ] && export PATH="$CUDA_HOME/bin:$PATH"
for _cuda_lib in "$CONDA_PREFIX/lib" "$CUDA_HOME/lib64"; do
    [ -d "$_cuda_lib" ] && export LD_LIBRARY_PATH="${_cuda_lib}:${LD_LIBRARY_PATH:-}"
done
unset _cuda_lib
# 3090 = sm_86. 官方在 A100/H100 用 9.0+PTX；不 pin 会为所有 arch 编译（慢十几倍）。
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.6}"

# --- Paths ---
# Official code (Linux fs on WSL; 00a clones it). Three pinned git deps live in
# $FLASHWORLD_DEPS (gsplat / spz / diffusers) so pip never hits the network for them.
FLASHWORLD_DIR="${FLASHWORLD_DIR:-$REPO_DIR/../FlashWorld}"
FLASHWORLD_REPO="${FLASHWORLD_REPO:-https://github.com/imlixinyang/FlashWorld.git}"
FLASHWORLD_DEPS="${FLASHWORLD_DEPS:-$HOME/repos/_flashworld_deps}"

# Weight root + output.
# WSL: proxy.env 写 FLASHWORLD_MODEL_DIR / FLASHWORLD_RESULTS_DIR（项目专属名）。
# 权重布局（HF cache 格式，app.py 里 model_id 是字符串，靠 HF_HOME 解析）：
#   $FLASHWORLD_MODEL_DIR/hf_cache/hub/models--Wan-AI--Wan2.2-TI2V-5B-Diffusers/...
#   $FLASHWORLD_MODEL_DIR/hf_cache/hub/models--imlixinyang--FlashWorld/...
MODEL_DIR="${MODEL_DIR:-${FLASHWORLD_MODEL_DIR:-$REPO_DIR/../../model/flashworld}}"
RESULTS_DIR="${RESULTS_DIR:-${FLASHWORLD_RESULTS_DIR:-$REPO_DIR/../flashworld_results}}"

export HF_HOME="${HF_HOME:-$MODEL_DIR/hf_cache}"
WAN_MODEL_ID="${WAN_MODEL_ID:-Wan-AI/Wan2.2-TI2V-5B-Diffusers}"
FW_REPO_ID="${FW_REPO_ID:-imlixinyang/FlashWorld}"

# ckpt 在 HF cache 里的快照路径（sha 随仓库更新会变，故用 glob）
FW_CKPT="${FW_CKPT:-$(ls -d "$HF_HOME"/hub/models--imlixinyang--FlashWorld/snapshots/*/model.ckpt 2>/dev/null | head -1)}"
WAN_SNAPSHOT="${WAN_SNAPSHOT:-$(ls -d "$HF_HOME"/hub/models--Wan-AI--Wan2.2-TI2V-5B-Diffusers/snapshots/*/ 2>/dev/null | head -1)}"

export REPO_DIR FLASHWORLD_DIR FLASHWORLD_REPO FLASHWORLD_DEPS \
       MODEL_DIR RESULTS_DIR WAN_MODEL_ID FW_REPO_ID FW_CKPT WAN_SNAPSHOT CUDA_HOME

# _env.sh — shared setup: proxy + CA bundle + conda env activation + GPU + paths.
# Sourced by 00a/01/02. Expects SCRIPT_DIR (this dir) to be set by the caller.
#
# TaoFlowForge（阿里 Taobao3D，arXiv 2609.37139）— 单图 → 原生三角网格（三阶段
# 级联流匹配：64³ 占据 → 512³ 顶点 → 拓扑解码）。官方仓只含推理代码 + Gradio。
# 权重: ModelScope Taobao3D/TaoFlowForge（stage0/1/2.pt），Apache-2.0，非 gated。
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

# Activate the taoflowforge conda env (00a creates it: python=3.10 + torch 2.5.1 cu121,
# 与官方 README 测试配置一致).
CONDA_ENV="${CONDA_ENV:-taoflowforge}"
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

# --- Paths ---
# Official repo alibaba/Taobao3D（仓里 TaoFlowForge/ 子目录才是本项目代码；
# `python -m taoflowforge.cli` 必须以该子目录为 cwd 运行——仓根没有 taoflowforge 包）。
TAOFLOWFORGE_REPO="${TAOFLOWFORGE_REPO:-https://github.com/alibaba/Taobao3D.git}"
TAOFLOWFORGE_DIR="${TAOFLOWFORGE_DIR:-$REPO_DIR/../Taobao3D}"
TFF_PKG_DIR="$TAOFLOWFORGE_DIR/TaoFlowForge"

# Weight root + output.
# WSL: proxy.env 写 TAOFLOWFORGE_MODEL_DIR / TAOFLOWFORGE_RESULTS_DIR（项目专属名，
# 避免 set -a source 后污染其它项目）。
MODEL_DIR="${MODEL_DIR:-${TAOFLOWFORGE_MODEL_DIR:-$REPO_DIR/../../model/taoflowforge}}"
RESULTS_DIR="${RESULTS_DIR:-${TAOFLOWFORGE_RESULTS_DIR:-$REPO_DIR/../taoflowforge_results}}"

# 三个 stage checkpoint（ModelScope 快照下到哪层不保证，01 会归一到这三个路径）
TFF_STAGE0="${TFF_STAGE0:-$MODEL_DIR/stage0.pt}"
TFF_STAGE1="${TFF_STAGE1:-$MODEL_DIR/stage1.pt}"
TFF_STAGE2="${TFF_STAGE2:-$MODEL_DIR/stage2.pt}"
TFF_MODELSCOPE_ID="${TFF_MODELSCOPE_ID:-Taobao3D/TaoFlowForge}"

# DINOv3-L/16 目录（--dino-checkpoint；发布版 stage0 未内嵌 DINO 权重，必配）。
# ModelScope 镜像 facebook/dinov3-vitl16-pretrain-lvd1689m（非 gated，
# config.json / model.safetensors 齐全）。
TFF_DINO_DIR="${TFF_DINO_DIR:-/mnt/d/wheel/dinov3-vitl16}"

# 可选：RMBG-2.0 抠背景（HF gated，需自行申请）；不设置则关闭背景移除。
TFF_REMBG_DIR="${TFF_REMBG_DIR:-}"

export REPO_DIR TAOFLOWFORGE_REPO TAOFLOWFORGE_DIR TFF_PKG_DIR \
       MODEL_DIR RESULTS_DIR TFF_STAGE0 TFF_STAGE1 TFF_STAGE2 \
       TFF_MODELSCOPE_ID TFF_DINO_DIR TFF_REMBG_DIR

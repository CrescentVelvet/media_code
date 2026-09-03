# _env.sh — shared setup: proxy + CA bundle + conda env activation + GPU + paths.
# Sourced by 00/01/02/03. Expects SCRIPT_DIR (this dir) to be set by the caller.
#
# Reuses the vggt-omega conda env (default `doll`, torch>=2.3). No new env.
# VGGT-Omega provides the feed-forward model (poses + depth -> point cloud);
# gaussian-splatting (original 3DGS) provides train.py / render.py.
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

# Activate the existing conda env (torch already installed; reuse).
CONDA_ENV="${CONDA_ENV:-vggt_human}"
export CONDA_ENV
if ! command -v conda >/dev/null 2>&1; then
    # Fallback: try common conda locations (WSL/local dev without `conda init`).
    # On servers conda is already on PATH, so this block never triggers.
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
conda activate "$CONDA_ENV" 2>/dev/null || true  # env may not exist yet (00 creates it)

# Pin GPU (0-indexed) via GPU=N.
if [ -n "${GPU:-}" ]; then
    export CUDA_VISIBLE_DEVICES="$GPU"
fi

# CUDA library paths (libcupti.so.12 etc.).
for _cuda_lib in \
    "/usr/local/cuda/extras/CUPTI/lib64" \
    "/usr/local/cuda/lib64" \
    "$CONDA_PREFIX/lib"; do
    [ -d "$_cuda_lib" ] && export LD_LIBRARY_PATH="${_cuda_lib}:${LD_LIBRARY_PATH:-}"
done

# --- Paths ---
# VGGT-Omega official code (sibling of media_code; 00 clones it).
VGGT_DIR="${VGGT_DIR:-$REPO_DIR/../vggt-omega}"
VGGT_REPO="${VGGT_REPO:-https://github.com/facebookresearch/vggt-omega.git}"

# Original 3DGS official code (sibling of media_code; 00 clones it).
GS_DIR="${GS_DIR:-$REPO_DIR/../gaussian-splatting}"
GS_REPO="${GS_REPO:-https://github.com/graphdeco-inria/gaussian-splatting.git}"

# Shared weight root (code-dir's parent, same as other algos).
MODEL_DIR="${MODEL_DIR:-$REPO_DIR/../../model/VGGT-Omega}"
# Separate weights root for denoiser models (DiffBIR / SwinIR, not VGGT-Omega).
WEIGHTS_ROOT="${WEIGHTS_ROOT:-$REPO_DIR/../../model}"

# Output (siblings of media_code, per AGENTS.md convention).
RESULTS_DIR="${RESULTS_DIR:-$REPO_DIR/../vggt_human_results}"

# DiffBIR (denoising, step 04; optional — only when DENOISER=diffbir).
DIFFBIR_DIR="${DIFFBIR_DIR:-$REPO_DIR/../DiffBIR}"
DIFFBIR_REPO="${DIFFBIR_REPO:-https://github.com/XPixelGroup/DiffBIR.git}"
DIFFBIR_CKPT="${DIFFBIR_CKPT:-$WEIGHTS_ROOT/DiffBIR/cldm.pth}"
DIFFBIR_CONFIG="${DIFFBIR_CONFIG:-$DIFFBIR_DIR/configs/inference/cldm.yaml}"

# SwinIR (denoising, step 04; optional — only when DENOISER=swinir).
SWINIR_DIR="${SWINIR_DIR:-$REPO_DIR/../SwinIR}"
SWINIR_REPO="${SWINIR_REPO:-https://github.com/JingyunLiang/SwinIR.git}"
SWINIR_CKPT="${SWINIR_CKPT:-$WEIGHTS_ROOT/SwinIR/model_gaussian_gray_denoising_blind.pth}"

# HYPIR (face enhancement, step 01/06; HYPIR deps installed in vggt_human env).
HYPIR_DIR="${HYPIR_DIR:-$REPO_DIR/../HYPIR}"
HYPIR_MODEL_DIR="${HYPIR_MODEL_DIR:-$WEIGHTS_ROOT/HYPIR}"
HYPIR_BASE_MODEL="${HYPIR_BASE_MODEL:-$HYPIR_MODEL_DIR/sd2_base}"
HYPIR_WEIGHT="${HYPIR_WEIGHT:-$HYPIR_DIR/experiments/beauty_ppr50k_20260721/checkpoint-1000/ema_state_dict.pth}"

# Pose optimization (step 04/07; PoseAdjuster + PoseRefineModule)
export POSE_ADJUST="${POSE_ADJUST:-1}"
export POSE_REFINE="${POSE_REFINE:-1}"
export REFINE_INTRINSIC="${REFINE_INTRINSIC:-0}"
export POSE_REFINE_WEIGHT="${POSE_REFINE_WEIGHT:-0.01}"
export POSE_REFINE_LR_Q="${POSE_REFINE_LR_Q:-1e-3}"
export POSE_REFINE_LR_T="${POSE_REFINE_LR_T:-1e-3}"
export POSE_REFINE_LR_I="${POSE_REFINE_LR_I:-1e-4}"
export GRAVITY_PRIOR="${GRAVITY_PRIOR:-0}"

# Dynamic mask & filtering (step 04 预处理; P0-1/P0-2)
# 默认 0（与其它增强一致：默认干净基线，显式开启）。
# ⚠️ 曾默认 1 + prompt 含 person，静态人物数据集会把主体当"动态"删点。
#   prompt 用 DYNAMIC_PROMPTS 覆盖（默认 "TV screen monitor"）。
export ENABLE_DYNAMIC_MASK="${ENABLE_DYNAMIC_MASK:-0}"
export ENABLE_DYNAMIC_FILTER="${ENABLE_DYNAMIC_FILTER:-0}"
export ENABLE_MLP_DYNAMIC="${ENABLE_MLP_DYNAMIC:-0}"   # 保留占位，当前无代码引用
export DYNAMIC_PROMPTS="${DYNAMIC_PROMPTS:-TV screen monitor}"
export DYNAMIC_THRESHOLD="${DYNAMIC_THRESHOLD:-0.3}"
export DYNAMIC_DILATE_PX="${DYNAMIC_DILATE_PX:-5}"
export SAM2_MODEL_PATH="${SAM2_MODEL_PATH:-$MODEL_DIR/sam2}"
export DINO_MODEL_PATH="${DINO_MODEL_PATH:-$MODEL_DIR/dinov2}"

# Depth-normal consistency (step 04, P1-1)
# 默认 0 = 走官方基线 train.py。曾默认 1，导致不明说就静默切到增强分支
# （且会盖掉其它增强开关），与「默认干净基线、增强逐个显式开启」的设计冲突。
export USE_DEPTH_NORMAL="${USE_DEPTH_NORMAL:-0}"
export DEPTH_NORMAL_WEIGHT="${DEPTH_NORMAL_WEIGHT:-0.05}"

export REPO_DIR VGGT_DIR VGGT_REPO GS_DIR GS_REPO MODEL_DIR RESULTS_DIR \
       WEIGHTS_ROOT DIFFBIR_DIR DIFFBIR_REPO DIFFBIR_CKPT DIFFBIR_CONFIG \
       SWINIR_DIR SWINIR_REPO SWINIR_CKPT \
       HYPIR_DIR HYPIR_MODEL_DIR HYPIR_BASE_MODEL HYPIR_WEIGHT \
       ENABLE_DYNAMIC_MASK ENABLE_DYNAMIC_FILTER ENABLE_MLP_DYNAMIC \
       SAM2_MODEL_PATH DINO_MODEL_PATH \
       USE_DEPTH_NORMAL DEPTH_NORMAL_WEIGHT

# _env.sh — shared setup: proxy + CA bundle + conda env + GPU + paths。
# 被 00/01/02/... 各 .sh source。调用方需先设 SCRIPT_DIR。
#
# flame_human = FLAME 2020 参数化头模型 + DECA 初值 + 表情驱动的 3DGS 人脸链路。
# 与 vggt_human 的关系：本目录不重做 SfM/COLMAP，输入直接吃上游产出的
# 03_source（COLMAP 场景 + 相机）与 person mask（SegTrack/SAM3）。

REPO_DIR="$(dirname "$SCRIPT_DIR")"

# 可选代理（仓根 proxy.env，gitignored）
if [ -f "$REPO_DIR/proxy.env" ]; then
    set -a; # shellcheck disable=SC1090
    source "$REPO_DIR/proxy.env"; set +a
fi

[ -n "${http_proxy:-}" ]  && export HTTP_PROXY="$http_proxy"
[ -n "${https_proxy:-}" ] && export HTTPS_PROXY="$https_proxy"

# --- 公司代理 TLS 拦截绕过（pip/hf/git）---
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

# --- conda env（conda 不在 PATH 时自动找常见安装位置）---
CONDA_ENV="${CONDA_ENV:-flame_human}"
export CONDA_ENV
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
    echo "ERROR: conda not found on PATH (need env '$CONDA_ENV')." >&2
    echo "       Install miniconda or run: source ~/miniconda3/etc/profile.d/conda.sh" >&2
    exit 1
fi
# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV" 2>/dev/null || true  # env 可能还没建（00 负责）

# GPU 选卡（GPU=N，0-indexed）
if [ -n "${GPU:-}" ]; then
    export CUDA_VISIBLE_DEVICES="$GPU"
fi

# CUDA 库路径（libcupti 等）
# 仅在 env 激活成功（CONDA_PREFIX 指向本 env）时才拼，避免 env 未建时
# 把 base 的 lib 挂进来引起库冲突。
if [ -n "${CONDA_PREFIX:-}" ] && [ -d "$CONDA_PREFIX/lib" ]; then
    case ":${LD_LIBRARY_PATH:-}:" in
        *":$CONDA_PREFIX/lib:"*) ;;  # 已有，不重复拼
        *) export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}" ;;
    esac
fi
unset _cuda_lib

# --- 官方代码 ---
# DECA：提供 shape/expr/pose 初值（只取初值，不取 landmark，见 NOTES）
DECA_DIR="${DECA_DIR:-$REPO_DIR/../DECA}"
DECA_REPO="${DECA_REPO:-https://github.com/yfeng95/DECA.git}"
# 原版 3DGS（rasterization / simple_knn 子模块复用；训练循环自己写，因为要带 local_exp）
GS_DIR="${GS_DIR:-$REPO_DIR/../gaussian-splatting}"
GS_REPO="${GS_REPO:-https://github.com/graphdeco-inria/gaussian-splatting.git}"

# --- 权重 ---
MODEL_DIR="${MODEL_DIR:-$REPO_DIR/../../model/flame_human}"
WEIGHTS_ROOT="${WEIGHTS_ROOT:-$REPO_DIR/../../model}"

# FLAME 2020（需官网注册下载，见 download_urls.md）
FLAME_MODEL="${FLAME_MODEL:-$MODEL_DIR/FLAME2020/generic_model.pkl}"
# MediaPipe 468 点 → FLAME 顶点的重心嵌入（必须，缺了 4.3 没法算 landmark 残差）
FLAME_LM468_EMBEDDING="${FLAME_LM468_EMBEDDING:-$MODEL_DIR/flame_lm468_embedding.npz}"
# DECA 预训练权重
DECA_CKPT="${DECA_CKPT:-$MODEL_DIR/deca_model.tar}"

# HYPIR（人脸增强，阶段五/阶段八；与 vggt_human 共用同一份）
HYPIR_DIR="${HYPIR_DIR:-$REPO_DIR/../HYPIR}"
HYPIR_MODEL_DIR="${HYPIR_MODEL_DIR:-$WEIGHTS_ROOT/HYPIR}"
HYPIR_BASE_MODEL="${HYPIR_BASE_MODEL:-$HYPIR_MODEL_DIR/sd21_base}"
HYPIR_WEIGHT="${HYPIR_WEIGHT:-$HYPIR_MODEL_DIR/HYPIR_sd2.pth}"

# --- 输入 / 输出 ---
# 上游 vggt_human 的输出（用 UPSTREAM_DIR 一次性改，避免每步都带两个路径）
UPSTREAM_DIR="${UPSTREAM_DIR:-$REPO_DIR/../vggt_human_results}"
# COLMAP 场景（含 images/ 与 sparse 相机）
SOURCE_DIR="${SOURCE_DIR:-$UPSTREAM_DIR/03_source}"
# SegTrack/SAM3 的 person mask（阶段二做 face↔body 关联）
PERSON_MASKS_DIR="${PERSON_MASKS_DIR:-$UPSTREAM_DIR/03_sam3_person_masks}"
# 本链路输出
RESULTS_DIR="${RESULTS_DIR:-$REPO_DIR/../flame_human_results}"

export REPO_DIR DECA_DIR DECA_REPO DECA_CKPT GS_DIR GS_REPO \
       MODEL_DIR WEIGHTS_ROOT FLAME_MODEL FLAME_LM468_EMBEDDING \
       HYPIR_DIR HYPIR_MODEL_DIR HYPIR_BASE_MODEL HYPIR_WEIGHT \
       UPSTREAM_DIR SOURCE_DIR PERSON_MASKS_DIR RESULTS_DIR

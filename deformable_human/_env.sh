# _env.sh — shared setup: proxy + CA bundle + conda env activation.
# Sourced by all deformable_human/*.sh. Expects SCRIPT_DIR (this dir) set by caller.
REPO_DIR="$(dirname "$SCRIPT_DIR")"

# 1. 代理（从 proxy.env 读；本机一般只有 HF_ENDPOINT / PIP_INDEX_URL，没有 http_proxy）
if [ -f "$REPO_DIR/proxy.env" ]; then
    set -a; # shellcheck disable=SC1090
    source "$REPO_DIR/proxy.env"; set +a
fi
[ -n "${http_proxy:-}" ]  && export HTTP_PROXY="$http_proxy"
[ -n "${https_proxy:-}" ] && export HTTPS_PROXY="$https_proxy"

# 2. CA bundle（公司代理 TLS 拦截；本机通常用系统 bundle 即可）
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

# 3. conda env 激活
# 本目录 env 由 00a_setup_env.sh 从 vggt_human **clone** 而来（torch 2.5.1+cu121 +
# conda cuda-nvcc 12.1 工具链随 clone 继承），不要与 deformable_gaussians 的
# py3.7/torch1.13 独立 env 混用——那是服务器侧的取舍，见 README_wsl.md。
CONDA_ENV="${CONDA_ENV:-deformable_human}"
export CONDA_ENV
# Fallback: conda 不在 PATH 时自动找常见安装位置（WSL 本机未 conda init 时触发）
if ! command -v conda >/dev/null 2>&1; then
    for _cb in "$HOME/miniconda3" "$HOME/anaconda3" "/opt/conda"; do
        if [ -f "$_cb/etc/profile.d/conda.sh" ]; then
            # shellcheck disable=SC1090
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
conda activate "$CONDA_ENV" 2>/dev/null || true  # env 可能还没建（00a 负责创建）

# 4. GPU 选卡（cuda:0 in-process == 物理 GPU $GPU）
if [ -n "${GPU:-}" ]; then
    export CUDA_VISIBLE_DEVICES="$GPU"
fi

# 5. 路径（都用 ${VAR:-default} 允许外部覆盖；WSL 上由 proxy.env 覆盖）
# WSL 路径纪律（README_wsl.md）：官方仓与训练输出必须在 Linux fs（~/repos、~/output），
# /mnt/c 只读代码、/mnt/d 只放最终产物。proxy.env 用项目专属变量覆盖默认：
#   DEFORMABLE_HUMAN_RESULTS_DIR → $HOME/output/deformable_human_results
#   DG_DIR                       → $HOME/repos/Deformable-3D-Gaussians（通用名，与
#                                  deformable_gaussians 共用同一官方仓）
MODEL_DIR="${MODEL_DIR:-$REPO_DIR/../../model}"
RESULTS_DIR="${RESULTS_DIR:-${DEFORMABLE_HUMAN_RESULTS_DIR:-$REPO_DIR/../deformable_human_results}}"
DG_DIR="${DG_DIR:-$HOME/repos/Deformable-3D-Gaussians}"
# Stage A 复用的兄弟编排目录（只读调用，不复制代码）
VGGT_HUMAN_DIR="${VGGT_HUMAN_DIR:-$REPO_DIR/vggt_human}"
DG_ORCH_DIR="${DG_ORCH_DIR:-$REPO_DIR/deformable_gaussians}"
# 本地 wheel 缓存（所有算法共用；pip --find-links 用）
WHEELS_DIR="${WHEELS_DIR:-/mnt/d/wheel}"
export REPO_DIR MODEL_DIR RESULTS_DIR DG_DIR VGGT_HUMAN_DIR DG_ORCH_DIR WHEELS_DIR

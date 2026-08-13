# _env.sh — shared setup: proxy + CA bundle + conda env activation + GPU + paths.
# Sourced by 00/01/02/03/04/run_all. Expects SCRIPT_DIR (this dir) to be set by the caller.
#
# World-R1 用一个 conda env (world_r1, CPython 3.10) 跑全部步骤:
# reward server (DA3 + Qwen3-VL) + Flow-GRPO 训练 + 推理。
REPO_DIR="$(dirname "$SCRIPT_DIR")"

# 1. 代理（从 proxy.env 读）
if [ -f "$REPO_DIR/proxy.env" ]; then
    set -a; # shellcheck disable=SC1090
    source "$REPO_DIR/proxy.env"; set +a
fi

[ -n "${http_proxy:-}" ]  && export HTTP_PROXY="$http_proxy"
[ -n "${https_proxy:-}" ] && export HTTPS_PROXY="$https_proxy"

# 2. CA bundle（公司代理 TLS 拦截）
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

# 强制离线——所有模型预下载（公司代理封 HF）；reward server 的 DA3 / Qwen3-VL
# 都从 HF cache 加载, 预下载后离线模式防 SSL 报错。01 下载脚本临时关掉它下载。
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"

# 3. conda env 激活
CONDA_ENV="${CONDA_ENV:-world_r1}"
export CONDA_ENV
if ! command -v conda >/dev/null 2>&1; then
    echo "ERROR: conda not found on PATH (need env '$CONDA_ENV')." >&2
    exit 1
fi
# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV" 2>/dev/null || true  # env 可能还没建（00 负责创建）

# 4. GPU 选卡
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

# 5. 路径（都用 ${VAR:-default} 允许外部覆盖）

# World-R1 官方代码（clone 到 media_code 的 sibling）
WORLD_R1_DIR="${WORLD_R1_DIR:-$REPO_DIR/../World-R1}"

# 权重根（各算法共享，在 code-dir 上一级）
MODEL_DIR="${MODEL_DIR:-$REPO_DIR/../../model}"

# HF cache 根——DA3 / Qwen3-VL / HPSv2 等从 HF 加载的模型都缓存在这里。
# reward server 的 worker process 用 from_pretrained("repo_id"), 设好 HF_HOME
# 后会从本地 cache 加载（配合 HF_HUB_OFFLINE=1）。
export HF_HOME="${HF_HOME:-$MODEL_DIR/huggingface}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/transformers}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-$HF_HOME/hub}"

# 基础视频模型（Wan2.1-T2V-14B-Diffusers, 用于 RL 训练 + 推理）
WAN_MODEL_PATH="${WAN_MODEL_PATH:-$MODEL_DIR/Wan2.1-T2V-14B-Diffusers}"

# 输出（在 media_code 的 sibling）
RESULTS_DIR="${RESULTS_DIR:-$REPO_DIR/../world_r1_results}"

# 训练产物（checkpoint / 日志, 在 media_code 的 sibling）
EXPERIMENTS_DIR="${EXPERIMENTS_DIR:-$REPO_DIR/../world_r1_experiments}"

export REPO_DIR WORLD_R1_DIR MODEL_DIR WAN_MODEL_PATH RESULTS_DIR EXPERIMENTS_DIR

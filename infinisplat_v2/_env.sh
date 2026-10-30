# _env.sh — shared setup: proxy + conda env + GPU + paths.
# Sourced by 00/01/02/03/run_all. Expects SCRIPT_DIR (this dir) to be set by caller.
#
# 本机（WSL）路径策略遵循 AGENTS.md：
#   官方仓 ~/repos/<repo>（Linux fs，clone/编译快）
#   权重   ~/model/infinisplat_v2
#   输出   ~/output/infinisplat_v2_results（跑完用 08_move_output.sh 搬到 /mnt/d）
REPO_DIR="$(dirname "$SCRIPT_DIR")"

# Optional proxy (gitignored proxy.env at repo root).
if [ -f "$REPO_DIR/proxy.env" ]; then
    set -a; # shellcheck disable=SC1090
    source "$REPO_DIR/proxy.env"; set +a
fi
[ -n "${http_proxy:-}" ]  && export HTTP_PROXY="$http_proxy"
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
    : "${CURL_CA_BUNDLE:=$CA_FILE}"
    export REQUESTS_CA_BUNDLE SSL_CERT_FILE GIT_SSL_CAINFO PIP_CERT CURL_CA_BUNDLE
fi

export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"
# HF 官方站/hf-mirror 在本机不可达，权重走 aifasthub（见 01_download_models.sh）。
export HF_BASE_URL="${HF_BASE_URL:-https://aifasthub.com}"

CONDA_ENV="${CONDA_ENV:-infinisplat_v2}"
export CONDA_ENV
# Fallback: conda 不在 PATH 时自动找常见安装位置（WSL 未 conda init 时触发）
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
    exit 1
fi
# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV" 2>/dev/null || true  # env 可能还没建（00 负责）

if [ -n "${GPU:-}" ]; then
    export CUDA_VISIBLE_DEVICES="$GPU"
fi

# --- Paths（全部允许外部覆盖）---
# 官方仓（zju3dv/InfiniSplat，v2 在 InfiniSplat_v2/ 子目录）
INFINISPLAT_REPO="${INFINISPLAT_REPO:-$HOME/repos/InfiniSplat}"
INFINISPLAT_V2="${INFINISPLAT_V2:-$INFINISPLAT_REPO/InfiniSplat_v2}"
# DA3 官方仓（钉死 commit 41736238）
DA3_REPO="${DA3_REPO:-$HOME/repos/Depth-Anything-3}"
# 权重根：weights/da3/{config.json,model.safetensors}
MODEL_DIR="${MODEL_DIR:-$HOME/model/infinisplat_v2}"
WEIGHTS_DIR="${WEIGHTS_DIR:-$MODEL_DIR/da3}"
RESULTS_DIR="${RESULTS_DIR:-$HOME/output/infinisplat_v2_results}"

export INFINISPLAT_REPO INFINISPLAT_V2 DA3_REPO MODEL_DIR WEIGHTS_DIR RESULTS_DIR

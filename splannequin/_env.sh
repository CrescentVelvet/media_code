# _env.sh — shared setup: proxy + CA bundle + conda env activation + GPU + paths.
# Sourced by all splannequin/*.sh scripts. Expects SCRIPT_DIR (this dir) set by caller.
#
# splannequin-gs 官方仓基于 4DGaussians/HexPlane 代码库（WACV 2026）。
# 官方 pin：python=3.8 + torch==1.13.1(cu117) + gcc-10；
# WSL 实现：conda 提供 cuda-nvcc 11.7 + gcc 11（nvcc 11.7 支持 host gcc<=11），免 sudo。
REPO_DIR="$(dirname "$SCRIPT_DIR")"

# 1. 代理（从 proxy.env 读；家里直连时 proxy.env 里没有 proxy 变量则不生效）
if [ -f "$REPO_DIR/proxy.env" ]; then
    set -a; # shellcheck disable=SC1090
    source "$REPO_DIR/proxy.env"; set +a
fi
[ -n "${http_proxy:-}" ]  && export HTTP_PROXY="$http_proxy"
[ -n "${https_proxy:-}" ] && export HTTPS_PROXY="$https_proxy"

# 2. CA bundle（公司代理 TLS 拦截；家里无此文件则走系统默认）
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

# 3. conda env 激活
CONDA_ENV="${CONDA_ENV:-splannequin}"
export CONDA_ENV
# Fallback: conda 不在 PATH 时自动找常见安装位置（WSL 本机未 conda init 时触发）
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
conda activate "$CONDA_ENV" 2>/dev/null || true  # env 可能还没建（00a 负责创建）

# 4. GPU 选卡
if [ -n "${GPU:-}" ]; then
    export CUDA_VISIBLE_DEVICES="$GPU"
fi

# 5. 路径（${VAR:-default} 允许外部覆盖）
# 官方仓：WSL 放 Linux fs（编译快）；GitHub 直连不通时由 00a 从 tarball 落盘
SPLANNEQUIN_DIR="${SPLANNEQUIN_DIR:-$HOME/repos/splannequin-gs}"
SPLANNEQUIN_REPO="${SPLANNEQUIN_REPO:-https://github.com/chien90190/splannequin-gs.git}"

# 数据/实验输出：官方脚本默认写仓内 ./data ./exp，这里统一外置
DATA_DIR="${DATA_DIR:-${SPLANNEQUIN_DATA_DIR:-$HOME/output/splannequin_results/data}}"
EXP_DIR="${EXP_DIR:-${SPLANNEQUIN_EXP_DIR:-$HOME/output/splannequin_results/exp}}"

# 源码 tarball 缓存（GitHub 不通时 Windows 侧迅雷/curl 下载后经 /mnt/d 传入）
SRC_CACHE="${SRC_CACHE:-/mnt/d/wheel/splannequin_src}"

# EvaluationNet（aesthetic 评测用，手动下载放这里；Google Drive id=1IXBtvwn8fMCmDRczQExIR3ZSDhinoqbi）
MODEL_DIR="${MODEL_DIR:-${SPLANNEQUIN_MODEL_DIR:-$HOME/model/splannequin}}"

export SPLANNEQUIN_DIR SPLANNEQUIN_REPO DATA_DIR EXP_DIR SRC_CACHE MODEL_DIR

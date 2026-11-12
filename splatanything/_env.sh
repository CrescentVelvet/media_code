# _env.sh — shared setup: proxy + conda env + GPU + paths.
# Sourced by 00/01/02/… Expects SCRIPT_DIR (this dir) to be set by caller.
#
# 本机（WSL）路径策略遵循 AGENTS.md：
#   官方仓 ~/repos/SplatAnything（Linux fs，clone 快）
#   权重   /mnt/d/wheel/splatanything_ms（D 盘权重池，不复制到 Linux fs）
#   输出   ~/output/splatanything_results（跑完用 08_move_output.sh 搬到 /mnt/d）
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
# HF 官方站本机直连不通（连接被 reset），权重走镜像；aifasthub 与 hf-mirror 均可达。
export HF_BASE_URL="${HF_BASE_URL:-https://aifasthub.com}"
# 镜像站在国内，绕开 v2rayN 代理直连更快（proxy.env 会设 http_proxy）。
# ⚠️ 关键是 CDN 域名：镜像站对 resolve 请求回 302，真实数据在 us.aws.cdn.hf.co /
#    cas-bridge.xethub.hf.co。这两个域名不排除的话会被塞进 v2rayN，实测 57 KB/s
#    vs 直连 1.7 MB/s（差 30×）。只写 aifasthub.com 是不够的。
export no_proxy="${no_proxy:-},.hf.co,.xethub.hf.co,aifasthub.com,hf-mirror.com,modelscope.cn,pypi.tuna.tsinghua.edu.cn"
export NO_PROXY="$no_proxy"

CONDA_ENV="${CONDA_ENV:-splatanything}"
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
# 官方仓 zju3dv/SplatAnything（稀疏检出，不含 examples/assets 大图）
SPLATANYTHING_DIR="${SPLATANYTHING_DIR:-$HOME/repos/SplatAnything}"
# 权重：worldmirror2.safetensors + SplatAnything.ckpt + skyseg.onnx
SPLATANYTHING_MODEL_DIR="${SPLATANYTHING_MODEL_DIR:-/mnt/d/wheel/splatanything_ms}"
SPLATANYTHING_RESULTS_DIR="${SPLATANYTHING_RESULTS_DIR:-$HOME/output/splatanything_results}"

export SPLATANYTHING_DIR SPLATANYTHING_MODEL_DIR SPLATANYTHING_RESULTS_DIR

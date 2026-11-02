#!/usr/bin/env bash
# idol 共享环境设置，被本目录所有 .sh source。
# 关键：所有模型/安装包落在 /mnt/d/wheel（= D:\wheel），复用免重下。

REPO_DIR="$(dirname "$SCRIPT_DIR")"

# ---------------- 路径（全部可用环境变量覆盖） ----------------
IDOL_REPO="${IDOL_REPO:-$HOME/repos/idol-sdk}"          # 官方代码（WSL Linux fs，编译快）
WHEEL_DIR="${WHEEL_DIR:-/mnt/d/wheel}"                   # 模型 + 安装包长期落地盘
IDOL_MODELS="${IDOL_MODELS:-$WHEEL_DIR/idol_ms}"         # 模型快照目录（01 导出的软链集合）
RESULTS_DIR="${RESULTS_DIR:-$HOME/output/idol_results}"  # 跑完用 08 搬到 /mnt/d/output/
export REPO_DIR IDOL_REPO WHEEL_DIR IDOL_MODELS RESULTS_DIR

# ---------------- HF / pip 缓存落地 D:\wheel ----------------
# HF_HOME 一设，models--* 全在 D 盘，重复复现不会重新下载
export HF_HOME="${HF_HOME:-$WHEEL_DIR/hf_home_idol}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-$WHEEL_DIR/pip_cache}"
# hf-xet 会接管大文件下载走 Xet 协议，在「WSL + 宿主代理」下极慢
# （实测 2 分钟 65M ≈ 0.54 MB/s，同一通道 curl 单连接能到 1.97 MB/s）。
# 置 1 强制回落到普通 HTTP CDN 下载。
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"

# HF_HUB_ENABLE_HF_TRANSFER 默认关：实测 hf_transfer 在「WSL + 宿主 HTTP 代理」下会卡死
# （2 分钟只落 129K，incomplete 文件长期 0 字节），换回 python requests 后 3.4s 即通。
# 若日后走直连或换机场代理，可 HF_HUB_ENABLE_HF_TRANSFER=1 打开试。
export HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-0}"
export HF_HUB_DOWNLOAD_TIMEOUT="${HF_HUB_DOWNLOAD_TIMEOUT:-60}"
# 离线开关：HF_HUB_OFFLINE=1 时完全不碰网络（缓存已在 D 盘时用）
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-0}"

# ---------------- HF token（读 media_code/proxy.env，不落盘） ----------------
if [ -z "${HF_TOKEN:-}" ] && [ -f "$REPO_DIR/proxy.env" ]; then
    HF_TOKEN="$(sed -n 's/^[[:space:]]*export[[:space:]]\+HF_TOKEN[[:space:]]*=[[:space:]]*//p' \
        "$REPO_DIR/proxy.env" | head -1 | tr -d "\r\"'")"
    export HF_TOKEN
fi

# ---------------- 代理：HF 直连不通(WSL 实测 curl 000)，走宿主 v2rayN 10808 ----------------
# 网关 IP 每次开机可能变，动态取；端口实测只有 10808 开。
# 国内源必须 no_proxy 排除，否则绕代理反而更慢。
_VPN_GW="$(ip route show default 2>/dev/null | head -1 | awk '{print $3}')"
if [ -z "${http_proxy:-}" ] && [ -n "$_VPN_GW" ] && timeout 2 bash -c "</dev/tcp/$_VPN_GW/10808" 2>/dev/null; then
    export http_proxy="http://$_VPN_GW:10808"
    export https_proxy="$http_proxy"
fi
export no_proxy="localhost,127.0.0.1,${_VPN_GW},pypi.tuna.tsinghua.edu.cn,mirrors.tuna.tsinghua.edu.cn,mirrors.aliyun.com,mirrors.cloud.tencent.com,repo.anaconda.com"
export NO_PROXY="$no_proxy"

# ---------------- pip 源（有代理时国内源仍更快） ----------------
export PIP_INDEX_URL="${PIP_INDEX_URL:-https://pypi.tuna.tsinghua.edu.cn/simple}"
export PIP_TRUSTED_HOST="${PIP_TRUSTED_HOST:-pypi.tuna.tsinghua.edu.cn}"

# ---------------- conda env ----------------
CONDA_ENV="${CONDA_ENV:-idol}"
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
    echo "       Install miniconda or run: source ~/miniconda3/etc/profile.d/conda.sh" >&2
    exit 1
fi
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV" 2>/dev/null || true

# ---------------- C 编译器（torch.compile / triton 必需） ----------------
# 本机 WSL 无系统 gcc（apt 要 sudo 密码），改用 conda-forge 的 gcc：
#   conda install -n idol -c conda-forge gcc_linux-64 gxx_linux-64 -y
# 它只提供 x86_64-conda-linux-gnu-gcc，没有 gcc 这个名字，而 triton 硬找 "gcc"，
# 不报错就报 "Failed to find C compiler"（InductorError）。用 CC/CXX 指明即可。
if [ -n "${CONDA_PREFIX:-}" ] && [ -x "$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-gcc" ]; then
    export CC="${CC:-$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-gcc}"
    export CXX="${CXX:-$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-g++}"
fi

# ---------------- GPU ----------------
if [ -n "${GPU:-}" ]; then
    export CUDA_VISIBLE_DEVICES="$GPU"
fi

# ---------------- 下载通道自检函数（各脚本共用） ----------------
idol_check_hf_token () {
    if [ -z "${HF_TOKEN:-}" ]; then
        echo "❌ ERROR: HF_TOKEN 未设置。两个图像编码器是 gated 仓库，必须带 token。" >&2
        echo "   在 $REPO_DIR/proxy.env 里写 export HF_TOKEN=hf_xxx" >&2
        return 1
    fi
    return 0
}

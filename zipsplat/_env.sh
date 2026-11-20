# zipsplat/_env.sh — 共享环境设置（被本目录所有 .sh 脚本 source）
SCRIPT_DIR_ENV="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MEDIA_CODE_ROOT="$(cd "$SCRIPT_DIR_ENV/.." && pwd)"
REPO_DIR="$MEDIA_CODE_ROOT"
export REPO_DIR MEDIA_CODE_ROOT

# 代理 / CA bundle；国内镜像必须由 proxy.env 的 no_proxy 排除。
if [ -f "$MEDIA_CODE_ROOT/proxy.env" ]; then
    # 不 source 可能包含 Windows 凭据/CRLF 的共享配置；仅读取代理 URL 和 no_proxy。
    # printf -v 在 local 同名的脚本函数中会污染调用者变量；安全解析后统一导出。
    declare -A _proxy_values=()
    while IFS='=' read -r _key _value; do
        _key="${_key//[[:space:]]/}"
        [[ "$_key" =~ ^[a-zA-Z_][a-zA-Z0-9_]*$ ]] || continue
        _value="${_value//$'\r'/}"
        _value="${_value#"${_value%%[![:space:]]*}"}"
        _value="${_value%"${_value##*[![:space:]]}"}"
        case "$_key" in
            http_proxy|https_proxy|HTTP_PROXY|HTTPS_PROXY)
                [[ "$_value" =~ ^(https?|socks5h?)://[^[:space:]]+$ ]] || continue
                _proxy_values["$_key"]="$_value"
                ;;
            no_proxy|NO_PROXY)
                [[ "$_value" =~ ^[a-zA-Z0-9.,:*_-]+$ ]] || continue
                _proxy_values["$_key"]="$_value"
                ;;
        esac
    done < "$MEDIA_CODE_ROOT/proxy.env"
    for _key in "${!_proxy_values[@]}"; do export "$_key=${_proxy_values[$_key]}"; done
    unset _key _value _proxy_values
fi
[ -n "${http_proxy:-}" ] && export HTTP_PROXY="$http_proxy"
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
    export REQUESTS_CA_BUNDLE SSL_CERT_FILE GIT_SSL_CAINFO PIP_CERT
fi
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"

CONDA_ENV="${CONDA_ENV:-zipsplat}"
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
    exit 1
fi
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV" 2>/dev/null || true

if [ -n "${GPU:-}" ]; then
    export CUDA_VISIBLE_DEVICES="$GPU"
fi

export ZIPSPLAT_REPO="${ZIPSPLAT_REPO:-$HOME/repos/ZipSplat}"
export MODEL_DIR="${ZIPSPLAT_MODEL_DIR:-$HOME/model/zipsplat}"
export WEIGHT_PATH="${WEIGHT_PATH:-$MODEL_DIR/zipsplat-da3g-504.tar}"
export RESULTS_DIR="${ZIPSPLAT_RESULTS_DIR:-$HOME/output/zipsplat_results}"
export WHEEL_DIR="${WHEEL_DIR:-/mnt/d/wheel}"
export WHEEL_ZIPSPLAT_DIR="${WHEEL_ZIPSPLAT_DIR:-$WHEEL_DIR/zipsplat}"
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.6}"

# gsplat JIT 需要完整 CUDA toolkit 头；clone env 后优先用同机已有的完整 toolkit。
if [ -n "${CONDA_PREFIX:-}" ]; then
    export CUDA_HOME="${CUDA_HOME:-$CONDA_PREFIX}"
    for _inc in "$CUDA_HOME/targets/x86_64-linux/include" \
        "$HOME/miniconda3/envs/splatanything/targets/x86_64-linux/include"; do
        if [ -f "$_inc/crt/host_defines.h" ]; then
            export CPATH="$_inc${CPATH:+:$CPATH}"
            export CPLUS_INCLUDE_PATH="$_inc${CPLUS_INCLUDE_PATH:+:$CPLUS_INCLUDE_PATH}"
            break
        fi
    done
    _lib="$CUDA_HOME/targets/x86_64-linux/lib"
    [ -d "$_lib" ] && export LIBRARY_PATH="$_lib:$_lib/stubs${LIBRARY_PATH:+:$LIBRARY_PATH}"
    [ -d "$_lib" ] && export LD_LIBRARY_PATH="$_lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
    unset _inc _lib
fi

# structsplat/_env.sh — 共享环境设置（被本目录所有 .sh 脚本 source）

SCRIPT_DIR_ENV="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MEDIA_CODE_ROOT="$(cd "$SCRIPT_DIR_ENV/.." && pwd)"

# 1. 代理（从仓根 proxy.env 读）
if [ -f "$MEDIA_CODE_ROOT/proxy.env" ]; then
    set -a; source "$MEDIA_CODE_ROOT/proxy.env"; set +a
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

# 3. conda env 激活
CONDA_ENV="${CONDA_ENV:-structsplat}"
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
conda activate "$CONDA_ENV" 2>/dev/null || true  # env 可能还没建（00a 负责创建）

# 4. CUDA / host compiler —— gsplat 与 deepspeed 编译的前置条件
#
# ⚠️ 本机 WSL 实测（2026-11-19）：**系统里没有 /usr/bin/gcc**，只有 conda-forge 的
#    gcc_linux-64 12.4.0，其可执行文件名为 x86_64-conda-linux-gnu-gcc，**没有建 gcc
#    这个别名**。于是 nvcc 找 host compiler 时报：
#        gcc: No such file or directory
#        nvcc fatal : Failed to preprocess host compiler properties.
#    30 个 .cu 全部编不出来 → gsplat._C = None。这是 gsplat 编译失败的真正根因，
#    不是「CUDA 头文件路径不对」（补 CPATH 只解决了 torch 侧 include 的问题）。
#
# 两个修法叠加：
#   (a) 在 $CONDA_PREFIX/bin 下补 gcc/g++/cc/c++ 四个软链 → nvcc 自动找到；
#   (b) 显式 CC/CXX 指向 conda 的编译器，torch cpp_extension 用它查 ABI。
#
# 另：conda 的 CUDA 头文件在 targets/x86_64-linux/include，而 torch 的 cpp_extension
#     只加 $CUDA_HOME/include（那里只有 GL/X11 等系统头）→ 必须补 CPATH。
if [ -x "${CONDA_PREFIX:-}/bin/nvcc" ]; then
    export CUDA_HOME="${CUDA_HOME:-$CONDA_PREFIX}"
    export PATH="$CUDA_HOME/bin:$PATH"
    export CPATH="$CONDA_PREFIX/targets/x86_64-linux/include${CPATH:+:$CPATH}"
    export CPLUS_INCLUDE_PATH="$CONDA_PREFIX/targets/x86_64-linux/include${CPLUS_INCLUDE_PATH:+:$CPLUS_INCLUDE_PATH}"
    export LIBRARY_PATH="$CONDA_PREFIX/targets/x86_64-linux/lib${LIBRARY_PATH:+:$LIBRARY_PATH}"

    # (a) 补 gcc 别名（幂等；只在缺失时建，绝不覆盖真实编译器）
    _CC="$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-gcc"
    _CXX="$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-g++"
    if [ -x "$_CC" ] && [ -x "$_CXX" ]; then
        for _pair in "gcc:$_CC" "cc:$_CC" "g++:$_CXX" "c++:$_CXX"; do
            _n="${_pair%%:*}"; _t="${_pair#*:}"
            [ -e "$CONDA_PREFIX/bin/$_n" ] || ln -sfn "$_t" "$CONDA_PREFIX/bin/$_n"
        done
        unset _pair _n _t
    fi
    unset _CC _CXX

    # (b) 显式声明编译器（只在本机确实没有系统 gcc 时才这么做）
    if ! command -v gcc >/dev/null 2>&1; then
        if [ -x "$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-gcc" ]; then
            export CC="${CC:-$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-gcc}"
            export CXX="${CXX:-$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-g++}"
            export CUDACXX="${CUDACXX:-$CONDA_PREFIX/bin/nvcc}"
            export NVCC_PREPEND_FLAGS="${NVCC_PREPEND_FLAGS:--ccbin $CC}"
        fi
    fi

    # RTX 3090 = sm_86；不设置会用默认全架构列表，编译时间暴涨
    export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.6}"
    # deepspeed 顶层 import 时若无 CUDA_HOME 会抛 MissingCUDAException
    export DS_BUILD_OPS="${DS_BUILD_OPS:-0}"
fi

# 5. GPU 选卡
if [ -n "${GPU:-}" ]; then
    export CUDA_VISIBLE_DEVICES="$GPU"
fi

# 6. 路径（都用 ${VAR:-default} 允许外部覆盖）
export STRUCTSPLAT_REPO="${STRUCTSPLAT_REPO:-$HOME/repos/StructSplat}"
# 官方 yaml 里 ckpts/vggt、ckpts/dinov3_convnext_large 是相对路径 → 必须就地放置/软链
export SS_CKPT_DIR="$STRUCTSPLAT_REPO/ckpts"
export SS_DATA_DIR="${SS_DATA_DIR:-$STRUCTSPLAT_REPO/data}"
export RESULTS_DIR="${STRUCTSPLAT_RESULTS_DIR:-$HOME/output/structsplat_results}"
# 权重复用池（2026-11-05 峰哥约定）：下载模型前先在 D:\wheel 搜，
# 新下载的模型落盘到 D:\wheel 下对应子目录，消费侧一律软链，不复制
export WHEEL_DIR="${WHEEL_DIR:-/mnt/d/wheel}"
export WHEEL_STRUCTSPLAT_DIR="${WHEEL_STRUCTSPLAT_DIR:-$WHEEL_DIR/structsplat_ms}"
export WHEEL_VGGT1B_DIR="${WHEEL_VGGT1B_DIR:-$WHEEL_DIR/structsplat_ms/VGGT-1B}"
export WHEEL_DINOV3CNX_DIR="${WHEEL_DINOV3CNX_DIR:-$WHEEL_DIR/structsplat_ms/dinov3_convnext_large}"

#!/usr/bin/env bash
# _gsplat_env.sh — gsplat 渲染前置（03 与 06 共用）。需先 source _env.sh（用 $CONDA_ENV/$CONDA_PREFIX）。
#
# gsplat 首次运行会 JIT 编译 CUDA kernel，需要四样东西：
#   1. nvcc        —— conda install -n $CONDA_ENV -c nvidia cuda-nvcc=12.6
#   2. gcc/g++     —— 本机 WSL 没装，默认借用 infinisplat_v2 env 的 conda gcc（GCC_DIR 可改）
#   3. glm 头文件  —— gsplat 1.5.2 不内置（1.5.3 才带），git clone 到 $GLM_DIR
#   4. TORCH_CUDA_ARCH_LIST —— 不设会编所有架构，编译时间 ×10，看起来像卡死

: "${GLM_DIR:=$HOME/repos/glm_src}"
: "${GCC_DIR:=$HOME/miniconda3/envs/infinisplat_v2/bin}"

if ! python -c "import gsplat" 2>/dev/null; then
    echo "📦 安装 gsplat 1.5.2（1.5.3 与 torch 2.7 的 JIT 接口不兼容）"
    env -u http_proxy -u https_proxy -u HTTP_PROXY -u HTTPS_PROXY \
        python -m pip install "gsplat==1.5.2" \
        --index-url "${PIP_INDEX:-https://pypi.tuna.tsinghua.edu.cn/simple}" 2>&1 | tail -3 || {
        echo "❌ gsplat 安装失败" >&2; exit 1; }
fi

if ! command -v nvcc >/dev/null 2>&1; then
    if [ -x "$CONDA_PREFIX/bin/nvcc" ]; then
        export CUDA_HOME="$CONDA_PREFIX"
        export PATH="$CUDA_HOME/bin:$PATH"
    else
        echo "⚠️  没有 nvcc，gsplat 会 JIT 失败：conda install -n $CONDA_ENV -c nvidia cuda-nvcc=12.6" >&2
    fi
fi
export CUDA_HOME="${CUDA_HOME:-$CONDA_PREFIX}"
if [ -d "$GCC_DIR" ] && ! command -v gcc >/dev/null 2>&1; then
    export PATH="$GCC_DIR:$PATH"
fi
if [ -d "$GLM_DIR/glm" ]; then
    export CPATH="$GLM_DIR:$CPATH"
else
    echo "⚠️  缺 glm 头文件（gsplat 1.5.2 编译要 <glm/gtc/type_ptr.hpp>）" >&2
    echo "    git clone --depth 1 https://github.com/g-truc/glm.git $GLM_DIR" >&2
fi
# conda 的 cuda 头在 targets/<arch>/include，torch 只找 $CUDA_HOME/include
for d in "$CUDA_HOME/targets/x86_64-linux/include" "$CUDA_HOME/include"; do
    [ -d "$d" ] && export CPATH="$d:$CPATH"
done
if [ -z "${TORCH_CUDA_ARCH_LIST:-}" ]; then
    TORCH_CUDA_ARCH_LIST="$(python - <<'PY'
import torch
if torch.cuda.is_available():
    maj, mi = torch.cuda.get_device_capability(0)
    print(f'{maj}.{mi}')
PY
)"
    [ -n "$TORCH_CUDA_ARCH_LIST" ] && export TORCH_CUDA_ARCH_LIST
fi
echo "  🔧 CUDA_HOME=$CUDA_HOME  arch=${TORCH_CUDA_ARCH_LIST:-auto}  glm=${GLM_DIR}"

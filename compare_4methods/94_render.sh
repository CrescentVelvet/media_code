#!/usr/bin/env bash
# 94 渲染包装：gsplat 是 JIT 编译的，需要三样东西同时到位，缺一样报错都极不直观：
#   1. conda activate        → 提供 CC/CXX(conda-gcc)、ninja、python
#   2. CUDA_HOME + CPATH     → 缺了报 torch/.../CUDAStream.h: cuda_runtime_api.h: No such file
#   3. LIBRARY_PATH          → 缺了链接期找不到 libcudart
#
# ⚠️ 不要依赖 $CONDA_PREFIX：经 wsl.exe 传参时变量会被提前吃掉（静默变空）。
#    所以这里用绝对路径，可用 QUERYSPLAT_ENV 覆盖。
# 不用 set -u：conda 的 activate.d 脚本会引用未定义变量（SYS_SYSROOT），加上就炸。
set -eo pipefail
ENVS="${QUERYSPLAT_ENV:-$HOME/miniconda3/envs/querysplat}"
CONDA_SH="$HOME/miniconda3/etc/profile.d/conda.sh"

if [ -f "$CONDA_SH" ]; then
  # shellcheck disable=SC1090
  source "$CONDA_SH"
  conda activate "$ENVS"
else
  export PATH="$ENVS/bin:$PATH"
  export CC="$ENVS/bin/x86_64-conda-linux-gnu-cc"
  export CXX="$ENVS/bin/x86_64-conda-linux-gnu-c++"
fi

export CUDA_HOME="$ENVS"
export CPATH="$ENVS/targets/x86_64-linux/include"
export LIBRARY_PATH="$ENVS/targets/x86_64-linux/lib"

exec python "$(dirname "$0")/94_render_da3_gs.py" "$@"

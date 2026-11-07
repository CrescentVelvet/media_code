#!/usr/bin/env bash
# 判定 gsplat 是否内部 sigmoid（沿用 94_render.sh 的 JIT 环境）
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
exec python "$(dirname "$0")/_probe_alpha.py"

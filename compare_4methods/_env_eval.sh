#!/bin/bash
# 评测环境包装：94_render.sh 同款。
# 解决 querysplat env 重编译 gsplat CUDA 扩展时找不到 CUDA 头/库的问题：
#   - 头文件在 nvidia/cuda_runtime/include
#   - 库 + stubs 在 targets/x86_64-linux/lib
# 设好 CUDA_HOME / CPATH / LIBRARY_PATH / LD_LIBRARY_PATH 后，JIT 能一次编成功，
# 之后 .so 时间戳刷新，torch 不再触发重编译。
set -e
source ~/miniconda3/etc/profile.d/conda.sh
conda activate querysplat

Q=/home/velvet/miniconda3/envs/querysplat
T=$Q/targets/x86_64-linux
# 完整 CUDA toolkit 头（含 crt/）：借用 splatanything env；自己的 runtime 只有 nvidia/cuda_runtime
CUDA_TOOLKIT=/home/velvet/miniconda3/envs/splatanything/targets/x86_64-linux

# 头文件：优先完整 toolkit（含 crt/），兜底 nvidia/cuda_runtime
CUDA_INC=""
for c in "$CUDA_TOOLKIT/include" "$T/include" "$Q/lib/python3.11/site-packages/nvidia/cuda_runtime/include"; do
  [ -f "$c/crt/host_defines.h" ] && CUDA_INC="$c" && break
done
[ -z "$CUDA_INC" ] && echo "❌ 找不到含 crt/ 的 CUDA 头" && exit 1

export CUDA_HOME="${CUDA_HOME:-$Q}"
export CPATH="$CUDA_INC:$CPATH"
export LIBRARY_PATH="$T/lib:$T/lib/stubs:$LIBRARY_PATH"
export LD_LIBRARY_PATH="$T/lib:$LD_LIBRARY_PATH"

echo "CUDA_INC=$CUDA_INC"
echo "libcuda stub: $(ls $T/lib/stubs/libcuda.so 2>/dev/null)"

cd /mnt/c/code/media_code
exec "$@"

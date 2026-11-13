# 用法: source /data/00950754/envs/activate_splatanything.sh
#
# H100 集群（迪拜 ca-uae）个人环境激活脚本。
# 由 00_setup_env.sh 生成 / 可手动编辑。

source /data/conda/etc/profile.d/conda.sh
conda activate /data/00950754/envs/splatanything

export PIP_CACHE_DIR="/data/00950754/.pip_cache"

# ---------- gsplat JIT 编译环境 ----------
export CUDA_HOME="/data/00950754/envs/splatanything"
export PATH="$CUDA_HOME/bin:$PATH"

# conda 只装了 x86_64-conda-linux-gnu-gcc，没有 gcc/g++ 这两个名字，
# 而 nvcc 默认找 `gcc`/`g++` → 必须补 dev 别名，否则 JIT 报
# "unable to find a compatible host compiler"。
_GCC_DIR="$CUDA_HOME/bin"
for _pair in "gcc:gcc" "g++:g++" "cc:gcc" "c++:g++" "ld:ld"; do
    _name="${_pair%%:*}"; _src="${_pair##*:}"
    if [ ! -e "$_GCC_DIR/$_name" ] && [ -e "$_GCC_DIR/x86_64-conda-linux-gnu-$_src" ]; then
        ln -sf "x86_64-conda-linux-gnu-$_src" "$_GCC_DIR/$_name" 2>/dev/null
    fi
done
unset _pair _name _src _GCC_DIR

export CC="$CUDA_HOME/bin/x86_64-conda-linux-gnu-gcc"
export CXX="$CUDA_HOME/bin/x86_64-conda-linux-gnu-g++"
export GCC="$CC"
export GXX="$CXX"

# gsplat 1.5.2 不自带 glm 头文件；cuda 头在 targets/<arch>/include（不在 $CUDA_HOME/include）
export CPATH="/data/00950754/repos/glm_src:$CUDA_HOME/targets/x86_64-linux/include:$CPATH"
export LIBRARY_PATH="$CUDA_HOME/targets/x86_64-linux/lib:$CUDA_HOME/lib64:$LIBRARY_PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/targets/x86_64-linux/lib:$CUDA_HOME/lib64:$LD_LIBRARY_PATH"

# H100 = sm_90。不钉死的话 torch 会尝试编所有架构，编译时间 ×10 看起来像卡死。
# 只有在没有 GPU 的登录节点才需要手动指定；有 GPU 时会自动探测。
if [ -z "${TORCH_CUDA_ARCH_LIST:-}" ] && ! command -v nvidia-smi >/dev/null 2>&1; then
    export TORCH_CUDA_ARCH_LIST="9.0"
fi

# gsplat JIT 用 ninja 并行编译，缓存到 /data（别塞满 51G 的家目录）
export TORCH_EXTENSIONS_DIR="/data/00950754/.torch_extensions"
mkdir -p "$TORCH_EXTENSIONS_DIR"

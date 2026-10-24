#!/usr/bin/env bash
# 00b_build_ext.sh — 只重编两个 CUDA 扩展（glm 补齐后/换卡后/编译失败后用）。
# 环境已在 00a 建好时跑这个，不必重跑整个 00a。
set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/_env.sh"

DGR="$SPLANNEQUIN_DIR/submodules/depth-diff-gaussian-rasterization"
SK="$SPLANNEQUIN_DIR/submodules/simple-knn"

if [ ! -f "$DGR/third_party/glm/glm/glm.hpp" ]; then
    echo "❌ glm missing: $DGR/third_party/glm" >&2
    echo "   Run 00a_setup_env.sh first (it fetches glm tarball from \$SRC_CACHE)." >&2
    exit 1
fi

export CUDA_HOME="${CUDA_HOME:-$CONDA_PREFIX}"
export PATH="$CUDA_HOME/bin:$PATH"
_gcc="$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-gcc"
_gpp="$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-g++"
[ -x "$_gcc" ] && [ -x "$_gpp" ] && { export CC="$_gcc"; export CXX="$_gpp"; }
# RTX 3090 = sm_86；只编本机架构
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.6}"

echo "🔨 building CUDA ext: depth-diff-gaussian-rasterization"
echo "  nvcc: $(nvcc --version | tail -1 | xargs)"
echo "  CC:   $CC"
rm -rf "$DGR/build" "$DGR/dist" "$DGR"/*.egg-info
pip install --no-build-isolation --no-deps "$DGR" || {
    echo "❌ depth-diff-gaussian-rasterization build failed" >&2; exit 1; }

echo "🔨 building CUDA ext: simple-knn"
rm -rf "$SK/build" "$SK/dist" "$SK"/*.egg-info
pip install --no-build-isolation --no-deps "$SK" || {
    echo "❌ simple-knn build failed" >&2; exit 1; }

python -c "import diff_gaussian_rasterization, simple_knn; print('🎉 CUDA exts importable')" || {
    echo "❌ import check failed" >&2; exit 1; }

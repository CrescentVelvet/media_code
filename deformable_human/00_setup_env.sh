#!/usr/bin/env bash
# 00_setup_env.sh —【服务器】deformable_human 环境搭建。
#
# 与 00a（WSL）的差异：官方仓 clone 到 media_code 的 sibling（$REPO_DIR/../）、
# GitHub 直连（GIT_URL_PREFIX 默认为空）、pip 走默认源 + trusted-host。
# env 同样默认从 donor（CLONE_FROM，默认 vggt_human）clone，保证 torch/CUDA 工具链一致。
#
# 用法：
#   CLONE_FROM=vggt_human INSTALL_DEPS=1 BUILD_CUDA=1 bash deformable_human/00_setup_env.sh
set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/_env.sh"

CLONE_FROM="${CLONE_FROM:-vggt_human}"
DG_DIR="${DG_DIR:-$REPO_DIR/../Deformable-3D-Gaussians}"  # 服务器：sibling of media_code
GIT_URL_PREFIX="${GIT_URL_PREFIX:-}"                       # 服务器一般直连 GitHub
DG_REPO_URL="${GIT_URL_PREFIX}https://github.com/ingra14m/Deformable-3D-Gaussians.git"

echo "🚀 [00] deformable_human 环境搭建（服务器）"
echo "  🐍 env:        $CONDA_ENV  (clone from: $CLONE_FROM)"
echo "  📦 官方仓:     $DG_DIR"
echo ""

# ── 1. conda env（clone donor）────────────────────────────────────────────
if [ -d "$(conda info --base)/envs/$CONDA_ENV" ]; then
    echo "⏭️  [1] env '$CONDA_ENV' 已存在，跳过 clone"
else
    if [ ! -d "$(conda info --base)/envs/$CLONE_FROM" ]; then
        echo "❌ ERROR: donor env '$CLONE_FROM' 不存在。CLONE_FROM=<已有torch2.5的env> 重跑，" >&2
        echo "       或参考 deformable_gaussians/_env.sh 头部注释新建 py3.7/torch1.13 独立 env。" >&2
        exit 1
    fi
    echo "📦 [1] conda create -n $CONDA_ENV --clone $CLONE_FROM"
    conda create -y -n "$CONDA_ENV" --clone "$CLONE_FROM" || { echo "❌ [1] FAILED" >&2; exit 1; }
fi
conda activate "$CONDA_ENV" || { echo "❌ conda activate $CONDA_ENV 失败" >&2; exit 1; }

# ── 2. clone 官方仓 ─────────────────────────────────────────────────────────
if [ -f "$DG_DIR/train.py" ]; then
    echo "⏭️  [2] 官方仓已存在: $DG_DIR"
else
    echo "📦 [2] clone $DG_REPO_URL → $DG_DIR"
    LD_LIBRARY_PATH= git clone --recursive "$DG_REPO_URL" "$DG_DIR" || \
        LD_LIBRARY_PATH= git -c http.sslVerify=false clone --recursive "$DG_REPO_URL" "$DG_DIR"
    [ -f "$DG_DIR/train.py" ] || { echo "❌ [2] FAILED" >&2; exit 1; }
fi
if [ -n "$GIT_URL_PREFIX" ]; then
    ( cd "$DG_DIR" && git config --local \
        "url.${GIT_URL_PREFIX}https://github.com/.insteadOf" "https://github.com/" )
fi
if [ ! -f "$DG_DIR/submodules/simple-knn/setup.py" ] || \
   [ ! -f "$DG_DIR/submodules/depth-diff-gaussian-rasterization/setup.py" ]; then
    ( cd "$DG_DIR" && LD_LIBRARY_PATH= git submodule update --init --recursive ) || \
    ( cd "$DG_DIR" && LD_LIBRARY_PATH= git -c http.sslVerify=false submodule update --init --recursive )
fi

# ── 3. Python deps ──────────────────────────────────────────────────────────
if [ "${INSTALL_DEPS:-0}" = "1" ]; then
    REQ="$DG_DIR/requirements.txt"
    [ -f "$REQ" ] || { echo "❌ ERROR: $REQ 不存在" >&2; exit 1; }
    TMP_REQ="$(mktemp).txt"
    grep -v -iE '^[[:space:]]*submodules/' "$REQ" \
        | grep -v -iE '^[[:space:]]*(torch|torchvision)([=<>!~]|$|[[:space:]])' > "$TMP_REQ"
    echo "📦 [3] pip install requirements（submodule/torch pin 已过滤）"
    PIP_FLAGS=(--trusted-host pypi.org --trusted-host pypi.python.org \
        --trusted-host files.pythonhosted.org --timeout 600 --retries 10)
    pip install "${PIP_FLAGS[@]}" -r "$TMP_REQ" || { echo "❌ [3] FAILED" >&2; rm -f "$TMP_REQ"; exit 1; }
    rm -f "$TMP_REQ"
fi

# ── 4. CUDA 子模块编译 ──────────────────────────────────────────────────────
if [ "${BUILD_CUDA:-1}" = "1" ]; then
    : "${CUDA_HOME:=$CONDA_PREFIX}"   # nvcc 随 clone 继承；服务器系统 CUDA 则改指 /usr/local/cuda
    export CUDA_HOME
    export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.6}"
    if [ ! -x "$CUDA_HOME/bin/nvcc" ]; then
        echo "❌ ERROR: nvcc 不在 $CUDA_HOME/bin/（设 CUDA_HOME=/usr/local/cuda 或 donor env 需含 cuda-nvcc）" >&2
        exit 1
    fi
    echo "🔧 [4] nvcc: $("$CUDA_HOME/bin/nvcc" --version | tail -1 | xargs)  arch: $TORCH_CUDA_ARCH_LIST"
    PIP_FLAGS=(--trusted-host pypi.org --trusted-host pypi.python.org \
        --trusted-host files.pythonhosted.org --timeout 600 --retries 10)
    echo "🔧 [4a] build simple-knn"
    pip install "${PIP_FLAGS[@]}" --no-build-isolation "$DG_DIR/submodules/simple-knn" || { echo "❌ [4a] FAILED" >&2; exit 1; }
    echo "🔧 [4b] build depth-diff-gaussian-rasterization（fork）"
    pip install "${PIP_FLAGS[@]}" --no-build-isolation "$DG_DIR/submodules/depth-diff-gaussian-rasterization" || {
        echo "❌ [4b] FAILED（fork 老代码 vs torch 2.5.1，见 README「可能遇到的问题」）" >&2; exit 1; }
fi

# ── 5. 验证 ────────────────────────────────────────────────────────────────
echo ""
echo "🔍 [5] 验证"
python - <<'PY'
import sys
import torch
print(f"  torch: {torch.__version__}  cuda: {torch.version.cuda}  available: {torch.cuda.is_available()}")
if not torch.cuda.is_available():
    sys.exit("❌ torch.cuda 不可用")
for mod in ("diff_gaussian_rasterization", "simple_knn"):
    try:
        m = __import__(mod)
        print(f"  ✅ {mod}: {m.__file__}")
    except ImportError as e:
        print(f"  ⚠️ {mod} 不可 import: {e}（BUILD_CUDA=1 重跑）")
PY

echo ""
echo "🎉 [00] Done. env '$CONDA_ENV' 就绪。"

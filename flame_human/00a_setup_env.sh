#!/usr/bin/env bash
# 00a_setup_env.sh — WSL 本机环境搭建（conda env + 官方仓 clone + 依赖）。
#
# 服务器版见 00_setup_env.sh；本文件是 WSL 变体（conda 可能未 init、走 ~/repos）。
# 只装依赖与 clone 官方仓；权重下载在 01_download_models.sh。
set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/_env.sh"

PY_VER="${PY_VER:-3.10}"
TORCH_VER="${TORCH_VER:-2.1.2}"
TORCHV_VER="${TORCHV_VER:-0.16.2}"
CUDA_TAG="${CUDA_TAG:-cu118}"

echo "🚀 [flame_human 00a] WSL 环境搭建"
echo "  🐍 conda env : $CONDA_ENV (python $PY_VER)"
echo "  🎮 CUDA tag  : $CUDA_TAG  torch $TORCH_VER / torchvision $TORCHV_VER"
echo "  📦 DECA      : $DECA_DIR"
echo "  📦 3DGS      : $GS_DIR"

# ── 1. conda env ────────────────────────────────────────────────────────────
if ! conda env list | grep -qE "^${CONDA_ENV}\s"; then
    echo "📦 creating conda env '$CONDA_ENV' ..."
    conda create -y -n "$CONDA_ENV" "python=$PY_VER" || { echo "❌ conda create failed"; exit 1; }
fi
# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV" || { echo "❌ conda activate failed"; exit 1; }

# ── 2. torch ────────────────────────────────────────────────────────────────
python -c "import torch, sys; sys.exit(0 if torch.__version__.startswith('$TORCH_VER') else 1)" 2>/dev/null \
    && echo "⏭️  torch $TORCH_VER 已装" \
    || {
        echo "📦 installing torch $TORCH_VER+$CUDA_TAG ..."
        pip install --index-url "https://download.pytorch.org/whl/$CUDA_TAG" \
            "torch==$TORCH_VER" "torchvision==$TORCHV_VER" || { echo "❌ torch install failed"; exit 1; }
    }

# ── 3. 基础依赖 ─────────────────────────────────────────────────────────────
echo "📦 installing base deps ..."
pip install -q numpy scipy opencv-python pillow pyyaml tqdm \
    scikit-image scikit-learn matplotlib || echo "  ⚠️ some base deps failed" >&2

# smplx：加载 FLAME2020.pkl（FLAME 走 SMPL-X 同一套 loader）
pip install -q smplx || echo "  ⚠️ smplx install failed" >&2
# chumpy：DECA 的 .pkl 反序列化需要
pip install -q "chumpy==0.70" --no-build-isolation || echo "  ⚠️ chumpy install failed" >&2

# MediaPipe：468 点 2D landmark（阶段三的主观测，DECA 只给 68 点不够用）
pip install -q mediapipe || echo "  ⚠️ mediapipe install failed" >&2

# plyfile / trimesh：读写高斯 PLY 与 mesh
pip install -q plyfile trimesh || echo "  ⚠️ plyfile/trimesh failed" >&2

# ── 4. PyTorch3D（point-to-triangle 距离 + KNN；阶段七切分要用）──────────────
python -c "import pytorch3d" 2>/dev/null \
    && echo "⏭️  pytorch3d 已装" \
    || {
        echo "📦 installing pytorch3d (可能较久) ..."
        # 先用 fbaipublicfiles 的预编译 wheel，失败再退回源码编译
        pip install --no-index --no-cache-dir pytorch3d \
            -f "https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/py310_cu118_pyt1120/download.html" \
            || pip install "git+https://github.com/facebookresearch/pytorch3d.git" \
            || echo "  ⚠️ pytorch3d install failed — 阶段七需退回纯 torch 实现" >&2
    }

# ── 5. 官方仓 ───────────────────────────────────────────────────────────────
clone_repo() {
    local dir="$1" repo="$2" name="$3"
    if [ -d "$dir/.git" ]; then
        echo "⏭️  $name 已存在: $dir"
    else
        echo "📦 cloning $name -> $dir"
        mkdir -p "$(dirname "$dir")"
        git clone "$repo" "$dir" \
            || git -c http.sslVerify=false clone "$repo" "$dir" \
            || { echo "  ⚠️ clone $name failed" >&2; return 1; }
    fi
}
clone_repo "$DECA_DIR" "$DECA_REPO" "DECA"
clone_repo "$GS_DIR" "$GS_REPO" "gaussian-splatting"

# ── 6. 3DGS 子模块（diff-gaussian-rasterization / simple-knn）───────────────
if [ -d "$GS_DIR" ]; then
    echo "📦 installing 3DGS submodules ..."
    ( cd "$GS_DIR" && pip install -q submodules/diff-gaussian-rasterization ) \
        || echo "  ⚠️ diff-gaussian-rasterization failed" >&2
    ( cd "$GS_DIR" && pip install -q submodules/simple-knn ) \
        || echo "  ⚠️ simple-knn failed" >&2
fi

# ── 7. 验证 ─────────────────────────────────────────────────────────────────
echo ""
echo "🔍 验证："
python - <<'PY' 2>&1 | sed 's/^/  /'
import importlib
for m, need in [("torch", True), ("smplx", True), ("chumpy", True),
                ("mediapipe", True), ("scipy", True), ("cv2", True),
                ("plyfile", True), ("trimesh", False), ("pytorch3d", False)]:
    try:
        mod = importlib.import_module(m)
        v = getattr(mod, "__version__", "")
        print(f"  ✅ {m} {v}")
    except Exception as e:
        print(f"  {'❌' if need else '⚠️ '} {m}: {type(e).__name__}")
PY

echo ""
echo "📁 目录："
[ -d "$DECA_DIR/.git" ] && echo "  ✅ DECA code" || echo "  [---] DECA code"
[ -d "$GS_DIR/.git" ]   && echo "  ✅ 3DGS code" || echo "  [---] 3DGS code"
echo ""
echo "🎉 env 就绪。下一步：bash 01_download_models.sh"

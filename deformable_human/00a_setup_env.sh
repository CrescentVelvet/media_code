#!/usr/bin/env bash
# 00a_setup_env.sh —【WSL 本机】deformable_human 环境从零搭建。
#
# 做四件事：
#   1) conda env：从 vggt_human clone（继承 torch 2.5.1+cu121 + conda cuda-nvcc 12.1
#      工具链 + 已编译的包缓存）。CLONE_FROM 可换其他 donor env。
#   2) clone 官方仓 ingra14m/Deformable-3D-Gaussians → ~/repos/（Linux fs，编译快；
#      GitHub 直连不通，走 ghfast.top 前缀，submodule 用本地 url.insteadOf 改写）。
#   3) INSTALL_DEPS=1：装官方 requirements.txt（过滤 submodule 本地路径行与 torch pin）。
#   4) BUILD_CUDA=1（默认开）：编译两个 CUDA 子模块 simple-knn +
#      depth-diff-gaussian-rasterization。⚠️ 后者是**fork**，包名同样是
#      diff_gaussian_rasterization，装完会**遮蔽** clone 带来的 editable 版（指向
#      ~/repos/gaussian-splatting）——这是刻意的，且只影响本 env，vggt_human env 不动。
#      编译链来自 clone 的 conda cuda-nvcc（CUDA_HOME 默认指 $CONDA_PREFIX）。
#
# 用法：
#   bash deformable_human/00a_setup_env.sh                 # env 已建好+仓已 clone 时只做验证
#   INSTALL_DEPS=1 BUILD_CUDA=1 bash deformable_human/00a_setup_env.sh   # 首次完整搭建
#   INSTALL_COLMAP=1 bash deformable_human/00a_setup_env.sh              # 附带装 colmap（conda-forge，CPU 版）
set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/_env.sh"

CLONE_FROM="${CLONE_FROM:-vggt_human}"
DG_DIR="${DG_DIR:-$HOME/repos/Deformable-3D-Gaussians}"   # WSL：强制 Linux fs
GIT_URL_PREFIX="${GIT_URL_PREFIX:-https://ghfast.top/}"   # 本机 GitHub 直连不通
DG_REPO_URL="${GIT_URL_PREFIX}https://github.com/ingra14m/Deformable-3D-Gaussians.git"

echo "🚀 [00a] deformable_human 环境搭建（WSL）"
echo "  🐍 env:        $CONDA_ENV  (clone from: $CLONE_FROM)"
echo "  📦 官方仓:     $DG_DIR"
echo ""

# ── 0. conda ToS（conda 26.x 非交互报错，先 accept；失败忽略）──────────────
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main 2>/dev/null || true
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r 2>/dev/null || true

# ── 1. conda env（clone）──────────────────────────────────────────────────
if [ -d "$(conda info --base)/envs/$CONDA_ENV" ]; then
    echo "⏭️  [1] env '$CONDA_ENV' 已存在，跳过 clone（要重建先手动: conda env remove -n $CONDA_ENV）"
else
    if [ ! -d "$(conda info --base)/envs/$CLONE_FROM" ]; then
        echo "❌ ERROR: donor env '$CLONE_FROM' 不存在，无法 clone。" >&2
        echo "       先确认: conda env list；或 CLONE_FROM=<其他env> 重跑" >&2
        exit 1
    fi
    echo "📦 [1] conda create -n $CONDA_ENV --clone $CLONE_FROM （大 env，耐心等）"
    conda create -y -n "$CONDA_ENV" --clone "$CLONE_FROM"
    if [ $? -ne 0 ]; then
        echo "❌ [1] FAILED: conda clone 失败" >&2
        exit 1
    fi
fi
# 重新激活（_env.sh 里 env 可能还不存在，activate 被 || true 跳过）
conda activate "$CONDA_ENV" || { echo "❌ conda activate $CONDA_ENV 失败" >&2; exit 1; }

# ── 2. clone 官方仓（含 submodule）─────────────────────────────────────────
if [ -f "$DG_DIR/train.py" ]; then
    echo "⏭️  [2] 官方仓已存在: $DG_DIR（更新请手动 git -C $DG_DIR pull）"
else
    echo "📦 [2] clone $DG_REPO_URL → $DG_DIR"
    mkdir -p "$(dirname "$DG_DIR")"
    # LD_LIBRARY_PATH= 前缀：conda libffi 与系统 libp11-kit 冲突会搞崩 git（AGENTS.md §6）
    LD_LIBRARY_PATH= git clone --recursive "$DG_REPO_URL" "$DG_DIR" || {
        echo "⚠️  --recursive clone 失败，尝试普通 clone + 单独 submodule（sslVerify=false 兜底）" >&2
        LD_LIBRARY_PATH= git clone "$DG_REPO_URL" "$DG_DIR" || \
            LD_LIBRARY_PATH= git -c http.sslVerify=false clone "$DG_REPO_URL" "$DG_DIR"
    }
    [ -f "$DG_DIR/train.py" ] || { echo "❌ [2] FAILED: clone 后 $DG_DIR/train.py 不存在" >&2; exit 1; }
fi
# submodule 用本地 url.insteadOf 走 ghfast.top（不污染全局 git config）
if [ -n "$GIT_URL_PREFIX" ]; then
    ( cd "$DG_DIR" && git config --local \
        "url.${GIT_URL_PREFIX}https://github.com/.insteadOf" "https://github.com/" )
fi
if [ ! -f "$DG_DIR/submodules/simple-knn/setup.py" ] || \
   [ ! -f "$DG_DIR/submodules/depth-diff-gaussian-rasterization/setup.py" ]; then
    echo "📦 [2b] submodule update --init --recursive"
    ( cd "$DG_DIR" && LD_LIBRARY_PATH= git submodule update --init --recursive ) || \
    ( cd "$DG_DIR" && LD_LIBRARY_PATH= git -c http.sslVerify=false submodule update --init --recursive )
fi

# ── 2c. proxy.env 幂等追加 DG_DIR（让 01-08 自动拿到 Linux fs 路径）─────────
PROXY_ENV_FILE="$REPO_DIR/proxy.env"
if [ -f "$PROXY_ENV_FILE" ] && ! grep -q '^DG_DIR=' "$PROXY_ENV_FILE"; then
    echo "DG_DIR=$DG_DIR" >> "$PROXY_ENV_FILE"
    echo "  ✅ proxy.env 追加: DG_DIR=$DG_DIR（deformable_gaussians 也共用此键，同一官方仓）"
fi

# ── 3. Python deps（过滤 submodule 本地路径行 + torch pin）───────────────────
if [ "${INSTALL_DEPS:-0}" = "1" ]; then
    REQ="$DG_DIR/requirements.txt"
    [ -f "$REQ" ] || { echo "❌ ERROR: $REQ 不存在" >&2; exit 1; }
    TMP_REQ="$(mktemp).txt"
    grep -v -iE '^[[:space:]]*submodules/' "$REQ" \
        | grep -v -iE '^[[:space:]]*(torch|torchvision)([=<>!~]|$|[[:space:]])' > "$TMP_REQ"
    echo "📦 [3] pip install requirements（submodule/torch pin 已过滤；--find-links 本地 wheel 优先）"
    PIP_FLAGS=(-i https://mirrors.aliyun.com/pypi/simple --find-links "$WHEELS_DIR" --timeout 600 --retries 5)
    pip install "${PIP_FLAGS[@]}" -r "$TMP_REQ"
    if [ $? -ne 0 ]; then
        echo "❌ [3] FAILED: pip install 失败（输出往上翻）" >&2
        rm -f "$TMP_REQ"
        exit 1
    fi
    rm -f "$TMP_REQ"
fi

# ── 3b. colmap（可选；conda-forge 是 CPU 版，跑 01 时记得 USE_GPU=0）─────────
if [ "${INSTALL_COLMAP:-0}" = "1" ]; then
    if command -v colmap >/dev/null 2>&1; then
        echo "⏭️  [3b] colmap 已在 PATH: $(command -v colmap)"
    else
        echo "📦 [3b] conda install colmap（conda-forge，CPU 版；无 sudo 的唯一省事路径）"
        conda install -y -c conda-forge colmap
    fi
fi

# ── 4. CUDA 子模块编译 ──────────────────────────────────────────────────────
if [ "${BUILD_CUDA:-1}" = "1" ]; then
    # nvcc 随 clone 继承（vggt_human env 里装过 cuda-nvcc 12.1）；CUDA_HOME 指 env
    : "${CUDA_HOME:=$CONDA_PREFIX}"
    export CUDA_HOME
    # 3090 = sm_86；显式指定避免对全 arch 编译（快很多，也避开老 setup.py 的怪默认）
    export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.6}"
    if [ ! -x "$CUDA_HOME/bin/nvcc" ]; then
        echo "❌ ERROR: $CUDA_HOME/bin/nvcc 不存在。" >&2
        echo "       donor env 没带 cuda-nvcc？检查: conda list -n $CLONE_FROM | grep cuda" >&2
        exit 1
    fi
    echo "🔧 [4] nvcc: $("$CUDA_HOME/bin/nvcc" --version | tail -1 | xargs)  arch: $TORCH_CUDA_ARCH_LIST"
    PIP_FLAGS=(-i https://mirrors.aliyun.com/pypi/simple --find-links "$WHEELS_DIR" --timeout 600 --retries 5)

    echo "🔧 [4a] build simple-knn"
    pip install "${PIP_FLAGS[@]}" --no-build-isolation "$DG_DIR/submodules/simple-knn" || {
        echo "❌ [4a] FAILED" >&2; exit 1; }

    echo "🔧 [4b] build depth-diff-gaussian-rasterization（fork，遮蔽 editable 版是刻意的）"
    pip install "${PIP_FLAGS[@]}" --no-build-isolation "$DG_DIR/submodules/depth-diff-gaussian-rasterization" || {
        echo "❌ [4b] FAILED: fork 光栅化器（2023 年代码）在 torch 2.5.1 上编译失败。" >&2
        echo "       排查方向见 README_wsl.md「可能遇到的问题」；兜底是退回 py3.7/torch1.13 独立 env" >&2
        echo "       （参考 deformable_gaussians/_env.sh 头部注释的服务器方案）。" >&2
        exit 1; }
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
try:
    import diff_gaussian_rasterization as dgr
    print(f"  ✅ diff_gaussian_rasterization: {dgr.__file__}")
    # editable 版（gaussian-splatting）会显示 ~/repos/ 路径；fork 编译版应在 env site-packages
    if "site-packages" not in dgr.__file__ and "BUILD_CUDA" not in __import__("os").environ.get("SKIP_MARKER", ""):
        print("  ⚠️ 注意: 当前 import 到的不是 fork 编译版（仍指向 editable），"
              "train.py 需要 fork 的 depth 输出，请确认 BUILD_CUDA=1 已跑过")
except ImportError as e:
    print(f"  ⚠️ diff_gaussian_rasterization 不可 import: {e}（BUILD_CUDA=1 重跑）")
try:
    import simple_knn
    print(f"  ✅ simple_knn: {simple_knn.__file__}")
except ImportError as e:
    print(f"  ⚠️ simple_knn 不可 import: {e}（BUILD_CUDA=1 重跑）")
PY

echo ""
echo "🎉 [00a] Done. env '$CONDA_ENV' 就绪。"
echo "  💡 建议: pip cache purge   # 清 HTTP 缓存释放 vhdx 空间"
echo "  → 下一步: 准备数据  bash deformable_human/01_prepare_data.sh"

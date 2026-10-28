#!/usr/bin/env bash
# 00a_setup_env.sh — WSL Ubuntu 24.04 环境安装（TaoFlowForge 推理链路）。
#
# 做什么：
#   1. 建 conda env `taoflowforge`（python=3.10，与官方 README 测试配置一致）
#   2. pip 装 torch 2.5.1 + torchvision 0.20.1（PyPI 默认 wheel = cu121）
#   3. clone 官方仓 alibaba/Taobao3D 到 ~/repos/Taobao3D
#   4. pip install -r TaoFlowForge/requirements.txt（纯 wheel，无 JIT 编译，
#      不需要 nvcc/gcc —— open3d/embreex 都是预编译 wheel）
#   5. pip 装 modelscope（01 下权重用）
#   6. 把 TAOFLOWFORGE_* 路径追加进 proxy.env（若尚未存在）
#
# 用法：
#   bash taoflowforge/00a_setup_env.sh
#   SKIP_CLONE=1 bash taoflowforge/00a_setup_env.sh   # 已有 ~/repos/Taobao3D
set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"

# ── 1. Source conda ────────────────────────────────────────────────────────
CONDA_BASE="$HOME/miniconda3"
if [ ! -f "$CONDA_BASE/etc/profile.d/conda.sh" ]; then
    echo "❌ ERROR: miniconda3 not found at $CONDA_BASE" >&2
    exit 1
fi
# shellcheck disable=SC1091
source "$CONDA_BASE/etc/profile.d/conda.sh"

conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main 2>/dev/null || true
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r 2>/dev/null || true

# ── 2. WSL 路径默认值（Linux fs：仓 ~/repos，输出 ~/output；权重放 D 盘读一次即可）──
# 在 source _env.sh 之前 export，让它的 ${VAR:-default} 取到这些值。
export TAOFLOWFORGE_DIR="${TAOFLOWFORGE_DIR:-$HOME/repos/Taobao3D}"
export TAOFLOWFORGE_MODEL_DIR="${TAOFLOWFORGE_MODEL_DIR:-/mnt/d/wheel/taoflowforge_ms}"
export TAOFLOWFORGE_RESULTS_DIR="${TAOFLOWFORGE_RESULTS_DIR:-$HOME/output/taoflowforge_results}"

# ── 3. 追加 TAOFLOWFORGE_* 路径到 proxy.env（已有则不动）───────────────────
PROXY_ENV="$REPO_DIR/proxy.env"
if ! grep -q "TAOFLOWFORGE_MODEL_DIR" "$PROXY_ENV" 2>/dev/null; then
    echo "📦 appending TAOFLOWFORGE_* path overrides -> $PROXY_ENV"
    cat >> "$PROXY_ENV" <<'ENVEOF'

# ── taoflowforge（阿里 Taobao3D 单图→原生网格；仅 taoflowforge/_env.sh 读取）───
# 官方仓在 Linux fs；权重放 D 盘（推理时读一次，不占 Linux fs 空间）
export TAOFLOWFORGE_DIR="$HOME/repos/Taobao3D"
export TAOFLOWFORGE_MODEL_DIR="/mnt/d/wheel/taoflowforge_ms"
export TAOFLOWFORGE_RESULTS_DIR="${TAOFLOWFORGE_RESULTS_DIR:-$HOME/output/taoflowforge_results}"
ENVEOF
    echo "  ✅ appended"
else
    echo "⏭️  proxy.env 已有 TAOFLOWFORGE_* 配置: $PROXY_ENV"
fi

# ── 4. Source _env.sh（拿到 proxy.env + conda activate + 路径）──────────────
# shellcheck disable=SC1091
source "$SCRIPT_DIR/_env.sh"

echo "=== [00a] Setup TaoFlowForge env on WSL ==="
echo "  🤖 official repo: $TAOFLOWFORGE_DIR"
echo "  🏋️ weights dir:    $MODEL_DIR"
echo "  💾 output dir:    $RESULTS_DIR"
echo "  🏠 conda env:     $CONDA_ENV"
echo ""

# ── 5. 建 conda env（python=3.10，官方测试版本）───────────────────────────
_new_prefix="$(conda info --base 2>/dev/null)/envs/$CONDA_ENV"
if [ -d "$_new_prefix/conda-meta" ] && [ -f "$_new_prefix/conda-meta/history" ]; then
    echo "--- conda env '$CONDA_ENV' already exists at $_new_prefix ---"
else
    echo "📦 creating conda env '$CONDA_ENV' (python=3.10)..."
    conda create -y -n "$CONDA_ENV" python=3.10 || {
        echo "❌ conda create failed" >&2
        exit 1
    }
    echo "  ✅ created"
fi

# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"
echo "  🐍 python: $(python --version 2>&1) at $(which python)"

# ── 6. PyTorch 2.5.1 + torchvision 0.20.1（PyPI 默认 wheel = cu121）────────
WHEELS_DIR="${WHEELS_DIR:-/mnt/d/wheel}"
mkdir -p "$WHEELS_DIR"
PIP_FLAGS=(-i "${PIP_INDEX_URL:-https://pypi.tuna.tsinghua.edu.cn/simple}"
    --find-links "$WHEELS_DIR" --timeout 600 --retries 5)
TORCH_VERSION="${TORCH_VERSION:-2.5.1}"
TORCHVISION_VERSION="${TORCHVISION_VERSION:-0.20.1}"

if ! python -c "import torch; assert torch.__version__.startswith('$TORCH_VERSION'); assert torch.cuda.is_available()" 2>/dev/null; then
    echo "📦 installing PyTorch $TORCH_VERSION + torchvision $TORCHVISION_VERSION ..."
    pip install "${PIP_FLAGS[@]}" "torch==$TORCH_VERSION" "torchvision==$TORCHVISION_VERSION" || {
        echo "❌ torch install failed" >&2
        echo "  备用：pip install --index-url https://download.pytorch.org/whl/cu121 torch==$TORCH_VERSION torchvision==$TORCHVISION_VERSION" >&2
        exit 1
    }
fi
python -c "import torch; print(f'  ✅ torch {torch.__version__}  cuda={torch.version.cuda}  available={torch.cuda.is_available()}')" || exit 1

# ── 7. Clone 官方仓 ───────────────────────────────────────────────────────
if [ "${SKIP_CLONE:-0}" != "1" ] && [ ! -d "$TAOFLOWFORGE_DIR/.git" ]; then
    mkdir -p "$(dirname "$TAOFLOWFORGE_DIR")"
    echo "📦 cloning Taobao3D -> $TAOFLOWFORGE_DIR"
    git clone "$TAOFLOWFORGE_REPO" "$TAOFLOWFORGE_DIR" || \
        git -c http.sslVerify=false clone "$TAOFLOWFORGE_REPO" "$TAOFLOWFORGE_DIR" || {
        echo "❌ clone failed" >&2
        exit 1
    }
else
    echo "⏭️  Taobao3D 已存在: $TAOFLOWFORGE_DIR"
fi

if [ ! -d "$TFF_PKG_DIR/taoflowforge" ]; then
    echo "❌ ERROR: $TFF_PKG_DIR/taoflowforge 不存在（官方仓结构变了？）" >&2
    exit 1
fi

# ── 8. 装推理依赖（requirements.txt：全预编译 wheel，无需编译器）──────────
if [ ! -f "$TFF_PKG_DIR/requirements.txt" ]; then
    echo "❌ ERROR: $TFF_PKG_DIR/requirements.txt 不存在" >&2
    exit 1
fi
echo "📦 installing TaoFlowForge requirements ..."
# ⚠️ 上游 bug（2026-10 实测）：requirements.txt 钉死 Pillow==12.0.0，
# 但 gradio==5.50.0 要求 pillow<12.0，新 pip 解析器直接 ResolutionImpossible。
# 兼容处理：拷一份到 /tmp，把 Pillow 钉到 11.x（gradio 允许 >=8.0,<12.0），不改上游文件。
REQ_FIXED="$(mktemp /tmp/tff_req_XXXX.txt)"
sed 's/^Pillow==12\.0\.0$/Pillow==11.3.0/' "$TFF_PKG_DIR/requirements.txt" > "$REQ_FIXED"
if ! ( cd "$TFF_PKG_DIR" && pip install "${PIP_FLAGS[@]}" -r "$REQ_FIXED" ); then
    echo "❌ requirements 安装失败" >&2
    rm -f "$REQ_FIXED"
    exit 1
fi
rm -f "$REQ_FIXED"

# modelscope CLI（01 下权重用；不在官方 requirements 里）
if ! python -c "import modelscope" 2>/dev/null; then
    echo "📦 installing modelscope ..."
    pip install "${PIP_FLAGS[@]}" modelscope || {
        echo "⚠️ modelscope 安装失败 —— 01 下载脚本将不可用（也可手工/迅雷下权重）" >&2
    }
fi

# open3d 需要系统 libgomp.so.1；新 libgcc-ng(15.x) 不再捆绑它，需单独装（WSL 无 sudo 时 conda 装）
if ! python -c "import open3d" 2>/dev/null; then
    echo "📦 installing libgomp (open3d runtime dep) ..."
    conda install -y -c conda-forge libgomp || echo "⚠️ libgomp 安装失败" >&2
fi

# ── 9. conda init bash ────────────────────────────────────────────────────
if ! grep -q "miniconda3" "$HOME/.bashrc" 2>/dev/null; then
    echo "📦 running 'conda init bash' ..."
    conda init bash >/dev/null 2>&1 && echo "  ✅ done（重开 shell 或 source ~/.bashrc 后生效）"
else
    echo "⏭️  conda already initialized in ~/.bashrc"
fi

# ── 10. 验证 ──────────────────────────────────────────────────────────────
echo ""
echo "--- verification ---"
python -c "import torch; print(f'  ✅ torch {torch.__version__} cuda={torch.cuda.is_available()}')" 2>/dev/null || echo "  [MISS] torch"
python -c "import trimesh, open3d, transformers; print('  ✅ trimesh/open3d/transformers')" 2>/dev/null || echo "  [MISS] trimesh/open3d/transformers"
python -c "import modelscope" 2>/dev/null && echo "  ✅ modelscope" || echo "  [---] modelscope（01 需要）"
( cd "$TFF_PKG_DIR" && python -c "import taoflowforge; print('  ✅ taoflowforge package importable')" ) 2>/dev/null || echo "  [MISS] taoflowforge package"
for f in "$TFF_STAGE0" "$TFF_STAGE1" "$TFF_STAGE2"; do
    [ -f "$f" ] && echo "  ✅ $(basename "$f")" || echo "  [---] $(basename "$f") 未下载 —— 跑 taoflowforge/01_download_models.sh"
done

echo ""
echo "🎉 [00a] Done. Next:"
echo "  1. bash taoflowforge/01_download_models.sh   # ModelScope 下 stage0/1/2 权重"
echo "  2. GPU=0 INPUT=/path/to/image.png bash taoflowforge/02_run_inference.sh"

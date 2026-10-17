#!/usr/bin/env bash
# Gen3R WSL 本机构建 env（无 sudo，全部走 conda + pip）
#
# 与 00_setup_env.sh 的差异：本机 conda 从零建 env，torch 走 PyPI 阿里镜像，
# 官方仓 clone 到 ~/repos（Linux fs），不依赖服务器共享盘。
set -o pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/_env.sh"

INSTALL_DEPS="${INSTALL_DEPS:-1}"
PY_VER="${PY_VER:-3.11.2}"
# 阿里源实测可用（清华源间歇性打不开/极慢，见 AGENTS.md 第 6 节）
# 注意：不继承 proxy.env 的 PIP_INDEX_URL，那里面配的是清华源
PIP_INDEX="${PIP_INDEX:-https://mirrors.aliyun.com/pypi/simple}"
WHEELS_DIR="${WHEELS_DIR:-/mnt/d/wheel}"

echo "🚀 [00a] Gen3R WSL env setup"
echo "  📁 官方仓: $GEN3R_DIR"
echo "  💾 权重:   $MODEL_DIR"
echo "  🐍 conda:  $CONDA_ENV ($PY_VER)"
echo "  📦 pip:    $PIP_INDEX"

# ── 1) clone 官方仓（GitHub 本机直连可用）──
if [ ! -d "$GEN3R_DIR/gen3r" ]; then
    echo "📦 cloning Gen3R → $GEN3R_DIR"
    mkdir -p "$(dirname "$GEN3R_DIR")"
    LD_LIBRARY_PATH= git clone https://github.com/JaceyHuang/Gen3R "$GEN3R_DIR" || {
        echo "❌ clone failed" >&2; exit 1; }
else
    echo "⏭️  repo exists: $GEN3R_DIR"
fi

# ── 2) conda env ──
if ! conda env list | awk '{print $1}' | grep -qx "$CONDA_ENV"; then
    echo "📦 creating conda env $CONDA_ENV (python=$PY_VER)"
    conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main >/dev/null 2>&1
    conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r >/dev/null 2>&1
    conda create -n "$CONDA_ENV" python="$PY_VER" -y || { echo "❌ conda create failed" >&2; exit 1; }
else
    echo "⏭️  env exists: $CONDA_ENV"
fi

if [ "$INSTALL_DEPS" != "1" ]; then
    echo "⏭️  INSTALL_DEPS=0, skip pip install"
    echo "🎉 Done (env only)."
    exit 0
fi

PY="$HOME/miniconda3/envs/$CONDA_ENV/bin/python"
PIP_FLAGS=(-i "$PIP_INDEX" --timeout 600 --retries 5)
[ -d "$WHEELS_DIR" ] && PIP_FLAGS+=(--find-links "$WHEELS_DIR")

# ── 3) torch（优先用 /mnt/d/wheel 里的本地 wheel：本机镜像单线程 ~0.4MB/s，
#       联网装 torch+cu12 依赖要 1.5h+，本地 wheel 2 分钟）──
# 实测走通组合：torch 2.8.0+cu128 + torchvision 0.23.0（本地 wheel），
# flash_attn 2.8.3.post2（预编译 wheel 也在 /mnt/d/wheel）。
TORCH_VERSION="${TORCH_VERSION:-2.8.0}"
TV_VERSION="${TV_VERSION:-0.23.0}"
echo "📦 installing torch ${TORCH_VERSION} + torchvision ${TV_VERSION}"
if ls "$WHEELS_DIR"/torch-"${TORCH_VERSION}"*.whl >/dev/null 2>&1 && \
   ls "$WHEELS_DIR"/torchvision-"${TV_VERSION}"*.whl >/dev/null 2>&1; then
    echo "   📁 使用本地 wheel: $WHEELS_DIR"
    # torch 的纯 py 依赖本地没有 → 先联网装，再 --no-index 装 torch 本体
    # （--no-index + torch 一起会因 filelock/sympy 等缺失直接失败）
    "$PY" -m pip install "${PIP_FLAGS[@]}" \
        filelock typing-extensions sympy networkx jinja2 fsspec mpmath MarkupSafe \
        "numpy==1.26.0" pillow || {
        echo "❌ torch 依赖安装失败" >&2; exit 1; }
    "$PY" -m pip install --no-index --find-links "$WHEELS_DIR" \
        "torch==${TORCH_VERSION}" "torchvision==${TV_VERSION}" || {
        echo "❌ torch install failed" >&2; exit 1; }
else
    echo "   🌐 从镜像安装（$PIP_INDEX）"
    "$PY" -m pip install "${PIP_FLAGS[@]}" \
        "torch==${TORCH_VERSION}" "torchvision==${TV_VERSION}" || {
        echo "❌ torch install failed" >&2; exit 1; }
fi
# torchaudio 未被 Gen3R 代码引用（grep 无命中），不装

# ── 4) 推理依赖（不装 pytorch3d：只被 eval_utils 用于评估指标）──
# ⚠️ opencv-python-headless 必须 4.10：5.x 会把 numpy 顶到 2.x，和 open3d/torch 冲突。
echo "📦 installing inference deps"
"$PY" -m pip install "${PIP_FLAGS[@]}" \
    diffusers==0.33.1 transformers==4.46.3 accelerate==1.0.1 safetensors \
    einops==0.8.0 imageio==2.35.1 imageio-ffmpeg==0.5.1 numpy==1.26.0 \
    open3d==0.18.0 easydict==1.13 pillow "opencv-python-headless==4.10.0.84" || {
    echo "❌ deps install failed" >&2; exit 1; }

# ── 4b) open3d 需要 libgomp.so.1（conda 安装，WSL 无 sudo）──
if ! conda list -n "$CONDA_ENV" libgomp 2>/dev/null | grep -q libgomp; then
    echo "📦 installing libgomp (open3d 依赖)"
    conda install -n "$CONDA_ENV" -c conda-forge libgomp -y || {
        echo "❌ libgomp install failed" >&2; exit 1; }
fi

# ── 4c) flash-attn：没有它 SDPA 走 math 后端（仅支持 fp16/bf16 的高效后端），
#        49 帧×560² 在 24GB 卡上单步 >4min；有了它 9.5s/step ──
if ls "$WHEELS_DIR"/flash_attn-*.whl >/dev/null 2>&1; then
    echo "📦 installing flash-attn（本地 wheel）"
    "$PY" -m pip install --no-deps --no-index --find-links "$WHEELS_DIR" flash-attn || {
        echo "⚠️ flash-attn wheel 安装失败（无对应版本？），继续走 SDPA fallback"; }
else
    echo "⚠️ $WHEELS_DIR 无 flash_attn wheel，跳过（可用 SDPA，但慢且吃显存）"
fi

# ── 4d) 打本仓的 24GB 卡适配补丁（bf16 直载 + 编码器卸载 + 步数覆盖）──
if [ -f "$SCRIPT_DIR/patch_infer_bf16.py" ]; then
    echo "📦 applying media_code patches"
    python3 "$SCRIPT_DIR/patch_infer_bf16.py" || true
    python3 "$SCRIPT_DIR/patch_pipeline_vram.py" || true
fi

# ── 5) 验证 ──
echo "🔍 verifying imports"
cd "$GEN3R_DIR" || exit 1
"$PY" - <<'EOF'
import torch, diffusers, transformers, open3d, imageio
print(f"✅ torch {torch.__version__} cuda={torch.cuda.is_available()} "
      f"dev={torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu'}")
print(f"✅ diffusers {diffusers.__version__} transformers {transformers.__version__}")
import gen3r  # noqa: F401
from gen3r.pipeline import Gen3RPipeline
print("✅ gen3r pipeline import ok")
EOF
if [ $? -ne 0 ]; then
    echo "❌ import check failed" >&2
    exit 1
fi

echo "🎉 Done. 下一步：bash gen3r/01_download_models.sh"

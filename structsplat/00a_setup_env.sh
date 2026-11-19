#!/usr/bin/env bash
# structsplat/00a_setup_env.sh — WSL 建 env（从零 or 克隆 querysplat 复用已编译 gsplat）
#
# 为什么克隆：官方 requirements.txt 钉 torch==2.4.0+cu118 + gsplat==1.5.3（源码编译），
# 本机 querysplat env 已有 torch 2.8.0+cu128 且 gsplat 1.5.3 的 CUDA 扩展已编译
# （缓存键 py311_cu128，克隆后仍命中）。省掉 gsplat 半小时级编译 + 大概率失败的风险。
set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/_env.sh"

BASE_ENV="${BASE_ENV:-querysplat}"
PY_VER="3.11"   # 必须与 BASE_ENV 一致，否则 gsplat 编译缓存 (py311_cu128) 不命中

echo "🚀 [00a] 建 conda env '$CONDA_ENV'（克隆自 '$BASE_ENV'）"

# ---------- 1. clone 官方仓 ----------
if [ ! -d "$STRUCTSPLAT_REPO/.git" ]; then
    echo "📦 clone 官方仓 → $STRUCTSPLAT_REPO"
    git clone --recursive https://github.com/J-C-Zhao/StructSplat.git "$STRUCTSPLAT_REPO"
else
    echo "⏭️  官方仓已存在: $STRUCTSPLAT_REPO"
    # 子模块可能没拉
    if [ ! -d "$STRUCTSPLAT_REPO/third_party/vggt/vggt" ]; then
        echo "📦 补拉子模块 third_party/vggt"
        git -C "$STRUCTSPLAT_REPO" submodule update --init --recursive
    fi
fi

# ---------- 2. env ----------
EXISTING=$(conda env list | awk '{print $1}' | grep -x "$CONDA_ENV")
if [ -n "$EXISTING" ]; then
    echo "⏭️  env '$CONDA_ENV' 已存在，跳过创建"
else
    if [ "$CONDA_ENV" = "$BASE_ENV" ]; then
        echo "❌ CONDA_ENV 不能等于 BASE_ENV（会原地污染）" >&2; exit 1
    fi
    BASE_PATH="$HOME/miniconda3/envs/$BASE_ENV"
    if [ ! -d "$BASE_PATH" ]; then
        echo "❌ 源 env 不存在: $BASE_PATH（设 BASE_ENV 指向可用的 env）" >&2; exit 1
    fi
    echo "📦 conda create --clone $BASE_ENV → $CONDA_ENV"
    conda create -y -n "$CONDA_ENV" --clone "$BASE_ENV" || { echo "❌ clone 失败" >&2; exit 1; }
fi

conda activate "$CONDA_ENV" || { echo "❌ 激活 $CONDA_ENV 失败" >&2; exit 1; }
echo "🐍 python: $(which python) → $(python -V 2>&1)"

# ---------- 3. 补依赖（不重装 torch / gsplat） ----------
# 官方 pin: transformers==5.0.0 (dinov3)、deepspeed、ml_collections、
#           pytorch-lightning、torchmetrics、lpips、prefetch_generator
# 注：deepspeed 只在训练路径用到（evaluation.py 也需要 import），必须装；
#     但不需要它的 CUDA op —— 设 DS_BUILD_OPS=0 走纯 python 路径。
export DS_BUILD_OPS="${DS_BUILD_OPS:-0}"

REQ_LOCAL="$SCRIPT_DIR/requirements_runtime.txt"
if [ ! -f "$REQ_LOCAL" ]; then
    echo "❌ 缺 $REQ_LOCAL" >&2; exit 1
fi

echo "📦 pip install -r requirements_runtime.txt"
python -m pip install -r "$REQ_LOCAL" -i "${PIP_INDEX_URL:-https://pypi.tuna.tsinghua.edu.cn/simple}" \
    --trusted-host "${PIP_TRUSTED_HOST:-pypi.tuna.tsinghua.edu.cn}" || {
    echo "❌ pip install 失败" >&2; exit 1; }

# ---------- 4. 自检 ----------
echo "🔍 依赖自检"
python - <<'PY'
import importlib, sys
mods = ["torch", "gsplat", "deepspeed", "transformers", "ml_collections",
        "lpips", "prefetch_generator", "pytorch_lightning", "torchmetrics", "einops"]
bad = []
for m in mods:
    try:
        mod = importlib.import_module(m)
        print(f"  ✅ {m:22s} {getattr(mod, '__version__', '?')}")
    except Exception as e:
        print(f"  ❌ {m:22s} {type(e).__name__}: {e}")
        bad.append(m)
if bad:
    sys.exit("❌ 缺依赖: " + ", ".join(bad))
PY
[ $? -ne 0 ] && exit 1

echo "🔍 gsplat CUDA 扩展自检（克隆 env 的关键验证）"
python - <<'PY'
import torch
from gsplat.cuda._backend import _C
assert _C is not None, "❌ gsplat CUDA 扩展未加载 —— 编译缓存键可能不匹配"
from gsplat import rasterization
means = torch.tensor([[0.,0.,2.]], device="cuda")
quats = torch.tensor([[1.,0.,0.,0.]], device="cuda")
scales = torch.tensor([[0.1,0.1,0.1]], device="cuda")
opac = torch.tensor([0.9], device="cuda")
cols = torch.tensor([[1.,0.,0.]], device="cuda")
vm = torch.eye(4, device="cuda")[None].clone(); vm[:,2,3] = 2.0
K = torch.tensor([[[100.,0.,32.],[0.,100.,32.],[0.,0.,1.]]], device="cuda")
out,_,_ = rasterization(means, quats, scales, opac, cols, vm, K, 64, 64)
print(f"  ✅ gsplat rasterization OK  out={tuple(out.shape)} sum={float(out.sum()):.4f}")
PY
[ $? -ne 0 ] && { echo "❌ gsplat 自检失败" >&2; exit 1; }

echo "🎉 [00a] env 就绪: $CONDA_ENV"
echo "   下一步: bash structsplat/01_download_models.sh"

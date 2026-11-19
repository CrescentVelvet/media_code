#!/usr/bin/env bash
# structsplat/00b_fix_toolchain.sh — 修复 gsplat 编译链（本机 WSL 无系统 gcc）
#
# 本机实测（2026-11-19）：/usr/bin/gcc 不存在，只有 conda-forge 的
# x86_64-conda-linux-gnu-gcc。nvcc 找 host compiler 失败导致 gsplat 全部 .cu 编译失败。
# 本脚本：
#   1. 补 gcc/g++/cc/c++ 软链
#   2. 装 pillow-heif（读手机 .heic 必需）
#   3. 清 gsplat 编译缓存后重编并跑渲染冒烟
#
# 用法：bash structsplat/00b_fix_toolchain.sh
set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/_env.sh"

echo "============================================================"
echo " 修复编译链 (env=$CONDA_ENV)"
echo "============================================================"
echo "  CONDA_PREFIX : $CONDA_PREFIX"
echo "  CUDA_HOME    : ${CUDA_HOME:-<unset>}"
echo "  CC           : ${CC:-<unset>}"
echo "  CXX          : ${CXX:-<unset>}"
echo -n "  which gcc    : "; command -v gcc || echo "<none>"
echo -n "  which nvcc   : "; command -v nvcc || echo "<none>"

# ---------- 1. 确认编译器就位 ----------
echo ""
echo "=== [1/3] host compiler 检查"
gcc --version 2>&1 | head -1 || { echo "❌ gcc 仍不可用"; exit 1; }
g++ --version 2>&1 | head -1 || { echo "❌ g++ 仍不可用"; exit 1; }
nvcc --version 2>&1 | grep -E "release" || echo "⚠️  nvcc 版本未取到"

# ---------- 2. pillow-heif ----------
echo ""
echo "=== [2/3] pillow-heif（.heic 读取）"
python - <<'PYEOF'
try:
    import pillow_heif
    print("  ✅ 已安装:", pillow_heif.__version__)
except ImportError:
    print("  ⬇️  未安装，正在安装…")
    import subprocess, sys
    r = subprocess.run([sys.executable, "-m", "pip", "install", "-q", "pillow-heif"],
                       capture_output=True, text=True)
    if r.returncode != 0:
        print("  ❌ 安装失败:", r.stderr[-600:])
        sys.exit(1)
    import pillow_heif
    print("  ✅ 安装完成:", pillow_heif.__version__)
PYEOF
[ $? -ne 0 ] && echo "  ⚠️  pillow-heif 不可用，HEIC 输入会失败"

# ---------- 3. 重编 gsplat ----------
echo ""
echo "=== [3/3] 重编 gsplat CUDA 扩展（首次约 5-10 分钟）"
CACHE="$HOME/.cache/torch_extensions/py$(python -c 'import sys;print(f"{sys.version_info.major}{sys.version_info.minor}")')_cu$(python -c 'import torch;print(torch.version.cuda.replace(".",""))' 2>/dev/null)/gsplat_cuda"
echo "  缓存: $CACHE"
if [ -d "$CACHE" ]; then
    rm -rf "$CACHE"
    echo "  已清缓存"
fi

t0=$(date +%s)
python - <<'PYEOF'
import sys, torch
print(f"  torch {torch.__version__}  cuda {torch.version.cuda}  avail {torch.cuda.is_available()}")
from gsplat.cuda._backend import _C
if _C is None:
    print("❌ FAIL: gsplat._C is None（CUDA 扩展未编出）")
    sys.exit(1)
print(f"✅ gsplat._C 加载成功: {type(_C).__name__}")

# 渲染冒烟
from gsplat import rasterization
N, S, H, W = 200, 2, 96, 96
dev = "cuda"
means = (torch.rand(N, 3, device=dev) - 0.5) * 2.0
quats = torch.nn.functional.normalize(torch.randn(N, 4, device=dev), dim=-1)
scales = torch.rand(N, 3, device=dev) * 0.05 + 0.01
opac = torch.rand(N, device=dev)
colors = torch.rand(N, 3, device=dev)
eye = torch.eye(4, device=dev); eye[2, 3] = 3.0
vm = eye.unsqueeze(0).repeat(S, 1, 1)
K = torch.tensor([[100., 0, W/2], [0, 100., H/2], [0, 0, 1.]], device=dev).unsqueeze(0).repeat(S, 1, 1)
out, alpha, meta = rasterization(means=means, quats=quats, scales=scales,
                                 opacities=opac, colors=colors,
                                 viewmats=vm, Ks=K, width=W, height=H,
                                 render_mode="RGB+D", packed=False,
                                 rasterize_mode="classic")
print(f"✅ 渲染冒烟通过: out {tuple(out.shape)}  alpha {tuple(alpha.shape)}")
print(f"   RGB mean={out[..., :3].mean().item():.4f}  depth mean={out[..., 3].mean().item():.4f}")
PYEOF
RC=$?
t1=$(date +%s)
echo ""
if [ $RC -eq 0 ]; then
    echo "🎉 编译链修复完成，耗时 $((t1-t0))s"
else
    echo "❌ 仍然失败（exit=$RC，耗时 $((t1-t0))s）" >&2
    exit $RC
fi

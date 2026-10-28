#!/usr/bin/env bash
# 04_check_weights.sh — 下载完成后的权重体检：文件尺寸核对 + stage0 DINO key 探测。
#
# 用途：
#   1. 核对三个 stage*.pt 尺寸是否与 ModelScope 公布一致（迅雷/手工下载容易截断）
#   2. 探测 stage0.pt 是否内嵌 dino_encoder.* 权重 —— 没内嵌则首次推理必报
#      RuntimeError，需补 --dino-checkpoint（见 README Notes）
#
# 用法：
#   bash taoflowforge/04_check_weights.sh
set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/_env.sh"

# ModelScope 公布尺寸（字节），2026-10-28 实测
declare -A EXPECT=( [stage0.pt]=5470000000 [stage1.pt]=14500000000 [stage2.pt]=6030000000 )
# 容差：公布值是四舍五入的 GB/MB，允许 ±1% 偏差
TOL_PCT=1

echo "🚀 [04] 权重体检: $MODEL_DIR"
echo ""

_fail=0
for name in stage0.pt stage1.pt stage2.pt; do
    f="$MODEL_DIR/$name"
    if [ ! -f "$f" ]; then
        echo "  ❌ $name 不存在"
        _fail=1
        continue
    fi
    sz=$(stat -c %s "$f")
    exp="${EXPECT[$name]}"
    dev=$(( (sz - exp) * 100 / exp ))
    [ $dev -lt 0 ] && dev=$(( -dev ))
    if [ $dev -le $TOL_PCT ]; then
        echo "  ✅ $name  $((sz/1024/1024)) MiB（公布 $((exp/1000000)) MB，偏差 ${dev}%）"
    else
        echo "  ⚠️  $name  $((sz/1024/1024)) MiB，与公布值偏差 ${dev}% —— 可能下载截断"
        _fail=1
    fi
done

[ $_fail -ne 0 ] && { echo ""; echo "❌ 尺寸校验未过，先补齐权重再跑推理" >&2; exit 1; }

echo ""
echo "🔍 探测 stage0.pt 是否内嵌 DINO 权重（mmap 只读 key，不占内存）..."
python - <<'PY'
import os, torch
p = os.environ["TFF_STAGE0"]
# DrvFS(/mnt/d) 上 mmap 可能不支持，失败则普通加载（stage0 5.5GB，内存够）
try:
    ckpt = torch.load(p, map_location="cpu", mmap=True, weights_only=True)
except (OSError, RuntimeError):
    ckpt = torch.load(p, map_location="cpu", weights_only=True)
sd = ckpt.get("model_state_dict", ckpt)
dino = sum(1 for k in sd if k.startswith("dino_encoder."))
print(f"  dino_encoder.* 张量数: {dino}")
if dino:
    print("  ✅ stage0 已内嵌 DINOv3 —— 推理无需 --dino-checkpoint")
else:
    print("  ⚠️ 未内嵌 DINO —— 首跑会报 RuntimeError，需补：")
    print("    --dino-checkpoint <DINOv3-L/16 HF目录>")
    print("    下载：ModelScope 镜像 facebook/dinov3-vitl16-pretrain-lvd1689m（非 gated）")
PY

echo ""
echo "🎉 [04] Done."

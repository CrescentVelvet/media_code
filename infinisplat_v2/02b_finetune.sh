#!/usr/bin/env bash
# 02b_finetune.sh — 推理之后的后处理精调（位置冻结的 3DGS photometric finetune）。
#
# 为什么做成默认：Step 3 实测同样 32 视角、同样高斯数，只跑 3000 步纯光度优化
# （161 s / 6.3 GB）就能把 σ_eff 1.50 → 0.50 px、留出 PSNR +2.11 dB（7/7 视角全正）。
# 这笔开销相对"推理 26 s 但要多拍 2 倍视角"便宜得多，所以接在 02 后面默认开。
#
# 用法：
#   # a) 独立跑在已有结果上（02 会自动带这些变量调本脚本）
#   RESULT=~/output/infinisplat_v2_results/jiajia \
#     IMAGES_DIR=~/output/infinisplat_v2_results/_captures/jiajia/images \
#     bash 02b_finetune.sh
#
#   # b) 换步数 / 做消融
#   FINETUNE_ITERS=1000 bash 02b_finetune.sh
#   FINETUNE_EXTRA_ARGS="--opt sh" bash 02b_finetune.sh    # 只调颜色
#
# 输出：<RESULT>/ft/{scene.ply, cameras.npz, result.json, finetune.json}
#       —— 刻意不覆盖原始导出，两版并存才能做对照（15 / 16 都吃这个布局）。
#
# Env:
#   RESULT=             # 02 的输出目录（含 scene.ply / cameras.npz / result.json）
#   IMAGES_DIR=         # 与该重建同源的原图目录（GT 来源，缺一不可）
#   FT_SUBDIR=ft        # 精调结果落在 <RESULT>/<FT_SUBDIR>
#   FINETUNE_ITERS=3000 # 0 = 干跑（只 eval，验证本脚本与 03 口径一致）
#   FINETUNE_EXTRA_ARGS=# 直接透传给 14_finetune_gs.py，如 "--opt sh --lr-sh 1e-2"
#   FINETUNE_FORCE=0    # 1 = 即使 finetune.json 已存在也重跑
set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/_env.sh"

: "${FT_SUBDIR:=ft}"
: "${FINETUNE_ITERS:=3000}"
: "${FINETUNE_EXTRA_ARGS:=}"
: "${FINETUNE_FORCE:=0}"

if [ -z "${RESULT:-}" ]; then
    if [ -z "${OUT_NAME:-}" ]; then
        echo "❌ ERROR: 需要 RESULT（02 的输出目录）或 OUT_NAME" >&2
        exit 1
    fi
    RESULT="$RESULTS_DIR/$OUT_NAME"
fi
if [ ! -f "$RESULT/scene.ply" ] || [ ! -f "$RESULT/result.json" ]; then
    echo "❌ ERROR: $RESULT 缺 scene.ply / result.json（先跑 02_run_inference.sh）" >&2
    exit 1
fi
if [ -z "${IMAGES_DIR:-}" ] || [ ! -d "$IMAGES_DIR" ]; then
    echo "❌ ERROR: 需要 IMAGES_DIR（精调的 GT 来源，不能用别的场景的图）" >&2
    exit 1
fi
if ! python -c "import gsplat" 2>/dev/null; then
    echo "❌ ERROR: gsplat 未安装（精调需要可微光栅化器）" >&2
    echo "      WITH_GSPLAT=1 bash $SCRIPT_DIR/00_setup_env.sh" >&2
    exit 1
fi

FT_DIR="$RESULT/$FT_SUBDIR"
if [ -f "$FT_DIR/finetune.json" ] && [ "$FINETUNE_FORCE" != "1" ]; then
    echo "⏭️  已有精调结果，跳过（FINETUNE_FORCE=1 可强制重跑）: $FT_DIR"
    exit 0
fi

echo "🔧 [02b] 后处理精调（位置冻结 / 位姿不动 / 只调 SH·scale·opacity·quat）"
echo "  💾 输入: $RESULT"
echo "  🖼️  GT:   $IMAGES_DIR"
echo "  💾 输出: $FT_DIR"
echo "  ⚙️  iters: $FINETUNE_ITERS ${FINETUNE_EXTRA_ARGS:+(extra: $FINETUNE_EXTRA_ARGS)}"

START=$(date +%s)
# shellcheck disable=SC2086
python "$SCRIPT_DIR/14_finetune_gs.py" --result "$RESULT" --images-dir "$IMAGES_DIR" \
    --out "$FT_DIR" --iters "$FINETUNE_ITERS" $FINETUNE_EXTRA_ARGS
RC=$?
ELAPSED=$(( $(date +%s) - START ))
if [ $RC -ne 0 ]; then
    echo "❌ 精调失败 (exit $RC)" >&2
    exit 1
fi

python - "$FT_DIR" <<'PY'
import json, sys
from pathlib import Path
d = Path(sys.argv[1])
r = json.loads((d / "finetune.json").read_text())
ini, fin = r["init"]["train"], r["final"]["train"]
hol_i, hol_f = r["init"]["hold"], r["final"]["hold"]
print(f"  ✅ 自洽 PSNR {ini['psnr']:.3f} → {fin['psnr']:.3f}  ({fin['psnr']-ini['psnr']:+.3f} dB)")
if hol_i and hol_f:
    print(f"  🎯 留出 PSNR {hol_i['psnr']:.3f} → {hol_f['psnr']:.3f}  ({hol_f['psnr']-hol_i['psnr']:+.3f} dB)"
          f"  ← 只有这个数能报")
print(f"  📉 半径对数均值变化 {r['scale_log_delta_mean']:+.4f}  不透明度 {r['opacity_mean_init']:.3f} → {r['opacity_mean']:.3f}")
print(f"  🖼️  图像配对: {r.get('image_pairing', 'n/a')}")
print(f"  🔁 写盘回读 Δ{r['ply_roundtrip']['delta']:+.3f} dB（>0.05 即 PLY 字段写错）")
PY

echo "⏱️  精调耗时 ${ELAPSED}s"
echo "🎉 Done: $FT_DIR"

#!/usr/bin/env bash
# 04_run_dataset.sh — 批量跑"自有物体数据集"（每物体一个子目录，K 张图、无位姿），
# 逐物体记录 GPU 峰值显存、端到端耗时、高斯数与自洽渲染 PSNR。
#
# 用法：
#   DATASET_ROOT=/mnt/d/dataset/物体三图数据 bash 04_run_dataset.sh
#   DATASET_ROOT=... LONG_SIDE=1022 DO_EVAL=0 bash 04_run_dataset.sh
#
# 输出：
#   $RESULTS_DIR/<OUT_PREFIX>_<物体名>/{scene.ply, cameras.npz, result.json}
#   $RESULTS_DIR/<OUT_PREFIX>_bench.csv   逐物体指标
#   $RESULTS_DIR/<OUT_PREFIX>_bench.md    汇总表
#
# Env (all optional):
#   DATASET_ROOT=      数据集根（每个子目录 = 一个物体）
#   OUT_PREFIX=        输出目录前缀，默认 threeview
#   LONG_SIDE=518      长边像素
#   POSE_MODE=         da3-pose（默认，无位姿）/ provided（需每物体带 cameras.npz）
#   MIN_VIEWS=2        少于该视角数跳过（模型最少视 2 张）
#   MAX_VIEWS=8        上限视角数
#   DO_EVAL=1          跑完对输入视角自洽渲染，记 PSNR/SSIM
#   GPU_ID=0           采样哪张卡的显存
set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/_env.sh"

: "${DATASET_ROOT:=/mnt/d/dataset/物体三图数据}"
: "${OUT_PREFIX:=threeview}"
: "${LONG_SIDE:=518}"
: "${POSE_MODE:=da3-pose}"
: "${MIN_VIEWS:=2}"
: "${MAX_VIEWS:=8}"
: "${DO_EVAL:=1}"
: "${GPU_ID:=${GPU:-0}}"

if [ ! -d "$DATASET_ROOT" ]; then
    echo "❌ ERROR: DATASET_ROOT 不存在: $DATASET_ROOT" >&2
    exit 1
fi

CSV="$RESULTS_DIR/${OUT_PREFIX}_bench.csv"
MD="$RESULTS_DIR/${OUT_PREFIX}_bench.md"
TMP="$RESULTS_DIR/${OUT_PREFIX}_logs"
mkdir -p "$TMP"
echo "case,views,long_side,pose_mode,sec,peak_mem_mb,base_mem_mb,gaussians,psnr,ssim" > "$CSV"

echo "🚀 [04] 批量推理 — $DATASET_ROOT"
echo "  📐 长边 $LONG_SIDE | 上限视角 $MAX_VIEWS | 位姿 $POSE_MODE"
echo "  💾 输出前缀 $OUT_PREFIX | 评测 $DO_EVAL"

for d in "$DATASET_ROOT"/*/; do
    [ -d "$d" ] || continue
    name=$(basename "$d")
    n=$(find "$d" -maxdepth 1 -type f \( -iname '*.jpg' -o -iname '*.jpeg' -o -iname '*.png' -o -iname '*.webp' \) | wc -l)
    if [ "$n" -lt "$MIN_VIEWS" ]; then
        echo ""
        echo "⏭️  $name 只有 $n 张 (< $MIN_VIEWS)，跳过"
        continue
    fi

    out_name="${OUT_PREFIX}_${name}"
    rm -rf "${RESULTS_DIR:?}/$out_name"   # 同名覆盖，保证指标与目录一一对应

    echo ""
    echo "🖼️  $name ($n 张)"
    export IMAGES_DIR="$d" POSE_MODE OUT_NAME="$out_name" LONG_SIDE MAX_VIEWS HTML=0
    # 计时 + 显存采样交给 python 包装器（shell 后台采样的竞态已踩过）
    python "$SCRIPT_DIR/bench_run.py" --gpu "$GPU_ID" --out "$TMP/bench_$name.json" -- \
        bash "$SCRIPT_DIR/02_run_inference.sh" > "$TMP/run_$name.log" 2>&1
    rc=$?
    grep -E "⏱️|✅ 高斯|📄 scene.ply" "$TMP/run_$name.log" | sed 's/^/  /'

    if [ $rc -ne 0 ]; then
        echo "  ❌ 推理失败 (exit $rc)，日志: $TMP/run_$name.log"
        tail -8 "$TMP/run_$name.log" | sed 's/^/     /'
        continue
    fi

    res="$RESULTS_DIR/$out_name"
    read -r sec peak base <<< "$(python - "$TMP/bench_$name.json" <<'PY'
import json, sys
d = json.load(open(sys.argv[1], encoding='utf-8'))
print(d.get('sec'), d.get('peak_mb'), d.get('base_mb'))
PY
)"

    if [ "$DO_EVAL" = "1" ]; then
        RESULT="$res" IMAGES_DIR="$d" MODE=rerender LONG_SIDE="$LONG_SIDE" \
            bash "$SCRIPT_DIR/03_eval_render.sh" > "$TMP/eval_$name.log" 2>&1
        [ $? -ne 0 ] && echo "  ⚠️ 渲染评测失败（$TMP/eval_$name.log）"
    fi

    read -r gauss psnr ssim <<< "$(python - "$res" <<'PY'
import json, sys
from pathlib import Path
d = Path(sys.argv[1])
st = json.loads((d / 'result.json').read_text()).get('stats', {})
m = d / 'eval' / 'metrics_rerender.json'
p = s = float('nan')
if m.exists():
    mm = json.loads(m.read_text())
    # render_check.py 写的是 psnr_mean/ssim_mean；兼容另一种写法
    p, s = mm.get('psnr_mean', mm.get('mean_psnr', p)), mm.get('ssim_mean', mm.get('mean_ssim', s))
print(f"{st.get('final_count', 0)} {p:.3f} {s:.4f}")
PY
)"
    echo "  ✅ 高斯 ${gauss} | 自洽 PSNR ${psnr} SSIM ${ssim}"
    echo "$name,$n,$LONG_SIDE,$POSE_MODE,$sec,$peak,$base,$gauss,$psnr,$ssim" >> "$CSV"
done

python - "$CSV" "$MD" <<'PY'
import csv, sys
from pathlib import Path
rows = list(csv.DictReader(open(sys.argv[1], encoding='utf-8')))
if not rows:
    sys.exit("（无有效结果）")
w = [("case", "物体"), ("views", "视角"), ("sec", "端到端(s)"),
     ("peak_mem_mb", "显存峰值(MB)"), ("gaussians", "高斯数"),
     ("psnr", "自洽PSNR"), ("ssim", "自洽SSIM")]
lines = ["| " + " | ".join(t for _, t in w) + " |", "|" + "---|" * len(w)]
for r in rows:
    cells = []
    for k, _ in w:
        v = r.get(k, "")
        if k == "gaussians" and v.isdigit():
            v = f"{int(v):,}"
        cells.append(v)
    lines.append("| " + " | ".join(cells) + " |")
txt = "\n".join(lines)
Path(sys.argv[2]).write_text(txt + "\n", encoding='utf-8', newline='\n')
print()
print(txt)
PY

echo ""
echo "🎉 Done. CSV: $CSV"

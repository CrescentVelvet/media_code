#!/usr/bin/env bash
# 06_nvs_holdout.sh — 对"少视角数据集"做真实 NVS 评测：用 N_INPUT 张重建，渲染留出的那几张。
#
# 为什么需要它：全视角输入的"自洽渲染"PSNR 天然高（高斯中心就是从那些像素反投影的），
# 只能证明属性学对 + 渲染器接对，不等于新视角能力。要量化跨视角必须留出。
#
# 做法（位姿全程同源，避免引入额外误差）：
#   1) 全量 K 张用 da3-pose 跑一次 → cameras.npz（RDNorm 空间，K 张都在）
#   2) 切出前 N_INPUT 张 + 它们的相机 → provided 模式重建（不重估位姿）
#   3) 用留出视角的相机渲染 → 与真图比 PSNR/SSIM
#
# 两处坑（都已实测踩过，别改回去）：
#   a) cameras.npz 里的 K 在 **processed 尺度**（长边 518），而 provided 入口把它当**原图像素**。
#      直接把 npz 的 K 喂进去会让视场角差 7 倍（3840/518），重建几何全错但自洽渲染仍然"看起来很对"。
#      所以喂之前必须把 fx/fy/cx/cy 乘 (原图长边 / LONG_SIDE)，底行 [0,0,1] 不动。
#   b) 重建输出空间与全量空间只差**纯尺度**（实测相机朝向夹角 0°，第一相机都在原点）。
#      别用 2 点 Umeyama 拟合 Sim(3)：2 个点只能约束 4 个自由度，解出来的 R 会绕连线乱转，
#      相机被转歪后渲染出来是一面白墙。只按基线长度比缩放目标相机中心即可。
#
# 用法：
#   DATASET_ROOT=/mnt/d/dataset/物体三图数据 bash 06_nvs_holdout.sh
#   DATASET_ROOT=... CASES="手表 鼠标" N_INPUT=2 bash 06_nvs_holdout.sh
#
# Env:
#   DATASET_ROOT=      数据集根（子目录 = 一个物体）；CASES 给了就不扫
#   CASES=             空格分隔的物体名，只跑这些
#   N_INPUT=2          输入视角数（≥2，否则无法用基线比定尺度）
#   N_TARGET=1         留出视角数（取序列末尾）
#   LONG_SIDE=518      长边像素
#   OUT_PREFIX=nvshold 输出目录前缀
#   REUSE_ALL=1        复用已存在的 <OUT_PREFIX>all_<case> 全量推理结果（省 20 s/物体）
set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/_env.sh"

: "${DATASET_ROOT:=/mnt/d/dataset/物体三图数据}"
: "${N_INPUT:=2}"
: "${N_TARGET:=1}"
: "${LONG_SIDE:=518}"
: "${OUT_PREFIX:=nvshold}"
: "${REUSE_ALL:=1}"

if [ "$N_INPUT" -lt 2 ]; then
    echo "❌ ERROR: N_INPUT 至少 2（单视角无法用基线比确定尺度）" >&2
    exit 1
fi

if [ -n "${CASES:-}" ]; then
    read -r -a case_list <<< "$CASES"
else
    case_list=()
    for d in "$DATASET_ROOT"/*/; do
        [ -d "$d" ] && case_list+=("$(basename "$d")")
    done
fi

CSV="$RESULTS_DIR/${OUT_PREFIX}_bench.csv"
echo "case,views_in,views_target,psnr,ssim,alpha,gaussians,sec" > "$CSV"

echo "🚀 [06] 少视角 NVS（$N_INPUT 输入 → $N_TARGET 留出）"
echo "  📐 长边 $LONG_SIDE | 输出前缀 $OUT_PREFIX"

for name in "${case_list[@]}"; do
    SRC="$DATASET_ROOT/$name"
    N=$(find "$SRC" -maxdepth 1 -type f \( -iname '*.jpg' -o -iname '*.jpeg' -o -iname '*.png' -o -iname '*.webp' \) 2>/dev/null | wc -l)
    if [ "$N" -lt $((N_INPUT + N_TARGET)) ]; then
        echo ""
        echo "⏭️  $name 只有 $N 张，不足 $((N_INPUT + N_TARGET))，跳过"
        continue
    fi
    echo ""
    echo "🖼️  $name ($N 张 → $N_INPUT 输入 / $N_TARGET 留出)"

    ALL="$RESULTS_DIR/${OUT_PREFIX}all_$name"
    if [ "$REUSE_ALL" = "1" ] && [ -f "$ALL/cameras.npz" ]; then
        echo "  ⏭️  复用全量推理: $ALL"
    else
        rm -rf "${RESULTS_DIR:?}/${OUT_PREFIX}all_$name"
        IMAGES_DIR="$SRC" POSE_MODE=da3-pose OUT_NAME="${OUT_PREFIX}all_$name" \
            LONG_SIDE="$LONG_SIDE" MAX_VIEWS=8 HTML=0 bash "$SCRIPT_DIR/02_run_inference.sh" \
            > "$RESULTS_DIR/${OUT_PREFIX}_all_$name.log" 2>&1 || {
            echo "  ❌ 全量推理失败（$RESULTS_DIR/${OUT_PREFIX}_all_$name.log）" >&2; continue; }
    fi

    D="$RESULTS_DIR/_${OUT_PREFIX}_split/$name"
    rm -rf "$D"; mkdir -p "$D/in" "$D/tgt"
    python - "$ALL/cameras.npz" "$SRC" "$D" "$LONG_SIDE" "$N_INPUT" "$N_TARGET" <<'PY'
import shutil, sys
from pathlib import Path
import numpy as np
from PIL import Image
cams_path, src, dst = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
long_side, n_in, n_tgt = int(sys.argv[4]), int(sys.argv[5]), int(sys.argv[6])
files = sorted(p for p in src.iterdir() if p.is_file() and p.suffix.lower() in {'.jpg', '.jpeg', '.png', '.webp'})
z = np.load(cams_path)
c2w, K = z["c2w"], z["intrinsics"]
in_files, tgt_files = files[:n_in], files[-n_tgt:]
if len(c2w) < len(files):
    sys.exit(f"❌ 全量相机数 {len(c2w)} < 图像数 {len(files)}")
with Image.open(files[0]) as im:
    ow, oh = im.size
ratio = max(ow, oh) / long_side      # 坑 a：processed 尺度 → 原图尺度
Ki = K.copy(); Ki[:, :2] *= ratio     # 只缩 fx/fy/cx/cy，底行保持 [0,0,1]
for p in in_files:
    shutil.copy(p, dst / "in" / p.name)
for p in tgt_files:
    shutil.copy(p, dst / "tgt" / p.name)
np.savez(dst / "in" / "cameras.npz", c2w=c2w[:n_in].astype(np.float32),
         intrinsics=Ki[:n_in].astype(np.float32))
np.savez(dst / "tgt" / "cameras.npz", c2w=c2w[-n_tgt:].astype(np.float32),
         intrinsics=K[-n_tgt:].astype(np.float32))
print(f"  ✂️  输入 {[p.name for p in in_files]} → 留出 {[p.name for p in tgt_files]} (K×{ratio:.3f})")
PY

    OUT="$RESULTS_DIR/${OUT_PREFIX}_$name"
    rm -rf "${RESULTS_DIR:?}/${OUT_PREFIX}_$name"
    IMAGES_DIR="$D/in" CAMERAS="$D/in/cameras.npz" POSE_MODE=provided OUT_NAME="${OUT_PREFIX}_$name" \
        LONG_SIDE="$LONG_SIDE" MAX_VIEWS=8 HTML=0 bash "$SCRIPT_DIR/02_run_inference.sh" \
        > "$RESULTS_DIR/${OUT_PREFIX}_run_$name.log" 2>&1 || {
        echo "  ❌ 重建失败（$RESULTS_DIR/${OUT_PREFIX}_run_$name.log）" >&2; continue; }
    grep -E "⏱️|✅ 高斯" "$RESULTS_DIR/${OUT_PREFIX}_run_$name.log" | sed 's/^/  /'

    # 坑 b：只按基线比缩放目标相机，不拟合旋转
    python - "$OUT/cameras.npz" "$D/in/cameras.npz" "$D/tgt/cameras.npz" "$D/tgt_scaled.npz" <<'PY'
import sys
import numpy as np
inp, ref, tgt, out = sys.argv[1:5]
ci = np.load(ref)["c2w"][:, :3, 3]
co = np.load(inp)["c2w"][:, :3, 3]
s = np.linalg.norm(co[0] - co[1]) / np.linalg.norm(ci[0] - ci[1])
z = np.load(tgt)
c2w = z["c2w"].astype(np.float32).copy()
c2w[:, :3, 3] *= s
np.savez(out, c2w=c2w, intrinsics=z["intrinsics"])
print(f"  🔗 尺度比 {s:.4f}")
PY

    python "$SCRIPT_DIR/render_check.py" --result "$OUT" --cameras "$D/tgt_scaled.npz" \
        --images-dir "$D/tgt" --out "$OUT/eval" --tag holdout --eps2d 0.01 2>&1 \
        | grep -E "\[0|mean" | sed 's/^/  /'

    python - "$OUT" "$name" "$N_INPUT" "$N_TARGET" >> "$CSV" <<'PY'
import json, sys
from pathlib import Path
d, name, nin, ntgt = Path(sys.argv[1]), sys.argv[2], sys.argv[3], sys.argv[4]
st = json.loads((d / 'result.json').read_text()).get('stats', {})
mm = json.loads((d / 'eval' / 'metrics_holdout.json').read_text())
pv = mm.get('per_view', {})
print(f"{name},{nin},{ntgt},{mm['psnr_mean']:.3f},{mm['ssim_mean']:.4f},,{st.get('final_count', 0)},")
PY
done

echo ""
echo "🎉 Done. CSV: $CSV"

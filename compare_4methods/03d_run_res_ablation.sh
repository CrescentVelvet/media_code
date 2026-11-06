#!/usr/bin/env bash
# R-3 训练分辨率消融：RES=2（720x960）vs RES=1（1439x1919），同一 init、同一 2k 步。
#
# 与历史实验的区别（这是本脚本存在的理由）：
#   历史上"训练分辨率"和"评测分辨率"是两个不同的量，却从没被分开测过：
#     训练 720x960 → 渲染/评测 378x504（94 取 npz 的 504 处理分辨率）
#   所以"人脸糊"无法归因。本轮把训练与渲染两个维度都钉死：
#     训练 2 档 × 渲染 3 档（378x504 / 756x1008 / 1439x1919）
#
# 关键：RENDER_SIZE 是**按该分辨率重新光栅化**（等比缩放内参），不是把低分辨率图放大。
#      所以三档各代表"该方法在这个交付分辨率下的真实产物"。
#
# 设计要点（对应 ERRATA 防再犯清单）：
#   A5 两组共享同一 init ckpt / 同一 colmap_scene / 同一 cameras.npz —— 唯一变量是 RES
#   B1 训练后必须验 xyz 真的动了（99c_check_trained.py），否则"跑完但没学"会静默通过
#   C2 跨分辨率比较必须传 --eval-size，否则两边在不同尺寸上算 PSNR（不可比）
#
# 用法（WSL）：
#   bash compare_4methods/03d_run_res_ablation.sh
# 只补跑某一档（已跑的组会因产物存在而跳过）：
#   RES_LIST=4 SKIP_TRAIN=1 bash compare_4methods/03d_run_res_ablation.sh
# 可选 env：
#   ITERS=2000     训练步数
#   RES_LIST="2 1" 要跑的训练分辨率因子（RES=1/2/4 → 1439x1919 / 720x960 / 360x480）
#   SRC=...        提供 init/场景/npz 的既有目录（默认 P1 米制目录）
#   BASE=...       输出根目录
#   SKIP_TRAIN=1   复用已有训练产物，只重渲/重评
set -o pipefail

CODE=/mnt/c/code/media_code
SRC="${SRC:-$HOME/output/da3_gs_results/佳佳抱枕_p1m}"
BASE="${BASE:-$HOME/output/da3_gs_results/佳佳抱枕_res_ab}"
IMAGES="$HOME/output/da3_results/_captures/佳佳抱枕/images"
ITERS="${ITERS:-2000}"
RES_LIST="${RES_LIST:-2 1}"
VIEWS=$(seq -s, 0 31)
SIZES=("378x504" "756x1008" "1439x1919")
INIT="$SRC/ffinit_ckpt/chkpnt_ffinit.pth"
SCENE="$SRC/colmap_scene"
NPZ="$SRC/cameras.npz"

LOG_FILE="$BASE/run.log"
mkdir -p "$BASE"
LOG() { echo "[$(date +%H:%M:%S)] $*"; }
die() { LOG "❌ $*"; exit 1; }
need() { [ -e "$1" ] || die "缺少产物：$1"; }

[ -f "$INIT" ] || die "缺 init ckpt: $INIT"
[ -d "$SCENE/images" ] || die "缺场景图像: $SCENE/images"
[ -f "$NPZ" ] || die "缺 cameras.npz: $NPZ"

# 峰值显存采样用（RES=1 有 OOM 风险，必须留证据；预注册 R-3.3）。
# ⚠️ 不要包成函数再用 $(...) 取 PID：$() 的子 shell 退出后采样进程不可靠。
LOG "配置：iters=$ITERS res=[$RES_LIST] sizes=${SIZES[*]} init=$INIT"
LOG "输出：$BASE"

# ── ① 训练各组（唯一变量 = RES） ─────────────────────────────────────────────
for R in $RES_LIST; do
  TAG="res$R"
  OUT="$BASE/final_$TAG"
  if [ "${SKIP_TRAIN:-0}" = "1" ] && [ -f "$OUT/point_cloud/iteration_$ITERS/point_cloud.ply" ]; then
    LOG "① 跳过训练 $TAG（产物已存在）"
    continue
  fi
  LOG "① 训练 $TAG  RES=$R  iters=$ITERS"
  MEMLOG="$BASE/mem_$TAG.csv"
  nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -l 5 > "$MEMLOG" 2>/dev/null &
  MPID=$!
  SCENE_DIR="$SCENE" OUT_DIR="$OUT" INIT_CKPT="$INIT" \
  ITERATIONS="$ITERS" RES="$R" DENSIFY=0 RENDER=0 CONDA_ENV=vggt_human \
    bash "$CODE/compare_4methods/03b_train_short.sh" > "$BASE/train_$TAG.log" 2>&1
  RC=$?
  kill "$MPID" 2>/dev/null
  if [ $RC -ne 0 ]; then
    tail -25 "$BASE/train_$TAG.log"
    die "训练失败 $TAG (rc=$RC) —— 若为 OOM，按 R-3.3 降级路径处理，不要静默改配置"
  fi
  tail -5 "$BASE/train_$TAG.log"
  MAXMEM=$(grep -E '^[0-9]+$' "$MEMLOG" 2>/dev/null | sort -n | tail -1)
  LOG "   峰值显存 ${MAXMEM:-?} MiB"
  need "$OUT/point_cloud/iteration_$ITERS/point_cloud.ply"
done

# ── ② B1 闸门：位置参数必须真的动过 ──────────────────────────────────────────
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate querysplat
for R in $RES_LIST; do
  TAG="res$R"
  LOG "② B1 闸门 $TAG"
  python "$CODE/compare_4methods/99c_check_trained.py" \
    "$INIT" "$BASE/final_$TAG/point_cloud/iteration_$ITERS/point_cloud.ply" \
    2>&1 | tee -a "$LOG_FILE" | tail -4
  [ "${PIPESTATUS[0]}" = "0" ] || die "B1 未通过：$TAG 训练没生效，停止（不解释、不继续）"
done

# ── ③ 每个训练产物在 3 档分辨率下渲染 + 评测 ─────────────────────────────────
for R in $RES_LIST; do
  TAG="res$R"
  PLY="$BASE/final_$TAG/point_cloud/iteration_$ITERS/point_cloud.ply"
  for SZ in "${SIZES[@]}"; do
    RD="$BASE/final_$TAG/render_$SZ"
    LOG "③ 渲染 $TAG @ $SZ"
    RENDER_SIZE="$SZ" bash "$CODE/compare_4methods/94_render.sh" \
      "$PLY" "$NPZ" "$RD" "$VIEWS" 2>&1 | tee -a "$LOG_FILE" | tail -2
    need "$RD/view_000.png"

    LOG "③ 评测 $TAG @ $SZ（--eval-size 与渲染尺寸一致，作为显式闸门）"
    python "$CODE/compare_4methods/99_eval_render_metrics.py" \
      --render "$RD" --ref "$IMAGES" --views "$VIEWS" \
      --eval-size "$SZ" --tag "res${R}_$SZ" \
      --out "$BASE/final_$TAG/metrics_eval$SZ.json" 2>&1 \
      | tee -a "$LOG_FILE" | grep -E "MEAN|✅"
  done
done

LOG "🎉 完成：$BASE"
LOG "汇总："
python - <<'PY'
import glob, json, os
base = os.path.expanduser("~/output/da3_gs_results/佳佳抱枕_res_ab")
print(f"{'组':<6}{'评测尺寸':<12}{'PSNR':>8}{'SSIM':>9}{'锐度':>9}{'ref锐度':>9}")
rows = []
for p in sorted(glob.glob(base + "/final_res*/metrics_eval*.json")):
    d = json.load(open(p))
    m = d["mean"]
    tag = os.path.basename(os.path.dirname(p)).replace("final_", "")
    refs = [r["sharp_ref"] for r in d["per_view"]]
    es = d.get("eval_size")
    sz = "x".join(map(str, es)) if es else "?"
    print(f"{tag:<6}{sz:<12}{m['psnr']:>8.2f}{m['ssim']:>9.4f}{m['sharp']:>9.1f}{sum(refs)/len(refs):>9.1f}")
    rows.append((tag, sz, m["psnr"]))
for sz in sorted({r[1] for r in rows}):
    g = {t: p for t, s, p in rows if s == sz}
    if "res1" in g and "res2" in g:
        print(f"  Δ(RES=1 − RES=2) @ {sz}: {g['res1'] - g['res2']:+.2f} dB")
PY

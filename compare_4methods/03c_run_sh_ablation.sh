#!/usr/bin/env bash
# P0-2 复测：前馈初值「DC-only」vs「保留 SH rest」，在**米制 + 短精调**下的 A/B。
#
# 为什么现在才做：P0 那次（+1.23 dB）是在 E-01 的错尺度下测的，不能沿用。
# 设计要点（对应 ERRATA.md 防再犯清单）：
#   A5 对照组/实验组共享**同一次前向**的相机与世界系 → 98 一次出 gs_ply(dc) + gs_ply_full
#   A2 两份 PLY 必须剪出同一批点（98 内有硬闸门，顶点数不等直接退出）
#   B1 训练后必须验 md5（iteration_0 vs iteration_N），否则"跑完了但没学"会静默通过
#   C2 只报"32 训练视角自一致性"，不冒充 NVS
#
# 判据用**产物是否存在**，不靠 grep 日志（grep 匹配不到会假装失败，且掩盖真实报错）。
#
# 用法（WSL）：
#   bash compare_4methods/03c_run_sh_ablation.sh                        # 主实验（原生 degree 时间表）
#   SKIP_FORWARD=1 SH_DEGREE_INIT=2 bash compare_4methods/03c_run_sh_ablation.sh   # 补充：SH 全程生效
# 可选 env：
#   ITERS=2000           训练步数
#   DENSIFY=0            0/1/2（与 P1 甜点一致，关）
#   BASE=...             输出根目录
#   SH_DEGREE_INIT=0     初始 active_sh_degree；非 0 时产物加 _d2 后缀，避免覆盖主实验
#   SKIP_FORWARD=1       复用 BASE 里已有的 PLY/npz，跳过前向与 colmap 场景重建
set -o pipefail

CODE=/mnt/c/code/media_code
BASE="${BASE:-$HOME/output/da3_gs_results/佳佳抱枕_sh_ab}"
IMAGES="$HOME/output/da3_results/_captures/佳佳抱枕/images"
ITERS="${ITERS:-2000}"
DENSIFY="${DENSIFY:-0}"
SHD="${SH_DEGREE_INIT:-0}"
SUF=""; [ "$SHD" != "0" ] && SUF="_d${SHD}"
VIEWS=$(seq -s, 0 31)
LOG_FILE="$BASE/run.log"
LOG() { echo "[$(date +%H:%M:%S)] $*"; }

die() { LOG "❌ $*"; exit 1; }
need() { [ -e "$1" ] || die "缺少产物：$1"; }

source "$HOME/miniconda3/etc/profile.d/conda.sh"
mkdir -p "$BASE"
LOG "配置：iters=$ITERS densify=$DENSIFY sh_degree_init=$SHD 后缀='$SUF' skip_forward=${SKIP_FORWARD:-0}"

# ── ① 一次前向，两个出口（dc / full-SH），米制 ────────────────────────────────
if [ "${SKIP_FORWARD:-0}" = "1" ]; then
  LOG "① 跳过前向（SKIP_FORWARD=1），复用 $BASE 里的 PLY/npz"
else
  LOG "① DA3 前向 + 双出口导出"
  ( conda activate da3_human && \
    DA3_MODEL_DIR=/mnt/d/wheel/da3_human_ms \
    IMAGES_DIR="$IMAGES" OUT_DIR="$BASE" \
    PROCESS_RES=504 REF_STRATEGY=first SAVE_FULL_SH=1 \
    python "$CODE/compare_4methods/98_da3_ff_init.py" ) 2>&1 | tee -a "$LOG_FILE" | tail -20
fi
need "$BASE/cameras.npz"
need "$BASE/gs_ply/0000.ply"
need "$BASE/gs_ply_full/0000.ply"

# ── ② COLMAP 场景（3DGS 要 images/ + sparse/0/） ──────────────────────────────
if [ "${SKIP_FORWARD:-0}" = "1" ] && [ -d "$BASE/colmap_scene/sparse/0" ]; then
  LOG "② 跳过 colmap 场景（已存在）"
else
  LOG "② 组装 COLMAP 场景"
  ( conda activate da3_human && \
    DA3_INFER_DIR="$BASE" SCENE_DIR="$BASE/colmap_scene" IMAGES_DIR="$IMAGES" \
    MAX_INIT_POINTS=20000 \
    python "$CODE/da3_human/build_da3_colmap_scene.py" ) 2>&1 | tee -a "$LOG_FILE" | tail -5
fi
need "$BASE/colmap_scene/sparse/0"
need "$BASE/colmap_scene/images"

# ── ③ 两份 ckpt：唯一的差别是 f_rest 是否为 0 ────────────────────────────────
for tag in dc full; do
  if [ "$tag" = "dc" ]; then PLY="$BASE/gs_ply/0000.ply"; OUT="$BASE/ffinit_ckpt_${tag}${SUF}"
  else PLY="$BASE/gs_ply_full/0000.ply"; OUT="$BASE/ffinit_ckpt_${tag}${SUF}"; fi
  LOG "③ PLY → ckpt ($tag$SUF)"
  ( conda activate querysplat && \
    PLY="$PLY" NPZ="$BASE/cameras.npz" OUT_DIR="$OUT" MIN_OPACITY=0.005 MAX_DIST=5.0 \
    SH_DEGREE_INIT="$SHD" \
    python "$CODE/compare_4methods/98b_ply_to_3dgs_ckpt.py" ) 2>&1 \
    | tee -a "$LOG_FILE" | tail -8
  need "$OUT/chkpnt_ffinit.pth"
done

# ── ④ 训练（两组同超参，只差 init） ──────────────────────────────────────────
for tag in dc full; do
  LOG "④ 训练 S_$tag$SUF  iters=$ITERS densify=$DENSIFY"
  SCENE_DIR="$BASE/colmap_scene" OUT_DIR="$BASE/final_${tag}${SUF}" \
  INIT_CKPT="$BASE/ffinit_ckpt_${tag}${SUF}/chkpnt_ffinit.pth" \
  ITERATIONS="$ITERS" DENSIFY="$DENSIFY" RENDER=0 CONDA_ENV=vggt_human \
    bash "$CODE/compare_4methods/03b_train_short.sh" > "$BASE/train_${tag}${SUF}.log" 2>&1
  tail -6 "$BASE/train_${tag}${SUF}.log"
  need "$BASE/final_${tag}${SUF}/point_cloud/iteration_$ITERS/point_cloud.ply"
done

# ── ⑤ md5 闸门（清单 B1）：训练后 PLY 必须与初值不同 ────────────────────────
for tag in dc full; do
  a=$(md5sum "$BASE/ffinit_ckpt_${tag}${SUF}/point_cloud/iteration_0/point_cloud.ply" | cut -d' ' -f1)
  b=$(md5sum "$BASE/final_${tag}${SUF}/point_cloud/iteration_$ITERS/point_cloud.ply" | cut -d' ' -f1)
  [ "$a" = "$b" ] && die "B1 未通过：S_$tag$SUF 训练后 PLY 与初值逐字节相同（参数没被训练）"
  LOG "✅ B1 通过 S_$tag$SUF：${a:0:8} → ${b:0:8}"
done

# ── ⑥ 渲染 + 评测 ────────────────────────────────────────────────────────────
for tag in dc full; do
  LOG "⑥ 渲染 S_$tag$SUF"
  bash "$CODE/compare_4methods/94_render.sh" \
    "$BASE/final_${tag}${SUF}/point_cloud/iteration_$ITERS/point_cloud.ply" \
    "$BASE/cameras.npz" "$BASE/final_${tag}${SUF}/render_all32" "$VIEWS" 2>&1 \
    | tee -a "$LOG_FILE" | tail -2
  LOG "⑥ 评测 S_$tag$SUF"
  ( conda activate querysplat && \
    python "$CODE/compare_4methods/99_eval_render_metrics.py" \
      --render "$BASE/final_${tag}${SUF}/render_all32" --ref "$IMAGES" --views "$VIEWS" \
      --tag "sh_ab_${tag}${SUF}" --out "$BASE/final_${tag}${SUF}/metrics_sh_ab.json" ) 2>&1 \
    | tee -a "$LOG_FILE" | tail -3
done

LOG "🎉 完成：$BASE"


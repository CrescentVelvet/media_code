#!/usr/bin/env bash
# hand_batch.sh — 批量跑 som_human 全链路：01 转换 → 01c COLMAP 精化 → 02 训练
#
# 背景：面向「一批手机采集包」（标准输入格式见 README「输入数据格式」节）。
# 2026-10-16 首次用于 /mnt/d/dataset/手部微动样例 的 10 个包（68~218 帧，合计 1382 帧）：
#   01 转换 10/10（46~101s，总 12min）｜01c 精化 10/10（128~767s，总 59.5min，全部帧注册）
#   ｜02 训练 10/10（10k + -r 2，总 100min，test PSNR 21.3~27.7）
#
# 用法:
#   bash som_human/hand_batch.sh <01|01c|02|all> [scene_filter(子串)]
#
# Env:
#   RAW_ROOT=/mnt/d/dataset/手部微动样例   采集包根目录（其下每个子目录 = 一个采集包）
#   PAR=1                                  并发数。⚠️ 02 训练每路显存 ~5GB，但链路 CPU 密集
#                                          （SoM 系数场/Adam/densify 记账 ∝N），2 路实测
#                                          吞吐仅 ~1.5×（单进程 7 it/s vs 单跑 19.6 it/s）。
#                                          01/01c 建议 PAR=1（COLMAP GPU SIFT 争用）。
#   FORCE=1                                忽略已有产物强制重跑
#   ITERATIONS=10000  EXTRA_TRAIN_ARGS="-r 2"
#   SCENE_MAP=<file>                       可选 TSV：<scene名>\t<采集包目录名>
#
# 命名：默认用下方 DEFAULT_MAP（UUID 前 8 位 + 拼音，保证纯 ASCII 且可读）；
#       未命中映射的包回退为 uuid 前 8 位。
# 幂等：每阶段产物存在则跳过（01: colmap_scene/sparse/0/images.txt；01c: colmap_refined/...；
#       02: model_colmap/point_cloud/iteration_<N>/point_cloud.ply）。
set -o pipefail

STAGE="${1:-all}"
FILTER="${2:-}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/_env.sh"

RAW_ROOT="${RAW_ROOT:-/mnt/d/dataset/手部微动样例}"
PAR="${PAR:-1}"
FORCE="${FORCE:-0}"
ITERATIONS="${ITERATIONS:-10000}"
ARGS="${EXTRA_TRAIN_ARGS:--r 2}"
LOG="${BATCH_LOG:-$HOME/output/som_human_results/_batch_logs}"
mkdir -p "$LOG"
SUMMARY="$LOG/summary.tsv"
[ -s "$SUMMARY" ] || printf "scene\tstage\trc\telapsed_s\tnote\n" > "$SUMMARY"

# 默认映射：scene<TAB>采集包目录名
DEFAULT_MAP='hand01_jiajia_baozhen	01e954de1b8a4cde9315c976f4dcb964-佳佳抱枕
hand02_sanren_jucan	10e3ec6291c04bedaad735309e4bc43b-三人聚餐
hand03_nansheng_shinei	3a8b3cc746304f49b9e3275e36aa9374-男生室内
hand04_jiajia_dun	4ce21aad7af7470c89335dc893215c82-佳佳蹲
hand05_nansheng_erlangtui	7fc5f4de2fe04157bc76cc26425fda70-男生二郎腿
hand06_nvsheng_xiezi	804efb01efe94cb9be68e7d90d97101c-女生写字
hand07_nansheng_shouji	83112686acbe49b9809fbfe9358bcd6f-男生手机
hand08_nvsheng_tanqin	9a7ffcda6a704fc39b5181f564561da7-女生弹琴
hand09_nvsheng_niutou	d9b953493b88456380ab9ee2fe31bc18-女生扭头
hand10_shuangren_zhaoshou	f5cf3ba0d4014ce79a82fcaf7ecae053-双人招手'

MAP="$DEFAULT_MAP"
[ -n "${SCENE_MAP:-}" ] && [ -f "$SCENE_MAP" ] && MAP="$(cat "$SCENE_MAP")"

echo "🚀 [hand_batch] stage=$STAGE  root=$RAW_ROOT  PAR=$PAR  iter=$ITERATIONS  args='$ARGS'"

run_stage() {   # $1=scene $2=stage $3=raw_dir
  local SC="$1" ST="$2" RAW="$3" T0 RC
  case "$ST" in
    01)  T0=$(date +%s)
         SCENE_NAME="$SC" RAW_DIR="$RAW" bash "$SCRIPT_DIR/01_convert_capture.sh" \
             > "$LOG/$SC.$ST.log" 2>&1 ;;
    01c) T0=$(date +%s)
         SCENE_NAME="$SC" bash "$SCRIPT_DIR/01c_colmap_refine.sh" \
             > "$LOG/$SC.$ST.log" 2>&1 ;;
    02)  T0=$(date +%s)
         SCENE_NAME="$SC" SOURCE_PATH="$RESULTS_DIR/$SC/colmap_refined" \
         MODEL_PATH="$RESULTS_DIR/$SC/model_colmap" ITERATIONS="$ITERATIONS" \
         EXTRA_TRAIN_ARGS="$ARGS" bash "$SCRIPT_DIR/02_train.sh" \
             > "$LOG/$SC.$ST.log" 2>&1 ;;
  esac
  RC=$?
  local EL=$(( $(date +%s) - T0 ))
  printf "%s\t%s\t%d\t%d\t%s\n" "$SC" "$ST" "$RC" "$EL" "$([ $RC -eq 0 ] && echo OK || echo FAIL)" >> "$SUMMARY"
  echo "[$SC][$ST] rc=$RC ${EL}s"
  [ $RC -ne 0 ] && tail -6 "$LOG/$SC.$ST.log"
  return $RC
}

run_scene() {   # 一个采集包的完整链路（按 STAGE 执行对应阶段）
  local SC="$1" RAW="$2"
  echo "########## $SC  ($(basename "$RAW")) ##########"
  case "$STAGE" in
    01|all)
      if [ "$FORCE" = 1 ] || [ ! -f "$RESULTS_DIR/$SC/colmap_scene/sparse/0/images.txt" ]; then
        run_stage "$SC" 01 "$RAW" || return 1
      else echo "  [skip] 01 已存在"; fi ;;
  esac
  case "$STAGE" in
    01c|all)
      if [ "$FORCE" = 1 ] || [ ! -f "$RESULTS_DIR/$SC/colmap_refined/sparse/0/images.txt" ]; then
        run_stage "$SC" 01c "$RAW" || return 1
      else echo "  [skip] 01c 已存在"; fi ;;
  esac
  case "$STAGE" in
    02|all)
      if [ "$FORCE" = 1 ] || [ ! -f "$RESULTS_DIR/$SC/model_colmap/point_cloud/iteration_$ITERATIONS/point_cloud.ply" ]; then
        run_stage "$SC" 02 "$RAW" || return 1
      else echo "  [skip] 02 已存在"; fi ;;
  esac
  return 0
}

running=0
while IFS=$'\t' read -r SC RAWNAME; do
  SC="${SC%$'\r'}"; RAWNAME="${RAWNAME%$'\r'}"
  [ -z "$SC" ] && continue
  [ -n "$FILTER" ] && case "$SC" in *"$FILTER"*) ;; *) continue ;; esac
  RAW="$RAW_ROOT/$RAWNAME"
  [ -d "$RAW" ] || { echo "⚠️ 采集包不存在，跳过: $RAW"; continue; }
  run_scene "$SC" "$RAW" &
  running=$((running + 1))
  if [ "$running" -ge "$PAR" ]; then wait -n; running=$((running - 1)); fi
done <<< "$MAP"
wait

echo; echo "########## 汇总 ##########"
column -t -s$'\t' "$SUMMARY" 2>/dev/null || cat "$SUMMARY"
echo
echo "ply 清单：bash som_human/hand_batch.sh 02（产出在 \$RESULTS_DIR/<scene>/model_colmap/point_cloud/iteration_$ITERATIONS/point_cloud.ply）"

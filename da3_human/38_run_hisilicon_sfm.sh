#!/usr/bin/env bash
# 38_run_hisilicon_sfm.sh — 用每例最终 COLMAP SfM 相机运行 DA3 高斯重建。
# 输出 CASE/depthanything3_sfm.ply；不覆盖 RGB-only 或 stage1 相机版结果。
set -o pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/_env.sh"

: "${ROOT:=/mnt/d/dataset/测试数据/海思端侧重建拍摄数据}"
: "${CASE_FILTER:=}"
: "${PROCESS_RES:=504}"
: "${FORCE:=0}"
: "${OUT_ROOT:=}"
: "${PLY_NAME:=depthanything3_sfm.ply}"
: "${OUT_SUBDIR:=depthanything3_sfm}"
export ROOT CASE_FILTER PROCESS_RES FORCE DA3_MODEL_DIR OUT_ROOT PLY_NAME OUT_SUBDIR

if [ ! -f "$DA3_MODEL_DIR/model.safetensors" ]; then
    echo "❌ 原生 DA3 权重缺失: $DA3_MODEL_DIR" >&2
    exit 1
fi
if [ ! -d "$ROOT" ]; then
    echo "❌ 数据集不存在: $ROOT" >&2
    exit 1
fi
LOG_DIR="$ROOT"
if [ -n "$OUT_ROOT" ]; then
    LOG_DIR="$OUT_ROOT"
    mkdir -p "$OUT_ROOT" "$LOG_DIR/logs"
elif [ -n "$CASE_FILTER" ]; then
    LOG_DIR="$ROOT/$CASE_FILTER/depthanything3_sfm"
fi
mkdir -p "$LOG_DIR/logs"
echo "🚀 [38] 使用最终 SfM 相机的原生 DA3 重建"
echo "  🖼️ 输入: $ROOT（每例 colmap_sfm_results/sparse/0）"
echo "  🤖 权重: $DA3_MODEL_DIR"
echo "  📐 process_res=$PROCESS_RES, 去畸变 + 完整 Sim(3) + 不裁剪 + 全 SH"
echo "  💾 输出: CASE/depthanything3_sfm.ply"
echo "  📜 日志: $LOG_DIR/logs/da3_sfm_inference.log"
python "$SCRIPT_DIR/38_run_hisilicon_sfm.py" > "$LOG_DIR/logs/da3_sfm_inference.log" 2>&1
RC=$?
tail -n 30 "$LOG_DIR/logs/da3_sfm_inference.log"
if [ "$RC" -ne 0 ]; then
    echo "❌ 有案例失败，详见日志与各案例 summary" >&2
    tail -n 40 "$LOG_DIR/logs/da3_sfm_inference.log" >&2
    exit "$RC"
fi
echo "🎉 Done."

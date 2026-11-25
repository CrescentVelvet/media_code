#!/usr/bin/env bash
# 36_run_hisilicon_cameras.sh — 使用原始 stage1 相机的海思 DA3 高斯重建。
# 输出 CASE/depthanything3_cameras.ply，不覆盖 RGB-only 的 depthanything3.ply。
set -o pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/_env.sh"

: "${ROOT:=/mnt/d/dataset/测试数据/海思端侧重建拍摄数据}"
: "${CASE_FILTER:=}"
: "${PROCESS_RES:=504}"
: "${FORCE:=0}"
export ROOT CASE_FILTER PROCESS_RES FORCE DA3_MODEL_DIR

if [ ! -f "$DA3_MODEL_DIR/model.safetensors" ]; then
    echo "❌ 原生 DA3 权重缺失: $DA3_MODEL_DIR" >&2
    exit 1
fi
if [ ! -d "$ROOT" ]; then
    echo "❌ 数据集不存在: $ROOT" >&2
    exit 1
fi
LOG_DIR="$ROOT"
if [ -n "$CASE_FILTER" ]; then
    LOG_DIR="$ROOT/$CASE_FILTER/depthanything3_cameras"
fi
mkdir -p "$LOG_DIR/logs"
echo "🚀 [36] 输入 stage1 相机的原生 DA3 重建"
echo "  🖼️ 输入: $ROOT（每案例 10 图 + 原始 JSON 位姿/内参）"
echo "  🤖 权重: $DA3_MODEL_DIR"
echo "  📐 process_res=$PROCESS_RES, 输入世界坐标 + 不裁剪 + 全 SH"
echo "  💾 输出: CASE/depthanything3_cameras.ply"
echo "  📜 日志: $LOG_DIR/logs/da3_inference.log"
python "$SCRIPT_DIR/36_run_hisilicon_cameras.py" > "$LOG_DIR/logs/da3_inference.log" 2>&1
RC=$?
tail -n 30 "$LOG_DIR/logs/da3_inference.log"
if [ "$RC" -ne 0 ]; then
    echo "❌ 有案例失败，详见日志与各案例 summary" >&2
    exit "$RC"
fi
echo "🎉 Done."

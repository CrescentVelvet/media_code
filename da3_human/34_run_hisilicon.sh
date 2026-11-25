#!/usr/bin/env bash
# 34_run_hisilicon.sh — 海思十视角案例的原生 DA3 批量高斯重建。
# 输出：每个 CASE/depthanything3.ply；相机/深度/日志在 CASE/depthanything3/。
# 输入复用已验收的 splatanything/images，不使用其中相机或几何结果。
set -o pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/_env.sh"

: "${ROOT:=/mnt/d/dataset/测试数据/海思端侧重建拍摄数据}"
: "${CASE_FILTER:=}"
: "${PROCESS_RES:=504}"
: "${FORCE:=0}"
: "${OUT_SUBDIR:=depthanything3}"
export ROOT CASE_FILTER PROCESS_RES FORCE OUT_SUBDIR DA3_MODEL_DIR

if [ ! -f "$DA3_MODEL_DIR/model.safetensors" ]; then
    echo "❌ 原生 DA3 权重缺失: $DA3_MODEL_DIR" >&2
    exit 1
fi
if [ ! -d "$ROOT" ]; then
    echo "❌ 数据集不存在: $ROOT" >&2
    exit 1
fi
# 指定单案例时明确检查输出父目录；批量使用数据集根本身。
LOG_DIR="$ROOT"
if [ -n "$CASE_FILTER" ]; then
    LOG_DIR="$ROOT/$CASE_FILTER/$OUT_SUBDIR"
fi
mkdir -p "$LOG_DIR/logs"
echo "🚀 [34] 原生 DA3 海思批量重建"
echo "  🖼️ 输入: $ROOT（每案例 10 张已登记 RGB）"
echo "  🤖 权重: $DA3_MODEL_DIR"
echo "  📐 process_res=$PROCESS_RES, 不裁剪 + 全 SH + 米制"
echo "  💾 输出: CASE/depthanything3.ply"
echo "  📜 日志: $LOG_DIR/logs/da3_inference.log"
python "$SCRIPT_DIR/34_run_hisilicon.py" > "$LOG_DIR/logs/da3_inference.log" 2>&1
RC=$?
tail -n 25 "$LOG_DIR/logs/da3_inference.log"
if [ "$RC" -ne 0 ]; then
    echo "❌ 有案例失败，详见日志与各案例 summary" >&2
    exit "$RC"
fi
echo "🎉 Done."

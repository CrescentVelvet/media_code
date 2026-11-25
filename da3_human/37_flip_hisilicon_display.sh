#!/usr/bin/env bash
# 输入位姿版PLY补绕世界X轴180度的显示变换；不改原两版重建。
set -o pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/_env.sh"
: "${ROOT:=/mnt/d/dataset/测试数据/海思端侧重建拍摄数据}"
: "${CASE_FILTER:=}"
: "${FORCE:=0}"
export ROOT CASE_FILTER FORCE
LOG_DIR="$ROOT"
if [ -n "$CASE_FILTER" ]; then
    LOG_DIR="$ROOT/$CASE_FILTER/depthanything3_cameras_upright"
fi
mkdir -p "$LOG_DIR/logs"
python "$SCRIPT_DIR/37_flip_hisilicon_display.py" > "$LOG_DIR/logs/display_transform.log" 2>&1
RC=$?
tail -n 25 "$LOG_DIR/logs/display_transform.log"
exit "$RC"

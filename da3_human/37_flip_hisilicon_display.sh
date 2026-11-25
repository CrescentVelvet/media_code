#!/usr/bin/env bash
# 一次性收敛：已转正PLY替换相机版原文件；先备份，不重新推理。
set -o pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/_env.sh"
: "${ROOT:=/mnt/d/dataset/测试数据/海思端侧重建拍摄数据}"
: "${BACKUP_ROOT:=/mnt/d/output/_backup/da3_cameras_pre_rotation_20261126}"
: "${CASE_FILTER:=}"
export ROOT BACKUP_ROOT CASE_FILTER
LOG_DIR="/mnt/d/output/_backup"
: "${PYTHON_BIN:=$(command -v python || true)}"
if [ -z "$PYTHON_BIN" ]; then
    PYTHON_BIN="$HOME/miniconda3/envs/da3_human/bin/python"
fi
if [ ! -x "$PYTHON_BIN" ]; then
    echo "❌ 未找到可用Python，请显式设置 PYTHON_BIN" >&2
    exit 1
fi
mkdir -p "$LOG_DIR/logs"
"$PYTHON_BIN" "$SCRIPT_DIR/37_flip_hisilicon_display.py" > "$LOG_DIR/logs/da3_cameras_in_place_rotation.log" 2>&1
RC=$?
tail -n 25 "$LOG_DIR/logs/da3_cameras_in_place_rotation.log"
exit "$RC"

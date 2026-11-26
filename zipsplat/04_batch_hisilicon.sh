#!/usr/bin/env bash
# 04_batch_hisilicon.sh — 海思端侧拍摄包：每个 images 子目录十图 → zipsplat.ply
set -o pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export ZIPSPLAT_RESULTS_DIR="${ZIPSPLAT_RESULTS_DIR:-/mnt/d/output/results/zipsplat_results/hisilicon_10v}"
source "$SCRIPT_DIR/_env.sh"

export DATASET_ROOT="${DATASET_ROOT:-/mnt/d/dataset/测试数据/海思端侧重建拍摄数据}"
# 默认将 zipsplat.ply 放回各场景 images 同级目录；原始图像/相机不动。
export FINAL_DIR="${FINAL_DIR:-$DATASET_ROOT}"
export RESULTS_DIR="$ZIPSPLAT_RESULTS_DIR"
export MAX_VIEWS=10
export BA="${BA:-1}"
export MIN_FREE_MIB="${MIN_FREE_MIB:-14000}"
export WAIT_SECONDS="${WAIT_SECONDS:-0}"

python "$SCRIPT_DIR/batch_hisilicon.py"
RC=$?
if [ $RC -ne 0 ]; then echo "❌ 批量重建未全部完成 exit=$RC，见 $RESULTS_DIR/batch_summary.json" >&2; exit $RC; fi
echo "🎉 全部完成：$RESULTS_DIR/batch_summary.json"

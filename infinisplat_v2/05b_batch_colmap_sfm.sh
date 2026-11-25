#!/usr/bin/env bash
# 05b_batch_colmap_sfm.sh — 对每个 case/images 独立运行完整 COLMAP SfM。
# Env: DATASET_ROOT, CASE_REGEX, FORCE, MAX_CASES, EXPECTED_IMAGES, COLMAP_ENV,
#      ANALYSIS_PYTHON, USE_GPU, OUTPUT_DIRNAME.
set -o pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
COLMAP_ENV="${COLMAP_ENV:-colmap}"
COLMAP_HOME="${COLMAP_HOME:-$HOME/miniconda3/envs/$COLMAP_ENV}"
COLMAP_BIN="$COLMAP_HOME/bin/colmap"
ANALYSIS_PYTHON="${ANALYSIS_PYTHON:-$COLMAP_HOME/bin/python}"
export LD_LIBRARY_PATH="$COLMAP_HOME/lib:${LD_LIBRARY_PATH:-}"
if [ ! -x "$COLMAP_BIN" ] || [ ! -x "$ANALYSIS_PYTHON" ]; then
    echo "❌ ERROR: 需要 $COLMAP_ENV 环境中的 colmap 和 Python。" >&2
    exit 1
fi
for cmd in feature_extractor exhaustive_matcher mapper bundle_adjuster model_converter; do
    if ! "$COLMAP_BIN" "$cmd" -h > /dev/null 2>&1; then
        echo "❌ ERROR: COLMAP 命令不可用: $cmd（检查 LD_LIBRARY_PATH）。" >&2
        exit 1
    fi
done
exec "$ANALYSIS_PYTHON" "$SCRIPT_DIR/run_colmap_sfm.py"

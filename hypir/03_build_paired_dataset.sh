#!/usr/bin/env bash
# 03b_build_paired_dataset.sh — 由 HQ/LQ 两个人脸裁剪文件夹构建「配对训练 parquet」。
#
# 作用：把 HQ(高质量) 和 LQ(低质量) 两个文件夹按「同名文件」配对，生成一张
#       parquet 表（列：hq_path, lq_path, prompt），供 PairedFaceDataset 读取。
#       配对依据是文件名完全相同——crop_faces_paired.py 已保证 hq/ 与 lq/ 同名。
#
# 必填：
#   HQ_DIR=/.../ppr10k_faces_20260703/hq   LQ_DIR=/.../ppr10k_faces_20260703/lq
# 常用覆盖：
#   PARQUET_OUT=/.../paired.parquet   输出路径（默认放在 HQ_DIR 同级）
#   PROMPT=""                         每张图的文本提示（""=空文本训练，HYPIR 默认）
#   MIN_SIDE=0                        丢弃 LQ 短边小于该像素数的配对（0=不丢弃）
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/_env.sh"   # 激活 conda 环境、设代理/CA、按 GPU=N 选卡

: "${HQ_DIR:?set HQ_DIR (e.g. .../ppr10k_faces_20260703/hq)}"
: "${LQ_DIR:?set LQ_DIR (e.g. .../ppr10k_faces_20260703/lq)}"
# 默认把 parquet 输出到 HQ_DIR 的同级目录，文件名 hypir_paired.parquet
PARQUET_OUT="${PARQUET_OUT:-$(dirname "$HQ_DIR")/hypir_paired.parquet}"
PARQUET_OUT="$(mkdir -p "$(dirname "$PARQUET_OUT")" && cd "$(dirname "$PARQUET_OUT")" && pwd)/$(basename "$PARQUET_OUT")"

# 构表只需 polars + pillow；缺了就现装（带公司代理的 trusted-host 兜底）。
if ! python -c "import polars, PIL" 2>/dev/null; then
    echo "--- installing polars pillow ---"
    pip install --trusted-host pypi.org --trusted-host pypi.python.org \
        --trusted-host files.pythonhosted.org --timeout 600 --retries 10 polars pillow
fi

export HQ_DIR LQ_DIR PARQUET_OUT
python "$SCRIPT_DIR/build_paired_dataset.py"

echo "=== [03b] Done. Paired parquet: $PARQUET_OUT ==="
echo "    Next: PARQUET_PATH=$PARQUET_OUT bash $SCRIPT_DIR/04_train_paired.sh"

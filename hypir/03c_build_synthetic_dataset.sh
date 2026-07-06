#!/usr/bin/env bash
# 03c_build_synthetic_dataset.sh — 只输入 HQ 文件夹 -> parquet，配 HYPIR 官方在线退化训练。
#
# 与 03b(真实 LQ+HQ 配对) 对照：这里只给 HQ，训练时由 RealESRGANDataset +
# RealESRGANBatchTransform 在线合成 LQ（HYPIR 默认退化：blur/sinc/noise/jpeg
# 两阶段，每 epoch 随机刷新），不存 LQ 文件、增强更多样、省盘。
#
# 产出 parquet 列：image_path(绝对) + prompt。RealESRGANDataset 按 image_path_key/
# prompt_key 读取（官方 sd2_train.yaml 默认即这两个键）。
#
# 必填：HQ_DIR=/.../hq
# 常用：PARQUET_OUT=/.../synthetic.parquet  PROMPT=""  (空文本训练, HYPIR 默认)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/_env.sh"   # 激活 conda、设代理/CA、按 GPU=N 选卡

: "${HQ_DIR:?set HQ_DIR (e.g. .../ppr10k_faces/hq)}"
# 默认 parquet 输出到 HQ_DIR 同级目录
PARQUET_OUT="${PARQUET_OUT:-$(dirname "$HQ_DIR")/hypir_synthetic.parquet}"
PARQUET_OUT="$(mkdir -p "$(dirname "$PARQUET_OUT")" && cd "$(dirname "$PARQUET_OUT")" && pwd)/$(basename "$PARQUET_OUT")"

# 建表只需 polars + pillow；缺了就现装（带公司代理的 trusted-host 兜底）。
if ! python -c "import polars, PIL" 2>/dev/null; then
    echo "--- installing polars pillow ---"
    pip install --trusted-host pypi.org --trusted-host pypi.python.org \
        --trusted-host files.pythonhosted.org --timeout 600 --retries 10 polars pillow
fi

export HQ_DIR PARQUET_OUT PROMPT="${PROMPT:-}"
python "$SCRIPT_DIR/build_synthetic_dataset.py"

echo "=== [03c] Done. HQ-only parquet: $PARQUET_OUT ==="
echo "    Next (官方默认·在线退化·从零训): PARQUET_PATH=$PARQUET_OUT CROP_TYPE=random bash $SCRIPT_DIR/04_train.sh"

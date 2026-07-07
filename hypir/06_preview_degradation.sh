#!/usr/bin/env bash
# 06_preview_degradation.sh — 预览 HYPIR 合成退化：HQ -> LQ。
#
# 输入 HQ 文件夹，用官方 RealESRGANDataset(生成核) + RealESRGANBatchTransform
# (两阶段 blur/sinc/noise/jpeg) 合成退化 LQ 并存盘，供肉眼检查训练时在线合成的
# 退化效果。退化参数完全取自官方 configs/sd2_train.yaml（无复制）。
#
# 关键：queue_size 强制 0——跳过训练池，直接返回当前 HQ 的 LQ（否则满 256 后会
# 返回随机缓存样本，不是当前 HQ 的）。HQ 任意尺寸，resize 到 OUT_SIZE(512) 再退化。
#
# 必填：HQ_DIR=/.../hq
# 常用：LQ_OUT=/.../lq_preview  NUM_PER_IMAGE=4(每张 HQ 出 4 种随机退化)  DEVICE=cuda
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/_env.sh"   # 激活 conda、设代理/CA、按 GPU=N 选卡

: "${HQ_DIR:?set HQ_DIR (e.g. .../ppr10k_faces_20260703/hq)}"
LQ_OUT="${LQ_OUT:-$(dirname "$HQ_DIR")/lq_preview}"

# 退化预览只需 polars + pillow + omegaconf + torch；缺了就现装。
if ! python -c "import polars, PIL, omegaconf, torch" 2>/dev/null; then
    echo "--- installing polars pillow omegaconf ---"
    pip install --trusted-host pypi.org --trusted-host pypi.python.org \
        --trusted-host files.pythonhosted.org --timeout 600 --retries 10 polars pillow omegaconf
fi

export HQ_DIR LQ_OUT
export OUT_SIZE="${OUT_SIZE:-512}" SEED="${SEED:-231}"
export NUM_PER_IMAGE="${NUM_PER_IMAGE:-1}" DEVICE="${DEVICE:-cpu}"

python "$SCRIPT_DIR/preview_degradation.py"

echo "=== [06] Done. LQ preview: $LQ_OUT ==="
echo "    对比：把 $LQ_OUT/*.png 和原 HQ 并排看，即可判断退化强度是否合适。"

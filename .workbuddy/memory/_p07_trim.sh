#!/usr/bin/env bash
# 验证训练：只用垂手段 105 帧重建 p07，对照原 160 帧版本
set -u
R=/home/velvet/output/som_human_results/hand10_shuangren_zhaoshou
SOURCE_PATH="$R/colmap_trim105/persons/p07" \
MODEL_PATH="$R/model_p07_trim" \
FG_MASK_DIR="$R/colmap_trim105/persons/p07/masks" \
FG_INIT_FILTER=1 \
ITERATIONS=10000 \
MAX_GAUSSIANS=600000 \
EXTRA_TRAIN_ARGS="-r 2" \
bash /mnt/c/code/media_code/som_human/02_train.sh
echo "TRAIN_RC=$?"

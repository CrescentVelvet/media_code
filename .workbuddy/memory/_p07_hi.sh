#!/usr/bin/env bash
# 一次性对照实验：p07 提高容量（全分辨率 + 30k iter），看手部细节是否回来
set -u
cd /c/code/media_code 2>/dev/null || cd /mnt/c/code/media_code
R=/home/velvet/output/som_human_results/hand10_shuangren_zhaoshou
SOURCE_PATH="$R/colmap_refined/persons/p07" \
MODEL_PATH="$R/model_p07_hi" \
FG_MASK_DIR="$R/colmap_refined/persons/p07/masks" \
FG_INIT_FILTER=1 \
ITERATIONS=30000 \
MAX_GAUSSIANS=600000 \
SAVE_ITERATIONS="10000 20000 30000" \
TEST_ITERATIONS="10000 30000" \
EXTRA_TRAIN_ARGS="-r 1" \
bash /mnt/c/code/media_code/som_human/02_train.sh
echo "TRAIN_RC=$?"

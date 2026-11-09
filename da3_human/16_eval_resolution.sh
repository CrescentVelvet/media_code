#!/usr/bin/env bash
# 16_eval_resolution.sh — 用「各自分辨率自己的相机」做 in-sample 评估，比较分辨率收益。
# 为什么要自洽：不同 process_res 下 DA3 估计的 FOV 不同（504p fx/W=0.9780 vs 672p 0.9566，
# 差 2.2%），拿 504p 的相机去渲染 672p 的高斯会因视锥不匹配白丢 ~2.9 dB（实测）。
set +u
R=/home/velvet/output/da3_human_results
SCRIPT=/mnt/c/code/media_code/da3_human/07_eval_da3_gs.py

source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate vggt_human || { echo "conda activate 失败"; exit 1; }

for D in "$@"; do
  echo "=============================================================="
  echo "=== $D"
  echo "=============================================================="
  PLY="$R/$D/gs_max/0000.ply" NPZ="$R/$D/cameras.npz" SF=1.0 \
    python "$SCRIPT" eval
done

#!/usr/bin/env bash
# AUDIT-9 一键：先导官方口径 npz（da3_human），再用 gsplat 复刻渲染（querysplat）
#
# 用法：bash compare_4methods/audit_9_official_render.sh
set -o pipefail
B=/home/velvet/output/da3_gs_results
SC="$B/佳佳抱枕_p1m"
NPZ="$SC/official_dump.npz"
OUT="$SC/audit_official"

mkdir -p "$OUT"

echo "================ 阶段 1/2：导出官方口径高斯（da3_human）================"
if [ -f "$NPZ" ] && [ "${FORCE_DUMP:-0}" != "1" ]; then
  echo "已存在 $NPZ（FORCE_DUMP=1 可强制重跑）"
else
  DA3_MODEL_DIR="${DA3_MODEL_DIR:-/mnt/d/wheel/da3_human_ms}" \
  IMAGES_DIR="${IMAGES_DIR:-/home/velvet/output/da3_results/_captures/佳佳抱枕/images}" \
  OUT_NPZ="$NPZ" PROCESS_RES="${PROCESS_RES:-504}" REF_STRATEGY="${REF_STRATEGY:-first}" \
    /home/velvet/miniconda3/envs/da3_human/bin/python \
    "$(dirname "$0")/audit_9a_dump_official.py" || exit 1
fi

echo
echo "================ 阶段 2/2：复刻官方 render_3dgs（querysplat）================"
ENVS=/home/velvet/miniconda3/envs/querysplat
if [ -f /home/velvet/miniconda3/etc/profile.d/conda.sh ]; then
  # shellcheck disable=SC1090
  source /home/velvet/miniconda3/etc/profile.d/conda.sh
  conda activate "$ENVS"
fi
export CUDA_HOME="$ENVS"
export CPATH="$ENVS/targets/x86_64-linux/include"
export LIBRARY_PATH="$ENVS/targets/x86_64-linux/lib"

NPZ="$NPZ" OUT="$OUT" VIEWS="${VIEWS:-0,8,16}" OLD_NPZ="${OLD_NPZ:-$SC/cameras.npz}" \
  exec python "$(dirname "$0")/audit_9b_render_official.py"

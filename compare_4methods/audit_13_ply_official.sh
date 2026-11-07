#!/usr/bin/env bash
# AUDIT-13 一键：I-01/I-02 归位 —— 把「PLY 交付口径」抬回官方水平
#
# 阶段 1：da3_human 跑一次前向，导出 2x2 组合（裁剪/不裁剪 × DC/全SH）
# 阶段 2：querysplat 用 gsplat 逐档渲染 32 视角，与内存官方口径对账
#
# 用法：
#   bash compare_4methods/audit_13_ply_official.sh
#   SKIP_FORWARD=1 bash ...            # 复用已有 PLY
#   VIEWS=0,8,16 bash ...              # 只跑 3 个视角（快）
set -o pipefail
B=/home/velvet/output/da3_gs_results
SC_REF="$B/佳佳抱枕_p1m"        # 参照物：cameras.npz / official_dump.npz（audit_9 产的）
SC_NEW="$B/佳佳抱枕_p1m_off"    # 本次产物（不覆盖参照物）
OUT="$SC_NEW/official_export"
CODE=/mnt/c/code/media_code
PY_DA3=/home/velvet/miniconda3/envs/da3_human/bin/python
ENVS=/home/velvet/miniconda3/envs/querysplat

mkdir -p "$OUT"

echo "================ 阶段 1/2：一次前向，导出 2x2 组合（da3_human）================"
if [ "${SKIP_FORWARD:-0}" = "1" ] && [ -f "$OUT/i12_fullpts_sh/0000.ply" ]; then
  echo "复用已有 PLY（SKIP_FORWARD=1）"
else
  DA3_MODEL_DIR="${DA3_MODEL_DIR:-/mnt/d/wheel/da3_human_ms}" \
  IMAGES_DIR="${IMAGES_DIR:-/home/velvet/output/da3_results/_captures/佳佳抱枕/images}" \
  OUT_DIR="$SC_NEW" PROCESS_RES="${PROCESS_RES:-504}" REF_STRATEGY="${REF_STRATEGY:-first}" \
  EXPORT_OFFICIAL=1 \
    "$PY_DA3" "$CODE/compare_4methods/98_da3_ff_init.py" || exit 1
fi

for f in "$SC_NEW/gs_ply/0000.ply" "$OUT/i01_fullpts_dc/0000.ply" \
         "$OUT/i02_pruned_sh/0000.ply" "$OUT/i12_fullpts_sh/0000.ply"; do
  [ -f "$f" ] || { echo "❌ 缺 $f"; exit 1; }
done

echo
echo "================ 阶段 2/2：PLY 渲染对账（querysplat）================"
if [ -f /home/velvet/miniconda3/etc/profile.d/conda.sh ]; then
  # shellcheck disable=SC1090
  source /home/velvet/miniconda3/etc/profile.d/conda.sh
  conda activate "$ENVS"
fi
export CUDA_HOME="$ENVS"
export CPATH="$ENVS/targets/x86_64-linux/include"
export LIBRARY_PATH="$ENVS/targets/x86_64-linux/lib"

SC="$SC_NEW" CAM="$SC_REF/cameras.npz" MEM_NPZ="$SC_REF/official_dump.npz" \
OUT="$OUT" VIEWS="${VIEWS:-all}" \
  exec python "$CODE/compare_4methods/audit_13_ply_official.py"

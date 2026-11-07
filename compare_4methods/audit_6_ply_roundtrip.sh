#!/usr/bin/env bash
# AUDIT-6 运行器：内存高斯 ↔ 导出 PLY 逐位对账（需 da3_human env，跑一次前向 ~20 s）
# 参数必须与 98 完全一致，否则 PLY 不是同一次前向的产物。
# 用法：bash compare_4methods/audit_6_ply_roundtrip.sh
set -o pipefail
export DA3_MODEL_DIR="${DA3_MODEL_DIR:-/mnt/d/wheel/da3_human_ms}"
export IMAGES_DIR="${IMAGES_DIR:-$HOME/output/da3_results/_captures/佳佳抱枕/images}"
export OUT_DIR="${OUT_DIR:-$HOME/output/da3_gs_results/佳佳抱枕_p1m}"
export PROCESS_RES="${PROCESS_RES:-504}"
export REF_STRATEGY="${REF_STRATEGY:-first}"
exec "$HOME/miniconda3/envs/da3_human/bin/python" "$(dirname "$0")/audit_6_ply_roundtrip.py"

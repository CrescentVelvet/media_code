#!/usr/bin/env bash
# 22_run_n1.sh -- 显式执行 Rosemary 的 SA P0 / Zip FF 同协议对照（不训练）。
set -o pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONDA_ENV=splatanything
source "$SCRIPT_DIR/_env.sh"
source "$SCRIPT_DIR/_gsplat_env.sh"
OUT="${OUT:-/mnt/d/output/results/splatanything_results/n1_head_compare_20261122/rosemary_v27}"
VALIDATION_OUT="${VALIDATION_OUT:-${OUT}_render_validation}"
"$CONDA_PREFIX/bin/python" "$SCRIPT_DIR/22_n1_head_compare.py" \
  --sa-result /home/velvet/output/splatanything_results/pose_rosemary/rosemary33v \
  --zip-result /mnt/d/output/results/zipsplat_results/rosemary3_ff \
  --gt-pkg /mnt/d/dataset/human_fbx_gt/rosemary_face_v3 \
  --source-scene /home/velvet/output/splatanything_results/_scenes/rosemary33v \
  --target-scene /home/velvet/output/splatanything_results/_scenes/rosemary3_t2 \
  --landmarks /mnt/d/output/results/splatanything_results/geometry_audit_v2_20261122/landmarks_native/landmarks_native.json \
  --roi-qa /mnt/d/output/results/splatanything_results/geometry_audit_v2_20261122/roi_visual_qa.json \
  --n0-audit /mnt/d/output/results/splatanything_results/geometry_audit_v2_20261122/five_view_rigid_v28/geometry_audit_v2.json \
  --zip-roundtrip-audit "${ZIP_ROUNDTRIP_AUDIT:-/mnt/d/output/results/splatanything_results/n1_head_compare_20261122/zip_native_roundtrip/zip_native_roundtrip.json}" \
  --official-repo /home/velvet/repos/SplatAnything \
  --render-validation-dir "$VALIDATION_OUT" \
  --out "$OUT"

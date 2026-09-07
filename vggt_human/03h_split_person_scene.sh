#!/usr/bin/env bash
# 03h_split_person_scene.sh — 3DGS 点云按 人/场景 拆分 (头已由 head_gs 03g 替代)
#
# 输入:
#   $RESULTS_DIR/04b_model_3dgs_ba/point_cloud/iteration_30000/point_cloud.ply  (GAUSSIAN_PLY)
#   $RESULTS_DIR/03_sam3_person_masks/     ← SAM3 video 模式 prompt=person 的 mask
#   $RESULTS_DIR/03_sam3_face_masks/       ← 清洗后的 face mask (用于 track↔pid 对齐)
#   $RESULTS_DIR/03e_head_3dmm/            ← head_fit.json + head_mesh_p*.npz (头包围盒)
# 输出: $RESULTS_DIR/03h_person_scene_split/
#   body_gs_p{00,01,02}.ply + scene_gs.ply + split_stats.json
# 注: 不用 -u —— proxy.env 引用了可能未定义的 PYTHONPATH (与 03f 等保持一致)
set -eo pipefail
PYTHONPATH="${PYTHONPATH:-}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/_env.sh"

RESULTS_DIR="${RESULTS_DIR:?RESULTS_DIR 必填}"
SOURCE_DIR="${SOURCE_DIR:-$RESULTS_DIR/03b_source_ba}"
GAUSSIAN_PLY="${GAUSSIAN_PLY:-$RESULTS_DIR/04b_model_3dgs_ba/point_cloud/iteration_30000/point_cloud.ply}"
PERSON_MASKS_DIR="${PERSON_MASKS_DIR:-$RESULTS_DIR/03_sam3_person_masks}"
FACE_MASKS_DIR="${FACE_MASKS_DIR:-$RESULTS_DIR/03_sam3_face_masks}"
HEAD_FIT_JSON="${HEAD_FIT_JSON:-$RESULTS_DIR/03e_head_3dmm/head_fit.json}"
HEAD_MESH_DIR="${HEAD_MESH_DIR:-$RESULTS_DIR/03e_head_3dmm}"
SPLIT_OUT_DIR="${SPLIT_OUT_DIR:-$RESULTS_DIR/03h_person_scene_split}"
PERSONS="${PERSONS:-0,1,2}"

cd "$SCRIPT_DIR"
RESULTS_DIR="$RESULTS_DIR" SOURCE_DIR="$SOURCE_DIR" GAUSSIAN_PLY="$GAUSSIAN_PLY" \
PERSON_MASKS_DIR="$PERSON_MASKS_DIR" FACE_MASKS_DIR="$FACE_MASKS_DIR" \
HEAD_FIT_JSON="$HEAD_FIT_JSON" HEAD_MESH_DIR="$HEAD_MESH_DIR" \
SPLIT_OUT_DIR="$SPLIT_OUT_DIR" PERSONS="$PERSONS" \
python split_person_scene.py

#!/usr/bin/env bash
# 07_init_body_gs.sh — 阶段七：BodyGaussian 初始化（point-to-triangle + 显式补缝）。
#
# 与参考实现的差异（详见脚本头注释）：
#   切分距离从「KNN 到最近顶点」改为「点到 mesh 表面」——前者受顶点密度支配，
#   换 FLAME（20971→5023）会让删除半径缩到 49%，问题从空洞翻转为重叠重影。
#   补缝从「指望 alpha blending + densification 兜底」改为显式生成桥接高斯。
set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/_env.sh"

echo "🚀 [flame_human 07] 阶段七 BodyGaussian 初始化"
echo "  🧍 body 点云: $UPSTREAM_DIR/03h_person_scene_split/body_gs_p{pid}.ply"
echo "  🗿 头部网格  : ${MESH_DIR:-$RESULTS_DIR/06_avatar_gs}"
echo "  💾 输出      : ${OUT_DIR:-$RESULTS_DIR/07_body_gs}"

if [ ! -d "${MESH_DIR:-$RESULTS_DIR/06_avatar_gs}" ]; then
    echo "❌ 缺头部网格目录，先跑 bash 06_init_avatar_gs.sh"
    exit 1
fi

BODY_PLY="${BODY_PLY:-$UPSTREAM_DIR/03h_person_scene_split/body_gs_p{pid}.ply}" \
MESH_DIR="${MESH_DIR:-$RESULTS_DIR/06_avatar_gs}" \
OUT_DIR="${OUT_DIR:-$RESULTS_DIR/07_body_gs}" \
PERSONS="${PERSONS:-0,1,2}" \
MIN_DIST_MM="${MIN_DIST_MM:-20}" \
BRIDGE_MM="${BRIDGE_MM:-25}" \
BRIDGE_STEPS="${BRIDGE_STEPS:-3}" \
BRIDGE_N="${BRIDGE_N:-1500}" \
SEED="${SEED:-0}" \
    python "$SCRIPT_DIR/init_body_gs.py"
if [ $? -ne 0 ]; then
    echo "❌ FAILED: 阶段七失败"
    exit 1
fi

echo "🎉 Done. 下一步：bash 08_train.sh"

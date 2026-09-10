#!/usr/bin/env bash
# 07a_split_body_scene.sh — 从上游 3DGS 场景模型提取 body 点云（07 的前置）。
#
# 上游 vggt_human 的 03h 依赖 04b_model_3dgs_ba（30k iter）+ 03e 头网格等
# 一串前置产物，verify 数据上没有。这里用现有素材替代：
#   场景高斯 = model_3dgs_nn/iteration_7000（完整训练）
#   person mask = 本链路 01c 产物
#   头网格 = 本链路 06 产物（avatar_mesh_p{pid}.npz）
set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/_env.sh"

echo "🧍 [flame_human 07a] 从场景高斯提取 body 点云"

if [ ! -d "$UPSTREAM_DIR" ]; then
    echo "❌ UPSTREAM_DIR 不存在: $UPSTREAM_DIR"
    exit 1
fi

OUT_DIR="${OUT_DIR:-$RESULTS_DIR/07_body_gs_src}" \
PERSONS="${PERSONS:-0}" \
    python "$SCRIPT_DIR/split_body_scene.py"
if [ $? -ne 0 ]; then
    echo "❌ FAILED: 07a 失败"
    exit 1
fi

echo "🎉 Done. 下一步：bash 07_init_body_gs.sh"

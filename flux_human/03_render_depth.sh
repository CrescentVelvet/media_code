#!/usr/bin/env bash
# 03_render_depth.sh — 用 SMPL mesh (reference.npz 的 vertices+faces) 渲染 N 个
# 视角的骨骼深度图, 作为 04 Flux ControlNet(depth) 的 condition。
#
# 纯 flux_human env (pyrender + trimesh, 不依赖 sam_3d_body 代码)。
# 输入: $SMPL_OUT/reference.npz (02 导出)
# 输出:
#   $DEPTH_DIR/depth_<view>.npy      float32 深度 (米)
#   $DEPTH_DIR/depth_<view>.png      uint8 可视化 (ControlNet 输入)
#   $DEPTH_DIR/cameras.npz           每视角相机位姿 (给 05 重建用, known pose)
set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/_env.sh"

RESULTS_DIR="${RESULTS_DIR:-$REPO_DIR/../flux_human_results}"
SMPL_OUT="${SMPL_OUT:-$RESULTS_DIR/smpl}"
DEPTH_DIR="${DEPTH_DIR:-$RESULTS_DIR/depth}"

REF_NPZ="${REF_NPZ:-$SMPL_OUT/reference.npz}"
NUM_VIEWS="${NUM_VIEWS:-24}"            # orbit 视角数 (技术方案 24-36)
ELEVATION="${ELEVATION:--10}"           # 相机俯仰角 (度; 负=俯视)
IMG_SIZE="${IMG_SIZE:-768}"             # 渲染分辨率 (方形)
CAMERA_DIST="${CAMERa_DIST:-}"          # 相机距离 (空=按 mesh 尺度自动算)
FOV_DEG="${FOV_DEG:-35}"                # 相机垂直 FOV (度)

echo "=== [03] 渲染骨骼深度图 (flux_human env) ==="
echo "  参考数据:   $REF_NPZ"
echo "  输出目录:   $DEPTH_DIR  (depth_*.npy + .png + cameras.npz)"
echo "  视角数:     $NUM_VIEWS  (orbit 一圈, elevation=${ELEVATION}°)"
echo "  渲染尺寸:   ${IMG_SIZE}x${IMG_SIZE}  fov=${FOV_DEG}°"

# --- checks ---
if [ ! -f "$REF_NPZ" ]; then
    echo "❌ ERROR: $REF_NPZ 不存在. 先跑 02: GPU=0 bash flux_human/02_extract_smpl.sh" >&2
    exit 1
fi
if ! python -c "import pyrender, trimesh" 2>/dev/null; then
    echo "❌ ERROR: pyrender/trimesh 未装. INSTALL_DEPS=1 bash flux_human/00_setup_env.sh" >&2
    exit 1
fi

mkdir -p "$DEPTH_DIR"

export REF_NPZ DEPTH_DIR NUM_VIEWS ELEVATION IMG_SIZE CAMERA_DIST FOV_DEG

python "$SCRIPT_DIR/render_depth.py"
if [ $? -ne 0 ]; then
    echo "❌ render_depth.py 失败" >&2; exit 1
fi

echo "🎉 [03] Done. 深度图: $DEPTH_DIR/depth_*.png"
echo "    Next: GPU=0 bash $SCRIPT_DIR/04_generate_views.sh  (Flux1 多视角生成)"

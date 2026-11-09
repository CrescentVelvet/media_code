#!/usr/bin/env bash
# run_all.sh — 一键：生成轨迹 → 推理（单图 → 3DGS）
#
# 用法：
#   GPU=0 IMAGE=/path/to/scene.jpg bash scene_splatter/run_all.sh
#   # 自定义轨迹：
#   GPU=0 IMAGE=... YAW_SPAN=90 NUM_FRAMES=200 bash scene_splatter/run_all.sh
#   # 用现成轨迹（跳过生成）：
#   GPU=0 IMAGE=... CAMERA_PATH=/path/camera.pickle.gz bash scene_splatter/run_all.sh
set -o pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/_env.sh"

IMAGE="${IMAGE:-}"
if [ -z "$IMAGE" ] || [ ! -f "$IMAGE" ]; then
    echo "❌ ERROR: IMAGE 未设置或不存在: $IMAGE" >&2
    exit 1
fi

# 轨迹：没给 CAMERA_PATH 就自动生成（放 RESULTS_DIR/cameras/，按图像名区分）
if [ -z "${CAMERA_PATH:-}" ]; then
    STEM="$(basename "$IMAGE")"; STEM="${STEM%.*}"
    CAMERA_PATH="$RESULTS_DIR/cameras/$STEM.pickle.gz"
    mkdir -p "$(dirname "$CAMERA_PATH")"
    if [ ! -f "$CAMERA_PATH" ]; then
        echo "🚀 [run_all] 生成相机轨迹 → $CAMERA_PATH"
        OUT="$CAMERA_PATH" \
        NUM_FRAMES="${NUM_FRAMES:-100}" YAW_SPAN="${YAW_SPAN:-40}" \
        RADIUS="${RADIUS:-1.0}" FOV_Y="${FOV_Y:-60}" \
        python "$SCRIPT_DIR/02_gen_traj.py" || { echo "❌ 轨迹生成失败" >&2; exit 1; }
        OUT="$CAMERA_PATH" python "$SCRIPT_DIR/02_gen_traj.py" --selftest || true
    else
        echo "⏭️  轨迹已存在: $CAMERA_PATH（YAW_SPAN 等参数不会生效，删掉重生成才生效）"
    fi
fi

echo "🚀 [run_all] 开始推理"
IMAGE="$IMAGE" CAMERA_PATH="$CAMERA_PATH" \
PER_VIDEO_LENGTH="${PER_VIDEO_LENGTH:-16}" NUM_OVERLAP="${NUM_OVERLAP:-2}" \
CAM_INTERVAL="${CAM_INTERVAL:-2}" \
bash "$SCRIPT_DIR/02_run_inference.sh"

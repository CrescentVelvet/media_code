#!/usr/bin/env bash
# 02_extract_smpl.sh — 抽帧 -> 调 sam_3d_body 抽 SMPL -> 选静止参考帧 T* -> 导出。
#
# 跨 env 设计:
#   - 抽帧 + 编排: flux_human env (本脚本)
#   - sam_3d_body 推理 + extract_smpl.py (需 import sam_3d_body 取 faces): sam_3d_body env
#     (用 conda run -n sam_3d_body 调用, 不污染 flux_human env)
#
# 输入: VIDEO (视频文件) 或 FRAMES_DIR (已抽好的帧目录)
# 输出:
#   $RESULTS_DIR/frames/<frame_%06d>.jpg   抽帧
#   $SAM3D_RESULTS/                        sam_3d_body 推理 (result/mesh/npz)
#   $SMPL_OUT/reference.npz               参考帧 T* 的 SMPL 数据 (供 03/04 用)
#       含 vertices, cam_t, faces, kp3d/kp2d, focal, bbox, ref_image, frame_idx
set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/_env.sh"

RESULTS_DIR="${RESULTS_DIR:-$REPO_DIR/../flux_human_results}"
SAM3D_DIR="${SAM3D_DIR:-$REPO_DIR/../sam-3d-body}"
SAM3D_MODEL_DIR="${SAM3D_MODEL_DIR:-$REPO_DIR/../../model/sam-3d-body}"

FRAMES_DIR="${FRAMES_DIR:-$RESULTS_DIR/frames}"
SAM3D_RESULTS="${SAM3D_RESULTS:-$RESULTS_DIR/sam3d}"
SMPL_OUT="${SMPL_OUT:-$RESULTS_DIR/smpl}"

VIDEO="${VIDEO:-}"
FPS="${FPS:-2}"                         # 抽帧率 (人体微动 2fps 足够捕捉动作)
INFERENCE_TYPE="${INFERENCE_TYPE:-body}"  # body=跳过手部解码器 (更快; 重建不需要手部细化)
WINDOW="${WINDOW:-5}"                   # 选参考帧的滑动窗口 (pose 变化最小)

# sam_3d_body 权重路径 (dinov3 默认骨干)
SAM3D_CKPT="$SAM3D_MODEL_DIR/sam-3d-body-dinov3/model.ckpt"
SAM3D_MHR="$SAM3D_MODEL_DIR/sam-3d-body-dinov3/assets/mhr_model.pt"
SAM3D_FOV="$SAM3D_MODEL_DIR/moge-2-vitl-normal"

echo "=== [02] SMPL 提取 (sam_3d_body env: sam_3d_body) ==="
echo "  视频文件:   ${VIDEO:-❌ 未提供 (用 FRAMES_DIR 代替)}"
echo "  抽帧目录:   $FRAMES_DIR  (fps=$FPS)"
echo "  sam3d 输出: $SAM3D_RESULTS  (result/mesh/npz)"
echo "  SMPL 导出:  $SMPL_OUT/reference.npz"
echo "  窗口大小:   $WINDOW  (选 pose 变化最小的参考帧)"

# --- checks ---
if ! conda env list 2>/dev/null | grep -qE "(^| )sam_3d_body( |$)"; then
    echo "❌ ERROR: conda env 'sam_3d_body' not found. 先建它: cd sam_3d_body && INSTALL_DEPS=1 bash 00_setup_env.sh" >&2
    exit 1
fi
if [ ! -f "$SAM3D_CKPT" ]; then
    echo "❌ ERROR: sam_3d_body 权重未找到: $SAM3D_CKPT" >&2
    echo "   先下权重: HF_TOKEN=hf_xxx bash sam_3d_body/01_download_models.sh" >&2
    exit 1
fi

# --- 1. 抽帧 (如果给 VIDEO) ---
if [ -n "$VIDEO" ]; then
    if [ ! -f "$VIDEO" ]; then
        echo "❌ ERROR: 视频文件不存在: $VIDEO" >&2; exit 1
    fi
    if [ -z "$(ls -A "$FRAMES_DIR" 2>/dev/null)" ]; then
        mkdir -p "$FRAMES_DIR"
        echo "🎬 抽帧: $VIDEO -> $FRAMES_DIR (fps=$FPS)"
        ffmpeg -hide_banner -loglevel error -i "$VIDEO" -vf "fps=$FPS" \
            "$FRAMES_DIR/frame_%06d.jpg" -y || {
            echo "❌ ffmpeg 抽帧失败" >&2; exit 1; }
        echo "✅ 抽帧完成: $(ls "$FRAMES_DIR" | wc -l) 帧"
    else
        echo "⏭️ 帧目录已有图, 跳过抽帧: $FRAMES_DIR"
    fi
fi

if [ -z "$(ls -A "$FRAMES_DIR" 2>/dev/null)" ]; then
    echo "❌ ERROR: 无帧图像. 设 VIDEO=xxx.mp4 或放帧到 FRAMES_DIR=$FRAMES_DIR" >&2
    exit 1
fi

# --- 2. 调 sam_3d_body 推理 (跨 env: sam_3d_body) ---
if [ -z "$(ls -A "$SAM3D_RESULTS/npz" 2>/dev/null)" ]; then
    echo "🚀 调 sam_3d_body 推理 (env: sam_3d_body, type=$INFERENCE_TYPE) ..."
    conda run -n sam_3d_body --no-capture-output \
        env INPUT_DIR="$FRAMES_DIR" \
            OUTPUT_DIR="$SAM3D_RESULTS" \
            MODEL_DIR="$SAM3D_MODEL_DIR" \
            CHECKPOINT_PATH="$SAM3D_CKPT" \
            MHR_PATH="$SAM3D_MHR" \
            FOV_PATH="$SAM3D_FOV" \
            INFERENCE_TYPE="$INFERENCE_TYPE" \
            GPU="${GPU:-}" \
        bash "$REPO_DIR/sam_3d_body/02_run_inference.sh"
    if [ $? -ne 0 ]; then
        echo "❌ sam_3d_body 推理失败" >&2; exit 1
    fi
else
    echo "⏭️ sam_3d_body npz 已存在, 跳过推理: $SAM3D_RESULTS/npz"
fi

# --- 3. extract_smpl.py 选参考帧 + 导出 (sam_3d_body env: 要 import sam_3d_body 取 faces) ---
echo "🎯 选静止参考帧 + 导出 SMPL 数据 ..."
export RESULTS_DIR SAM3D_RESULTS SMPL_OUT SAM3D_DIR FRAMES_DIR
export CHECKPOINT_PATH="$SAM3D_CKPT"
export MHR_PATH="$SAM3D_MHR"
export WINDOW

conda run -n sam_3d_body --no-capture-output python "$SCRIPT_DIR/extract_smpl.py"
if [ $? -ne 0 ]; then
    echo "❌ extract_smpl.py 失败" >&2; exit 1
fi

echo "🎉 [02] Done. SMPL 参考帧数据: $SMPL_OUT/reference.npz"
echo "    Next: GPU=0 bash $SCRIPT_DIR/03_render_depth.sh  (渲染骨骼深度图)"

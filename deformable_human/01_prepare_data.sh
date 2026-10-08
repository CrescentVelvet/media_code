#!/usr/bin/env bash
# 01_prepare_data.sh — Stage A：单目视频 → Deformable-GS 可训练的 COLMAP 场景。
#
# 两步，全部**委托兄弟目录的现成脚本**（文件做接口，不复制代码）：
#   1) 视频 → 帧序列：调 vggt_human/01a_video_to_frames.sh
#      （ffmpeg 抽帧 + BLUR_THRESHOLD 拉普拉斯模糊门——输入帧模糊正是本课题要防的，
#        阈值内嵌在 01a 里，模糊帧直接丢弃）
#   2) 帧序列 → COLMAP 场景：调 deformable_gaussians/03_colmap_pose.sh
#      （重命名纯数字帧号 → SfM → 去畸变，输出 images/ + sparse/0/*.bin）
#
# Phase 2（MHR 锚定，默认关）：ANCHOR_EXPORT=1 预留——将从 vggt_human 的
#   03e_detect_landmarks / 03f_fit_head_3dmm 适配出「单目逐帧 MHR + 657 landmarks」
#   导出 anchors/。⚠️ 尚未实现，当前只打印提示不执行。
#
# 输入：VIDEO_PATH=/path/to/xxx.mp4（单目、人物有动作）
# 输出：$DATA_ROOT/$SCENE_NAME/
#         frames/image/         抽帧（模糊帧已剔除）
#         colmap_scene/         images/ + sparse/0/{cameras,images,points3D}.bin
#
# Env (all optional, defaults shown):
#   VIDEO_PATH=            # 单目视频文件（必填）
#   SCENE_NAME=human_seq   # 场景名（决定输出子目录）
#   DATA_ROOT=             # 数据集根（默认 $RESULTS_DIR/datasets）
#   VIDEO_FPS=6            # 抽帧 fps（动态序列要比 vggt_human 默认的 2 密）
#   BLUR_THRESHOLD=100     # 模糊门（0=关；透传给 01a）
#   FORCE_RECOLMAP=0       # 1=COLMAP 场景已存在也强制重跑
#   USE_GPU=1              # COLMAP SIFT 用 GPU（conda-forge colmap 是 CPU 版→设 0）
#   ANCHOR_EXPORT=0        # 1=Phase 2 锚定导出（未实现，占位）
set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/_env.sh"

VIDEO_PATH="${VIDEO_PATH:-}"
SCENE_NAME="${SCENE_NAME:-human_seq}"
DATA_ROOT="${DATA_ROOT:-$RESULTS_DIR/datasets}"
FRAMES_DIR="$DATA_ROOT/$SCENE_NAME/frames"
SCENE_DIR="$DATA_ROOT/$SCENE_NAME/colmap_scene"
VIDEO_FPS="${VIDEO_FPS:-6}"
BLUR_THRESHOLD="${BLUR_THRESHOLD:-100}"
USE_GPU="${USE_GPU:-1}"

echo "🚀 [01] Stage A: 单目视频 → COLMAP 场景"
echo "  🎬 视频:       $VIDEO_PATH"
echo "  💾 输出:       $DATA_ROOT/$SCENE_NAME/"
echo "  📐 video_fps:  $VIDEO_FPS   blur_gate: $BLUR_THRESHOLD"
echo ""

# ── 前置检查 ───────────────────────────────────────────────────────────────
# VIDEO_PATH 与「预置帧」二选一：$FRAMES_DIR/image/ 已有帧时跳过抽帧（复用既有采集），
# 此时 VIDEO_PATH 可以不传。
if [ -d "$FRAMES_DIR/image" ] && [ "$(ls -A "$FRAMES_DIR/image" 2>/dev/null | wc -l)" -gt 0 ]; then
    PRESEEDED=1
else
    PRESEEDED=0
fi
if [ "$PRESEEDED" = "0" ]; then
    if [ -z "$VIDEO_PATH" ]; then
        echo "❌ ERROR: VIDEO_PATH 未设置（单目视频文件路径；或预置帧到 $FRAMES_DIR/image/）" >&2
        exit 1
    fi
    if [ ! -f "$VIDEO_PATH" ]; then
        echo "❌ ERROR: 视频不存在: $VIDEO_PATH" >&2
        exit 1
    fi
fi
if [ ! -f "$VGGT_HUMAN_DIR/01a_video_to_frames.sh" ]; then
    echo "❌ ERROR: 找不到 $VGGT_HUMAN_DIR/01a_video_to_frames.sh（VGGT_HUMAN_DIR 可覆盖）" >&2
    exit 1
fi
if [ ! -f "$DG_ORCH_DIR/03_colmap_pose.sh" ]; then
    echo "❌ ERROR: 找不到 $DG_ORCH_DIR/03_colmap_pose.sh（DG_ORCH_DIR 可覆盖）" >&2
    exit 1
fi

# ── 1) 视频 → 帧（委托 vggt_human/01a，它自己 source 自己的 env）─────────────
if [ "$PRESEEDED" = "1" ]; then
    echo "⏭️  [1] $FRAMES_DIR/image/ 已有预置帧（$(ls -A "$FRAMES_DIR/image" | wc -l) 张），跳过抽帧"
else
    echo "🎬 [1] 抽帧: 调 vggt_human/01a_video_to_frames.sh"
    INPUT_DIR="$VIDEO_PATH" \
    OUTPUT_DIR="$FRAMES_DIR" \
    VIDEO_FPS="$VIDEO_FPS" \
    BLUR_THRESHOLD="$BLUR_THRESHOLD" \
    bash "$VGGT_HUMAN_DIR/01a_video_to_frames.sh"
    if [ $? -ne 0 ]; then
        echo "❌ [1] FAILED: 抽帧失败" >&2
        exit 1
    fi
fi

# ── 2) 帧 → COLMAP 场景（委托 deformable_gaussians/03；CONDA_ENV 继承本目录）──
echo "🗺️  [2] COLMAP SfM: 调 deformable_gaussians/03_colmap_pose.sh"
INPUT_DIR="$FRAMES_DIR" \
OUTPUT_SCENE="$SCENE_DIR" \
SCENE_NAME="$SCENE_NAME" \
USE_GPU="$USE_GPU" \
FORCE_RECOLMAP="${FORCE_RECOLMAP:-0}" \
bash "$DG_ORCH_DIR/03_colmap_pose.sh"
if [ $? -ne 0 ]; then
    echo "❌ [2] FAILED: COLMAP 失败。动态人体的常见死因：" >&2
    echo "    - 人体占画面太大且动得快 → 背景匹配点不够，mapper 掉帧" >&2
    echo "    - 模糊帧没剔干净（BLUR_THRESHOLD 调高重跑第 1 步）" >&2
    echo "    - conda-forge colmap 是 CPU 版：USE_GPU=0" >&2
    exit 1
fi

# ── 3) Phase 2 锚定导出（占位，未实现）─────────────────────────────────────
if [ "${ANCHOR_EXPORT:-0}" = "1" ]; then
    echo ""
    echo "⚠️  [3] ANCHOR_EXPORT=1：MHR 锚定导出尚未实现（Phase 2）。"
    echo "    计划：适配 vggt_human 的 03e_detect_landmarks / 03f_fit_head_3dmm，"
    echo "    对单目逐帧导出 MHR 参数 + 657 landmarks → $DATA_ROOT/$SCENE_NAME/anchors/。"
    echo "    当前跳过，不影响 vanilla baseline。"
fi

# ── 验证输出 ───────────────────────────────────────────────────────────────
echo ""
echo "🔍 验证"
n_frames="$(find "$FRAMES_DIR/image" -maxdepth 1 -type f 2>/dev/null | wc -l)"
echo "  🖼️  帧数: $n_frames ($FRAMES_DIR/image/)"
for f in "$SCENE_DIR/images" "$SCENE_DIR/sparse/0/cameras.bin" "$SCENE_DIR/sparse/0/images.bin" "$SCENE_DIR/sparse/0/points3D.bin"; do
    [ -e "$f" ] && echo "  ✅ $f" || echo "  ❌ $f MISSING" >&2
done

echo ""
echo "🎉 [01] Done. 数据集就绪: $SCENE_DIR"
echo "  → 训练: GPU=0 SOURCE_PATH=$SCENE_DIR bash deformable_human/02_train.sh"

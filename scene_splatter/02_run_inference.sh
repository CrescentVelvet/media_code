#!/usr/bin/env bash
# 02_run_inference.sh — Scene Splatter 推理（单图 + 相机轨迹 → 3DGS）
#
# 输出（搬到 $RESULTS_DIR/02_infer/<时间戳>/）：
#   scene.ply            最终 3D 高斯（media_code 补丁导出；官方原版无此文件）
#   flash3d_video.mp4    Flash3D 初始化的环绕渲染
#   render_video_{i}.mp4 每轮迭代后的轨迹渲染（最后一段 = 最终质量）
#   confidence_maps/     逐帧置信度图（像素级动量融合权重）
#
# ⚠️ 配置不走命令行：scenesplatter.py 内部 viewcrafter 复用 sys.argv 做 parse_args，
#    hydra CLI 覆盖会炸（unrecognized arguments）。只能改 configs/config.yaml（本脚本代劳）。
#
# Env:
#   IMAGE             输入单图（必填）
#   CAMERA_PATH       轨迹 pickle.gz（必填；run_all.sh 可自动生成）
#   PER_VIDEO_LENGTH  每段视频帧数，默认 16（对齐 ckpt 训练长度）
#   NUM_OVERLAP       段间重叠帧数，默认 2
#   CAM_INTERVAL      轨迹抽稀间隔，默认 2
set -o pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/_env.sh"

IMAGE="${IMAGE:-}"
CAMERA_PATH="${CAMERA_PATH:-}"
PER_VIDEO_LENGTH="${PER_VIDEO_LENGTH:-16}"
NUM_OVERLAP="${NUM_OVERLAP:-2}"
CAM_INTERVAL="${CAM_INTERVAL:-2}"

if [ -z "$IMAGE" ] || [ ! -f "$IMAGE" ]; then
    echo "❌ ERROR: IMAGE 未设置或不存在: $IMAGE" >&2; exit 1
fi
if [ -z "$CAMERA_PATH" ] || [ ! -f "$CAMERA_PATH" ]; then
    echo "❌ ERROR: CAMERA_PATH 未设置或不存在（先用 02_gen_traj.py 生成）" >&2; exit 1
fi
if [ ! -f "$SS_DIR/scenesplatter.py" ]; then
    echo "❌ ERROR: 官方仓未 clone: $SS_DIR（先跑 00a_setup_env.sh）" >&2; exit 1
fi
if [ ! -f "$SS_DIR/flash3d/model_re10k_v2.pth" ]; then
    echo "❌ ERROR: 权重未就绪（先跑 01_download_models.sh）" >&2; exit 1
fi

echo "🚀 [02] Scene Splatter 推理"
echo "  🖼️  image:   $IMAGE"
echo "  📷 camera:  $CAMERA_PATH"
echo "  📐 video:   len=$PER_VIDEO_LENGTH overlap=$NUM_OVERLAP interval=$CAM_INTERVAL"
echo "  💾 results: $RESULTS_DIR/02_infer/"
echo "  🎮 GPU:     ${CUDA_VISIBLE_DEVICES:-all}"
echo ""
echo "⏱️  预期耗时：分钟级~小时级（迭代轮数 = 抽稀后帧数 / $((PER_VIDEO_LENGTH - NUM_OVERLAP))，"
echo "   每轮 5000 步 3DGS 优化 + 一段视频生成；3090 实测待填 EXPERIMENTS.md）"

# ── 写 configs/config.yaml（备份一次原文件）──
if [ ! -f "$SS_DIR/configs/config.yaml.media_code_bak" ]; then
    cp "$SS_DIR/configs/config.yaml" "$SS_DIR/configs/config.yaml.media_code_bak"
fi
IMAGE_ABS="$(python3 -c 'import os,sys;print(os.path.abspath(sys.argv[1]))' "$IMAGE")"
CAMERA_ABS="$(python3 -c 'import os,sys;print(os.path.abspath(sys.argv[1]))' "$CAMERA_PATH")"
python3 - "$SS_DIR/configs/config.yaml" "$IMAGE_ABS" "$CAMERA_ABS" \
    "$PER_VIDEO_LENGTH" "$NUM_OVERLAP" "$CAM_INTERVAL" <<'EOF'
import re, sys
path, image, camera, pl, ov, it = sys.argv[1:7]
with open(path, "r", encoding="utf-8") as f:
    src = f.read()
def set_key(src, key, value):
    return re.sub(rf"^(\s*{key}:).*$", rf"\g<1> {value}", src, count=1, flags=re.M)
src = set_key(src, "image_path", f'"{image}"')
src = set_key(src, "camera_path", f'"{camera}"')
src = set_key(src, "per_video_length", pl)
src = set_key(src, "num_overlap_frames", ov)
src = set_key(src, "camera_sample_interval", it)
with open(path, "w", encoding="utf-8", newline="\n") as f:
    f.write(src)
print("✅ config.yaml 已更新")
EOF

# ── 跑（必须在官方仓根目录；hydra chdir 进 exp/，../ 相对路径才能解析到仓根）──
cd "$SS_DIR" || { echo "❌ ERROR: $SS_DIR 不存在" >&2; exit 1; }
python scenesplatter.py
STATUS=$?

if [ $STATUS -ne 0 ]; then
    echo "❌ FAILED (exit $STATUS)" >&2
    exit $STATUS
fi

# ── 产物搬到 RESULTS_DIR（官方写在仓内 results/<时间戳>/，Linux fs 要搬到 D 盘由 08 负责）──
LATEST="$(ls -1dt "$SS_DIR"/results/*/ 2>/dev/null | head -1)"
if [ -z "$LATEST" ]; then
    echo "⚠️ 未找到 results 输出目录（pipeline 可能有异常，上方日志为准）" >&2
    exit 1
fi
DEST="$RESULTS_DIR/02_infer/$(basename "$LATEST")"
mkdir -p "$(dirname "$DEST")"
mv "$LATEST" "$DEST"
echo "🎉 Done. 产物目录: $DEST"
ls -lh "$DEST"

#!/usr/bin/env bash
# 03_colmap_pose.sh — 拍摄图像序列 → COLMAP NeRF-DS 格式场景。
#
# 用系统 colmap 二进制跑 SfM（feature_extractor → exhaustive_matcher → mapper
# → image_undistorter），把一组拍摄图像（INPUT_DIR/image/*.jpg）转成
# Deformable-3D-Gaussians 期望的 COLMAP 格式场景：
#   <output_scene>/
#     images/                 去畸变后的图像（colmap image_undistorter 输出）
#     sparse/0/
#       cameras.bin           相机内参（PINHOLE / SIMPLE_PINHOLE）
#       images.bin            相机外参（每帧 qvec + tvec）
#       points3D.bin           稀疏 3D 点（用于初始化高斯点云）
#
# ⚠️ 关键约束（Deformable-GS dataset_readers.py:136 的 `fid = int(image_name)
#    /(num_frames-1)`）：图像文件名必须是**纯数字**（如 00000.jpg），否则
#    `int()` 会抛 ValueError 中断训练。本脚本会按字典序排序输入图像，重命名
#    为 00000.jpg, 00001.jpg, ... 复制到 <output_scene>/input/。所以**拍摄时
#    请按时间顺序命名图像**（如 frame_0001.jpg, frame_0002.jpg, ...）——
#    重命名后的字典序就是时序，变形 MLP 才能正确学到随时间的形变。
#
# pipeline 参考 Deformable-GS 仓里的官方 convert.py（基于 MipNeRF-360 的 shell
# converter），改写成 bash 调系统 colmap CLI（不依赖官方 Python wrapper）。
#
# 与 wan22_rotate/05_3dgs_recon.sh 的 Pi3 路径不同：Deformable-GS 假设帧间
# 物体在动（NeRF-DS 设定），Pi3 是静态场景位姿估计器，动态场景上会糊；必须
# 用 COLMAP SfM 自己估位姿。所以 03 是独立步骤，不复用 Pi3。
#
# 输入：INPUT_DIR/image/*.jpg（拍摄原图像序列，跟 wan22_rotate INPUT_DIR 同模式）
# 输出：<OUTPUT_SCENE>/{images/, sparse/0/{cameras,images,points3D}.bin}
#
# Env (all optional, defaults shown):
#   INPUT_DIR=              # 拍摄图像目录（含 image/ 子文件夹）
#   OUTPUT_SCENE=           # 输出场景目录（COLMAP NeRF-DS 格式）
#   COLMAP_EXECUTABLE=      # colmap 二进制（默认从 PATH 找 "colmap"）
#   CAMERA_MODEL=OPENCV     # 相机模型（OPENCV / SIMPLE_PINHOLE / PINHOLE）
#   SINGLE_CAMERA=1         # 1=单相机假设（多视角同设备拍，NeRF-DS 默认）
#   USE_GPU=1               # 1=SIFT 提特征+匹配用 GPU（快；CPU 慢很多）
#   SKIP_MATCHING=0         # 1=跳过 feature/matcher（已有 distorted/database.db）
#   KEEP_DISTORTED=0        # 1=保留 distorted/ 临时目录（排查用）
#   NUM_IMAGES_MAX=0        # >0=只取前 N 张（防 COLMAP mapper OOM；0=全用）
set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/_env.sh"

# ── Paths ─────────────────────────────────────────────────────────────────
# 默认输出在 $MODEL_DIR/data/real/<scene_name>/ 下（跟 D-NeRF 同根，便于用 run_all）
MODEL_DIR="${MODEL_DIR:-$REPO_DIR/../../model/deformable-3d-gaussians}"
SCENE_NAME="${SCENE_NAME:-real_scene}"
INPUT_DIR="${INPUT_DIR:-$REPO_DIR/../wan22_rotate_results}"
OUTPUT_SCENE="${OUTPUT_SCENE:-$MODEL_DIR/data/real/$SCENE_NAME}"

# colmap 二进制：优先 COLMAP_EXECUTABLE，否则从 PATH 找 "colmap"
COLMAP_EXECUTABLE="${COLMAP_EXECUTABLE:-colmap}"

# ── Params ────────────────────────────────────────────────────────────────
CAMERA_MODEL="${CAMERA_MODEL:-OPENCV}"
SINGLE_CAMERA="${SINGLE_CAMERA:-1}"
USE_GPU="${USE_GPU:-1}"
SKIP_MATCHING="${SKIP_MATCHING:-0}"
KEEP_DISTORTED="${KEEP_DISTORTED:-0}"
NUM_IMAGES_MAX="${NUM_IMAGES_MAX:-0}"

echo "🚀 [03] COLMAP SfM (拍摄图像序列 → NeRF-DS COLMAP 格式)"
echo "  📂 输入:        $INPUT_DIR/image/"
echo "  💾 输出场景:    $OUTPUT_SCENE"
echo "  🤖 colmap:     $COLMAP_EXECUTABLE"
echo "  📐 camera_model: $CAMERA_MODEL  single_camera: $SINGLE_CAMERA  use_gpu: $USE_GPU"
echo ""

# ── 0. Sanity checks ──────────────────────────────────────────────────────
# colmap 二进制
if ! command -v "$COLMAP_EXECUTABLE" >/dev/null 2>&1; then
    # COLMAP_EXECUTABLE 可能是绝对路径，command -v 对绝对路径也行；这里再保险一下
    if [ ! -x "$COLMAP_EXECUTABLE" ]; then
        echo "❌ ERROR: colmap 二进制不在 PATH 也非可执行文件: '$COLMAP_EXECUTABLE'" >&2
        echo "       安装: apt install colmap  (Ubuntu; 自带 CUDA 版要 build from source)" >&2
        echo "       或设: COLMAP_EXECUTABLE=/path/to/colmap" >&2
        exit 1
    fi
fi
COLMAP_VERSION="$("$COLMAP_EXECUTABLE" -h 2>&1 | head -1 || echo unknown)"
echo "  ✅ colmap: $COLMAP_VERSION"

# 输入图像
INPUT_IMG_DIR="$INPUT_DIR/image"
if [ ! -d "$INPUT_IMG_DIR" ]; then
    echo "❌ ERROR: 输入图像目录不存在: $INPUT_IMG_DIR" >&2
    echo "       期望布局: INPUT_DIR/image/*.jpg (跟 wan22_rotate INPUT_DIR 同模式)" >&2
    echo "       若图像直接在 INPUT_DIR/ 下，建个子文件夹 image/ 再放进去" >&2
    exit 1
fi

# 列出输入图像（字典序，即按拍摄时序）
shopt -s nullglob
INPUT_IMAGES=()
for f in "$INPUT_IMG_DIR"/*.jpg "$INPUT_IMG_DIR"/*.JPG "$INPUT_IMG_DIR"/*.png "$INPUT_IMG_DIR"/*.PNG; do
    [ -f "$f" ] && INPUT_IMAGES+=("$f")
done
shopt -u nullglob

if [ ${#INPUT_IMAGES[@]} -eq 0 ]; then
    echo "❌ ERROR: $INPUT_IMG_DIR/ 下没有任何 .jpg/.png 图像" >&2
    exit 1
fi
echo "  🖼️  找到 ${#INPUT_IMAGES[@]} 张输入图像"

# 按 NUM_IMAGES_MAX 截断（防 COLMAP mapper 在大场景 OOM）
if [ "$NUM_IMAGES_MAX" -gt 0 ] && [ ${#INPUT_IMAGES[@]} -gt "$NUM_IMAGES_MAX" ]; then
    echo "  ⚠️ NUM_IMAGES_MAX=$NUM_IMAGES_MAX → 截取前 $NUM_IMAGES_MAX 张（其余跳过）"
    INPUT_IMAGES=("${INPUT_IMAGES[@]:0:$NUM_IMAGES_MAX}")
fi

# Deformable-GS 数据目录已存在（已有 sparse/0）→ 跳过
if [ -d "$OUTPUT_SCENE/sparse/0" ] && [ "${FORCE_RECOLMAP:-0}" != "1" ]; then
    echo "⏭️  $OUTPUT_SCENE/sparse/0/ 已存在，跳过 COLMAP（FORCE_RECOLMAP=1 强制重跑）"
    echo "    训练可直接吃: -s $OUTPUT_SCENE"
    exit 0
fi
mkdir -p "$OUTPUT_SCENE"

# ── 1. 复制 + 重命名为纯数字序号 → <output_scene>/input/ ────────────────
# Deformable-GS dataset_readers.py:136 `fid = int(image_name)/(num_frames-1)`
# 要求图像文件名是纯数字；按字典序排序复制成 00000.jpg, 00001.jpg, ...
INPUT_STAGE="$OUTPUT_SCENE/input"
if [ -d "$INPUT_STAGE" ] && [ "${FORCE_RECOLMAP:-0}" = "1" ]; then
    echo "  🧹 FORCE_RECOLMAP=1 → 清理旧 input/ + distorted/"
    rm -rf "$INPUT_STAGE" "$OUTPUT_SCENE/distorted"
fi
mkdir -p "$INPUT_STAGE"

NEED_COPY=1
if [ -d "$INPUT_STAGE" ] && [ "$(find "$INPUT_STAGE" -maxdepth 1 -name '*.jpg' -o -name '*.png' 2>/dev/null | wc -l)" -ge ${#INPUT_IMAGES[@]} ]; then
    # 已有 input/，且张数一致 → 跳过复制（节省 IO）
    echo "  ⏭️  $INPUT_STAGE/ 已有 $(${#INPUT_IMAGES[@]}) 张图像，跳过复制"
    NEED_COPY=0
fi

if [ "$NEED_COPY" = "1" ]; then
    echo "  📦 复制 + 重命名 → $INPUT_STAGE/ (00000.jpg, 00001.jpg, ...)"
    echo "     按字典序排序（拍摄时序；图像名应按时间命名）"
    idx=0
    for f in "${INPUT_IMAGES[@]}"; do
        printf -v pad "%05d" "$idx"
        ext="${f##*.}"        # 保留原扩展名（jpg/png）
        cp -f "$f" "$INPUT_STAGE/${pad}.${ext,,}"   # ${ext,,} = 小写扩展名
        idx=$((idx + 1))
    done
    echo "  ✅ 复制 $idx 张 → $INPUT_STAGE/"
fi

# ── 2. COLMAP feature_extractor + exhaustive_matcher ─────────────────────
DISTORTED_DIR="$OUTPUT_SCENE/distorted"
mkdir -p "$DISTORTED_DIR/sparse"

if [ "$SKIP_MATCHING" = "1" ]; then
    echo "⏭️  [skip matching] SKIP_MATCHING=1（复用 $DISTORTED_DIR/database.db）"
    if [ ! -f "$DISTORTED_DIR/database.db" ]; then
        echo "❌ ERROR: SKIP_MATCHING=1 但 $DISTORTED_DIR/database.db 不存在" >&2
        exit 1
    fi
else
    DB_PATH="$DISTORTED_DIR/database.db"
    rm -f "$DB_PATH"

    echo "🔍 [2a] feature_extractor (~分钟级, 取决于图像数)"
    echo "  📂 image_path:  $INPUT_STAGE"
    echo "  💾 database:    $DB_PATH"
    FEATURE_FLAGS=(
        --database_path "$DB_PATH"
        --image_path "$INPUT_STAGE"
        --ImageReader.camera_model "$CAMERA_MODEL"
        --SiftExtraction.use_gpu "$USE_GPU"
    )
    [ "$SINGLE_CAMERA" = "1" ] && FEATURE_FLAGS+=(--ImageReader.single_camera 1)
    "$COLMAP_EXECUTABLE" feature_extractor "${FEATURE_FLAGS[@]}"
    if [ $? -ne 0 ]; then
        echo "❌ [2a] FAILED. feature_extractor 没跑完。" >&2
        echo "    常见原因: GPU 显存不够（设 USE_GPU=0 走 CPU，慢但能跑）；" >&2
        echo "              图像 EXIF 缺失（必加 SINGLE_CAMERA=1 或 CAMERA_MODEL 显式指定）" >&2
        exit 1
    fi
    echo "✅ [2a] feature_extractor done"

    echo "🔗 [2b] exhaustive_matcher (两两匹配, ~N² 复杂度)"
    MATCHER_FLAGS=(
        --database_path "$DB_PATH"
        --SiftMatching.use_gpu "$USE_GPU"
    )
    "$COLMAP_EXECUTABLE" exhaustive_matcher "${MATCHER_FLAGS[@]}"
    if [ $? -ne 0 ]; then
        echo "❌ [2b] FAILED. exhaustive_matcher 没跑完。" >&2
        echo "    图像太少 / 视角重叠加不足时匹配会失败。检查拍摄视角重合度。" >&2
        exit 1
    fi
    echo "✅ [2b] exhaustive_matcher done"
fi

# ── 3. COLMAP mapper (SfM 重建 → distorted/sparse/0/) ────────────────────
echo "🗺️  [3] mapper (incremental SfM → $DISTORTED_DIR/sparse/0/)"
MAPPER_FLAGS=(
    --database_path "$DISTORTED_DIR/database.db"
    --image_path "$INPUT_STAGE"
    --output_path "$DISTORTED_DIR/sparse"
    --Mapper.ba_global_function_tolerance=0.000001
)
"$COLMAP_EXECUTABLE" mapper "${MAPPER_FLAGS[@]}"
if [ $? -ne 0 ]; then
    echo "❌ [3] FAILED. mapper 没跑完。" >&2
    echo "    常见原因: 匹配点不足 / 视角重叠加不够 / 图像模糊。" >&2
    echo "    排查: colmap gui -> load $DISTORTED_DIR/database.db 看匹配矩阵" >&2
    exit 1
fi
if [ ! -d "$DISTORTED_DIR/sparse/0" ]; then
    echo "❌ [3] FAILED. mapper 跑了但 $DISTORTED_DIR/sparse/0/ 没生成（重建失败）。" >&2
    echo "    通常匹配点太少；试调整拍摄视角或加更多重叠。" >&2
    exit 1
fi
echo "✅ [3] mapper done → $DISTORTED_DIR/sparse/0/"

# ── 4. COLMAP image_undistorter (去畸变 → <output_scene>/{images, sparse/0}) ──
echo "🔧 [4] image_undistorter (去畸变 → $OUTPUT_SCENE/{images, sparse/0})"
UNDIST_FLAGS=(
    --image_path "$INPUT_STAGE"
    --input_path "$DISTORTED_DIR/sparse/0"
    --output_path "$OUTPUT_SCENE"
    --output_type COLMAP
)
"$COLMAP_EXECUTABLE" image_undistorter "${UNDIST_FLAGS[@]}"
if [ $? -ne 0 ]; then
    echo "❌ [4] FAILED. image_undistorter 没跑完。" >&2
    exit 1
fi

# image_undistorter 输出在 <output_scene>/sparse/{cameras,images,points3D}.bin
# Deformable-GS scene/__init__.py:45 期望 <source_path>/sparse/0/*.bin
# 官方 convert.py:63-71 把 sparse/ 里的文件移到 sparse/0/ 下
if [ ! -d "$OUTPUT_SCENE/sparse/0" ]; then
    mkdir -p "$OUTPUT_SCENE/sparse/0"
    for f in "$OUTPUT_SCENE/sparse"/*; do
        [ -f "$f" ] || continue
        base="$(basename "$f")"
        [ "$base" = "0" ] && continue
        mv "$f" "$OUTPUT_SCENE/sparse/0/$base"
    done
fi
echo "✅ [4] image_undistorter done → $OUTPUT_SCENE/{images, sparse/0/}"

# ── 5. 清理 distorted/ 临时目录 ──────────────────────────────────────────
if [ "$KEEP_DISTORTED" != "1" ]; then
    echo "🧹 [5] 清理 $DISTORTED_DIR/ (KEEP_DISTORTED=1 保留排查)"
    rm -rf "$DISTORTED_DIR"
else
    echo "  ⏭️  KEEP_DISTORTED=1 → 保留 $DISTORTED_DIR/ (colmap gui 可加载排查)"
fi

# ── 6. 验证输出 ───────────────────────────────────────────────────────────
echo ""
echo "🔍 [verify] 输出文件检查"
for f in \
    "$OUTPUT_SCENE/images" \
    "$OUTPUT_SCENE/sparse/0/cameras.bin" \
    "$OUTPUT_SCENE/sparse/0/images.bin" \
    "$OUTPUT_SCENE/sparse/0/points3D.bin"; do
    if [ -e "$f" ]; then
        if [ -d "$f" ]; then
            n_img="$(find "$f" -maxdepth 1 -type f \( -name '*.jpg' -o -name '*.png' \) | wc -l)"
            echo "  ✅ $f/ ($n_img 张图像)"
        else
            size="$(du -h "$f" | cut -f1)"
            echo "  ✅ $f ($size)"
        fi
    else
        echo "  ❌ $f MISSING" >&2
    fi
done

# 图像数 vs 重建相机数（mapper 可能丢弃部分视角）
n_images="$(find "$OUTPUT_SCENE/images" -maxdepth 1 -type f \( -name '*.jpg' -o -name '*.png' \) | wc -l)"
n_cameras_bin_size="$(stat -c%s "$OUTPUT_SCENE/sparse/0/images.bin" 2>/dev/null || echo 0)"
# images.bin 每行 = 一个相机；每条记录固定大小（SIMPLE_PINHOLE 24B + 8B name 长度字段 + ...）
# 不准；用 colmap model_converter 转 txt 后 wc -l 更可靠
if [ -x "$COLMAP_EXECUTABLE" ]; then
    TMP_TXT="$(mktemp -d)/images.txt"
    "$COLMAP_EXECUTABLE" model_converter \
        --input_path "$OUTPUT_SCENE/sparse/0" \
        --output_type TXT \
        --output_path "$TMP_TXT" 2>/dev/null || true
    if [ -f "$TMP_TXT" ]; then
        n_cameras="$(grep -v '^#' "$TMP_TXT" | grep -v '^$' | wc -l)"
        echo "  📊 重建相机数: $n_cameras / 输入图像 $n_images"
        if [ "$n_cameras" -lt "$n_images" ]; then
            echo "  ⚠️ mapper 丢弃了 $((n_images - n_cameras)) 个视角（匹配不足/重叠加不够）"
        fi
        rm -rf "$(dirname "$TMP_TXT")"
    fi
fi

echo ""
echo "🎉 [03] Done. COLMAP NeRF-DS 场景就绪。"
echo "  📁 $OUTPUT_SCENE/"
echo "     ├── images/                    $(find "$OUTPUT_SCENE/images" -maxdepth 1 -type f \( -name '*.jpg' -o -name '*.png' \) | wc -l) 张去畸变图像"
echo "     └── sparse/0/"
echo "         ├── cameras.bin            相机内参"
echo "         ├── images.bin             相机外参"
echo "         └── points3D.bin           稀疏 3D 点（高斯初始化）"
echo ""
echo "  → 训练: bash $SCRIPT_DIR/04_train_real.sh  (SOURCE_PATH=$OUTPUT_SCENE)"

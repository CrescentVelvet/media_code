#!/usr/bin/env bash
# 04_train_real.sh — 真实拍摄人体序列训练 Deformable-3D-Gaussians。
#
# 调官方 train.py 训练 deformation MLP + canonical 高斯。针对真实多视角图像
# 序列（NeRF-DS 设定：人物保持静止时有微动——呼吸/衣摆/姿势漂移），不传
# --is_blender，默认 20000 步（NeRF-DS/HyperNeRF 标配，非 D-NeRF 的 40000）。
#
# 与 run_all.sh 的训练段区别：
#   - run_all 默认 D-NeRF Blender 模式（--is_blender, 40000 步, IS_BLENDER=1）
#   - 04 默认 NeRF-DS 真实模式（不带 --is_blender, 20000 步, IS_BLENDER=0）
#   - run_all 会先调 00 装 deps + 01 下 D-NeRF + clone 仓——04 假设这些都做过了
#   - 04 SOURCE_PATH 默认指向 03 输出的 COLMAP 场景（$MODEL_DIR/data/real/<scene>）
#   - 04 加 WHITE_BG 选项（白底分割图训练；NeRF-DS 原始图像不用开）
#
# Pipeline (train.py):
#   1. Scene 加载 COLMAP sparse/0/*.bin + images/ → 初始化高斯（用 simple_knn 从
#      points3D.bin 建点云）+ 分 train/test split（--eval 时 llffhold=8）
#   2. DeformModel 初始化（MLP, warm_up=3000 步前变形量为 0）
#   3. 训练循环（默认 20000 步）:
#      - 随机取一帧 viewpoint_cam，读 fid
#      - warm_up 后调 deform.step(xyz, fid) → d_xyz/d_rotation/d_scaling
#      - 可微光栅化（depth-diff-gaussian-rasterization）→ 渲染图 + depth
#      - loss = (1-λ)*L1 + λ*(1-SSIM)  (λ_dssim=0.2)
#      - backward + 致密化/剪枝（前 15000 步）+ opacity 重置（每 3000 步）
#   4. test_iterations (默认 7000+10000-40000 每 1000) 时评测 + log tensorboard
#   5. save_iterations (默认 7000/10000/20000/30000/40000 + 末步) 存 checkpoint
#
# 输入：SOURCE_PATH/sparse/0/*.bin + SOURCE_PATH/images/*.jpg（03 输出）
# 输出：MODEL_PATH/point_cloud/iteration_<N>/point_cloud.ply（高斯点云）
#       + cfg_args（render/metrics 读它恢复 is_blender/is_6dof/source_path）
#       + deform/（变形 MLP 权重）+ input.ply + cameras.json + TensorBoard events
#
# Env (all optional, defaults shown):
#   SOURCE_PATH=           # COLMAP 场景目录（默认 03 输出 $MODEL_DIR/data/real/<scene>）
#   SCENE_NAME=real_scene  # 场景名（影响默认 SOURCE_PATH + MODEL_PATH）
#   MODEL_PATH=            # 训练输出目录（默认 $DG_DIR/output/real_<scene>）
#   ITERATIONS=20000      # 训练步数（NeRF-DS 默认；D-NeRF 用 40000）
#   IS_BLENDER=0          # 0=真实数据（默认）；1=D-NeRF 合成（加 --is_blender）
#   IS_6DOF=0             # 1=6DoF 变体（指标略高、更慢）
#   WHITE_BG=0            # 1=白底训练（输入是白底分割图时开；NeRF-DS 原图关）
#   EVAL=1                # 1=划分 train/test（评测必需；0=全 train 不留 test）
#   TEST_ITERATIONS=      # 评测步（默认 train.py: 5000,6000,7000,10000-40000 每1000）
#   SAVE_ITERATIONS=      # 存 checkpoint 步（默认 train.py: 7000,10000,20000,30000,40000）
#   SKIP_VERIFY=0         # 1=跳过 CUDA 扩展 import 校验（已知装好时省秒）
#   EXTRA_TRAIN_ARGS=    # 透传给 train.py（如 --sh_degree 2 --port 0）
set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/_env.sh"

# ── Paths ─────────────────────────────────────────────────────────────────
DG_DIR="${DG_DIR:-$REPO_DIR/../Deformable-3D-Gaussians}"
MODEL_DIR="${MODEL_DIR:-$REPO_DIR/../../model/deformable-3d-gaussians}"
SCENE_NAME="${SCENE_NAME:-real_scene}"
SOURCE_PATH="${SOURCE_PATH:-$MODEL_DIR/data/real/$SCENE_NAME}"
# 训练输出（与 D-NeRF 复现分开：用 real_ 前缀避免覆盖 D-NeRF 的 $DG_DIR/output/<scene>）
MODEL_PATH="${MODEL_PATH:-$DG_DIR/output/real_$SCENE_NAME}"

# ── Params ────────────────────────────────────────────────────────────────
ITERATIONS="${ITERATIONS:-20000}"
IS_BLENDER="${IS_BLENDER:-0}"
IS_6DOF="${IS_6DOF:-0}"
WHITE_BG="${WHITE_BG:-0}"
EVAL="${EVAL:-1}"
SKIP_VERIFY="${SKIP_VERIFY:-0}"
EXTRA_TRAIN_ARGS="${EXTRA_TRAIN_ARGS:-}"

echo "🚀 [04] 训练 Deformable-3D-Gaussians（真实拍摄序列, NeRF-DS 模式）"
echo "  🤖 代码:       $DG_DIR"
echo "  📂 数据源:     $SOURCE_PATH"
echo "  💾 输出:       $MODEL_PATH"
echo "  📐 iterations: $ITERATIONS  is_blender: $IS_BLENDER  is_6dof: $IS_6DOF"
echo "  🎨 white_bg:   $WHITE_BG  eval: $EVAL"
if [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
    echo "  🎮 GPU:        physical $CUDA_VISIBLE_DEVICES (cuda:0 in-process)"
else
    echo "  🎮 GPU:        default cuda:0 (= first visible)  [set GPU=N to pin]"
fi
echo ""

# ── 0. Sanity checks ──────────────────────────────────────────────────────
# 代码仓
if [ ! -f "$DG_DIR/train.py" ]; then
    echo "❌ ERROR: Deformable-GS 代码仓未找到: $DG_DIR/train.py" >&2
    echo "       首次跑先: bash $SCRIPT_DIR/run_all.sh  (会 clone 仓 + 装 env + 编 CUDA)" >&2
    echo "       或手动: git clone --recursive https://github.com/ingra14m/Deformable-3D-Gaussians.git $DG_DIR" >&2
    exit 1
fi

# 数据源（COLMAP 场景）
if [ ! -d "$SOURCE_PATH/sparse/0" ]; then
    echo "❌ ERROR: COLMAP 场景未找到: $SOURCE_PATH/sparse/0/" >&2
    echo "       先跑 step 03 COLMAP 位姿估计:" >&2
    echo "         INPUT_DIR=<拍摄图像目录> SCENE_NAME=$SCENE_NAME bash $SCRIPT_DIR/03_colmap_pose.sh" >&2
    echo "       或若已有 NeRF-DS 格式数据, 设 SOURCE_PATH=/path/to/scene" >&2
    exit 1
fi
if [ ! -d "$SOURCE_PATH/images" ]; then
    echo "❌ ERROR: $SOURCE_PATH/images/ 不存在（COLMAP 场景缺图像）" >&2
    exit 1
fi

# CUDA 扩展（train.py 会 import diff_gaussian_rasterization + simple_knn）
if [ "$SKIP_VERIFY" != "1" ]; then
    if ! python -c "import diff_gaussian_rasterization, simple_knn" 2>/dev/null; then
        echo "❌ ERROR: CUDA 扩展不可 import (diff_gaussian_rasterization / simple_knn)。" >&2
        echo "       编译: BUILD_CUDA=1 bash $SCRIPT_DIR/00_setup_env.sh" >&2
        echo "       (需 CUDA toolkit 11.6 匹配 torch 1.13.1+cu116)" >&2
        exit 1
    fi
fi

# ── 1. 准备输出目录 ────────────────────────────────────────────────────────
mkdir -p "$MODEL_PATH"

# ── 2. 组装 train.py 参数 ───────────────────────────────────────────────────
TRAIN_FLAGS=(
    -s "$SOURCE_PATH"
    -m "$MODEL_PATH"
    --iterations "$ITERATIONS"
)
[ "$EVAL" = "1" ] && TRAIN_FLAGS+=(--eval)
[ "$IS_BLENDER" = "1" ] && TRAIN_FLAGS+=(--is_blender)
[ "$IS_6DOF" = "1" ] && TRAIN_FLAGS+=(--is_6dof)
[ "$WHITE_BG" = "1" ] && TRAIN_FLAGS+=(--white_background)

# test_iterations / save_iterations 覆盖（默认 train.py 自带）
if [ -n "${TEST_ITERATIONS:-}" ]; then
    # shellcheck disable=SC2206
    TRAIN_FLAGS+=(--test_iterations $TEST_ITERATIONS)
fi
if [ -n "${SAVE_ITERATIONS:-}" ]; then
    # shellcheck disable=SC2206
    TRAIN_FLAGS+=(--save_iterations $SAVE_ITERATIONS)
fi

# 关 GUI server（默认 127.0.0.1:6009；启动会卡住等连接）
TRAIN_FLAGS+=(--port 0)

# 透传额外参数
if [ -n "$EXTRA_TRAIN_ARGS" ]; then
    # shellcheck disable=SC2086
    TRAIN_FLAGS+=($EXTRA_TRAIN_ARGS)
fi

echo "🏋️ [2] train.py ${TRAIN_FLAGS[*]}"
echo "    (warm_up=3000 步前变形量=0; 前 15000 步致密化; 每 3000 步 opacity 重置)"
echo "    TensorBoard: tensorboard --logdir $MODEL_PATH --port 6006"
echo ""

# ── 3. 训练 ────────────────────────────────────────────────────────────────
# train.py 用相对 import（scene/, gaussian_renderer/, utils/）→ 必须在 $DG_DIR 里跑
( cd "$DG_DIR" && python train.py "${TRAIN_FLAGS[@]}" )
if [ $? -ne 0 ]; then
    echo "❌ FAILED. train.py 没跑完。" >&2
    echo "    常见原因:" >&2
    echo "      - OOM: 降 NUM_IMAGES_MAX 重跑 03 / 或减 --iterations" >&2
    echo "      - fid ValueError: 图像文件名不是纯数字; 03 应已重命名为 00000.jpg 等" >&2
    echo "      - CUDA ext ABI 不兼容: 用官方 torch==1.13.1+cu116 (py3.7)" >&2
    exit 1
fi

# ── 4. 总结 ─────────────────────────────────────────────────────────────────
# 找最后一个 iteration 目录
LAST_ITER="$(ls -d "$MODEL_PATH/point_cloud/iteration_"* 2>/dev/null \
    | sort -V | tail -1 | sed 's/.*iteration_//')"
if [ -z "$LAST_ITER" ]; then
    echo "⚠️ 没找到 point_cloud/iteration_*/ 目录（训练可能没存 checkpoint）" >&2
    LAST_ITER="$ITERATIONS"
fi

echo ""
echo "🎉 [04] Done. 训练完成。"
echo "  📊 模型路径:    $MODEL_PATH"
echo "  🏋️ 高斯点云:    $MODEL_PATH/point_cloud/iteration_$LAST_ITER/point_cloud.ply"
echo "  🎭 变形 MLP:    $MODEL_PATH/deform/  (deform.save_weights 输出)"
echo "  📝 cfg_args:    $MODEL_PATH/cfg_args  (render.py/metrics.py 读它恢复参数)"
echo "  📷 cameras.json: $MODEL_PATH/cameras.json"
echo ""
echo "  → 渲染 + 评测: bash $SCRIPT_DIR/05_render_real.sh  (MODEL_PATH=$MODEL_PATH)"
echo "  → 或直接调 02:  MODEL_PATH=$MODEL_PATH MODE=original bash $SCRIPT_DIR/02_run_inference.sh"

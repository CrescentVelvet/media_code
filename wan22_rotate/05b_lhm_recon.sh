#!/usr/bin/env bash
# 05b_lhm_recon.sh — LHM feed-forward 单图人体高斯重建 (ICCV 2025)
# (单张正面图 → LHM 前馈网络 → 可动画人体高斯 + 网格)
#
# 与 05 (2DGS) / 05a (GOF) 并列的第三种方案。范式区别:
#   - 05/05a: 多视图视频 → Pi3+COLMAP → 逐场景优化高斯 → 提网格 (per-scene opt)
#   - 05b:    单张正面图 → LHM 前馈网络 → 可动画人体高斯 → 导出网格 (feed-forward)
# 输入是 step 01 的 segmented_image.png (正面白底人物图), 无需 Pi3/COLMAP/视频。
# 速度快 (单次前向, 数秒出高斯), 但精度依赖 LHM 预训练, 不如 05/05a 逐场景拟合。
#
# ⚠️ 用独立 conda env `lhm` (torch 2.3.0 / numpy 1.23.0), 与 wan22_rotate 的
#    torch 2.6.0 / numpy 1.26.4 不兼容。本脚本在 source _env.sh 前设 CONDA_ENV=lhm
#    切换到 lhm env。首次需 INSTALL_LHM=1 建 env + 装依赖 + 下权重 (见 README「首次准备」)。
#
# 两个子步骤:
#   5b-1) 导出网格 (默认): 单图 → LHM → 人体高斯 .ply (canonical pose, 静止姿态)
#   5b-2) 渲染动画 (可选, 默认 SKIP_ANIM=1): 单图 + 动作序列 → 旋转动画 .mp4
#         动作来源 (三选一, 用 MOTION_SEQ 指定):
#           (a) LHM 自带 motion 示例 (mimo1 等, 00 装时 LHM_DOWNLOAD_MOTION=1 下载)
#           (b) 从 step 02 的 rotate_360.mp4 用 video2motion.py 提取 SMPL-X (EXTRACT_MOTION=1)
#           (c) 任意 smplx_params 目录 (MOTION_SEQ=<dir>)
#
# LHM 的输出固定写到仓内 exps/{meshs,videos}/video_human_benchmark/human-lrm-<size>/<model>/
# (路径由 parse_configs() 用 experiment.parent/child + model basename 拼成, 见 LHM/runners/infer/human_lrm.py)。
# 本脚本跑完后把 mesh/视频 copy 到 $MODEL_DIR_LHM/ (与 05/05a 的 model_2dgs/ model_gof/ 并列)。
#
# Prerequisites:
#   - INSTALL_DEPS=1 INSTALL_LHM=1 bash wan22_rotate/00_setup_env.sh (first time)
#   - step 01 已跑 (有 $RESULTS_DIR/segmented_image.png)
#
# Env (all optional, defaults shown):
#   IMAGE_INPUT=             # 单张人像图 (default: $RESULTS_DIR/segmented_image.png)
#   OUTPUT_NAME=rotate_360   # 影响默认 MODEL_DIR_LHM 子目录名
#   RESULTS_DIR=             # 输出根 (default: ../wan22_rotate_results)
#   MODEL_NAME=LHM-500M-HF   # LHM 模型 (alt: LHM-1B-HF, LHM-MINI; 须已在 00 下好)
#   MOTION_SEQ=              # 动作目录 (含 smplx_params); 留空则动画步骤用 LHM 默认 mimo1
#   EXTRACT_MOTION=0         # 1=从 rotate_360.mp4 提取 SMPL-X 动作 (需 00 装 LHM_DOWNLOAD_POSE=1)
#   RENDER_FPS=30            # 动画输出 fps
#   MOTION_READ_FPS=30       # 读动作序列的 fps
#   SKIP_MESH=0              # 1=跳过网格导出 (5b-1)
#   SKIP_ANIM=1              # 1=跳过动画渲染 (5b-2, 默认跳过; 给 MOTION_SEQ 或 EXTRACT_MOTION=1 才开)
#   DEVICE=cuda
set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# LHM 用独立 env (torch 2.3.0, 与 wan22_rotate 的 torch 2.6.0 不兼容)。
# 在 source _env.sh 前设 CONDA_ENV=lhm, _env.sh 的 ${CONDA_ENV:-wan22_rotate} 会保留 lhm。
CONDA_ENV="${CONDA_ENV:-lhm}"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/_env.sh"

# ── Paths ─────────────────────────────────────────────────────────────────
OUTPUT_NAME="${OUTPUT_NAME:-rotate_360}"
# 输入: step 01 的分割图 (正面白底人物)。也可直接传任意单张人像图。
IMAGE_INPUT="${IMAGE_INPUT:-$RESULTS_DIR/segmented_image.png}"
# LHM 仓 (代码 + 相对 ./pretrained_models/ 读权重)
# (LHM_DIR / LHM_MODEL_DIR 已在 _env.sh 定义)
# 本流程输出 (与 05 的 model_2dgs/ 05a 的 model_gof/ 并列)
MODEL_DIR_LHM="${MODEL_DIR_LHM:-$RESULTS_DIR/${OUTPUT_NAME}/model_lhm}"

# ── Params ────────────────────────────────────────────────────────────────
MODEL_NAME="${MODEL_NAME:-LHM-500M-HF}"
EXTRACT_MOTION="${EXTRACT_MOTION:-0}"
MOTION_SEQ="${MOTION_SEQ:-}"
RENDER_FPS="${RENDER_FPS:-30}"
MOTION_READ_FPS="${MOTION_READ_FPS:-30}"
DEVICE="${DEVICE:-cuda}"
# 子步骤开关
SKIP_MESH="${SKIP_MESH:-0}"
SKIP_ANIM="${SKIP_ANIM:-1}"

echo "🚀 [05b] LHM feed-forward 人体高斯重建 (单图 → 高斯 + 网格)"
echo "  🤖 model:       $MODEL_NAME (env 'lhm', torch 2.3.0)"
echo "  🖼️  image_input: $IMAGE_INPUT"
echo "  💾 output:      $MODEL_DIR_LHM"
echo "  📐 skip_mesh: $SKIP_MESH  skip_anim: $SKIP_ANIM"
echo ""

# ── 0. Sanity checks ──────────────────────────────────────────────────────
# 0a. lhm env 存在
if ! conda env list 2>/dev/null | grep -qw "lhm"; then
    echo "❌ ERROR: conda env 'lhm' not found." >&2
    echo "       Run: INSTALL_DEPS=1 INSTALL_LHM=1 bash $SCRIPT_DIR/00_setup_env.sh" >&2
    exit 1
fi
# CONDA_ENV=lhm 已由 _env.sh 激活; 校验确实在 lhm env 里
_cur_env="$(python -c 'import sys; print(sys.prefix)' 2>/dev/null)"
if [ -z "$(echo "$_cur_env" | grep -o 'lhm')" ]; then
    echo "⚠️  current env ($_cur_env) 不是 lhm, 强制切换..." >&2
    conda activate lhm || { echo "❌ conda activate lhm 失败" >&2; exit 1; }
fi
echo "  [OK] env: $(python --version 2>&1 | cut -d' ' -f2) @ $(basename "$(python -c 'import sys;print(sys.prefix)')")"

# 0b. LHM 仓 + pretrained_models 软链
if [ ! -f "$LHM_DIR/LHM/launch.py" ]; then
    echo "❌ ERROR: LHM repo not found at $LHM_DIR (missing LHM/launch.py)." >&2
    echo "       Run: INSTALL_DEPS=1 INSTALL_LHM=1 bash $SCRIPT_DIR/00_setup_env.sh" >&2
    exit 1
fi
if [ ! -e "$LHM_DIR/pretrained_models" ]; then
    echo "⚠️  $LHM_DIR/pretrained_models 不存在, 建软链 -> $LHM_MODEL_DIR" >&2
    mkdir -p "$LHM_MODEL_DIR"
    ln -sfn "$LHM_MODEL_DIR" "$LHM_DIR/pretrained_models"
fi

# 0c. LHM prior models (SMPL-X / sapiens / sam2 / gagatracker / dense_sample_points)
if [ ! -d "$LHM_MODEL_DIR/human_model_files" ]; then
    echo "❌ ERROR: LHM prior models missing (no human_model_files/ at $LHM_MODEL_DIR)." >&2
    echo "       Run: INSTALL_DEPS=1 INSTALL_LHM=1 bash $SCRIPT_DIR/00_setup_env.sh" >&2
    echo "       (or 手动下 LHM_prior_model.tar 解压到 $LHM_MODEL_DIR)" >&2
    exit 1
fi
# 0d. LHM 主权重 (huggingface/ cache)
if [ -z "$(ls -d "$LHM_MODEL_DIR/huggingface/models--3DAIGC--"* 2>/dev/null)" ]; then
    echo "❌ ERROR: LHM model '$MODEL_NAME' not found in $LHM_MODEL_DIR/huggingface/." >&2
    echo "       Run: INSTALL_DEPS=1 INSTALL_LHM=1 bash $SCRIPT_DIR/00_setup_env.sh" >&2
    echo "       (or 手动: from huggingface_hub import snapshot_download; " >&2
    echo "                snapshot_download(repo_id='3DAIGC/$MODEL_NAME', cache_dir='$LHM_MODEL_DIR/huggingface'))" >&2
    exit 1
fi

# 0e. 输入图存在
if [ ! -f "$IMAGE_INPUT" ]; then
    echo "❌ ERROR: input image not found: $IMAGE_INPUT" >&2
    echo "       Run step 01 first (produces $RESULTS_DIR/segmented_image.png)," >&2
    echo "       or set IMAGE_INPUT=<path-to-single-portrait.png>" >&2
    exit 1
fi

# 0f. verify 关键 import
if ! python -c "import diff_gaussian_rasterization, simple_knn" 2>/dev/null; then
    echo "❌ ERROR: LHM CUDA exts not importable (diff_gaussian_rasterization / simple_knn)." >&2
    echo "       Run: INSTALL_DEPS=1 INSTALL_LHM=1 bash $SCRIPT_DIR/00_setup_env.sh" >&2
    exit 1
fi
python -c "import torch; print('  [OK] torch', torch.__version__, 'cuda', torch.version.cuda)" 2>/dev/null

mkdir -p "$MODEL_DIR_LHM"
IMAGE_BASE="$(basename "$IMAGE_INPUT")"
# LHM 网格文件名 = '_'.join(basename.split('.')[:-1]) + '.ply'（去最后扩展名, 其余 . 换 _）。
# 视频文件名 uid = basename.split('.')[0]（第一个 . 之前）。单扩展名文件两者一致。
IMAGE_STEM="$(basename "$IMAGE_INPUT" | sed 's/\.[^.]*$//' | tr '.' '_')"
echo ""
echo "  📁 image stem: $IMAGE_STEM  (mesh 文件名将 = $IMAGE_STEM.ply)"
echo ""

# ── (可选) 从 rotate_360.mp4 提取 SMPL-X 动作 ──────────────────────────────
# 用 LHM 自带 engine/pose_estimation/video2motion.py, 需 00 装 LHM_DOWNLOAD_POSE=1
# (yolov8x + vitpose + mmcv==1.3.9 + ultralytics + ViTPose)。
if [ "$EXTRACT_MOTION" = "1" ]; then
    SRC_VIDEO="${SRC_VIDEO:-$RESULTS_DIR/${OUTPUT_NAME}.mp4}"
    if [ ! -f "$SRC_VIDEO" ]; then
        echo "❌ ERROR: EXTRACT_MOTION=1 但源视频不存在: $SRC_VIDEO" >&2
        echo "       先跑 step 02 生成 ${OUTPUT_NAME}.mp4, 或设 SRC_VIDEO=<path>" >&2
        exit 1
    fi
    MOTION_OUT="${MOTION_OUT:-$MODEL_DIR_LHM/custom_motion}"
    echo "🎬 从视频提取 SMPL-X 动作 (video2motion.py)"
    echo "  🎞️  src:    $SRC_VIDEO"
    echo "  💾 motion: $MOTION_OUT"
    echo ""
    if [ ! -f "$LHM_MODEL_DIR/human_model_files/pose_estimate/yolov8x.pt" ] || \
       [ ! -f "$LHM_MODEL_DIR/human_model_files/pose_estimate/vitpose-h-wholebody.pth" ]; then
        echo "❌ ERROR: video2motion 权重缺失 (yolov8x.pt / vitpose-h-wholebody.pth)." >&2
        echo "       Run: INSTALL_DEPS=1 INSTALL_LHM=1 LHM_DOWNLOAD_POSE=1 bash $SCRIPT_DIR/00_setup_env.sh" >&2
        exit 1
    fi
    if ! python -c "import mmcv, ultralytics, vitpose" 2>/dev/null; then
        echo "❌ ERROR: video2motion 依赖缺失 (mmcv/ultralytics/ViTPose)." >&2
        echo "       Run: INSTALL_DEPS=1 INSTALL_LHM=1 LHM_DOWNLOAD_POSE=1 bash $SCRIPT_DIR/00_setup_env.sh" >&2
        exit 1
    fi
    ( cd "$LHM_DIR" && python ./engine/pose_estimation/video2motion.py \
        --video_path "$SRC_VIDEO" --output_path "$MOTION_OUT" )
    if [ $? -ne 0 ]; then
        echo "❌ FAILED. video2motion 未完成." >&2
        exit 1
    fi
    MOTION_SEQ="$MOTION_OUT/$(basename "$SRC_VIDEO" .mp4)/smplx_params"
    echo "✅ 动作提取完成: $MOTION_SEQ"
    echo ""
    # 有动作了就默认开动画
    [ "$SKIP_ANIM" = "1" ] && echo "ℹ️  EXTRACT_MOTION=1 已出动作, 自动开动画 (SKIP_ANIM=0)" && SKIP_ANIM=0
fi

# ── 5b-1) 导出网格 (canonical pose, 静止姿态) ─────────────────────────────
# LHM runner: export_mesh is not None -> infer_mesh() -> 输出 <image_stem>.ply
# 输出到 $LHM_DIR/exps/meshs/<relative_path>/<image_stem>.ply
if [ "$SKIP_MESH" = "1" ]; then
    echo "⏭️  [skip 5b-1] SKIP_MESH=1 (跳过网格导出)"
else
    echo "✂️  [5b-1] LHM mesh 导出 (canonical pose, feed-forward 单图 -> 高斯 -> 网格)"
    echo "  🖼️  image:  $IMAGE_INPUT"
    echo "  🤖 model: $MODEL_NAME"
    echo "  💾 输出到 LHM 仓内 exps/meshs/... (跑完 copy 到 $MODEL_DIR_LHM)"
    echo ""
    # 清旧 mesh (同名会覆盖, 但 glob 时避免捞到上次的)
    find "$LHM_DIR/exps/meshs" -name "${IMAGE_STEM}.ply" -delete 2>/dev/null || true
    ( cd "$LHM_DIR" && python -m LHM.launch infer.human_lrm \
        model_name="$MODEL_NAME" \
        image_input="$IMAGE_INPUT" \
        export_mesh=True \
        motion_seqs_dir=None \
        motion_img_dir=None \
        vis_motion=true \
        motion_img_need_mask=true )
    if [ $? -ne 0 ]; then
        echo "❌ [5b-1] FAILED. LHM mesh 导出未完成." >&2
        echo "       常见原因: (1) prior_model 缺文件 (dense_sample_points/sapiens 等)" >&2
        echo "                 (2) pytorch3d 未装 (见 00 verify)" >&2
        echo "                 (3) 显存不足 (试 MODEL_NAME=LHM-MINI, 16GB 可跑)" >&2
        exit 1
    fi
    # 找输出 mesh 并 copy 到 $MODEL_DIR_LHM
    FOUND_MESH="$(find "$LHM_DIR/exps/meshs" -name "${IMAGE_STEM}.ply" 2>/dev/null | head -1)"
    if [ -z "$FOUND_MESH" ] || [ ! -f "$FOUND_MESH" ]; then
        echo "⚠️  没找到输出 mesh ${IMAGE_STEM}.ply, 列出 exps/meshs 全部 .ply:" >&2
        find "$LHM_DIR/exps/meshs" -name '*.ply' 2>/dev/null | head -20 >&2
        echo "       (LHM 输出路径可能因 model_name 解析不同; 手动 copy 上面的 .ply)" >&2
    else
        cp -f "$FOUND_MESH" "$MODEL_DIR_LHM/${IMAGE_STEM}.ply"
        echo "✅ [5b-1] mesh 导出完成"
        echo "  🌐 mesh: $MODEL_DIR_LHM/${IMAGE_STEM}.ply  (源: $FOUND_MESH)"
    fi
    echo ""
fi

# ── 5b-2) 渲染动画 (可选; 需动作序列) ─────────────────────────────────────
# LHM runner: export_mesh=None + export_video=True -> infer_single() -> 输出 <image_stem>.mp4
# 动作: MOTION_SEQ 指定 (smplx_params 目录); 留空用 LHM 默认 mimo1
if [ "$SKIP_ANIM" = "1" ]; then
    echo "⏭️  [skip 5b-2] SKIP_ANIM=1 (跳过动画渲染; 设 MOTION_SEQ=<dir> 或 EXTRACT_MOTION=1 开启)"
else
    # 解析动作目录: 优先 MOTION_SEQ, 否则 LHM 自带 mimo1
    if [ -z "$MOTION_SEQ" ]; then
        MOTION_SEQ="$LHM_DIR/train_data/motion_video/mimo1/smplx_params"
    fi
    if [ ! -d "$MOTION_SEQ" ]; then
        echo "❌ ERROR: 动作目录不存在: $MOTION_SEQ" >&2
        echo "       设 MOTION_SEQ=<smplx_params dir>, 或 EXTRACT_MOTION=1 从视频提取," >&2
        echo "       或 Run: INSTALL_DEPS=1 INSTALL_LHM=1 LHM_DOWNLOAD_MOTION=1 bash 00_setup_env.sh (下 LHM 自带 motion)" >&2
        exit 1
    fi
    MOTION_NAME="$(basename "$(dirname "$MOTION_SEQ")")"
    echo "🎬 [5b-2] LHM 动画渲染 (单图 + 动作 -> 旋转动画)"
    echo "  🖼️  image:  $IMAGE_INPUT"
    echo "  🤖 model: $MODEL_NAME"
    echo "  💃 motion: $MOTION_SEQ  (name: $MOTION_NAME)"
    echo "  📐 render_fps: $RENDER_FPS  read_fps: $MOTION_READ_FPS"
    echo ""
    # 清旧视频
    find "$LHM_DIR/exps/videos" -name "${IMAGE_STEM}.mp4" -delete 2>/dev/null || true
    ( cd "$LHM_DIR" && python -m LHM.launch infer.human_lrm \
        model_name="$MODEL_NAME" \
        image_input="$IMAGE_INPUT" \
        export_video=True \
        motion_seqs_dir="$MOTION_SEQ" \
        motion_img_dir=None \
        vis_motion=true \
        motion_img_need_mask=true \
        render_fps="$RENDER_FPS" \
        motion_video_read_fps="$MOTION_READ_FPS" )
    if [ $? -ne 0 ]; then
        echo "❌ [5b-2] FAILED. LHM 动画渲染未完成." >&2
        echo "       常见原因: 动作序列格式不对 / 显存不足 (试 MODEL_NAME=LHM-MINI)" >&2
        exit 1
    fi
    FOUND_VIDEO="$(find "$LHM_DIR/exps/videos" -name "${IMAGE_STEM}.mp4" 2>/dev/null | head -1)"
    if [ -z "$FOUND_VIDEO" ] || [ ! -f "$FOUND_VIDEO" ]; then
        echo "⚠️  没找到输出视频 ${IMAGE_STEM}.mp4, 列出 exps/videos 全部 .mp4:" >&2
        find "$LHM_DIR/exps/videos" -name '*.mp4' 2>/dev/null | head -20 >&2
    else
        cp -f "$FOUND_VIDEO" "$MODEL_DIR_LHM/${IMAGE_STEM}_anim.mp4"
        echo "✅ [5b-2] 动画渲染完成"
        echo "  🎬 video: $MODEL_DIR_LHM/${IMAGE_STEM}_anim.mp4  (源: $FOUND_VIDEO)"
    fi
    echo ""
fi

echo "🎉 [05b] Done. LHM 人体高斯重建完成."
echo "  🌐 Mesh:   $MODEL_DIR_LHM/${IMAGE_STEM}.ply  (canonical pose, 用 MeshLab 看)"
echo "  🎬 Anim:   $MODEL_DIR_LHM/${IMAGE_STEM}_anim.mp4  (若 5b-2 跑了)"
echo ""
echo "  💡 LHM vs 05/05a: 单图前馈 (秒级), 不需 Pi3+COLMAP; 但精度依赖预训练,"
echo "     不如 05 (2DGS) / 05a (GOF) 的逐场景拟合。对比:"
echo "     05_3dgs_recon.sh  -> model_2dgs/test/ours_*/mesh.ply"
echo "     05a_3dgs_recon.sh -> model_gof/test/ours_*/fusion/mesh_binary_search_7.ply"
echo "     05b (本脚本)     -> model_lhm/${IMAGE_STEM}.ply"

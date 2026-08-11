# Wan2.2 Rotate — 环绕人物 → 正面选图 → 分割 → 360° 旋转视频

从一组环绕人物拍摄的图像中，自动挑选人物面向相机的正面图像，用 SAM 3D Body 识别并分割人物（背景置白），再用 Wan2.2-TI2V-5B + 已训练的 LoRA 生成 360° 旋转视频。

> 选正面图有三种方式（按需选一）：
> - **01 完整版** — SAM 3D Body 3D 姿态估计（`global_rot` 算正面评分；需 GATED 权重，最准）
> - **01b 简化版** — ViTDet 检测 + 按人物面积最大选图（不加载 3D body 模型，更快，但不保证正面）
> - **01c MediaPipe 版** — MediaPipe Face Mesh 算正面评分（鼻尖居中 + 双眼距离最大；CPU 即可，无需 GATED 权重，比 01b 更能锁定"正面"）

本目录只含编排脚本——SAM 3D Body 官方代码在 `../sam-3d-body`、DiffSynth-Studio 在 `../DiffSynth-Studio`，权重在各算法的 `$MODEL_DIR` 下。两步共用同一个 conda env（从 `doll` 克隆，装两套依赖）。

## 常用命令

> 假设已进入容器（脚本自动激活 `wan22_rotate` env）；`GPU=0` 按需换卡。首次跑前先做下方「首次准备」。

```bash
# ── 分步 ──
# 1a) 选图+分割 完整版（SAM 3D Body: 3D 姿态估计选正面图 + SAM 分割）
GPU=0 INPUT_DIR=../Reconstruction/dataset/B003_Human_Data_w_pose/test_task_id_3a8b3cc746304f49b9e3275e36aa9374 \
  RESULTS_DIR=../../output/wan22_rotate_results \
  bash wan22_rotate/01_pick_and_segment.sh
# 1b) 选图+分割 简化版（只 ViTDet 检测 + SAM 分割, 按人物面积最大选图, 不加载 3D body 模型, 更快）
GPU=0 INPUT_DIR=../Reconstruction/dataset/B003_Human_Data_w_pose/test_task_id_3a8b3cc746304f49b9e3275e36aa9374 \
  RESULTS_DIR=../../output/wan22_rotate_results \
  bash wan22_rotate/01b_pick_and_segment.sh
# 1c) 选图+分割 MediaPipe 版（Face Mesh 算正面评分: 鼻尖居中 + 双眼距离最大; CPU 即可, 无需 GATED 权重）
#     比 01b 更能锁定"正面"（01b 选面积最大, 可能选到侧面）; 比 01 快很多（不用 3D body 模型）
#     仅选图不分割: SKIP_SEGMENTATION=1（快速看哪帧是正面）
GPU=0 INPUT_DIR=../Reconstruction/dataset/B003_Human_Data_w_pose/test_task_id_3a8b3cc746304f49b9e3275e36aa9374 \
  RESULTS_DIR=../../output/wan22_rotate_results \
  bash wan22_rotate/01c_pick_and_segment.sh

# 2) 只做视频生成（用上一步的分割图）
#    模型: Wan2.2-TI2V-5B, 从 WAN_MODEL_PATH 直接加载 (ModelConfig(path=...))
GPU=0 SEGMENTED_IMAGE=../../output/wan22_rotate_results/segmented_image_centered.png \
  WEIGHT_PATH=../../model/Wan2.2-TI2V-5B_lora_add_data_reload/step-66900.safetensors \
  WAN_MODEL_PATH=../../model/Wan2.2-TI2V-5B \
  RESULTS_DIR=../../output/wan22_rotate_results \
  bash wan22_rotate/02_generate_video.sh

# 3) 拆分视频为 JPG 帧（输出到 <视频同名>/image/，匹配 INPUT_DIR/image/ 模式）
#    默认抽每一帧（FPS=0）；指定 FPS 则按该 fps 采样
GPU=0 VIDEO_PATH=../../output/wan22_rotate_results/rotate_360.mp4 \
  RESULTS_DIR=../../output/wan22_rotate_results \
  bash wan22_rotate/03_extract_frames.sh
  
# 输出结构：
#   rotate_360.mp4
#   rotate_360/                 # 同名目录
#     image/
#       00000.jpg, 00001.jpg, ...

# 4) Pi3 位姿估计（不进行三维重建，只出位姿 + 稠密点云）
#    复用 wan22_rotate env（已有 Pi3 全部依赖）；调 pi3_3dgs/pi3_recon.py --no_colmap
#    自动用步骤 03 的 JPG（若已跑），否则自动调 03 抽帧
GPU=0 PI3_CKPT=../../model/Pi3/model.safetensors \
  INPUT=../../output/wan22_rotate_results/rotate_360/image \
  RESULTS_DIR=../../output/wan22_rotate_results \
  bash wan22_rotate/04_pi3_pose.sh

# 输出：rotate_360/pi3/{predictions.npz, dense_cloud.ply, poses.json}
# poses.json 是人类可读的 c2w 4x4 矩阵（每帧一个），可用任何 JSON viewer 看

# 5) 三维高斯重建（Pi3 → COLMAP → 2DGS 训练 → 渲染 + 网格）
#    在 wan22_rotate env 里跑（首次需 INSTALL_2DGS=1 编 2DGS CUDA 扩展，见下方「首次准备」）
#    一键（Pi3 重跑带 COLMAP 导出 + 2DGS 训练 + 渲染 + 网格）：
GPU=0 PI3_CKPT=../../model/Pi3/model.safetensors \
  INPUT=../../output/wan22_rotate_results/rotate_360.mp4 \
  RESULTS_DIR=../../output/wan22_rotate_results \
  bash wan22_rotate/05_3dgs_recon.sh
# 分步（05_3dgs_recon.sh 内部三步，可单独跳过）：
# 5a) Pi3 推理 + COLMAP 导出（视频抽帧 → Pi3 → cameras/images/points3D.txt）
GPU=0 PI3_CKPT=../../model/Pi3/model.safetensors \
  INPUT=../../output/wan22_rotate_results/rotate_360.mp4 \
  RESULTS_DIR=../../output/wan22_rotate_results \
  SKIP_TRAIN=1 SKIP_RENDER=1 \
  bash wan22_rotate/05_3dgs_recon.sh
# 5b) 2DGS 训练（白底适配 wan22_rotate 分割图，默认 WHITE_BG=1）
GPU=0 PI3_CKPT=../../model/Pi3/model.safetensors \
  RESULTS_DIR=../../output/wan22_rotate_results \
  SKIP_PI3=1 \
  bash wan22_rotate/05_3dgs_recon.sh
# 5c) 渲染 + 提网格（无界 TSDF 适配人像在白色虚空中，默认 UNBOUNDED=1）
GPU=0 PI3_CKPT=../../model/Pi3/model.safetensors \
  RESULTS_DIR=../../output/wan22_rotate_results \
  SKIP_PI3=1 SKIP_TRAIN=1 \
  bash wan22_rotate/05_3dgs_recon.sh
# 输出：<RESULTS_DIR>/rotate_360/
#   pi3/{predictions.npz, dense_cloud.ply, poses.json, source/}   Pi3 + COLMAP 场景
#   model/point_cloud/iteration_<N>/point_cloud.ply                高斯点云
#   model/test/ours_<N>/{renders/*.png, mesh.ply}                 渲染图 + 网格

# ── 自定义 ──
# 换 prompt / 分辨率 / 帧数（portrait 默认 1248×704；landscape 用 704×1248）
GPU=0 INPUT_DIR=../Reconstruction/dataset/B003_Human_Data_w_pose/test_task_id_3a8b3cc746304f49b9e3275e36aa9374 \
  RESULTS_DIR=../../output/wan22_rotate_results \
  PROMPT="人物360度旋转展示，高质量。" \
  HEIGHT=1248 WIDTH=706 NUM_FRAMES=121 \
  bash wan22_rotate/02_generate_video.sh
# 跳过选图步骤，直接用已有图片生成视频
GPU=0 \
  SEGMENTED_IMAGE=../../output/wan22_rotate_results/segmented_image_centered.png \
  WEIGHT_PATH=../../model/Wan2.2-TI2V-5B_lora_add_data_reload/step-66900.safetensors \
  WAN_MODEL_PATH=../../model/Wan2.2-TI2V-5B \
  RESULTS_DIR=../../output/wan22_rotate_results \
  bash wan22_rotate/02_generate_video.sh
# 选出的图是背面？翻转正面判定方向
GPU=0 FRONTAL_SIGN=-1 \
  INPUT_DIR=../Reconstruction/dataset/B003_Human_Data_w_pose/test_task_id_3a8b3cc746304f49b9e3275e36aa9374 \
  RESULTS_DIR=../../output/wan22_rotate_results \
  bash wan22_rotate/01_pick_and_segment.sh
# 用 SAM2 分割器（需提前放好 sam2 仓库 + checkpoint）
GPU=0 SEGMENTOR_PATH=../sam2 \
  INPUT_DIR=../Reconstruction/dataset/B003_Human_Data_w_pose/test_task_id_3a8b3cc746304f49b9e3275e36aa9374 \
  RESULTS_DIR=../../output/wan22_rotate_results \
  bash wan22_rotate/01_pick_and_segment.sh
```

- 结果：分割图 → `../wan22_rotate_results/segmented_image.png`；视频 → `../wan22_rotate_results/rotate_360.mp4`；JPG 帧 → `../wan22_rotate_results/rotate_360/image/*.jpg`；Pi3 位姿 → `../wan22_rotate_results/rotate_360/pi3/{predictions.npz,poses.json,dense_cloud.ply}`；调试信息 → `frontal_scores.csv` + `debug_mask.png`。

## → 接入三维重建（Pi3 + 2D Gaussian Splatting）

步骤 04 只估位姿（`--no_colmap`，不做三维重建）。步骤 5 在**同一个 wan22_rotate env** 里完成 3DGS 重建（Pi3 重跑带 COLMAP 导出 + 2DGS 训练 + 渲染 + 网格），无需 pi3_3dgs 独立 env。首次需 `INSTALL_2DGS=1` 编 2DGS 的两个 CUDA 扩展（复用已有的 gxx_linux-64=12）。

> 也可用独立的 [`pi3_3dgs/`](../pi3_3dgs/) 流程（自带独立 env，参数更全），但本流程推荐直接用步骤 5。

## 首次准备

本流程建一份独立的 `wan22_rotate` env（**CPython 3.10**，匹配本地 cp310 torch/triton 轮子——不要用 3.11 或 clone doll，cp310 轮子装不进 3.11），把 sam_3d_body + diffsynth 两套依赖装在一起（detectron2 用 `--no-deps` 装，`networkx==3.2.1` 对 diffsynth 无影响；gcc12 用 `conda install --no-update-deps` 装防 conda 把 python 掉包成 GraalPy，否则 numpy 全坏）。

```bash
cd <your-code-dir>            # e.g. /data_3d/<uid>/code
git -c http.sslVerify=false clone https://github.com/CrescentVelvet/media_code.git
cd media_code && cp proxy.env.example proxy.env   # 填 http_proxy / https_proxy
# ⚠️ 确认 proxy.env 中 http_proxy / https_proxy 两行已取消注释并填好地址，
#    否则 pip 装依赖会报 "Network is unreachable"

# 1. 建 wan22_rotate env（⚠️ CPython 3.10，匹配本地 cp310 torch/triton 轮子；
#    不要用 3.11 或 clone doll——cp310 轮子装不进 3.11）
conda create -n wan22_rotate python=3.10 -y && conda activate wan22_rotate

# 2. 装两套依赖 + SAM2 + MediaPipe + 验证
#    INSTALL_DEPS=1 会用本地 cp310 轮子装 torch 2.6.0+cu124 + nvidia 依赖，
#    装 gcc12（--no-update-deps 防 GraalPy 掉包）、diffsynth、sam_3d_body、detectron2，
#    并 clone SAM2 仓库 + pip install + 下载 sam2.1_hiera_large.pt，
#    装 mediapipe（step 01c 用，CPU 即可，无需 GATED 权重）
INSTALL_DEPS=1 bash wan22_rotate/00_setup_env.sh

# 3. （步骤 5 用）装 2DGS 依赖 + 编 CUDA 扩展
#    INSTALL_2DGS=1 会 clone 2d-gaussian-splatting 仓 + 装 open3d/lpips/trimesh 等 +
#    编 simple-knn + diff-surfel-rasterization（复用已有的 gxx_linux-64=12 + 系统 CUDA toolkit）
INSTALL_DEPS=1 INSTALL_2DGS=1 bash wan22_rotate/00_setup_env.sh

# ⚠️ 如果上面编 CUDA 扩展失败（simple-knn 拉不下来 / CUDA 版本不匹配 / GLM 缺失），
#    00 脚本已自动处理多数情况（自动找 CUDA 12.4 路径、zip fallback 拉 simple-knn、
#    clone GLM）。但如果还是失败，按以下步骤手动修：
#
# a) simple-knn 拉不下来（gitlab.inria.fr 被封/慢）：
#    在 Windows 浏览器下载 zip，传到容器：
#      https://gitlab.inria.fr/bkerbl/simple-knn/-/archive/main/simple-knn-main.zip
#    解压到子模块目录：
cd /data_3d/w00950754/code/2d-gaussian-splatting/submodules
unzip /path/to/simple-knn-main.zip
mv simple-knn-main simple-knn
ls simple-knn/setup.py   # 确认存在
#
# b) CUDA 版本不匹配（detected 11.8 vs PyTorch 12.4）：
#    系统有多个 CUDA，/usr/local/cuda 指向 11.8。找到 12.4 的路径设给 CUDA_HOME：
#    ls -d /usr/local/cuda-12*  →  /usr/local/cuda-12.4
conda activate wan22_rotate
export CUDA_HOME=/usr/local/cuda-12.4       # ⚠️ 不是 /usr/local/cuda（那个是 11.8）
export PATH=$CUDA_HOME/bin:$PATH           # 让 12.4 的 nvcc 排在 PATH 前面
export CC=$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-gcc
export CXX=$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-g++
$CUDA_HOME/bin/nvcc --version | tail -1    # 确认输出 cuda_12.4
#
# c) GLM 缺失（glm/glm.hpp: No such file or directory）：
cd /data_3d/w00950754/code/2d-gaussian-splatting
git clone https://github.com/g-truc/glm.git third_party/glm
ls third_party/glm/glm/glm.hpp   # 确认存在
#
# d) 编译两个 CUDA 扩展（--no-deps --no-index 避免联网）：
pip install --no-build-isolation --no-deps --no-index \
  /data_3d/w00950754/code/2d-gaussian-splatting/submodules/simple-knn
pip install --no-build-isolation --no-deps --no-index \
  /data_3d/w00950754/code/2d-gaussian-splatting/submodules/diff-surfel-rasterization
python -c "import simple_knn, diff_surfel_rasterization; print('OK')"

# 3. 下权重（两边各自的下载脚本）
#    完整版 01 需要 SAM 3D Body 权重（GATED）；简化版 01b 不需要
HF_TOKEN=hf_xxx bash sam_3d_body/01_download_models.sh   # SAM 3D Body（GATED，需先 Request access）
bash wan22/01_verify_models.sh                           # Wan2.2（确认权重在位）

# ⚠️ 如果 HuggingFace 下载失败（代理拦截 huggingface.co），
#    从 ModelScope 手动下载 SAM 3D Body + MoGe2 权重，放到对应目录：
#    1) sam-3d-body-dinov3（含 model.ckpt, model_config.yaml, assets/mhr_model.pt）
mkdir -p ../../model/sam-3d-body/sam-3d-body-dinov3/assets
#       从 ModelScope 下载后放入：
#         ../../model/sam-3d-body/sam-3d-body-dinov3/model.ckpt
#         ../../model/sam-3d-body/sam-3d-body-dinov3/model_config.yaml
#         ../../model/sam-3d-body/sam-3d-body-dinov3/assets/mhr_model.pt
#    2) moge-2-vitl-normal（FOV 估计器权重）
mkdir -p ../../model/sam-3d-body/moge-2-vitl-normal
#       从 ModelScope 下载后放入：
#         ../../model/sam-3d-body/moge-2-vitl-normal/
#    确认文件存在：
ls ../../model/sam-3d-body/sam-3d-body-dinov3/model.ckpt
ls ../../model/sam-3d-body/sam-3d-body-dinov3/assets/mhr_model.pt
ls ../../model/sam-3d-body/moge-2-vitl-normal/
```

权重需已在 `$MODEL_DIR`（默认 `../../model`）下：
```
$MODEL_DIR/
  Wan2.2-TI2V-5B/                          # DiT + T5 + VAE (wan22 用, 01_verify_models.sh 自动建 Wan-AI 符号链接)
  Wan2.1-T2V-1.3B/                         # tokenizer
  Wan2.2-TI2V-5B_lora_add_data_reload/     # 训练好的 LoRA
    step-66900.safetensors
  sam-3d-body/
    sam-3d-body-dinov3/                    # SAM 3D Body ckpt + mhr_model
    moge-2-vitl-normal/                    # MoGe2 FOV estimator
```

LoRA 权重路径示例：`$MODEL_DIR/Wan2.2-TI2V-5B_lora_add_data_reload/step-66900.safetensors`。

---

以下为详细参考（流程原理 / 各步骤参数 / 排错 / 目录布局）。

## Pipeline（流程详解）

```
INPUT_DIR/image/               (环绕人物拍摄的多张图像)
    │
    ▼
[01] SAM 3D Body  (sam_3d_body env)
    │  ├─ ViTDet 人体检测          → bbox per image
    │  ├─ MoGe2 FOV 估计           → 相机内参
    │  ├─ DINOv3 编码 + MHR 解码   → 3D body (global_rot, pred_vertices, ...)
    │  ├─ 正面评分 (global_rot)     → 选最佳正面图
    │  └─ 人物分割                  → 背景置白
    │       ├─ 优先：HumanSegmentor (SAM2/SAM3, 需配 SEGMENTOR_PATH)
    │       ├─ 备选：3D mesh silhouette (pyrender 渲染 pred_vertices)
    │       └─ 兜底：bbox 矩形掩码
    │
    │  [01b] 简化版: 跳过 MoGe2/DINOv3/MHR, 只用 ViTDet + SAM2, 按面积最大选图
    │  [01c] MediaPipe 版: 用 Face Mesh 算正面评分 (鼻尖居中 + 双眼距离最大), CPU 即可,
    │       无需 GATED 权重; 分割复用 01b 的 ViTDet + SAM2; SKIP_SEGMENTATION=1 仅选图
    ▼
$RESULTS_DIR/segmented_image.png   (人物保留, 背景白色)
    │
    ▼
[02] Wan2.2-TI2V-5B I2V + LoRA  (wan22 env)
    │  ├─ 加载 pipeline (DiT + T5 + VAE, 从本地 $MODEL_DIR)
    │  ├─ 加载 LoRA (pipe.load_lora, alpha=1)
    │  └─ I2V 生成 (分割图作首帧 → 360° 旋转视频)
    ▼
$RESULTS_DIR/rotate_360.mp4
    │
    ▼
[03] 拆帧 → JPG  (opencv)
    │  └─ cv2.VideoCapture + imwrite(.jpg, quality=95)
    │       ├─ FPS=0 (默认): 抽每一帧 (源 fps, 约 81 张 @ 15fps×5.4s)
    │       └─ FPS=N: 按 N fps 采样 (step=round(src_fps/N))
    ▼
$RESULTS_DIR/rotate_360/image/*.jpg   (同名目录下的 image/ 文件夹)
                                      (匹配 INPUT_DIR/image/ 模式, 可喂回 01 或 pi3_3dgs)
    │
    ▼
[04] Pi3 位姿估计  (wan22_rotate env, 调 pi3_3dgs/pi3_recon.py --no_colmap)
    │  ├─ 确认步骤 03 的 JPG 存在 (否则自动调 03 抽帧)
    │  ├─ Pi3 前向 (1 次推理 → 所有视角的位姿 + 稠密点云 + 置信度)
    │  └─ --no_colmap: 跳过 COLMAP 导出 + 2DGS 训练 (不进行三维重建)
    ▼
$RESULTS_DIR/rotate_360/pi3/{predictions.npz, dense_cloud.ply, poses.json}
   predictions.npz  # 原始 Pi3 张量 (points, camera_poses, conf, images, ...)
   dense_cloud.ply  # 置信度过滤后的稠密点云 (MeshLab/SuperSplat 可看)
   poses.json       # 人类可读的 c2w 4x4 矩阵 (每帧一个, OpenCV 约定)
    │
    ▼
[05] 三维高斯重建  (wan22_rotate env, 调 05_3dgs_recon.sh)
    │  ├─ Pi3 重跑 (带 COLMAP 导出, 非 --no_colmap)
    │  │     ├─ cameras.txt   PINHOLE, fx=fy=max(W,H), cx=W/2, cy=H/2
    │  │     ├─ images.txt    c2w→w2c 四元数+平移
    │  │     └─ points3D.txt  稠密点云 + RGB (体素下采样到 ≤100k)
    │  ├─ 2DGS 训练 (高斯初始化 → 可微光栅化 → L1+SSIM+法向+深度正则 → 致密化)
    │  └─ 渲染 + TSDF fusion 提网格 (unbounded 适配白底人像)
    ▼
$PI3_3DGS_RESULTS/{source/, model/}
    source/images/ + sparse/0/*.txt              # COLMAP 场景
    model/point_cloud/iteration_<N>/point_cloud.ply   # 高斯点云
    model/test/ours_<N>/{renders/*.png, mesh.ply}      # 渲染 + 网格
```

### Step 01 — 选图 + 分割 (`01_pick_and_segment.sh` → `pick_and_segment.py`)

**正面图挑选**：对每张图跑 SAM 3D Body 的 `process_one_image`（3D 人体网格恢复），输出含 `global_rot`（全局旋转）。将其转为旋转矩阵 R，计算人物朝向 `forward = R @ [0,0,1]`（SMPL 静止姿态面 +Z）。正面朝相机时 `forward[2] < 0`（朝 -Z 即相机方向），评分 `score = -forward[2]`，取评分最高的图。

> 如果选出来的是背面而不是正面，说明旋转约定相反——设 `FRONTAL_SIGN=-1` 翻转。

`global_rot` 支持 3×3 旋转矩阵、轴角 (3,)、四元数 (4,) 三种格式，自动识别。

**人物分割**：按以下优先级尝试，任一成功即用：
1. **HumanSegmentor**（SAM2/SAM3）— 像素级精确掩码。需 `SEGMENTOR_PATH` 指向 sam2 仓库（含 `checkpoints/` + `configs/`）。脚本尝试 `__call__` 和 `predict` 两种接口。
2. **3D mesh silhouette** — 用 `Renderer.vertices_to_trimesh` 构建 trimesh，pyrender 离屏渲染为二值掩码。无需额外配置，但只覆盖身体模型形状（可能漏掉头发、宽松衣物）。
3. **bbox 矩形** — 最粗略的兜底掩码（带 `PADDING` 边距）。

掩码应用：人物区域保留原像素，背景区域设为白色 `(255,255,255)`（`WHITE_BG=0` 改为黑色）。

### Step 02 — 视频生成 (`02_generate_video.sh`)

直接调用已有的 `wan22/02_run_inference.sh`（它自己 source `wan22/_env.sh` 激活 `wan22` env），传入：
- `INPUT_IMAGE=$RESULTS_DIR/segmented_image.png`（I2V 模式）
- `WEIGHT_PATH=<lora>.safetensors`（`pipe.load_lora(dit, weight, alpha=1)`）
- `PROMPT`、`HEIGHT`/`WIDTH`/`NUM_FRAMES` 等

默认 portrait（1248×704），81 帧 @ 15fps ≈ 5.4 秒，足够一圈 360° 旋转。更多参数见 `wan22/README.md` 的 inference 部分。

### Step 03 — 拆帧 (`03_extract_frames.sh` → `extract_frames.py`)

把 step 02 生成的 `rotate_360.mp4` 拆成与视频同名的目录下的 `image/` 文件夹里的多张 JPG：

```
$RESULTS_DIR/
  rotate_360.mp4               # 源视频（不变）
  rotate_360/                  # 新增：同名目录
    image/
      00000.jpg
      00001.jpg
      ...
```

这样 `rotate_360/` 文件夹本身就是合法的 `INPUT_DIR`（含 `image/` 子文件夹），可：
- 喂回 step 01 重新选图 + 分割（`INPUT_DIR=$RESULTS_DIR/rotate_360`）
- 喂给 pi3_3dgs 做三维重建（`INPUT=$RESULTS_DIR/rotate_360`）
- 或任何接受 `INPUT_DIR/image/` 模式的算法

默认 `FPS=0` 抽每一帧（按源视频 fps，~81 张）；设 `FPS=N` 按 N fps 采样。`JPG_QUALITY=95`（视觉无损）。

### Step 04 — Pi3 位姿估计 (`04_pi3_pose.sh` → 调 `pi3_3dgs/pi3_recon.py --no_colmap`)

在 wan22_rotate env 里跑 [π³ (Pi3)](https://github.com/yyfz/Pi3)（ICLR 2026）前馈位姿 + 稠密点云估计——**不进行三维重建**（不导出 COLMAP 格式、不跑 2DGS 训练）。

**为何能在 wan22_rotate env 里跑**：Pi3 的依赖（torch 2.6.0+cu124、numpy 1.26.4、cv2、safetensors、plyfile）全部已在 wan22_rotate env 里（sam_3d_body + diffsynth 装过，`plyfile` 由 00_setup_env.sh 装或 `pip install plyfile`）。不需要 pi3_3dgs env（那个 env 只是为了编 2DGS 的 CUDA 光栅化扩展）。

**流程**：
1. 若 `$RESULTS_DIR/<video_name>/image/` 已存在（步骤 03 跑过）：直接用这些 JPG 作为 Pi3 输入。
2. 否则：自动调 `03_extract_frames.sh` 抽帧（"拆分视频为 JPG 帧"这一步）。
3. 调 `pi3_3dgs/pi3_recon.py --no_colmap`：Pi3 前向一次出所有视角的位姿（c2w 4x4 矩阵）+ 稠密点云 + 置信度。
4. 输出 `predictions.npz` + `dense_cloud.ply` + `poses.json`（人类可读）。

**`--no_colmap` 的含义**：跳过 `pi3_recon.py` 的第 5 步（COLMAP 文本格式导出）。该步只在不做 2DGS 训练时是冗余的（COLMAP 格式是 2DGS `train.py` 的输入）。同时跳过了 `open3d` 的体素下采样（该库不在 wan22_rotate env 里，只在 COLMAP 导出阶段需要）。

**`poses.json` 格式**：
```json
{
  "frame_names": ["00000.jpg", "00001.jpg", ...],
  "camera_poses_c2w": [[[...4x4...], [...], ...]],   // N 个 4x4 c2w 矩阵
  "num_frames": 81,
  "image_size": [1248, 704],
  "convention": "OpenCV (z forward, y down, x right)",
  "note": "c2w = camera-to-world 4x4 matrix. Invert for w2c ..."
}
```

**首次准备**：需要 Pi3 仓库（`../Pi3`）和权重（`../../model/Pi3/model.safetensors`）。脚本会自动 clone Pi3 仓库（如果不存在）；权重需手动下载（脚本会给出提示）：
```bash
# 下 Pi3 权重（公开，免 token；约 1GB）
mkdir -p ../../model/Pi3
wget --no-check-certificate -O ../../model/Pi3/model.safetensors \
  https://huggingface.co/yyfz233/Pi3/resolve/main/model.safetensors
```

### Step 05 — 三维高斯重建（`05_3dgs_recon.sh`）

步骤 04 只估位姿（`--no_colmap`，不做三维重建）。步骤 5 在**同一个 wan22_rotate env** 里完成 3DGS 重建：Pi3 重跑（带 COLMAP 导出）→ 2DGS 训练 → 渲染 + 网格。

**为何能用同一个 env**：2DGS 的两个 CUDA 扩展（`diff-surfel-rasterization` + `simple-knn`）需要 nvcc + gxx 编译。wan22_rotate env 已有 `gxx_linux-64=12`（为 detectron2 装的），只需系统有 CUDA toolkit（nvcc），`INSTALL_2DGS=1` 即可在 wan22_rotate env 里编这两个扩展。无需建独立 env。

**为何重跑 Pi3**：步骤 04 用了 `--no_colmap`（只出位姿 + 点云，不导 COLMAP 格式）。步骤 5 不带 `--no_colmap`，会额外导出 `cameras.txt` / `images.txt` / `points3D.txt`（2DGS `train.py` 的输入）。Pi3 推理只需 10-60 秒，重跑无妨。

**wan22_rotate 输入的适配**（`05_3dgs_recon.sh` 已默认设好）：
- `WHITE_BG=1`（默认）— 2DGS 训练用白底背景，与分割图一致
- `UNBOUNDED=1`（默认）— TSDF 无界模式，人像在白色虚空中
- `MESH_RES=2048`（默认）— 提高网格分辨率

**首次准备**：在 wan22_rotate env 里装 2DGS 依赖 + 编 CUDA 扩展（一次性）：
```bash
INSTALL_DEPS=1 INSTALL_2DGS=1 bash wan22_rotate/00_setup_env.sh
```
需系统有 CUDA 12.4 toolkit（`CUDA_HOME=/usr/local/cuda`，nvcc 可用）。

## Config (env vars, all optional)

### Paths & envs
| var | default | note |
| --- | --- | --- |
| `INPUT_DIR` | _(required)_ | 人物文件夹（含 `image/` 子文件夹） |
| `WEIGHT_PATH` | _(required for 02)_ | 训练好的 LoRA `.safetensors` |
| `GPU` | _(unset)_ | physical GPU id，e.g. `GPU=0` |
| `CONDA_ENV` | `wan22_rotate` | conda env（从 doll 克隆，装两套依赖） |
| `SAM3D_DIR` | `../sam-3d-body` | SAM 3D Body 官方代码 |
| `SAM3D_MODEL_DIR` | `../../model/sam-3d-body` | SAM 3D Body 权重 |
| `DIFFSYNTH_DIR` | `../DiffSynth-Studio` | DiffSynth-Studio 代码 |
| `WAN_MODEL_DIR` | `../../model` | Wan2.2 权重根 |
| `RESULTS_DIR` | `../wan22_rotate_results` | 输出目录 |

### Step 01 params
| var | default | note |
| --- | --- | --- |
| `HF_REPO_ID` | `facebook/sam-3d-body-dinov3` | SAM 3D Body 骨干（alt `facebook/sam-3d-body-vith`） |
| `CHECKPOINT_PATH` | `$SAM3D_MODEL_DIR/<repo>/model.ckpt` | SAM 3D Body checkpoint |
| `MHR_PATH` | `$SAM3D_MODEL_DIR/<repo>/assets/mhr_model.pt` | MHR asset |
| `DEVICE` | `cuda` | falls back to `cpu` if CUDA unavailable |
| `DETECTOR_NAME` | `vitdet` | `vitdet` \| `sam3` \| `` (disable → full-image bbox) |
| `DETECTOR_PATH` | _(unset)_ | ViTDet auto-downloads; set for offline |
| `SEGMENTOR_NAME` | `sam2` | `sam2` (needs `SEGMENTOR_PATH`) \| `sam3` \| `` (disable) |
| `SEGMENTOR_PATH` | _(unset)_ | sam2 repo dir w/ `checkpoints/` + `configs/` |
| `FOV_NAME` | `moge2` | `moge2` \| `` (disable → default FOV) |
| `FOV_PATH` | `$SAM3D_MODEL_DIR/moge-2-vitl-normal` | MoGe2 local dir |
| `BBOX_THRESH` | `0.8` | detector score threshold |
| `INFERENCE_TYPE` | `body` | `body` (skip hand decoder, faster) \| `full` \| `hand` |
| `FRONTAL_SIGN` | `1` | `-1` = flip front-facing criterion (if picks back-facing) |
| `WHITE_BG` | `1` | `1` = white background; `0` = black |
| `PADDING` | `0.1` | bbox padding ratio (for bbox mask fallback) |

### Step 01c params (MediaPipe 版)
| var | default | note |
| --- | --- | --- |
| `MP_MIN_CONFIDENCE` | `0.5` | MediaPipe Face Mesh 检测置信度阈值 |
| `SKIP_SEGMENTATION` | `0` | `1` = 仅选正面图不分割（快速看哪帧是正面） |
| `DETECTOR_NAME` | `vitdet` | 分割用 ViTDet（01c 选图不用它，分割复用 01b 逻辑） |
| `SEGMENTOR_NAME` | `sam2` | 分割用 SAM2（需配 `SEGMENTOR_PATH`） |
| `DEVICE` | `cuda` | MediaPipe 本身跑 CPU；此值给 ViTDet/SAM2 分割用 |
| `WHITE_BG` | `1` | `1` = 白底；`0` = 黑底 |
| `PADDING` | `0.1` | bbox 兜底掩码的边距比例 |

### Step 02 params
| var | default | note |
| --- | --- | --- |
| `PROMPT` | `人物360度旋转展示，高质量，细节清晰。` | 生成提示词 |
| `SEGMENTED_IMAGE` | `$RESULTS_DIR/segmented_image.png` | I2V 输入图 |
| `HEIGHT` / `WIDTH` | `1248` / `704` | portrait（landscape 用 `704`/`1248`） |
| `NUM_FRAMES` | `121` | 生成帧数（4k+1，Wan 约束） |
| `FPS` | `15` | 输出 mp4 帧率 |
| `OUTPUT_NAME` | `rotate_360` | 输出文件名（不含扩展名） |
| `LOW_VRAM` | `0` | `1` = 磁盘 offload（慢但省显存，详见 wan22 README） |

### Step 03 params
| var | default | note |
| --- | --- | --- |
| `VIDEO_PATH` | `$RESULTS_DIR/${OUTPUT_NAME:-rotate_360}.mp4` | 源视频路径 |
| `FPS` | `0` | `0` = 抽每一帧（按源 fps）；`N` = 按 N fps 采样 |
| `JPG_QUALITY` | `95` | JPG 质量 1-100（95 ≈ 视觉无损） |
| `START_FRAME` | `0` | 起始帧（跳过开头几帧） |
| `END_FRAME` | `-1` | 结束帧，`-1` = 到末尾 |

### Step 04 params
| var | default | note |
| --- | --- | --- |
| `INPUT` | _(auto-detected)_ | Pi3 输入；默认用步骤 03 的 JPG 夹，否则用视频（自动调 03 抽帧） |
| `VIDEO_PATH` | `$RESULTS_DIR/${OUTPUT_NAME:-rotate_360}.mp4` | 源视频（仅当 `INPUT` 未设且步骤 03 未跑时用） |
| `OUTPUT_DIR` | `$RESULTS_DIR/${OUTPUT_NAME:-rotate_360}/pi3` | Pi3 输出目录 |
| `FRAME_FPS` | `10` | 视频抽帧 fps（仅当需要抽帧时；步骤 03 已跑则忽略） |
| `FRAME_MAX` | `60` | 最大帧数（防 OOM；Pi3 显存随 N 线性增长） |
| `CONF_THRES` | `0.1` | sigmoid-conf 阈值，过滤低置信像素 |
| `PI3_DIR` | `../Pi3` | Pi3 官方代码（自动 clone 如果缺） |
| `PI3_CKPT` | `../../model/Pi3/model.safetensors` | Pi3 checkpoint（需手动下载） |
| `DEVICE` | `cuda` | 或 `cpu`（很慢） |

### Step 05 params
> 步骤 5 在 wan22_rotate env 里跑（`05_3dgs_recon.sh`），默认已适配白底分割输入。
| var | default | note |
| --- | --- | --- |
| `INPUT` | `$RESULTS_DIR/rotate_360.mp4` | 输入视频 / 图像夹 |
| `WHITE_BG` | `1` | `1` = 训练用白底（适配 wan22_rotate 分割图） |
| `UNBOUNDED` | `1` | `1` = 无界 TSDF（适配人像在白色虚空） |
| `MESH_RES` | `2048` | TSDF 体素分辨率 |
| `ITERATIONS` | `30000` | 2DGS 训练步数（7000 = 快速 demo） |
| `FRAME_FPS` | `10` | 视频抽帧 fps（Pi3 显存随帧数线性增长） |
| `FRAME_MAX` | `60` | 最大帧数（防 OOM） |
| `SKIP_PI3` | `0` | `1` = 跳过 5a（复用已有 COLMAP source/） |
| `SKIP_TRAIN` | `0` | `1` = 跳过 5b（复用已有 model/） |
| `SKIP_RENDER` | `0` | `1` = 跳过 5c |

## 可能遇到的问题

**1. 选出的图是背面而不是正面**
SAM 3D Body 的 `global_rot` 旋转约定可能与你的人物数据相反。设 `FRONTAL_SIGN=-1` 翻转评分方向，重跑 step 01。

**2. 分割用了 bbox 兜底（掩码是矩形）**
说明 HumanSegmentor 和 mesh silhouette 都失败了。改善方法：
- 配置 SAM2 分割器：设 `SEGMENTOR_PATH=/path/to/sam2_repo`（含 `checkpoints/sam2.1_hiera_large.pt` + `configs/`）。
- mesh silhouette 失败通常是 pyrender OpenGL 问题：确认 `PYOPENGL_PLATFORM=egl`（`_env.sh` 默认设了），或试 `PYOPENGL_PLATFORM=osmesa`（需 `apt install libosmesa6-dev` + 重装 PyOpenGL）。

**3. step 01 报 `import sam_3d_body` / `cv2` 失败**
`wan22_rotate` env 缺 sam_3d_body 依赖。`INSTALL_DEPS=1 bash wan22_rotate/00_setup_env.sh` 重装。

**4. step 02 报 `import diffsynth` 失败**
`wan22_rotate` env 缺 diffsynth。`INSTALL_DEPS=1 bash wan22_rotate/00_setup_env.sh` 重装。

**5. 视频生成 OOM**
- 降分辨率 / 帧数：`HEIGHT=480 WIDTH=832 NUM_FRAMES=49`。
- 开磁盘 offload：`LOW_VRAM=1`。
- 详见 `wan22/README.md` 排错 #3。

**6. 多人物图像**
脚本取 bbox 面积最大的那个人（离相机最近）。如需指定其他人，修改 `pick_and_segment.py` 的 `best = max(outputs, ...)` 逻辑。

**7. 跑 `.sh` 报 `syntax error near unexpected token ('`**
CRLF 行尾污染。`find wan22_rotate -name '*.sh' -exec sed -i 's/\r$//' {} +` 或 `git checkout -- wan22_rotate/*.sh`（`.gitattributes` 强制 LF）。

**8. NumPy 坏了 / `import numpy` 报 ABI 不兼容 / python 变成了 GraalPy**
根因：`conda install -c conda-forge gxx_linux-64`（装 detectron2 编译要的 gcc12）**不带 `--no-update-deps`** 时，conda 求解器会把 env 的 python 实现掉包成 **GraalPy**（满足 python 槽位的另一个 conda-forge 包），而 numpy/torch 是 CPython ABI 编译的——GraalPy 下全坏；本地 cp310 torch/triton 轮子更直接装不上。本仓 `00_setup_env.sh` 的 gxx 步骤**已加 `--no-update-deps`** 防掉包，env 也强制 CPython 3.10（建完即校验 `platform.python_implementation()=="CPython"`，gxx 装完再校验一次）。若 env 已被掉包成 GraalPy，**重建即可**：
```bash
python -c "import platform; print(platform.python_implementation())"   # 输出 GraalPy 即中招
conda env remove -n wan22_rotate
conda create -n wan22_rotate python=3.10 -y && conda activate wan22_rotate   # CPython 3.10
INSTALL_DEPS=1 bash wan22_rotate/00_setup_env.sh                          # gxx 带 --no-update-deps
python -c "import numpy, torch; print(numpy.__version__, torch.__version__)"  # 验证
```
> 铁律：往 `wan22_rotate` env 里 `conda install` 任何包都加 `--no-update-deps`，否则 GraalPy 会回来把 numpy 干掉。

**9. 步骤 5 编 CUDA 扩展报 `simple-knn 拉不下来` / `CUDA version (11.8) mismatches PyTorch (12.4)`**
见上方「首次准备」的 ⚠️ 手动修复步骤。要点：
- **simple-knn**：`gitlab.inria.fr` 可能被代理封。浏览器下载 zip 传上去，解压到 `2d-gaussian-splatting/submodules/simple-knn/`。
- **CUDA 版本不匹配**：系统有多个 CUDA，`/usr/local/cuda` 可能指向 11.8。用 `ls -d /usr/local/cuda-12*` 找到 12.4 路径，`export CUDA_HOME=/usr/local/cuda-12.4` + `export PATH=$CUDA_HOME/bin:$PATH`。
- **pip 联网报错**：加 `--no-build-isolation --no-deps --no-index` 三个 flag，彻底禁止联网。

## 目录布局
```
<code-dir>/
├── media_code/                  # 本仓
│   ├── proxy.env                # 代理 + 覆盖项, gitignored
│   ├── wan22/                   # Wan2.2 推理/训练脚本 (step 02 调用)
│   ├── sam_3d_body/             # SAM 3D Body 推理脚本 (step 01 调用)
│   └── wan22_rotate/            # ← 本目录（编排脚本）
├── sam-3d-body/                 # SAM 3D Body 官方代码
├── sam2/                        # SAM2 官方代码 + checkpoints (00 clone, 01b 分割用)
│   └── checkpoints/
│       └── sam2.1_hiera_large.pt
├── DiffSynth-Studio-Human/     # DiffSynth-Studio 官方代码 (本流程专用, 00 clone)
├── Pi3/                          # Pi3 官方代码 (step 04 自动 clone; pi3_3dgs 也用)
├── 2d-gaussian-splatting/        # 2DGS 官方代码 + CUDA 子模块 (step 05 → pi3_3dgs/00 clone)
├── wan22_experiments/           # LoRA 训练产物 (epoch-N.safetensors)
├── wan22_rotate_results/        # 本流程输出 (step 01-04)
    ├── segmented_image.png      #   正面图 (人物保留, 背景白)
    ├── front_facing_original.jpg#   原始正面图
    ├── frontal_scores.csv       #   各图正面评分
    ├── debug_mask.png           #   分割掩码 (调试)
    ├── rotate_360.mp4           #   360° 旋转视频 (step 02)
    └── rotate_360/              #   同名目录 (step 03/04)
        ├── image/               #     拆出的 JPG 帧 (step 03)
        │   ├── 00000.jpg
        │   ├── 00001.jpg
        │   └── ...
        └── pi3/                 #     Pi3 位姿估计输出 (step 04)
            ├── frames/          #       Pi3 实际用的帧 (copy of image/, 或抽帧)
            ├── predictions.npz  #       原始 Pi3 张量 (points, camera_poses, conf, ...)
            ├── dense_cloud.ply  #       置信度过滤的稠密点云 (MeshLab/SuperSplat 可看)
            └── poses.json       #       人类可读的 c2w 4x4 矩阵 (每帧一个)
└── pi3_3dgs_results/            # step 05 三维重建输出 (独立目录)
    ├── source/                   #   COLMAP 场景 (01_pi3_recon 导出)
    │   ├── images/               #     训练图
    │   └── sparse/0/
    │       ├── cameras.txt
    │       ├── images.txt
    │       └── points3D.txt
    ├── predictions.npz           #   原始 Pi3 张量
    ├── dense_cloud.ply           #   稠密点云 (调试)
    └── model/                    #   2DGS 训练产物
        ├── point_cloud/iteration_<N>/point_cloud.ply   # 高斯点云
        └── test/ours_<N>/
            ├── renders/*.png     #     渲染图
            └── mesh.ply          #     TSDF 网格
```

## Notes
- Official code & weights follow their own licenses (Wan2.2 = Apache 2.0; SAM 3D Body = SAM License). This folder only orchestrates; no official code is copied.
- `.gitattributes` (repo root) forces LF so Windows-pushed scripts run cleanly on Ubuntu.
- `proxy.env` (proxy creds / env overrides) is gitignored — never committed.

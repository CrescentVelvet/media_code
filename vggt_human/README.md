# VGGT-Human — VGGT-Omega 前馈重建 → COLMAP → 原版 3DGS 训练

用 [VGGT-Omega (CVPR 2026 Oral)](https://github.com/facebookresearch/vggt-omega) 做前馈三维重建（一次推理出相机位姿 + 深度图 → 反投影成密集点云），再转 COLMAP 格式喂 [原版 3DGS](https://github.com/graphdeco-inria/gaussian-splatting)（3D Gaussian Splatting）做迭代优化训练。VGGT-Omega 提供**快速、鲁棒的位姿 + 密集点云初始化**（前馈，无需特征匹配 / SfM），3DGS 在此基础上**优化出高质量高斯表示**（novel-view 合成 + 渲染）。

> 本目录与 [`pdfgs_human/`](../pdfgs_human/) 并列且**思路相同**：都是"前馈模型出位姿 + 点云 → COLMAP → 3DGS 训练"。区别在于前馈模型和 3DGS 变体不同：
> - pdfgs_human：**Pi3**（前馈位姿）→ COLMAP → **PDF-GS**（带 DINOv3 distractor filtering 的 3DGS，抗微动）
> - 本目录：**VGGT-Omega**（1B 大模型，前馈位姿 + 深度 + 置信度）→ COLMAP → **原版 3DGS**（无 distractor filtering，更简单）
>
> 两者不共用 env（本目录复用 `doll` env，pdfgs_human 用 `pdfgs` env）。

> 📄 详细文档：选型对比（VGGT-Omega vs Pi3、原版 3DGS vs PDF-GS）、流程原理详解、排错手册见 [NOTES.md](NOTES.md)；实验设计与结果（位姿 A/B、人脸 finetune 消融、近景注入等）见 [EXPERIMENTS.md](EXPERIMENTS.md)。

## 常用命令

> 假设已进入容器（脚本自动激活 `doll` env）；`GPU=0` 按需换卡。首次跑前先做下方「首次准备」。
> **铁律：每条命令都必须显式写出模型路径、输入路径、输出路径，不能全靠脚本里的默认值。** 用具体路径，不要用 `...` 占位。

```bash
# ── 分步 ──
# 0) clone 仓 + 装依赖 + 编 CUDA 扩展 (一次性)
# GPU=0 INSTALL_DEPS=1 bash vggt_human/00_setup_env.sh
GPU=0 INSTALL_DENOISER=1 INSTALL_DEPS=1 bash vggt_human/00_setup_env.sh

# 1a) 视频 → 图像夹 (抽帧, 喂给 01; 视频输入用, 不影响图像输入的原流程)
#     INPUT_DIR 指向单个视频文件, 输出 <OUTPUT_DIR>/image/ 子夹 (000000.png, ...)
#     结构与 test_task 一致, 01 的 INPUT_DIR 指向 <OUTPUT_DIR> 即可自动检测
GPU=0 \
INPUT_DIR=../../output/vggt_human_results/input_video/生成环绕人物视频.mp4 \
OUTPUT_DIR=../../output/vggt_human_results/01a_input_frames/生成环绕人物视频 \
VIDEO_FPS=2 \
bash vggt_human/01a_video_to_frames.sh

# 输出：<OUTPUT_DIR>/image/  # 抽帧后的散图 (test_task 结构)

# 1) 前处理人脸增强 (MediaPipe + HYPIR + 渐变融合, 对原始输入图)
#    INSTALL_DEPS=1 bash vggt_human/00_setup_env.sh 建好 vggt_human env (含 mediapipe + HYPIR)
#    HYPIR_WEIGHT 指向 beauty_ppr50k 训练的 checkpoint
#    视频输入: 先跑 01a 抽帧, INPUT_DIR 指向 01a 的输出 (01a_input_frames)
# 图像夹输入 (原流程): INPUT_DIR=../Reconstruction/dataset/B003_Human_Data_w_pose/test_task_id_3a8b3cc746304f49b9e3275e36aa9374
GPU=0 \
INPUT_DIR=../../output/vggt_human_results/01a_input_frames/生成环绕人物视频 \
RESULTS_DIR=../../output/vggt_human_results \
bash vggt_human/01_face_enhance.sh

# 输出：<RESULTS_DIR>/01_input_face/images/  # 人脸增强后的原始图

# 2) VGGT-Omega 前馈推理 (图像 -> 位姿+深度 -> predictions.npz + scene.ply)
#    INPUT_DIR 指向 step 01 的输出 (01_input_face)
GPU=0 \
INPUT_DIR=../../output/vggt_human_results/01_input_face \
MODEL_DIR=../../model/VGGT-Omega \
RESULTS_DIR=../../output/vggt_human_results \
MAX_POINTS=2000000 \
bash vggt_human/02_run_inference.sh

# 输出：<RESULTS_DIR>/02_vggt/<scene>/
#   predictions.npz   # 原始输出 (extrinsic w2c, intrinsic, world_points, depth_conf, images)
#   scene.ply          # 置信度过滤后的彩色点云 (供检查)
#   frames/            # 喂给模型的图 (复制/抽帧)

# 3) npz -> COLMAP 转换 (自适应置信度过滤 + 体素降采样 ~200k + 坐标系对齐)
GPU=0 \
TARGET_POINTS=200000 \
POSE_ADJUST=1 \
POSE_REFINE=1 \
RESULTS_DIR=../../output/vggt_human_results \
bash vggt_human/03_npz_to_colmap.sh

# 输出：<RESULTS_DIR>/03_source/
#   images/*.png                      # 训练图 (从 frames/ 复制, 内参自动缩放)
#   sparse/0/cameras.txt              # PINHOLE (VGGT-Omega 实际内参)
#   sparse/0/images.txt              # w2c (qw qx qy qz tx ty tz)
#   sparse/0/points3D.txt             # ~200k 初始点 (Otsu 过滤 + 体素降采样)

# 4) 原版 3DGS 训练 + 渲染
#    增强开关 (默认全开, 改 0 即关, 便于消融):
#      POSE_ADJUST=1          训练前位姿变换 (居中+重力对齐+尺度归一化)
#      POSE_REFINE=1          训练中位姿精炼 (可学四元数+平移+内参)
#      ENABLE_DYNAMIC_MASK=1  动态掩码生成 (P0-1, GroundingDINO+SAM2)
#      ENABLE_DYNAMIC_FILTER=1 动态点云过滤 (P0-2, 多视角投影投票)
#      ENABLE_MLP_DYNAMIC=1   DINOv2+MLP 在线动态掩码 (P0-3, 动态感知损失)
#      USE_DEPTH_NORMAL=1     深度-法线一致性约束 (P1-1)
GPU=0 \
ITERATIONS=30000 \
WHITE_BG=0 \
POSE_ADJUST=1 \
POSE_REFINE=1 \
ENABLE_DYNAMIC_MASK=1 \
ENABLE_DYNAMIC_FILTER=1 \
ENABLE_MLP_DYNAMIC=1 \
USE_DEPTH_NORMAL=1 \
RESULTS_DIR=../../output/vggt_human_results \
bash vggt_human/04_train_3dgs.sh

# 输出：<RESULTS_DIR>/04_model_3dgs/
#   point_cloud/iteration_30000/point_cloud.ply    # 最终高斯
#   train/ours_30000/renders/*.png                 # 重建渲染 (vs GT)
#   train/ours_30000/gt/*.png                      # GT

# ── 去噪增强 + 人脸后处理（可选，提升稀疏区域 + 人脸质量）──
# 5) 渲染新视角 → 去噪 → AdaIN → 增强COLMAP场景
#    DENOISER 可选: diffbir (扩散, 质量高) | swinir (前馈, 快) | none (跳过去噪)
#    首次用 DiffBIR/SwinIR 需先: INSTALL_DENOISER=1 INSTALL_DEPS=1 bash vggt_human/00_setup_env.sh
GPU=0 \
DENOISER=diffbir \
NUM_NOVEL_VIEWS=10 \
ITERATION=30000 \
RESULTS_DIR=../../output/vggt_human_results \
bash vggt_human/05_denoise_novel.sh

# 输出：<RESULTS_DIR>/
#   05_novel_renders/*.png     # 3DGS 渲染的新视角 (有伪影)
#   05_novel_alpha/*.png       # 覆盖度图 (低 alpha = 稀疏区)
#   05_novel_poses.json        # 虚拟相机参数
#   05_source_aug/             # 增强COLMAP场景 (原图 + 去噪图)

# 6) 后处理人脸增强 (对增强 COLMAP 场景中的图)
#    与 step 01 调用同一个 face_enhance.py, 但对 05_source_aug/images/ 做后处理
GPU=0 \
RESULTS_DIR=../../output/vggt_human_results \
bash vggt_human/06_face_enhance.sh

# 输出：<RESULTS_DIR>/06_source_aug_face/
#   images/  # 原图 + 去噪图, 人脸区域已 HYPIR 增强 + 渐变融合
#   sparse/0/  # COLMAP 相机/点云 (原样复制)

# 7) 用增强场景训练 3DGS (原图 + 去噪虚拟相机 + 前后处理人脸增强 共同监督)
#    增强开关同 step 4 (POSE_ADJUST/POSE_REFINE/ENABLE_DYNAMIC_*/USE_DEPTH_NORMAL)
GPU=0 \
ITERATION=30000 \
WHITE_BG=0 \
POSE_ADJUST=1 \
POSE_REFINE=1 \
ENABLE_DYNAMIC_MASK=1 \
ENABLE_DYNAMIC_FILTER=1 \
ENABLE_MLP_DYNAMIC=1 \
USE_DEPTH_NORMAL=1 \
RESULTS_DIR=../../output/vggt_human_results \
bash vggt_human/07_train_denoise.sh

# 输出：<RESULTS_DIR>/07_model_3dgs_denoise/
#   point_cloud/iteration_30000/point_cloud.ply    # 增强训练后的高斯
```

- 结果：前处理人脸增强 → `01_input_face/images/`；VGGT-Omega 推理 → `02_vggt/<scene>/predictions.npz`；COLMAP 场景 → `03_source/`；3DGS 高斯 → `04_model_3dgs/point_cloud/iteration_30000/point_cloud.ply`。

## 首次准备

本流程**创建 `vggt_human` conda env**（从 `doll` 克隆，继承 torch>=2.3 + 3DGS CUDA 扩展），额外安装 HYPIR 依赖（diffusers/transformers/peft）+ mediapipe。clone 原版 3DGS + HYPIR 仓库 + 编译 CUDA 扩展（diff-gaussian-rasterization + simple-knn）。所有步骤（01-07）共用 `vggt_human` env。

> ⚠️ VGGT-Omega 权重是 gated 仓库，需先通过 `vggt-omega/01_download_models.sh` 下载（申请访问 + HF_TOKEN）。

```bash
cd <your-code-dir>            # e.g. /data_3d/<uid>/code
git -c http.sslVerify=false clone https://github.com/CrescentVelvet/media_code.git
cd media_code && cp proxy.env.example proxy.env   # 填 http_proxy / https_proxy
# ⚠️ 确认 proxy.env 中 http_proxy / https_proxy 已取消注释并填好地址

# 前提: 已做过 vggt-omega 的首次准备 (doll env + VGGT-Omega 权重)
#   INSTALL_DEPS=1 bash vggt-omega/00_setup_env.sh
#   VARIANT=1b_512 MODEL_DIR=../../model/VGGT-Omega bash vggt-omega/01_download_models.sh

# clone 3DGS 仓 + 装依赖 + 编 CUDA 扩展 (一次性)
GPU=0 INSTALL_DEPS=1 bash vggt_human/00_setup_env.sh
```

需系统有 CUDA toolkit（nvcc 可用；00 会自动检测 CUDA 版本与 torch 匹配）。

### 权重目录布局

```
$MODEL_DIR/                         # 默认 ../../model (code-dir 上一级, 各算法共享)
  VGGT-Omega/                        # VGGT-Omega checkpoint (gated HF 下载, 复用 vggt-omega)
    vggt_omega_1b_512.pt             # 默认变体 (512px)
    vggt_omega_1b_256_text.pt        # 256px text-aligned 变体
```

外部 clone 的官方代码（00 自动 clone，sibling of media_code）：
```
<code-dir>/
  media_code/vggt_human/             # 本目录 (编排脚本)
  vggt-omega/                        # VGGT-Omega 官方代码 (vggt-omega/00 已 clone; 本目录 00 确认存在)
  gaussian-splatting/                # 原版 3DGS 官方代码 (本目录 00 clone)
    submodules/
      diff-gaussian-rasterization/   # 3DGS 光栅化器 (CUDA 扩展, main 分支)
      simple-knn/                    # KNN (CUDA 扩展)
    third_party/glm/                 # GLM 数学库 (diff-gaussian-rasterization 依赖)
```

---

以下为详细参考（各步骤参数 / 目录布局）。流程原理详解（逐步机制、Otsu 过滤、体素降采样、AdaIN、渐变融合等）见 [NOTES.md](NOTES.md)「Pipeline（流程详解）」。

## Config (env vars, all optional)

### Paths & envs
| var | default | note |
| --- | --- | --- |
| `INPUT_DIR` | `../vggt-omega/examples` | 图像文件夹 / 视频 / 场景文件夹（见 Step 01） |
| `GPU` | _(unset)_ | physical GPU id, e.g. `GPU=0` |
| `CONDA_ENV` | `vggt_human` | conda env（从 doll 克隆，含 3DGS + HYPIR + mediapipe） |
| `VGGT_DIR` | `../vggt-omega` | VGGT-Omega 官方代码 |
| `GS_DIR` | `../gaussian-splatting` | 原版 3DGS 官方代码（00 clone） |
| `MODEL_DIR` | `../../model/VGGT-Omega` | VGGT-Omega checkpoint（gated） |
| `RESULTS_DIR` | `../vggt_human_results` | 输出根 |
| `INSTALL_DEPS` | `0` | `1` = 00 装依赖 + 编 CUDA 扩展 |
| `INSTALL_DENOISER` | `0` | `1` = 00 额外 clone + 下权重 DiffBIR / SwinIR |
| `WEIGHTS_ROOT` | `../../model` | 去噪模型权重根（与 VGGT-Omega 分开） |
| `DIFFBIR_DIR` | `../DiffBIR` | DiffBIR 官方代码（DENOISER=diffbir 时需要） |
| `SWINIR_DIR` | `../SwinIR` | SwinIR 官方代码（DENOISER=swinir 时需要） |
| `HYPIR_DIR` | `../HYPIR` | HYPIR 官方代码（step 05 人脸增强用） |
| `HYPIR_WEIGHT` | `$HYPIR_DIR/experiments/beauty_ppr50k_20260721/checkpoint-1000/ema_state_dict.pth` | HYPIR LoRA checkpoint |
| `HYPIR_BASE_MODEL` | `$WEIGHTS_ROOT/HYPIR/sd2_base` | SD2 base model dir |

### Step 02 params
| var | default | note |
| --- | --- | --- |
| `VGGT_OUTPUT_DIR` | `$RESULTS_DIR/02_vggt` | Step 02 输出 |
| `VARIANT` | `1b_512` | checkpoint 变体 |
| `RESOLUTION` | `512` | 输入分辨率（`1b_256_text` 用 `256`） |
| `MODE` | `balanced` | `balanced` / `max_size` |
| `CONF_THRES` | `20` | scene.ply 深度置信度百分位（0-100） |
| `MAX_POINTS` | `2000000` | scene.ply 点数上限 |
| `VIDEO_FPS` | `1` | 视频输入抽帧 fps |

### Step 01a params (video → frames)
| var | default | note |
| --- | --- | --- |
| `INPUT_DIR` | _(required)_ | 单个视频文件路径 (.mp4/.mov/.avi/.mkv) |
| `OUTPUT_DIR` | `$RESULTS_DIR/01a_input_frames` | 抽帧输出父夹（帧在 `<OUTPUT_DIR>/image/`） |
| `VIDEO_FPS` | `2` | 抽帧 fps |

### Step 03 params
| var | default | note |
| --- | --- | --- |
| `SCENE_NAME` | _(auto)_ | 场景子夹名（空 = 自动检测 VGGT_OUTPUT_DIR 下第一个） |
| `SOURCE_DIR` | `$RESULTS_DIR/03_source` | COLMAP 输出 |
| `TARGET_POINTS` | `200000` | 体素降采样目标点数 |
| `ALIGN` | `1` | `1` = 居中场景到原点 |

### Step 04 params
| var | default | note |
| --- | --- | --- |
| `GAUSSIAN_DIR` | `$RESULTS_DIR/04_model_3dgs` | 高斯输出 |
| `ITERATIONS` | `30000` | 训练迭代数 |
| `RES` | _(unset)_ | `--resolution` 因子；不设 = 全分辨率 |
| `WHITE_BG` | `0` | `1` = 白底光栅化 |
| `SKIP_RENDER` | `0` | `1` = 跳过渲染 |
| `SKIP_METRICS` | `1` | `1` = 跳过 PSNR/SSIM/LPIPS |
| `TRAIN_EXTRA_ARGS` | _(empty)_ | 透传给 train.py 的额外参数 |

### Pose optimization params (step 04/07, default ON)
| var | default | note |
| --- | --- | --- |
| `POSE_ADJUST` | `1` | `1` = 训练前 PoseAdjuster（视线交点居中 + SVD 重力对齐 + 尺度归一化） |
| `POSE_REFINE` | `1` | `1` = 训练中 PoseRefineModule（可学四元数+平移+内参精炼） |
| `REFINE_INTRINSIC` | `0` | `1` = 同时学内参（fx/fy/cx/cy），0 = 只学位姿 |
| `POSE_REFINE_WEIGHT` | `0.01` | 位姿正则化损失权重（防偏离初始值太远） |
| `POSE_REFINE_LR_Q` | `1e-3` | 四元数学习率 |
| `POSE_REFINE_LR_T` | `1e-3` | 平移学习率 |
| `POSE_REFINE_LR_I` | `1e-4` | 内参学习率 |
| `GRAVITY_PRIOR` | `0` | `0` = SVD 估计重力方向，`1` = 用 [0,-1,0] |

### Dynamic mask & filtering & MLP params (step 04, P0-1/P0-2/P0-3, independently toggleable)
| var | default | note |
| --- | --- | --- |
| `ENABLE_DYNAMIC_MASK` | `1` | `1` = 训练前生成动态掩码（GroundingDINO + SAM2/SAM），`0` = 跳过 |
| `ENABLE_DYNAMIC_FILTER` | `1` | `1` = 过滤动态点云（多视角投影投票），`0` = 跳过（需 MASK 先开） |
| `ENABLE_MLP_DYNAMIC` | `1` | `1` = DINOv2+MLP 在线动态掩码学习 + 动态感知损失，`0` = 标准 L1+SSIM |
| `DYNAMIC_MASK_DIR` | `$GAUSSIAN_DIR/dynamic_mask` | 动态掩码 debug 输出目录 |
| `SAM2_MODEL_PATH` | `$MODEL_DIR/sam2` | SAM2 checkpoint 目录或 .pt 文件 |
| `SAM2_CONFIG` | `configs/sam2.1/sam2.1_hiera_large.yaml` | SAM2 config yaml |
| `SAM2_DIR` | _(unset)_ | sam2 仓目录（用于 config 路径解析） |
| `SAM_MODEL_ID` | `facebook/sam-vit-base` | SAM 回退的 HF model ID |
| `GROUNDING_DINO_ID` | `IDEA-Research/grounding-dino-tiny` | GroundingDINO HF model ID |
| `BOX_THRESHOLD` | `0.3` | GroundingDINO box 置信度阈值 |
| `TEXT_THRESHOLD` | `0.25` | GroundingDINO text 置信度阈值 |
| `DYNAMIC_THRESHOLD` | `0.3` | 动态点过滤比率阈值（in_mask/valid > 阈值则移除） |
| `DYNAMIC_DILATE_PX` | `5` | mask 膨胀像素数（投影前膨胀，避免边缘漏判） |
| `DINO_MODEL_PATH` | `$MODEL_DIR/dinov2` | DINOv2 ViT-S/14 reg checkpoint 目录或 .pth |
| `USE_DEPTH_NORMAL` | `1` | `1` = 启用深度-法线一致性约束（需 render 支持 render_depth） |
| `DEPTH_NORMAL_WEIGHT` | `0.05` | 深度-法线一致性 loss 权重 |
| var | default | note |
| --- | --- | --- |
| `DENOISER` | `none` | `diffbir` \| `swinir` \| `nafnet` \| `none`（可插拔，见 denoisers.py） |
| `NUM_NOVEL_VIEWS` | `10` | 插入多少个虚拟相机 |
| `ALPHA_THRESH` | `0.3` | 渲染 alpha 低于此值 = 稀疏区，需去噪 |
| `ADAIN_REF` | `nearest` | AdaIN 颜色参考：`nearest`（最近训练图） \| `mean`（全局均值色） |
| `ITERATION` | `30000` | 加载 04 的哪个 iteration 的 checkpoint |
| `GAUSSIAN_DIR` | `$RESULTS_DIR/04_model_3dgs` | 3DGS 模型目录（04 的输出） |
| `SOURCE_AUG_DIR` | `$RESULTS_DIR/05_source_aug` | 增强 COLMAP 输出 |

### Step 06 params (face enhance)
| var | default | note |
| --- | --- | --- |
| `HYPIR_WEIGHT` | `$HYPIR_DIR/experiments/beauty_ppr50k_20260721/checkpoint-1000/ema_state_dict.pth` | HYPIR LoRA checkpoint |
| `HYPIR_BASE_MODEL` | `$WEIGHTS_ROOT/HYPIR/sd2_base` | SD2 base model dir |
| `FACE_PADDING` | `0.2` | 人脸框放大比例 (0.2 = 20%) |
| `UPSCALE` | `1` | HYPIR upscale (1 = 不超分, 只增强) |
| `PATCH_SIZE` | `512` | HYPIR patch size |
| `STRIDE` | `256` | HYPIR stride |
| `SOURCE_FACE_DIR` | `$RESULTS_DIR/06_source_aug_face` | 输出目录 |

### Step 07 params (train on enhanced scene)
| var | default | note |
| --- | --- | --- |
| `SOURCE_AUG_DIR` | `$RESULTS_DIR/06_source_aug_face` | 增强场景（06 的输出; fallback: 05_source_aug） |
| `GAUSSIAN_DENOISE_DIR` | `$RESULTS_DIR/07_model_3dgs_denoise` | 增强训练的模型输出 |
| `ITERATIONS` | `30000` | 训练迭代数 |
| `RES` | _(unset)_ | `--resolution` 因子；不设 = 全分辨率 |
| `WHITE_BG` | `0` | `1` = 白底光栅化 |
| `MODEL_PATH` | _(unset)_ | 续训：从哪个 model_path 加载 checkpoint |
| `LOADED_ITER` | _(unset)_ | 续训：加载第几轮的 checkpoint |
| `SKIP_RENDER` | `0` | `1` = 跳过渲染 |
| `SKIP_METRICS` | `1` | `1` = 跳过 PSNR/SSIM/LPIPS |
| `TRAIN_EXTRA_ARGS` | _(empty)_ | 透传给 train.py 的额外参数 |

## 可能遇到的问题

**全部迁至 [NOTES.md](NOTES.md)「可能遇到的问题」**：CUDA 扩展编译失败、GLM 缺失、checkpoint not found、OOM、API 签名变化（6b）、CRLF 行尾、DiffBIR/SwinIR 未装、续训 checkpoint、mediapipe/HYPIR 缺失、人脸融合硬边等 12 条。

## 目录布局
```
<code-dir>/
├── media_code/                     # 本仓
│   ├── proxy.env                   # 代理 + HF_TOKEN, gitignored
│   └── vggt_human/                  # ← 本目录（编排脚本）
│       ├── _env.sh                 # 共享: 代理 + CA + conda + GPU + paths
│       ├── 00_setup_env.sh        # clone 3DGS 仓 + 装依赖 + 编 CUDA 扩展
│       ├── 01a_video_to_frames.sh # 视频 → 图像夹 (抽帧, 喂给 01; 视频输入用)
│       ├── 01_face_enhance.sh      # 前处理: MediaPipe+HYPIR 人脸增强 (原始输入图)
│       ├── 02_run_inference.sh     # VGGT-Omega 前馈推理
│       ├── 03_npz_to_colmap.sh     # npz -> COLMAP 转换
│       ├── 04_train_3dgs.sh        # 原版 3DGS 训练 + 渲染
│       ├── 05_denoise_novel.sh     # 渲染新视角 → 去噪 → AdaIN → 增强COLMAP
│       ├── 06_face_enhance.sh      # 后处理: MediaPipe+HYPIR 人脸增强 (增强场景图)
│       ├── 07_train_denoise.sh     # 增强场景训练 3DGS
│       ├── run_batch.py            # VGGT-Omega 批量重建 (vggt-omega 副本)
│       ├── npz_to_colmap.py        # npz -> COLMAP 转换 (Otsu + 体素降采样)
│       ├── render_novel.py         # 3DGS 渲染新视角 (stage 1 of 05)
│       ├── denoise_images.py       # 去噪 + AdaIN + 增强COLMAP (stage 2 of 05)
│       ├── denoisers.py            # 去噪模型注册表 (DiffBIR/SwinIR/none 可插拔)
│       ├── face_enhance.py         # MediaPipe + HYPIR + 渐变融合 (step 01/06)
│       ├── video_to_frames.py      # 视频抽帧 (step 01a, cv2)
│       ├── dynamic_mask.py         # 动态物体掩码 (GroundingDINO+SAM2, step 04)
│       ├── dynamic_filter.py       # 动态点云过滤 (多视角投影投票, step 04)
│       ├── noise_negating.py       # DINOv2+MLP 在线动态掩码学习 (step 04, P0-3)
│       ├── depth_normal_cons.py     # 深度-法线一致性约束 (step 04, P1-1)
│       └── train_pose.py           # 3DGS 训练 wrapper (PoseAdjuster + 动态掩码)
├── vggt-omega/                      # VGGT-Omega 官方代码 (vggt-omega/00 clone)
├── gaussian-splatting/             # 原版 3DGS 官方代码 (本目录 00 clone)
│   ├── submodules/
│   │   ├── diff-gaussian-rasterization/   # 3DGS 光栅化器 (CUDA 扩展)
│   │   └── simple-knn/                    # KNN (CUDA 扩展)
│   ├── third_party/glm/            # GLM 数学库
│   └── train.py / render.py / metrics.py
├── model/                          # 权重根 (共享)
│   └── VGGT-Omega/                 # checkpoint (gated HF 下载, 复用 vggt-omega)
│   ├── DiffBIR/                   # DiffBIR checkpoint (INSTALL_DENOISER=1 下载)
│   ├── SwinIR/                    # SwinIR checkpoint
│   └── HYPIR/                     # SD2 base model + beauty LoRA (00 下载)
├── DiffBIR/                         # DiffBIR 官方代码 (00 clone, DENOISER=diffbir 时)
├── SwinIR/                          # SwinIR 官方代码 (00 clone, DENOISER=swinir 时)
├── HYPIR/                           # HYPIR 官方代码 (00 clone, step 01/06 用)
└── output/vggt_human_results/      # 输出 (repo 外)
    ├── 01a_input_frames/           # step 01a: 视频抽帧输出 (test_task 结构)
    │   └── image/                  #   000000.png, 000001.png, ...
    ├── 01_input_face/              # step 01: 前处理人脸增强后的原始图
    │   └── images/                 #   人脸增强图
    ├── 02_vggt/<scene>/            # step 02: VGGT-Omega 推理
    │   ├── predictions.npz         #   原始输出
    │   ├── scene.ply               #   点云 (供检查)
    │   └── frames/                 #   训练图
    ├── 03_source/                  # step 03: COLMAP 场景
    │   ├── images/                 #   训练图 (复制)
    │   └── sparse/0/               #   cameras.txt / images.txt / points3D.txt
    ├── 03b_source_ba/              # step 03b: COLMAP BA 精修后的场景
    │   ├── images/                 #   训练图 (复制)
    │   └── sparse/0/               #   BA binary 输出
    │   └── sparse/0_text/          #   BA text 输出 (脚本读这份)
    ├── 04_model_3dgs/              # step 04: 3DGS 高斯
    │   ├── point_cloud/iteration_30000/point_cloud.ply   # 最终高斯
    │   └── train/ours_30000/       #   渲染 (重建 vs GT)
    ├── 04b_model_3dgs_ba/          # step 04 用 03b_source_ba 训练的高斯
    ├── 05_novel_renders/           # step 05 stage 1: 渲染的新视角
    ├── 05_novel_alpha/             #   覆盖度图 (低 alpha = 稀疏区)
    ├── 05_novel_poses.json         #   虚拟相机参数
    ├── 05_source_aug/              # step 05 stage 2: 增强 COLMAP 场景
    │   ├── images/                 #   原图 + 去噪虚拟视角图
    │   └── sparse/0/               #   原相机 + 虚拟相机
    ├── 06_source_aug_face/         # step 06: 人脸增强后的场景
    │   └── images/                 #   人脸已 HYPIR 增强 + 渐变融合
    ├── 06b_source_face/            # step 06: 对 03_source 做人脸增强的场景
    ├── 06b_face_masks/             # step 06b: MediaPipe bbox 人脸 loss mask
    ├── 06b_model_3dgs_face/        # step 06b: 方案一 finetune 后的高斯
    ├── 06c_sam2_face_masks/        # step 06c: SAM2 像素级人脸 mask (原始视角)
    ├── 06c_face_center.json        # step 06c: 人脸 3D 中心
    ├── 06c_closeup_renders/        # step 06c: 外插近景渲染
    ├── 06c_closeup_alpha/          #   近景覆盖度图
    ├── 06c_closeup_poses.json      #   近景相机参数
    ├── 06c_closeup_enhanced/       # step 06c: HYPIR 增强后的近景图
    │   └── images/
    ├── 06c_closeup_masks/          # step 06c: 近景图的人脸 mask
    ├── 06c_merged_face_images/     # step 06c: 原始 + 近景 合并训练集
    ├── 06c_merged_face_masks/      # step 06c: 合并的人脸 mask
    ├── 06c_model_3dgs_closeup/     # step 06c: 方案二 finetune 后的高斯
    └── 07_model_3dgs_denoise/      # step 07: 增强训练后的高斯
        └── point_cloud/iteration_30000/point_cloud.ply
```

### 输出目录命名规则

`$RESULTS_DIR/` 下的每个产物目录/文件都带**产生它的步骤号前缀**，按前缀排序即等于按 pipeline 顺序排序，一眼能看出"谁产出的、什么时候产出的"。

| 前缀 | 产出步骤 | 产物 |
|---|---|---|
| `01a_` | `01a_video_to_frames.sh` | `01a_input_frames/` |
| `01_` | `01_face_enhance.sh` | `01_input_face/` |
| `02_` | `02_run_inference.sh` | `02_vggt/` |
| `03_` | `03_npz_to_colmap.sh` | `03_source/` |
| `03b_` | `03b_colmap_ba.sh` | `03b_source_ba/` |
| `04_` | `04_train_3dgs.sh` | `04_model_3dgs/` |
| `04b_` | `04_train_3dgs.sh`（输入 `03b_source_ba`） | `04b_model_3dgs_ba/` |
| `05_` | `05_denoise_novel.sh` | `05_novel_*`、`05_source_aug/` |
| `06_` | `06_face_enhance.sh` | `06_source_aug_face/` |
| `06b_` | `06b_face_finetune.sh` | `06b_source_face/`、`06b_face_masks/`、`06b_model_3dgs_face/` |
| `06c_` | `06c_closeup_finetune.sh` | `06c_sam2_face_masks/`、`06c_face_center.json`、`06c_closeup_*`、`06c_merged_*`、`06c_model_3dgs_closeup/` |
| `06d_` | `06d_continue_train.sh` | `06d_model_3dgs_continue/` |
| `07_` | `07_train_denoise.sh` | `07_model_3dgs_denoise/` |

> 脚本内部已按此命名设好默认值，直接用环境变量覆盖即可（`RESULTS_DIR` / `SOURCE_DIR` / `GAUSSIAN_DIR` …）。改前缀只影响默认值，不影响已有产物。

## Notes
- **实验结论/消融数据/踩坑归档**：见 [`EXPERIMENTS.md`](EXPERIMENTS.md)（人脸 finetune 消融、06e/06g 近景注入、SSIM composite 修正、HYPIR 推理提速等）。
- Pipeline: VGGT-Omega（前馈位姿+深度）→ COLMAP（格式转换）→ 3DGS（优化训练）。前馈给初始化，优化给质量。
- **去噪增强（04，可选）**：3DGS 在稀疏视角区域有伪影 → 渲染新视角 → 去噪（DiffBIR/SwinIR 可切换）→ AdaIN 颜色校正 → 虚拟相机加入训练。`DENOISER=none` 关闭去噪。加新去噪模型：在 `denoisers.py` 写一个函数 + 注册到 `DENOISERS` 字典。
- **人脸增强（05，可选）**：MediaPipe 检测人脸 → HYPIR 美颜增强 → 二次衰减渐变 mask 无缝融合回原图。`HYPIR_WEIGHT` 指向 beauty_ppr50k 训练的 LoRA checkpoint。
- VGGT-Omega 的 `extrinsic` 是 w2c（OpenCV 约定），与 COLMAP 一致——无需 c2w→w2c 转换。`intrinsic` 是模型预测的实际内参——无需假设 fx=fy=max(W,H)。
- 自适应置信度过滤用 Otsu's method（最大化类间方差），比固定阈值更鲁棒。体素降采样每体素保留最高置信度点。
- 原版 3DGS 无 distractor filtering（不做微动过滤）。静态场景够用；有微动用 pdfgs_human（PDF-GS）。
- 无网格输出（3DGS 仓库无 `extract_mesh`）。要网格走 wan22_rotate step 05/05a/05b。
- `.gitattributes`（仓根）强制 LF。`proxy.env` gitignored。官方代码 & 权重遵循各自 license。

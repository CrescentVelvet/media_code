# VGGT-Human — VGGT-Omega 前馈重建 → COLMAP → 原版 3DGS 训练

用 [VGGT-Omega (CVPR 2026 Oral)](https://github.com/facebookresearch/vggt-omega) 做前馈三维重建（一次推理出相机位姿 + 深度图 → 反投影成密集点云），再转 COLMAP 格式喂 [原版 3DGS](https://github.com/graphdeco-inria/gaussian-splatting)（3D Gaussian Splatting）做迭代优化训练。VGGT-Omega 提供**快速、鲁棒的位姿 + 密集点云初始化**（前馈，无需特征匹配 / SfM），3DGS 在此基础上**优化出高质量高斯表示**（novel-view 合成 + 渲染）。

> 本目录与 [`pdfgs_human/`](../pdfgs_human/) 并列且**思路相同**：都是"前馈模型出位姿 + 点云 → COLMAP → 3DGS 训练"。区别在于前馈模型和 3DGS 变体不同：
> - pdfgs_human：**Pi3**（前馈位姿）→ COLMAP → **PDF-GS**（带 DINOv3 distractor filtering 的 3DGS，抗微动）
> - 本目录：**VGGT-Omega**（1B 大模型，前馈位姿 + 深度 + 置信度）→ COLMAP → **原版 3DGS**（无 distractor filtering，更简单）
>
> 两者不共用 env（本目录复用 `doll` env，pdfgs_human 用 `pdfgs` env）。

## 为什么用 VGGT-Omega + 原版 3DGS

### VGGT-Omega vs Pi3（前馈位姿）

| | VGGT-Omega | Pi3 |
|---|---|---|
| 模型规模 | 1B 参数 | ~300M |
| 输出 | 位姿 + 深度 + 置信度 | 位姿 + 点云 + 置信度 |
| 内参 | **模型预测**（每视图实际内参） | 无（假设 fx=fy=max(W,H)） |
| 外参 | w2c（直接用，无需 c2w→w2c 转换） | c2w（需转 w2c） |
| 点云 | 深度反投影（密集，N×H×W 个点） | 模型直接输出（需降采样） |

VGGT-Omega 的优势：**实际内参**（不是假设的）+ **置信度过滤**（Otsu 自适应阈值）+ **更大模型**（可能位姿更准）。Pi3 的优势：更轻量、Pi3 有专门的 3DGS 导出（`pi3_recon.py` 现成）。

### 原版 3DGS vs PDF-GS

| | 原版 3DGS | PDF-GS |
|---|---|---|
| distractor filtering | 无 | DINOv3 特征过滤微动像素 |
| DINOv3 依赖 | 无（免 gated 下载） | 必须（gated 或本地 vitl16） |
| 训练 | 30k iter，单 phase | 4 phase × 10k iter |
| 适合场景 | 静态场景 | 有微动（呼吸/头发飘动） |

如果拍摄时人体静止（或微动可忽略），原版 3DGS 更简单（无 DINOv3 依赖），训练更快。如果有微动，用 pdfgs_human（PDF-GS 抗微动）。

**结论**：静态场景 + 想试 VGGT-Omega 的位姿/深度质量 → 用本目录；有微动 → 用 pdfgs_human。

## 常用命令

> 假设已进入容器（脚本自动激活 `doll` env）；`GPU=0` 按需换卡。首次跑前先做下方「首次准备」。
> **铁律：每条命令都必须显式写出模型路径、输入路径、输出路径，不能全靠脚本里的默认值。** 用具体路径，不要用 `...` 占位。

```bash
# ── 分步 ──
# 0) clone 仓 + 装依赖 + 编 CUDA 扩展 (一次性)
# GPU=0 INSTALL_DEPS=1 bash vggt_human/00_setup_env.sh
GPU=0 INSTALL_DENOISER=1 INSTALL_DEPS=1 bash vggt_human/00_setup_env.sh

# 1) 前处理人脸增强 (MediaPipe + HYPIR + 渐变融合, 对原始输入图)
#    INSTALL_DEPS=1 bash vggt_human/00_setup_env.sh 建好 vggt_human env (含 mediapipe + HYPIR)
#    HYPIR_WEIGHT 指向 beauty_ppr50k 训练的 checkpoint
GPU=0 INPUT_DIR=../Reconstruction/dataset/B003_Human_Data_w_pose/test_task_id_3a8b3cc746304f49b9e3275e36aa9374 \
  RESULTS_DIR=../../output/vggt_human_results \
  bash vggt_human/01_face_enhance.sh

# 输出：<RESULTS_DIR>/input_face/images/  # 人脸增强后的原始图

# 2) VGGT-Omega 前馈推理 (图像 -> 位姿+深度 -> predictions.npz + scene.ply)
#    INPUT_DIR 指向 step 01 的输出 (input_face)
GPU=0 \
INPUT_DIR=../../output/vggt_human_results/input_face \
  MODEL_DIR=../../model/VGGT-Omega \
  RESULTS_DIR=../../output/vggt_human_results \
  MAX_POINTS=2000000 \
  bash vggt_human/02_run_inference.sh

# 输出：<RESULTS_DIR>/vggt/<scene>/
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

# 输出：<RESULTS_DIR>/source/
#   images/*.png                      # 训练图 (从 frames/ 复制, 内参自动缩放)
#   sparse/0/cameras.txt              # PINHOLE (VGGT-Omega 实际内参)
#   sparse/0/images.txt              # w2c (qw qx qy qz tx ty tz)
#   sparse/0/points3D.txt             # ~200k 初始点 (Otsu 过滤 + 体素降采样)

# 4) 原版 3DGS 训练 + 渲染
GPU=0 \
  ITERATIONS=30000 \
  WHITE_BG=0 \
  RESULTS_DIR=../../output/vggt_human_results \
  bash vggt_human/04_train_3dgs.sh

# 输出：<RESULTS_DIR>/model_3dgs/
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
#   novel_renders/*.png     # 3DGS 渲染的新视角 (有伪影)
#   novel_alpha/*.png       # 覆盖度图 (低 alpha = 稀疏区)
#   novel_poses.json        # 虚拟相机参数
#   source_aug/             # 增强COLMAP场景 (原图 + 去噪图)

# 6) 后处理人脸增强 (对增强 COLMAP 场景中的图)
#    与 step 01 调用同一个 face_enhance.py, 但对 source_aug/images/ 做后处理
GPU=0 \
  RESULTS_DIR=../../output/vggt_human_results \
  bash vggt_human/06_face_enhance.sh

# 输出：<RESULTS_DIR>/source_aug_face/
#   images/  # 原图 + 去噪图, 人脸区域已 HYPIR 增强 + 渐变融合
#   sparse/0/  # COLMAP 相机/点云 (原样复制)

# 7) 用增强场景训练 3DGS (原图 + 去噪虚拟相机 + 前后处理人脸增强 共同监督)
GPU=0 \
  ITERATION=30000 \
  WHITE_BG=0 \
  RESULTS_DIR=../../output/vggt_human_results \
  bash vggt_human/07_train_denoise.sh

# 输出：<RESULTS_DIR>/model_3dgs_denoise/
#   point_cloud/iteration_30000/point_cloud.ply    # 增强训练后的高斯
```

- 结果：前处理人脸增强 → `input_face/images/`；VGGT-Omega 推理 → `vggt/<scene>/predictions.npz`；COLMAP 场景 → `source/`；3DGS 高斯 → `model_3dgs/point_cloud/iteration_30000/point_cloud.ply`。

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

以下为详细参考（流程原理 / 各步骤参数 / 排错 / 目录布局）。

## Pipeline（流程详解）

```
INPUT_DIR/                           (一组场景图像 / 视频)
    │
    ▼
[01] 前处理人脸增强 (vggt_human env) — MediaPipe → HYPIR → 渐变融合
    │  ├─ MediaPipe BlazeFace → 人脸框 → 放大 20% → 裁剪
    │  ├─ HYPIR (SD2Enhancer + LoRA beauty_ppr50k) 增强裁剪图 (upscale=1)
    │  └─ 二次衰减渐变 mask: 中心=增强, 边缘=原图 → 无缝融合
    ▼
$RESULTS_DIR/input_face/images/      (人脸增强后的原始图)
    │
    ▼
[02] VGGT-Omega 前馈推理 (doll env)  — 一次前向 → 所有视图位姿 + 深度图
    │  ├─ VGGTOmega(images) -> pose_enc + depth + depth_conf + images
    │  ├─ encoding_to_camera -> 外参 w2c (N,3,4) + 内参 (N,3,3)
    │  └─ unproject_depth -> 世界坐标点云 + 置信度过滤 -> scene.ply
    ▼
$RESULTS_DIR/vggt/<scene>/
    predictions.npz      # extrinsic(w2c) + intrinsic + world_points + depth_conf + images
    scene.ply             # 置信度过滤后的彩色点云 (供检查)
    frames/               # 喂给模型的图 (复制/抽帧)
    │
    ▼
[03] npz -> COLMAP 转换 (doll env) — 自适应过滤 + 降采样 + 坐标对齐
    │  ├─ 加载 predictions.npz: extrinsic(w2c), intrinsic, world_points, depth_conf
    │  ├─ 复制 frames/ -> source/images/ (内参按原图尺寸缩放)
    │  ├─ 自适应置信度过滤: Otsu 阈值 (分离高/低置信度点)
    │  ├─ 体素降采样 -> ~200k 点 (每体素保留最高置信度点)
    │  ├─ 坐标对齐: 点云 + 相机平移居中到原点
    │  └─ 写 COLMAP 文本格式: cameras.txt + images.txt + points3D.txt
    ▼
$RESULTS_DIR/source/
    images/*.png                  # 训练图
    sparse/0/cameras.txt          # PINHOLE (VGGT-Omega 实际内参)
    sparse/0/images.txt           # w2c (qw qx qy qz tx ty tz)
    sparse/0/points3D.txt         # ~200k 初始点
    │
    ▼
[04] 3DGS 训练 (doll env) — 原版 gaussian-splatting
    │  ├─ train.py -s source -m model_3dgs --iterations 30000
    │  │   (L1+SSIM loss, adaptive density: split/clone/prune, 30k iter)
    │  └─ render.py -s source -m model_3dgs (渲染训练视角 vs GT)
    ▼
$RESULTS_DIR/model_3dgs/
    point_cloud/iteration_30000/point_cloud.ply    # 最终高斯
    train/ours_30000/renders/*.png                 # 重建渲染
    train/ours_30000/gt/*.png                      # GT
    │
     ▼ (可选: 去噪增强)
[05] 渲染新视角 → 去噪 → AdaIN → 增强COLMAP (doll env)
    │  ├─ Stage 1 (render_novel.py): 加载3DGS → 轨迹找间隙 → 插入虚拟相机 → 渲染 (black+white bg for alpha)
    │  └─ Stage 2 (denoise_images.py): alpha<阈值=稀疏 → DENOISER去噪 → AdaIN颜色校正 → 写增强COLMAP
    ▼
$RESULTS_DIR/source_aug/
    images/*.png + novel_*.png     # 原图 + 去噪虚拟视角图
    sparse/0/{cameras,images,points3D}.txt  # 原相机 + 虚拟相机
    │
    ▼
[06] 后处理人脸增强 (vggt_human env) — MediaPipe 检测 → HYPIR 美颜 → 渐变融合
    │  ├─ MediaPipe BlazeFace → 人脸框 → 放大 20% → 裁剪
    │  ├─ HYPIR (SD2Enhancer + LoRA beauty_ppr50k) 增强裁剪图
    │  └─ 二次衰减渐变 mask: 中心=增强, 边缘=原图 → 无缝融合
    ▼
$RESULTS_DIR/source_aug_face/
    images/  # 原图 + 去噪图 (人脸区域已增强+融合)
    sparse/0/  # COLMAP 原样复制
    │
    ▼
[07] 3DGS 训练 (增强场景) — 原图 + 去噪虚拟相机 + 前后处理人脸增强 共同监督
    │  └─ train.py -s source_aug_face -m model_3dgs_denoise --iterations 30000
    ▼
$RESULTS_DIR/model_3dgs_denoise/
    point_cloud/iteration_30000/point_cloud.ply    # 增强训练后的高斯
```

### Step 00 — clone 仓 + 装依赖 + 编 CUDA 扩展 (`00_setup_env.sh`)

复用 `doll` conda env（torch>=2.3 预装）。clone 两个官方仓：VGGT-Omega（vggt-omega/00 已 clone，本步确认存在）+ gaussian-splatting（含 submodules）。编译两个 CUDA 扩展：
- **diff-gaussian-rasterization**：3DGS 光栅化器（main 分支，原版无 antialiasing）
- **simple-knn**：KNN 查询（gitlab.inria.fr 可能被公司代理封，00 自动 fallback 到 GitHub 镜像）

需要 gcc 12（conda install gxx_linux-64=12 python=3.10）+ CUDA toolkit（nvcc）。00 自动检测 CUDA 版本与 torch 匹配，不匹配时自动找 `cuda-12.x`。

### Step 01 — 前处理人脸增强 (`01_face_enhance.sh` → `face_enhance.py`)

对原始输入图（`INPUT_DIR`）做前处理人脸增强。与 Step 06（后处理）调用**同一个 `face_enhance.py`**，区别是输入：01 对原始图（无 COLMAP 场景），06 对增强 COLMAP 场景中的图。`face_enhance.py` 自动适配输入结构（`images/` 子夹 / `image/` 子夹 / 散图夹）。输出到 `input_face/images/`，作为 Step 02 的输入。

### Step 02 — VGGT-Omega 前馈推理 (`02_run_inference.sh` → `run_batch.py`)

复用 vggt-omega 的 `run_batch.py`（副本）。模型加载一次，循环场景。`INPUT_DIR` 支持图像文件夹 / 视频 / 场景文件夹（批量）。每个场景产出 `predictions.npz`（extrinsic w2c + intrinsic + world_points + depth_conf + images，与官方 demo 同 keys）+ `scene.ply` + `frames/`。

> VGGT-Omega 的 `extrinsic` 是 **w2c**（world-to-camera [R | t]，OpenCV 约定），与 COLMAP 格式一致——无需 c2w→w2c 转换（Pi3 输出 c2w 需要转）。`intrinsic` 是模型预测的**实际内参**（不是 Pi3 假设的 fx=fy=max(W,H)）。

### Step 03 — npz → COLMAP 转换 (`03_npz_to_colmap.sh` → `npz_to_colmap.py`)

读 `predictions.npz`，输出 COLMAP 文本格式场景：

**自适应置信度过滤**：对 `depth_conf` 直方图做 Otsu's method（最大化类间方差），自动找到高/低置信度的自然分界点。若结果过疏（<5% 点保留）或过密（>80%），回退到百分位阈值。比固定阈值更鲁棒——不同场景的置信度分布差异大。

**体素降采样到 ~200k**：从场景包围盒体积和目标点数算出体素大小 `voxel_size = cbrt(volume / target)`。每个体素保留**最高置信度**的点。比随机采样更好——保留的是高质量点，而非随机子集。

**坐标系对齐**：点云减去质心居中到原点，相机平移相应调整（`t_new = R @ centroid + t`）。旋转不变。帮助 3DGS 训练稳定性（场景在原点附近，learning rate 和 densification 阈值更合理）。

**图像处理**：优先从 `frames/` 复制原图（质量更好），内参按原图 / 预处理图尺寸比缩放。若 `frames/` 不可用，从 npz 的 `images` 数组保存 PNG。

### Step 04 — 原版 3DGS 训练 (`04_train_3dgs.sh`)

在 `doll` env 里跑 `gaussian-splatting` 的 `train.py`（`cd $GS_DIR` 内跑，保证相对 import）。`-s $SOURCE_DIR -m $GAUSSIAN_DIR --iterations 30000`。L1+SSIM loss + adaptive density control（split/clone/prune）。渲染训练视角到 `train/ours_30000/{renders,gt}/`。

> 无 `--eval` → 无 held-out test split → `scene.getTestCameras()` 为空，"test" 集自动跳过。无网格输出（3DGS 仓库无 `extract_mesh`）。

### Step 05 — 渲染新视角 → 去噪 → AdaIN → 增强 COLMAP (`05_denoise_novel.sh`)

**两阶段 pipeline**（分进程执行，避免 GPU 显存冲突）：

**Stage 1 — `render_novel.py`**：从 step 04 的 checkpoint 加载 3DGS 高斯 → 解析 COLMAP 相机轨迹 → 按绕场景中心的方位角排序 → 找最大间隙 → 插入 `NUM_NOVEL_VIEWS` 个中间视角（位置线性插值 + 旋转 SLERP）→ 渲染每个新视角（黑底 + 白底两次渲染算 alpha）→ 保存 PNG + `novel_poses.json`。

**Stage 2 — `denoise_images.py`**：逐视角检查 alpha → `avg_alpha < ALPHA_THRESH` 的 = 稀疏区（3DGS 有伪影）→ `DENOISER` 去噪（DiffBIR / SwinIR / none 可切换）→ AdaIN 颜色校正（去噪图的均值/标准差对齐到最近训练图）→ 写增强 COLMAP 场景（原图 + 去噪图，原相机 + 虚拟相机）。

> **去噪模型可插拔**：`denoisers.py` 用 registry 模式，每个去噪器是一个函数 `(image, device) -> image`。加新模型只需写一个函数 + 注册到 `DENOISERS` 字典。`DENOISER=none` 跳过去噪（仅渲染 + AdaIN）。首次用 DiffBIR/SwinIR 需 `INSTALL_DENOISER=1` 让 00 clone 仓库 + 下权重。

### Step 06 — 后处理人脸增强 (`06_face_enhance.sh` → `face_enhance.py`)

**用 `vggt_human` env**（有 diffusers/transformers/peft + mediapipe）。对 `source_aug/images/`（或 `source/images/`）中的每张图：

1. **MediaPipe BlazeFace** 检测人脸框 → 放大 `FACE_PADDING`（默认 20%）后裁剪。
2. **HYPIR 增强**：裁剪图喂给 `SD2Enhancer`（加载 `HYPIR_WEIGHT` 指向的 beauty_ppr50k LoRA checkpoint），`upscale=1`（只增强不超分）。
3. **渐变融合**：二次衰减 mask（中心=1, 边缘=0）把增强结果无缝融合回原图——中心区域完全用 HYPIR 结果，边缘平滑过渡到原图，避免硬边。
4. COLMAP `sparse/` 原样复制（只增强图像，不改相机参数）。

> ⚠️ `vggt_human` env 需先通过 `INSTALL_DEPS=1 bash vggt_human/00_setup_env.sh` 建好（从 doll 克隆 + 装 HYPIR 依赖 + mediapipe + clone HYPIR 仓 + 下 SD2 base model）。`HYPIR_WEIGHT` 默认指向 `beauty_ppr50k_20260721/checkpoint-1000/ema_state_dict.pth`，可改。

### Step 07 — 增强场景训练 (`07_train_denoise.sh`)

在 `doll` env 里用增强 COLMAP 场景训练 3DGS（`-s $SOURCE_AUG_DIR -m $GAUSSIAN_DENOISE_DIR`）。原图提供 GT 监督，去噪虚拟相机提供稀疏区域的额外监督，前后处理人脸增强提供更好的面部质量。默认从头训；可选从 04 的 checkpoint 续训（需 04 加 `--checkpoint_iterations`）。

## Config (env vars, all optional)

### Paths & envs
| var | default | note |
| --- | --- | --- |
| `INPUT_DIR` | `../vggt-omega/examples` | 图像文件夹 / 视频 / 场景文件夹（见 Step 01） |
| `GPU` | _(unset)_ | physical GPU id, e.g. `GPU=0` |
| `CONDA_ENV` | `doll` | conda env（torch>=2.3 预装；复用不重下 torch） |
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
| `VGGT_OUTPUT_DIR` | `$RESULTS_DIR/vggt` | Step 02 输出 |
| `VARIANT` | `1b_512` | checkpoint 变体 |
| `RESOLUTION` | `512` | 输入分辨率（`1b_256_text` 用 `256`） |
| `MODE` | `balanced` | `balanced` / `max_size` |
| `CONF_THRES` | `20` | scene.ply 深度置信度百分位（0-100） |
| `MAX_POINTS` | `2000000` | scene.ply 点数上限 |
| `VIDEO_FPS` | `1` | 视频输入抽帧 fps |

### Step 03 params
| var | default | note |
| --- | --- | --- |
| `SCENE_NAME` | _(auto)_ | 场景子夹名（空 = 自动检测 VGGT_OUTPUT_DIR 下第一个） |
| `SOURCE_DIR` | `$RESULTS_DIR/source` | COLMAP 输出 |
| `TARGET_POINTS` | `200000` | 体素降采样目标点数 |
| `ALIGN` | `1` | `1` = 居中场景到原点 |

### Step 04 params
| var | default | note |
| --- | --- | --- |
| `GAUSSIAN_DIR` | `$RESULTS_DIR/model_3dgs` | 高斯输出 |
| `ITERATIONS` | `30000` | 训练迭代数 |
| `RES` | _(unset)_ | `--resolution` 因子；不设 = 全分辨率 |
| `WHITE_BG` | `0` | `1` = 白底光栅化 |
| `SKIP_RENDER` | `0` | `1` = 跳过渲染 |
| `SKIP_METRICS` | `1` | `1` = 跳过 PSNR/SSIM/LPIPS |
| `TRAIN_EXTRA_ARGS` | _(empty)_ | 透传给 train.py 的额外参数 |

### Step 05 params
| var | default | note |
| --- | --- | --- |
| `DENOISER` | `none` | `diffbir` \| `swinir` \| `nafnet` \| `none`（可插拔，见 denoisers.py） |
| `NUM_NOVEL_VIEWS` | `10` | 插入多少个虚拟相机 |
| `ALPHA_THRESH` | `0.3` | 渲染 alpha 低于此值 = 稀疏区，需去噪 |
| `ADAIN_REF` | `nearest` | AdaIN 颜色参考：`nearest`（最近训练图） \| `mean`（全局均值色） |
| `ITERATION` | `30000` | 加载 04 的哪个 iteration 的 checkpoint |
| `GAUSSIAN_DIR` | `$RESULTS_DIR/model_3dgs` | 3DGS 模型目录（04 的输出） |
| `SOURCE_AUG_DIR` | `$RESULTS_DIR/source_aug` | 增强 COLMAP 输出 |

### Step 06 params (face enhance)
| var | default | note |
| --- | --- | --- |
| `HYPIR_WEIGHT` | `$HYPIR_DIR/experiments/beauty_ppr50k_20260721/checkpoint-1000/ema_state_dict.pth` | HYPIR LoRA checkpoint |
| `HYPIR_BASE_MODEL` | `$WEIGHTS_ROOT/HYPIR/sd2_base` | SD2 base model dir |
| `FACE_PADDING` | `0.2` | 人脸框放大比例 (0.2 = 20%) |
| `UPSCALE` | `1` | HYPIR upscale (1 = 不超分, 只增强) |
| `PATCH_SIZE` | `512` | HYPIR patch size |
| `STRIDE` | `256` | HYPIR stride |
| `SOURCE_FACE_DIR` | `$RESULTS_DIR/source_aug_face` | 输出目录 |

### Step 07 params (train on enhanced scene)
| var | default | note |
| --- | --- | --- |
| `SOURCE_AUG_DIR` | `$RESULTS_DIR/source_aug_face` | 增强场景（06 的输出; fallback: source_aug） |
| `GAUSSIAN_DENOISE_DIR` | `$RESULTS_DIR/model_3dgs_denoise` | 增强训练的模型输出 |
| `ITERATIONS` | `30000` | 训练迭代数 |
| `RES` | _(unset)_ | `--resolution` 因子；不设 = 全分辨率 |
| `WHITE_BG` | `0` | `1` = 白底光栅化 |
| `MODEL_PATH` | _(unset)_ | 续训：从哪个 model_path 加载 checkpoint |
| `LOADED_ITER` | _(unset)_ | 续训：加载第几轮的 checkpoint |
| `SKIP_RENDER` | `0` | `1` = 跳过渲染 |
| `SKIP_METRICS` | `1` | `1` = 跳过 PSNR/SSIM/LPIPS |
| `TRAIN_EXTRA_ARGS` | _(empty)_ | 透传给 train.py 的额外参数 |

## 可能遇到的问题

**1. `00` 报 CUDA 扩展编译失败（diff-gaussian-rasterization / simple-knn）**
三个根因：
- **gcc 太老**：系统 gcc 版本太低编不过 CUDA 12.x。00 会 `conda install gxx_linux-64=12 python=3.10`。手动：
  ```bash
  conda install -y -c conda-forge gxx_linux-64=12 python=3.10
  python -c "import platform; print(platform.python_implementation())"  # 必须 CPython
  INSTALL_DEPS=1 bash vggt_human/00_setup_env.sh
  ```
- **nvcc 找不到 / CUDA 版本不匹配**：确认 `/usr/local/cuda/bin/nvcc` 存在。若 torch 是 cu118 但系统只有 cuda-12.x（或反过来），00 会自动找匹配版本。手动：
  ```bash
  export CUDA_HOME=/usr/local/cuda-12.4  # ⚠️ 不是 /usr/local/cuda
  INSTALL_DEPS=1 bash vggt_human/00_setup_env.sh
  ```
- **simple-knn clone 失败（gitlab.inria.fr 被封）**：00 自动 fallback 到 GitHub 镜像。手动：
  ```bash
  cd $GS_DIR/submodules && rm -rf simple-knn
  git clone https://github.com/yindaheng98/simple-knn.git simple-knn
  INSTALL_DEPS=1 bash vggt_human/00_setup_env.sh
  ```

**2. `00` 报 GLM 缺失（`glm/glm.hpp: No such file`）**
diff-gaussian-rasterization 依赖 GLM 头文件库。00 自动 clone。手动：
```bash
cd $GS_DIR && git clone https://github.com/g-truc/glm.git third_party/glm
INSTALL_DEPS=1 bash vggt_human/00_setup_env.sh
```

**3. `02` 报 checkpoint not found**
VGGT-Omega 权重是 gated。先通过 vggt-omega 下载：
```bash
GPU=0 VARIANT=1b_512 MODEL_DIR=../../model/VGGT-Omega bash vggt-omega/01_download_models.sh
```

**4. `02` 报 `torch.cuda.OutOfMemoryError`**
显存随帧数线性增长。降压：`RESOLUTION=256`、`MODE=max_size`，或喂更少帧。`run_batch.py` 会捕获 OOM 并继续。

**5. `03` 报 `predictions.npz not found`**
确认 step 02 已跑完，npz 在 `$RESULTS_DIR/vggt/<scene>/predictions.npz`。若 `SCENE_NAME` 自动检测错误，手动指定：
```bash
GPU=0 SCENE_NAME=image RESULTS_DIR=../../output/vggt_human_results bash vggt_human/03_npz_to_colmap.sh
```

**6. `04` 报 `import diff_gaussian_rasterization` 失败**
CUDA 扩展没编成。重跑：
```bash
INSTALL_DEPS=1 bash vggt_human/00_setup_env.sh
```

**7. 跑 `.sh` 报 `syntax error near unexpected token '('`**
CRLF 行尾污染。`find vggt_human -name '*.sh' -exec sed -i 's/\r$//' {} +`（`.gitattributes` 强制 LF）。

**8. `05` 报 DiffBIR / SwinIR 仓库或权重未找到**
首次用去噪模型需 clone 仓库 + 下权重：
```bash
INSTALL_DENOISER=1 INSTALL_DEPS=1 bash vggt_human/00_setup_env.sh
```
或只装某一个：确认 `DIFFBIR_DIR` / `SWINIR_DIR` 指向已 clone 的仓库，`DIFFBIR_CKPT` / `SWINIR_CKPT` 指向已下载的权重。用 `DENOISER=none` 可跳过去噪（仅渲染 + AdaIN，虚拟相机仍加入训练）。

**9. `05` 渲染报 `Cannot import name 'Camera'` / 3DGS API 变化**
3DGS 仓库的 `Camera` 类 API 可能因版本不同。`render_novel.py` 用标准 API（`Camera(colmap_id, R, T, FoVx, FoVy, image, ...)`），若报错检查 `$GS_DIR/scene/cameras.py` 的构造函数签名是否匹配。

**10. `07` 想从 04 续训但找不到 checkpoint**
3DGS 默认不保存 `.pth` checkpoint（只存 PLY）。续训需在 04 加 `CHECKPOINT_ITERATIONS=30000`（透传 `--checkpoint_iterations 30000`），然后：
```bash
GPU=0 MODEL_PATH=../../output/vggt_human_results/model_3dgs LOADED_ITER=30000 \
  RESULTS_DIR=../../output/vggt_human_results bash vggt_human/07_train_denoise.sh
```
不续训则从头训（增强场景有更多相机，结果通常更好）。

**11. `01/06` 报 `mediapipe not installed` 或 `HYPIR code not found`**
`vggt_human` env 未建好或缺少 HYPIR 依赖。运行：
```bash
INSTALL_DEPS=1 bash vggt_human/00_setup_env.sh   # 从 doll 克隆 + 装 HYPIR 依赖 + mediapipe + clone HYPIR + 下 SD2 base
GPU=0 bash vggt_human/06_face_enhance.sh
```

**12. `01/06` 人脸融合边缘有硬边**
调大 `FACE_PADDING`（默认 0.2 → 0.3）让裁剪区域更大，渐变 mask 覆盖更广。或检查 `create_feather_mask` 的衰减函数（二次衰减，可改为余弦衰减更平滑）。

> 通用：`proxy.env`（代理凭证 + `HF_TOKEN`）在仓内 gitignored，不入库。切勿把凭证写进脚本。

## 目录布局
```
<code-dir>/
├── media_code/                     # 本仓
│   ├── proxy.env                   # 代理 + HF_TOKEN, gitignored
│   └── vggt_human/                  # ← 本目录（编排脚本）
│       ├── _env.sh                 # 共享: 代理 + CA + conda + GPU + paths
│       ├── 00_setup_env.sh        # clone 3DGS 仓 + 装依赖 + 编 CUDA 扩展
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
│       └── face_enhance.py         # MediaPipe + HYPIR + 渐变融合 (step 01/06)
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
    ├── input_face/                 # step 01: 前处理人脸增强后的原始图
    │   └── images/                 #   人脸增强图
    ├── vggt/<scene>/               # step 02: VGGT-Omega 推理
    │   ├── predictions.npz          #   原始输出
    │   ├── scene.ply                #   点云 (供检查)
    │   └── frames/                  #   训练图
    ├── source/                      # step 03: COLMAP 场景
    │   ├── images/                  #   训练图 (复制)
    │   └── sparse/0/
    │       ├── cameras.txt          #   PINHOLE 内参
    │       ├── images.txt           #   w2c 外参
    │       └── points3D.txt          #   ~200k 初始点
    └── model_3dgs/                  # step 04: 3DGS 高斯
        ├── point_cloud/iteration_30000/point_cloud.ply   # 最终高斯
        └── train/ours_30000/        # 渲染 (重建 vs GT)
            ├── renders/*.png
            └── gt/*.png
    ├── novel_renders/               # step 05 stage 1: 渲染的新视角
    ├── novel_alpha/                #   覆盖度图 (低 alpha = 稀疏区)
    ├── novel_poses.json            #   虚拟相机参数
    ├── source_aug/                 # step 05 stage 2: 增强 COLMAP 场景
    │   ├── images/                 #   原图 + 去噪虚拟视角图
    │   └── sparse/0/               #   原相机 + 虚拟相机
    ├── source_aug_face/            # step 06: 人脸增强后的场景
    │   ├── images/                 #   原图 + 去噪图 (人脸已 HYPIR 增强 + 渐变融合)
    │   └── sparse/0/               #   COLMAP 原样复制
    └── model_3dgs_denoise/         # step 07: 增强训练后的高斯
        └── point_cloud/iteration_30000/point_cloud.ply
```

## Notes
- Pipeline: VGGT-Omega（前馈位姿+深度）→ COLMAP（格式转换）→ 3DGS（优化训练）。前馈给初始化，优化给质量。
- **去噪增强（04，可选）**：3DGS 在稀疏视角区域有伪影 → 渲染新视角 → 去噪（DiffBIR/SwinIR 可切换）→ AdaIN 颜色校正 → 虚拟相机加入训练。`DENOISER=none` 关闭去噪。加新去噪模型：在 `denoisers.py` 写一个函数 + 注册到 `DENOISERS` 字典。
- **人脸增强（05，可选）**：MediaPipe 检测人脸 → HYPIR 美颜增强 → 二次衰减渐变 mask 无缝融合回原图。`HYPIR_WEIGHT` 指向 beauty_ppr50k 训练的 LoRA checkpoint。
- VGGT-Omega 的 `extrinsic` 是 w2c（OpenCV 约定），与 COLMAP 一致——无需 c2w→w2c 转换。`intrinsic` 是模型预测的实际内参——无需假设 fx=fy=max(W,H)。
- 自适应置信度过滤用 Otsu's method（最大化类间方差），比固定阈值更鲁棒。体素降采样每体素保留最高置信度点。
- 原版 3DGS 无 distractor filtering（不做微动过滤）。静态场景够用；有微动用 pdfgs_human（PDF-GS）。
- 无网格输出（3DGS 仓库无 `extract_mesh`）。要网格走 wan22_rotate step 05/05a/05b。
- `.gitattributes`（仓根）强制 LF。`proxy.env` gitignored。官方代码 & 权重遵循各自 license。

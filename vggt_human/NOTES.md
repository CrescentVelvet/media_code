# vggt_human — 原理详解与排错手册（NOTES）

> 本文件从 README.md 迁入：选型对比、流程原理详解、各步骤机制、常见报错修法。
> 运行命令与参数速查看 [README.md](README.md)；实验设计与结果看 [EXPERIMENTS.md](EXPERIMENTS.md)。
> **人脸链路 v2（FLAME + 表情驱动 + 重心绑定）的设计方案与决策记录看
> [DESIGN_face_pipeline.md](DESIGN_face_pipeline.md)**（2026-09-08 评审产出，尚未实现）。

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

## Pipeline（流程详解）

```
INPUT_DIR/                           (一组场景图像 / 视频)
    │
    ▼
[01a] 视频抽帧 (可选, 视频输入用) — cv2 按 VIDEO_FPS 抽帧 -> <OUTPUT_DIR>/image/
    ▼
[01] 前处理人脸增强 (vggt_human env) — MediaPipe → HYPIR → 渐变融合
    │  ├─ MediaPipe BlazeFace → 人脸框 → 放大 20% → 裁剪
    │  ├─ HYPIR (SD2Enhancer + LoRA beauty_ppr50k) 增强裁剪图 (upscale=1)
    │  └─ 二次衰减渐变 mask: 中心=增强, 边缘=原图 → 无缝融合
    ▼
$RESULTS_DIR/01_input_face/images/      (人脸增强后的原始图)
    │
    ▼
[02] VGGT-Omega 前馈推理 (doll env)  — 一次前向 → 所有视图位姿 + 深度图
    │  ├─ VGGTOmega(images) -> pose_enc + depth + depth_conf + images
    │  ├─ encoding_to_camera -> 外参 w2c (N,3,4) + 内参 (N,3,3)
    │  └─ unproject_depth -> 世界坐标点云 + 置信度过滤 -> scene.ply
    ▼
$RESULTS_DIR/02_vggt/<scene>/
    predictions.npz      # extrinsic(w2c) + intrinsic + world_points + depth_conf + images
    scene.ply             # 置信度过滤后的彩色点云 (供检查)
    frames/               # 喂给模型的图 (复制/抽帧)
    │
    ▼
[03] npz -> COLMAP 转换 (doll env) — 自适应过滤 + 降采样 + 坐标对齐
    │  ├─ 加载 predictions.npz: extrinsic(w2c), intrinsic, world_points, depth_conf
    │  ├─ 复制 frames/ -> 03_source/images/ (内参按原图尺寸缩放)
    │  ├─ 自适应置信度过滤: Otsu 阈值 (分离高/低置信度点)
    │  ├─ 体素降采样 -> ~200k 点 (每体素保留最高置信度点)
    │  ├─ 坐标对齐: 点云 + 相机平移居中到原点
    │  └─ 写 COLMAP 文本格式: cameras.txt + images.txt + points3D.txt
    ▼
$RESULTS_DIR/03_source/
    images/*.png                  # 训练图
    sparse/0/cameras.txt          # PINHOLE (VGGT-Omega 实际内参)
    sparse/0/images.txt           # w2c (qw qx qy qz tx ty tz)
    sparse/0/points3D.txt         # ~200k 初始点
    │
    ▼
[04] 3DGS 训练 (doll env) — 原版 gaussian-splatting
    │  ├─ train.py -s 03_source -m 04_model_3dgs --iterations 30000
    │  │   (L1+SSIM loss, adaptive density: split/clone/prune, 30k iter)
    │  └─ render.py -s 03_source -m 04_model_3dgs (渲染训练视角 vs GT)
    ▼
$RESULTS_DIR/04_model_3dgs/
    point_cloud/iteration_30000/point_cloud.ply    # 最终高斯
    train/ours_30000/renders/*.png                 # 重建渲染
    train/ours_30000/gt/*.png                      # GT
    │
     ▼ (可选: 去噪增强)
[05] 渲染新视角 → 去噪 → AdaIN → 增强COLMAP (doll env)
    │  ├─ Stage 1 (render_novel.py): 加载3DGS → 轨迹找间隙 → 插入虚拟相机 → 渲染 (black+white bg for alpha)
    │  └─ Stage 2 (denoise_images.py): alpha<阈值=稀疏 → DENOISER去噪 → AdaIN颜色校正 → 写增强COLMAP
    ▼
$RESULTS_DIR/05_source_aug/
    images/*.png + novel_*.png     # 原图 + 去噪虚拟视角图
    sparse/0/{cameras,images,points3D}.txt  # 原相机 + 虚拟相机
    │
    ▼
[06] 后处理人脸增强 (vggt_human env) — MediaPipe 检测 → HYPIR 美颜 → 渐变融合
    │  ├─ MediaPipe BlazeFace → 人脸框 → 放大 20% → 裁剪
    │  ├─ HYPIR (SD2Enhancer + LoRA beauty_ppr50k) 增强裁剪图
    │  └─ 二次衰减渐变 mask: 中心=增强, 边缘=原图 → 无缝融合
    ▼
$RESULTS_DIR/06_source_aug_face/
    images/  # 原图 + 去噪图 (人脸区域已增强+融合)
    sparse/0/  # COLMAP 原样复制
    │
    ▼
[07] 3DGS 训练 (增强场景) — 原图 + 去噪虚拟相机 + 前后处理人脸增强 共同监督
    │  └─ train.py -s 06_source_aug_face -m 07_model_3dgs_denoise --iterations 30000
    ▼
$RESULTS_DIR/07_model_3dgs_denoise/
    point_cloud/iteration_30000/point_cloud.ply    # 增强训练后的高斯
```

### Step 00 — clone 仓 + 装依赖 + 编 CUDA 扩展 (`00_setup_env.sh`)

复用 `doll` conda env（torch>=2.3 预装）。clone 两个官方仓：VGGT-Omega（vggt-omega/00 已 clone，本步确认存在）+ gaussian-splatting（含 submodules）。编译两个 CUDA 扩展：
- **diff-gaussian-rasterization**：3DGS 光栅化器（dr_aa 分支，antialiasing）
- **simple-knn**：KNN 查询（gitlab.inria.fr 被 .gitmodules 替换为 GitHub 镜像）

需要 gcc 12（conda install gxx_linux-64=12 python=3.10）+ CUDA toolkit（nvcc）。00 自动检测 CUDA 版本与 torch 匹配，不匹配时自动找 `cuda-12.x`。

### Step 01a — 视频 → 图像夹 (`01a_video_to_frames.sh` → `video_to_frames.py`)

视频输入预处理：将单个视频文件（`.mp4/.mov/.avi/.mkv`）按 `VIDEO_FPS`（默认 2）抽帧成 `<OUTPUT_DIR>/image/`（`000000.png`、`000001.png`、…）。输出结构与 test_task 输入一致，01 的 `INPUT_DIR` 指向 `<OUTPUT_DIR>` 即可，`face_enhance.py` 自动检测 `image/` 子夹。所以视频输入走 `01a → 01 → 02 → 03 → 04` 全链路，原流程一行不改。用 cv2 抽帧（与 `run_batch.py` 的 `extract_frames` 同逻辑）。

### Step 01 — 前处理人脸增强 (`01_face_enhance.sh` → `face_enhance.py`)

对原始输入图（`INPUT_DIR`）做前处理人脸增强。与 Step 06（后处理）调用**同一个 `face_enhance.py`**，区别是输入：01 对原始图（无 COLMAP 场景），06 对增强 COLMAP 场景中的图。`face_enhance.py` 自动适配输入结构（`images/` 子夹 / `image/` 子夹 / 散图夹）。输出到 `01_input_face/images/`，作为 Step 02 的输入。

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

**Stage 1 — `render_novel.py`**：从 step 04 的 checkpoint 加载 3DGS 高斯 → 解析 COLMAP 相机轨迹 → 按绕场景中心的方位角排序 → 找最大间隙 → 插入 `NUM_NOVEL_VIEWS` 个中间视角（位置线性插值 + 旋转 SLERP）→ 渲染每个新视角（黑底 + 白底两次渲染算 alpha）→ 保存 PNG + `05_novel_poses.json`。

**Stage 2 — `denoise_images.py`**：逐视角检查 alpha → `avg_alpha < ALPHA_THRESH` 的 = 稀疏区（3DGS 有伪影）→ `DENOISER` 去噪（DiffBIR / SwinIR / none 可切换）→ AdaIN 颜色校正（去噪图的均值/标准差对齐到最近训练图）→ 写增强 COLMAP 场景（原图 + 去噪图，原相机 + 虚拟相机）。

> **去噪模型可插拔**：`denoisers.py` 用 registry 模式，每个去噪器是一个函数 `(image, device) -> image`。加新模型只需写一个函数 + 注册到 `DENOISERS` 字典。`DENOISER=none` 跳过去噪（仅渲染 + AdaIN）。首次用 DiffBIR/SwinIR 需 `INSTALL_DENOISER=1` 让 00 clone 仓库 + 下权重。

### Step 06 — 后处理人脸增强 (`06_face_enhance.sh` → `face_enhance.py`)

**用 `vggt_human` env**（有 diffusers/transformers/peft + mediapipe）。对 `05_source_aug/images/`（或 `03_source/images/`）中的每张图：

1. **MediaPipe BlazeFace** 检测人脸框 → 放大 `FACE_PADDING`（默认 20%）后裁剪。
2. **HYPIR 增强**：裁剪图喂给 `SD2Enhancer`（加载 `HYPIR_WEIGHT` 指向的 beauty_ppr50k LoRA checkpoint），`upscale=1`（只增强不超分）。
3. **渐变融合**：二次衰减 mask（中心=1, 边缘=0）把增强结果无缝融合回原图——中心区域完全用 HYPIR 结果，边缘平滑过渡到原图，避免硬边。
4. COLMAP `sparse/` 原样复制（只增强图像，不改相机参数）。

> ⚠️ `vggt_human` env 需先通过 `INSTALL_DEPS=1 bash vggt_human/00_setup_env.sh` 建好（从 doll 克隆 + 装 HYPIR 依赖 + mediapipe + clone HYPIR 仓 + 下 SD2 base model）。`HYPIR_WEIGHT` 默认指向 `beauty_ppr50k_20260721/checkpoint-1000/ema_state_dict.pth`，可改。

### Step 07 — 增强场景训练 (`07_train_denoise.sh`)

在 `doll` env 里用增强 COLMAP 场景训练 3DGS（`-s $SOURCE_AUG_DIR -m $GAUSSIAN_DENOISE_DIR`）。原图提供 GT 监督，去噪虚拟相机提供稀疏区域的额外监督，前后处理人脸增强提供更好的面部质量。默认从头训；可选从 04 的 checkpoint 续训（需 04 加 `--checkpoint_iterations`）。

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
确认 step 02 已跑完，npz 在 `$RESULTS_DIR/02_vggt/<scene>/predictions.npz`。若 `SCENE_NAME` 自动检测错误，手动指定：
```bash
GPU=0 SCENE_NAME=image RESULTS_DIR=../../output/vggt_human_results bash vggt_human/03_npz_to_colmap.sh
```

**6. `04` 报 `import diff_gaussian_rasterization` 失败**
CUDA 扩展没编成。重跑：
```bash
INSTALL_DEPS=1 bash vggt_human/00_setup_env.sh
```

**6b. `04` 报 `Camera.__init__() got an unexpected keyword argument` / `create_from_pcd()` 参数不匹配 / `training_setup()` 属性缺失**
gaussian-splatting 的 main 分支更新了 API 签名，`train_pose.py` 已适配最新版。如果服务器上 clone 的是旧版（之前能跑、现在重 clone 后报错），说明 main 分支更新了。改动包括：`Camera.__init__` 加了 `resolution/depth_params/invdepthmap/uid`；`create_from_pcd` 加了 `cam_infos`；`training_setup` 要完整 config 对象；`densify_and_prune` 加了 `max_screen_size/radii`。重新 clone 最新版 + 用最新 `train_pose.py` 即可：
```bash
rm -rf $GS_DIR
INSTALL_DEPS=1 bash vggt_human/00_setup_env.sh   # 服务器
# 或 WSL: rm -rf ~/repos/gaussian-splatting && bash vggt_human/00a_setup_env.sh
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
GPU=0 MODEL_PATH=../../output/vggt_human_results/04_model_3dgs LOADED_ITER=30000 \
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

## 可能遇到的问题（WSL 专属）

**1. `conda: command not found`（运行脚本时）**
00a 末尾执行 `conda init bash`。如果还没 `source ~/.bashrc`：
```bash
source ~/.bashrc
# 或每次手动 source：
source ~/miniconda3/etc/profile.d/conda.sh
```
`_env.sh` 已加 fallback：conda 不在 PATH 时自动找 `~/miniconda3`。

**2. `pip install` 很慢 / 超时**
默认 PyPI 在国内慢。用清华镜像：
```bash
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple <package>
```

**3. `huggingface.co` 连不上 / 下载超时**
用 HF 镜像。在 `proxy.env` 里加：
```bash
export HF_ENDPOINT="https://hf-mirror.com"
```
`huggingface_hub` 库会自动用镜像。VGGT-Omega 权重 4.6GB 约 3 分钟下完。

**4. CUDA 扩展编译失败（`nvcc: command not found`）**
00a 通过 conda 装 cuda-nvcc。如果失败：
```bash
conda activate vggt_human
which nvcc           # 检查
# 手动装：
conda install -y -c nvidia/label/cuda-12.1.1 cuda-nvcc cuda-cudart-dev cuda-cccl
```

**5. `torch.cuda.OutOfMemoryError`（step 02）**
RTX 3090 有 24GB，但 VGGT-Omega 1B 对帧数敏感。降压：
```bash
RESOLUTION=256 MODE=max_size GPU=0 ... bash vggt_human/02_run_inference.sh
```

**6. 编译极慢（超过 30 分钟）**
确认仓库在 Linux 文件系统（`~/repos/`）而非 `/mnt/c/` 或 `/mnt/d/`。drvfs 上编译慢 5-10x 且 symlink 可能坏。

**7. 想切 cu124**
```bash
CUDA_TOOLKIT_LABEL=nvidia/label/cuda-12.4.0 \
TORCH_INDEX_URL=https://download.pytorch.org/whl/cu124 \
bash vggt_human/00a_setup_env.sh
```

**8. WSL vhdx 空间不足**
跑完 pipeline 后执行 step 08 把结果搬到 D 盘：
```bash
bash vggt_human/08_move_output.sh
```
如果 vhdx 本身太大（即使删了文件也不缩小），在 Windows PowerShell 里压缩：
```powershell
wsl --shutdown
diskpart
# select vdisk file="C:\WSL\Ubuntu2404\ext4.vhdx"
# compact vdisk
```

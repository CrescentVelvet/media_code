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

**9. 99c 的 thetaBuffer/phiBuffer 曾不生效（2026-09-10 实测）→ 2026-09-15 定位并修复**

> **2026-09-15 结论修正（取代下面的原始记录）**
>
> 真因不是「六项白名单丢掉所有区间字段」，而是 **cgltf 封装侧原本不写
> `thetaBuffer` / `phiBuffer` 这两项**。99e 从旧 MP4 解回的 view_limits.json 实测
> 有 14 项，`minPhi/maxPhi`、`minTheta/maxTheta`、`minRadius/maxRadius` **都在**，
> 字段名与 99c 的输入 json 完全一致 → 区间字段一直是写进去的。
>
> 用户已**修改 cgltf 源码补上这两项的写入**：**新封装的 MP4 会带
> `thetaBuffer` / `phiBuffer`，旧 MP4 不带**（要看效果必须重新封装）。
>
> 由此：①「回弹锁死」这条路**重新打开**，不再是死路；②`RADIUS_RANGE_SCALE` /
> `PHI_MARGIN` / `THETA_MARGIN` 当时实测无效**另有原因**，待复查（区间既已在文件里，
> 就该查是不是被上游 encode 覆盖、写入位置不对，或测试样本不对）。
>
> **一次跑通三件事的验证法**：cgltf 补丁后的链路跑 99c（buffer=0、区间按 cfg 收窄）
> → 99e 解回 → 比对 view_limits.json：① `thetaBuffer/phiBuffer` 是否落值；
> ② `minPhi/maxPhi`/`minTheta/maxTheta` 是否等于 99c 打印的「新」值；
> ③ `minRadius/maxRadius` 是否向 init 半径内缩。

原始记录（2026-09-10，其中「丢弃字段清单」的表述已被本次修正取代，保留作历史）：
gltf_packer 往 UWA_viewing_parameters 里**只写六项白名单字段**：
`longitude、latitude、distance、gravity、target、boundingbox`，view_limits.json
里的其他字段（thetaBuffer/phiBuffer、minRadius/maxRadius、minPhi/maxPhi、
minTheta/maxTheta 等）**静默丢弃**。

即：UWA 格式的视角约束实际只有「水平角 / 垂直角 / 距离 / 重力方向 / 目标点 /
包围盒」六个自由度。99c 当前对 view_limits 的所有收窄改动（buffer 锁死回弹、
radius 区间内缩、Phi/Theta 区间收窄）都不会进 MP4——**只有 init_camera 的改动
（初始位置/朝向/FOV）能生效**。

受影响的 99c 参数（写了但不生效，等播放器/工具链侧开口子再启用）：
`THETA_BUFFER / PHI_BUFFER / RADIUS_RANGE_SCALE`（以及收窄本体 PHI_MARGIN /
THETA_MARGIN 系列同样不进成品）。

可能的出路（均未验证）：
- 反编译/分析图库 App 或 UWA 播放器，找 buffer 字段的真实消费端（用户在图库侧
  找到过 polarBuffer/azimuthBuffer 接口，说明播放器代码里存在这两个 key 的
  读取逻辑，只是 gltf_packer 不写）；
- 绕过 gltf_packer，直接改 GLB 二进制里的 UWA_viewing_parameters extension
  （JSONX pack 无校验，事后注入理论上可行）；
- 询问图库/UWA 工具链提供方是否有新版 gltf_packer 支持完整字段。

> **2026-09-15 补充**：本条的「反向拿不到三件套」结论已被改动推翻——用户直接修改
> **cgltf 源码增加 json 输出**后，99e 解封装能拿到三件套了，实际文件名是
> `cameras.json` / `init_cam.json` / `view_limit.json`（注意与官方
> `camera.json` / `init_camera.json` / `view_limits.json` 不同名，99e 已做候选名容错）。
>
> **2026-09-15 后续判读**：解出的 view_limits.json 是 **14 项**
> （`gravityCoordinate` / `min|maxPhi` / `min|maxTheta` / `min|maxRadius` /
> `min|max X Y Z` / `target`）——**不是只 dump 六项白名单**，区间字段确实在；
> 但 `thetaBuffer` / `phiBuffer` 不在（旧 MP4 未写，见本条正文修正）。
> 反算：init 相机 pos−target 半径 1.9089、极角(自 −Y) 58.900 vs `minTheta` 58.912，
> 疑似以初始相机为锚；区间跨度随场景变化（本样本 Theta 跨度 7.243°，历史半跨度
> 7.19 / 6.16 / 2.475），**不是固定余量推导**。
>
> **仍待定论**：GLB 的 `UWA_viewing_parameters` 原始块里到底是标量还是区间——
> 这决定 99c 的区间收窄能否生效。验证法见本条「一次跑通三件事」。

> **2026-09-15 判读（用户实测 view_limits.json / init_cam.json 内容）**：
> - `thetaBuffer` / `phiBuffer` **确认不在**解出的 view_limits.json 里 → 白名单结论对
>   这两个字段成立，「回弹锁死」这条路死掉（回弹是播放器内置行为，不走数据通路）。
> - 但解出的 view_limits.json **有** `minPhi/maxPhi/minTheta/maxTheta/minRadius/maxRadius`
>   （外加 `gravityCoordinate` / `target` / `min|max X Y Z`），共 14 项，字段名与 99c 的
>   输入 json **完全一致**。即白名单六项的映射是：`longitude→Phi 区间`、`latitude→Theta
>   区间`、`distance→Radius 区间`、`gravity→gravityCoordinate`、`target→target`、
>   `boundingbox→min|max XYZ`。
> - 实测样本 `temp_video_1789371030897`：Phi[127.510, 238.405]、Theta[58.912, 66.155]、
>   R[0.4318, 2.1590]；由 init_cam 反算 pos−target 半径 1.9089、极角(自 −Y 轴) 58.900，
>   与 minTheta 58.912 差 0.012° → **疑似以 init 相机为锚**，但精度不足以定论。
> - 区间跨度随内容变化（本样本 Theta 跨度 7.243°；此前多样本 Theta 半跨度
>   7.19 / 6.16 / 2.475）→ **不是固定余量推导**，跨度是可变量。
> - **仍待定论的一步**：直接 dump GLB 的 JSON chunk，看 `UWA_viewing_parameters` 原始块
>   里是 3 个标量还是 6 个区间字段。只有标量 → 区间是解封装时推导的，99c 收窄无用；
>   若原始块里就带区间 → **白名单结论需修正**，99c 的 RADIUS/THETA 有救，要回头查
>   为何实测无效（可能 99c 写的位置不对，或测试用了旧 mp4）。

**10. 99e 解封装报 `FileNotFoundError: /dev/shm/.../image0.bmp`（2026-09-15）**

真因不在 PLY 也不在 decode 逻辑，而是 **`astcenc` 缺可执行权限**，且错误被吞掉：

1. `decode.py` 用 shell 脚本调用 `src/xencode/tools/astcenc-sse2` 把 `.astc` 转 `.bmp`；
   该文件权限是 `-rw-rw-r--`（**没有 +x**），bash 直接 `Permission denied`
2. 这个调用的报错被**重定向到临时目录的日志**（`decode.py` 内部），而临时目录在
   `/dev/shm` 下、进程退出即删除 —— **错误被彻底吞掉**
3. astcenc 没产出 `image0.bmp`，于是 PIL 打开时报 `FileNotFoundError: image0.bmp`，
   表象离真因极远，几乎无法从报错反推

修复：`chmod +x <tool_dir>/src/xencode/tools/astcenc-sse2`（该工具目录里可能同时有
`astcenc-avx2` 等变体，decode.py 指定用的是 sse2）。

99e 已做前置拦截：启动时扫描 `TOOL_DIR` 下所有 `astcenc*` 检查 `os.X_OK`，缺执行位
直接报错并打印可复制的 `chmod +x` 命令；`ASTCENC_AUTOFIX=1` 可自动补。
`ASTCENC_PATH` 可显式指定路径（工具链版本不同时兜底）。

教训：**凡是「报错被重定向进临时目录」的子进程，都要在上游做前置检查**——这类静默
失败的表象（PIL 打不开文件）与真因（少一个执行位）之间没有任何线索链。

## ⚠️ 本机 Python 3.13 / 服务器 Python 3.10 的语法陷阱（2026-09-23 踩坑）

`vggt_human` env 是 **Python 3.10.20**（本机 WorkBuddy 内置的是 3.13）。3.12+ 才合法的写法
会在本机一路通过、到服务器 **SyntaxError 挂在 import 阶段**，报错行号还指向 f-string 内部：

```python
label += f" · 距锚点 {math.dist(原点, ctx["anchor"]):.2f} m"
                                       ^^^^^^^^^^^  3.10 报 f-string: unmatched '['
```

规则（已在 3.10.20 上逐条实测，不是照文档猜的）：

| 写法 | 3.10 |
|---|---|
| `f"{d["k"]}"`（单引号外层 + 内层同类引号） | ❌ SyntaxError |
| `f'{d['k']}'`（外层单引号同理） | ❌ SyntaxError |
| 表达式段里出现反斜杠（`f"{s.replace('\n','')}"`） | ❌ SyntaxError |
| `f'''{d["k"]}'''`（三引号外层 + 内层双引号） | ✅ 合法 |
| `f"{d['k']}"`（内外引号类型不同） | ✅ 合法 |
| `f"{f'{x}'}"`（嵌套 f-string，内外类型不同） | ✅ 合法 |

**自查工具**：`python vggt_human/99h_check_py_syntax.py`（扫 `vggt_human/*.py`，命中退出码 1）。
它用 tokenize 判定（注释与普通字符串天然排除），需 Python ≥ 3.12 运行。
`ast.parse(src, feature_version=(3,10))` **查不出这一类**——PEG 解析器不再按旧规则限制 f-string，
三种 feature_version 实测全部通过，所以别指望它兜底。
最强验证仍是用**真 3.10 解释器**跑一遍：WSL 里 `~/miniconda3/envs/vggt_human/bin/python <脚本>`。

## Remy（鸿蒙 3D 采集包）transforms.json 格式速查（2026-09-22 实测）

采集端 = 华为 / KIRI 的 **Remy**（HarmonyOS 独家的 3D 空间记忆 App，`pcd.ply` 头部有
`comment Created in Remy`）。每个采集 ID 目录 = `transforms.json` + `image/*.heic` + `pcd.ply`。

**顶层**：`version=1` / `platform=Harmony` / `platform_version`（`phone/HUAWEI/HUAWEI/地区码/
系统/机型代号/机型代号/API/版本/渠道`）/ `capture_mode`（模式编号，语义未核实）/ `camera_model=OPENCV`
/ `anchor_point`（环绕轴心）/ `ply_file_path`（相对本 json）/ `frames[]`。

**frames[] 每帧**：`w,h,cx,cy,fl_x,fl_y,k1,k2,k3,p1,p2,file_path,transform_matrix`。
`file_path` 的文件名是 **自开机的纳秒时间戳**（15 位），帧间隔 ≈ 0.107 s（155 帧 / 16.43 s ≈ 9.37 fps）。

**四条容易踩的坑（都已实测核对）**：

1. **`transform_matrix` 是 camera-to-world**：第 4 列 = 相机光心（米），不是 w2c 的 t。
   判据：按 c2w 解释时各帧光心到 `anchor_point` 的水平距离恒为 0.81–1.08 m（等距环绕），
   按 w2c 解释（C = −Rᵀt）则变成 7.2 m 且朝向角余弦只有 −0.38。
2. **相机前向 = −Z（OpenGL/Blender 约定），不是 OpenCV 的 +Z**。`camera_model:"OPENCV"`
   只声明畸变参数集（k1,k2,k3,p1,p2 的 Brown-Conrady 模型），**不决定外参轴向**。
   判据：取 look = −R[:,2]，与 (anchor − C) 的夹角余弦均值 0.9926；若按 +Z 为前向则为 −0.9926
   （等于所有相机都背对目标，不成立）。
3. **世界系 +Y 向上（重力对齐）**，相机 +Y 轴在世界 Y 的分量均值 0.889（min 0.805 / max 0.984）。
   不是 nerfstudio 默认的 Z-up，走 nerfstudio 数据解析要显式核对 orientation。
4. **位姿来自 AR Engine VIO，自带米制尺度**，不要再跑 COLMAP SfM（会引入重投影误差且丢尺度）。
   同一份 json 里 155 帧内参完全一致（唯一组合数 = 1）→ 单一标定，非逐帧估计。

**常用量**：对角线 82.9°、等效焦距 ≈ 24.5 mm（35mm 制，`f_equiv = 43.2666 × fl_px / hypot(w,h)`）；
`image/` 全是 HEIC（需 `01b_heic_to_jpg.sh` 或 pillow-heif 解码）；`pcd.ply` 是
binary_little_endian、xyz float32 + rgb uint8（15 B/点），点云与相机同处一个米制世界系。

**可视化**：`99g_plot_capture_trajectory.py` 把每个 ID 画成一份自包含 HTML + 一份独立 `.svg`，
标题与文件名都用文件夹 ID。默认风格 `quad`＝四联对照图 2×2（左列原始采集轨迹、右列视角约束，
见下节「四联对照图」）；`combo`＝上「俯视图（XZ，轨迹按帧序做蓝→紫→玫红时间渐变）」
+ 下「等轴测（含 Y 高度）」上下排列的双联图，两块绘图区统一 800×450 预留框、图例都铺在各自
绘图区下方占满整幅宽度，统计卡片竖排在右侧；另有 `minimal` / `darkspace` / `fov` / `iso` 单图风格，
以及视角约束范围的 `lim_*` / `vlimit` 系列（见下节）。

两条取景/采样规则（2026-09-22 补，都为解决「浅弧样例只看得见一条大轨迹」）：

- **俯视图取景 = 相机范围 ∪ 锚点±均值半径**。画了均值半径环就得让环完整入框；相机只沿锚点
  一侧走一小段弧时，只按相机范围取景会把锚点和环一起裁到框外。
- **点云先按视野过滤、再限流到 9000 点**（`cloud_in_view` / `cloud_screen_for`）。若只做全局
  抽稀，视野被放大时落在视野内的点只剩千分之几，点云会稀到看不见。等轴测的视野规则见
  `iso_extent`（只按相机活动范围，大场景点云不参与取景，否则轨迹被压成角落一小团）。
- **世界原点 (0,0,0) 也画出来了**（`draw_world_origin`）：原点在取景内就直接画标记，在取景外
  就画到绘图区边缘、用箭头指向它并标「距锚点 X m」。不把原点并进取景范围是有意的——Remy 的
  AR 世界原点是会话起点，实测离场景 0.5–5.1 m（10 个样本只有 4 个落在取景内），并进去会把
  整个场景压成一小团。框内标签会在四个方位里挑一个不压锚点/起终点标签的位置（`label_box`）。

### 视角约束范围（view limit）的画法（2026-09-28 补）

**本质**：以 target 为心的一圈**球壳扇块** —— 方位角 `az ∈ [az_lo, az_hi]`、仰角
`el ∈ [el_lo, el_hi]`、距离 `r ∈ [r_lo, r_hi]`。换算成 UWA `view_limits` 就是
`Phi / Theta / Radius` 三个区间，换算约定以 `99c_repack_view_limits.py:165-186` 为准
（与官方 pack 脚本一致）：

```
loc  = cam − target
pitch = asin(loc.y / |loc|)        Theta = 90 − pitch     （自 +Y 轴量起）
yaw   = atan2(−loc.x, loc.z)       Phi   = yaw            （x 取了负号）
R     = |loc|
```

**为什么不照抄那版 C++ 移植代码**（四点都是方法论层面的错，不是精度差一点）：

1. 它把相机中心当未知数做最小二乘射线交汇（`compute_sight_center`）。**这个思路本身是对的**
   ——2026-09-28 的实测推翻了我最初「有 anchor_point 就不必估」的判断（见下节「球心取哪个
   点」）。问题在实现：它把每帧的射线参数 λᵢ 也当未知数、凑一个 (3n)×(3+n) 的大矩阵，
   等价但没必要 —— 对 λᵢ 取最优后代回，就是 3×3 的正规方程
   `Σ(I − dᵢdᵢᵀ)·m = Σ(I − dᵢdᵢᵀ)·cᵢ`，`sight_center()` 直接解这个（实测两种写法结果差 < 1e-9）。
2. 它的 `vulkan_to_webgl` / `webgl_to_vulkan` 是**同一个函数** `[x, −y, −z]`，而且只在对
   camera 与 target **同时**施加的情况下参与角度计算 —— 刚体变换（det=+1，绕 X 转 180°），
   角度结果完全相同，属死代码。
3. 半径区间用固定比例 `0.9×avg / 1.2×avg`，是拍脑袋常数。实测某样本 `view_limits.json`
   是 `R[0.432, 2.159]` 而 init 半径 1.909，按固定比例只会给出 `[1.72, 2.29]`，量级都不对。
4. 极角区间带「跨度 > 10° 才把 init 值并进来」的条件（逻辑不自洽）；方位角解缠只在
   `|az| > 0.8π` 时才补偿（步进稍大就崩）；绘图用 `cos(phi)` 从 +Y 量起，而极角是
   `arccos(−vy)` 从 −Y 量起 → **画出来的球壳纵向是反的**。

**本实现**：区间 = 各帧实测 min/max ± 显式余量（`VL_AZ_PAD` / `VL_EL_PAD` / `VL_R_PAD`，
默认 3° / 3° / 8%），所以图上的区间能直接拿去和 99e 解出的 `view_limits.json` 对照。
方位角解缠改成「按帧序每步增量归一到 (−180, 180] 后累加」，慢速小步长、跨 ±180° 分支切、
甚至绕两圈都正确；跨度 ≥ `VL_FULL_AZ`（默认 300°）按整圈处理。

**六种画法**（`STYLE=lim_band|lim_edges|lim_angle|lim_rings|lim_hull|lim_shell`）
用 `STYLE=vlimit` 一页看完。实测 10 个样本：az 跨度 171°–270°、el 跨度 8.6°–33.2°、
ρ 0.54–1.92 m，无一触发整圈。
`hull` 是**语义不同的对照组**（相机水平位置的凸包外扩，不假设环绕），不是同一件事的另一种画法。

### 球心取哪个点：anchor_point vs 最小二乘视线汇聚中心（2026-09-28 实测）

约束球心有两个候选，由 `VL_TARGET` 切换（默认 `sight`）：

| 候选 | 来源 |
|---|---|
| `sight`（**默认**） | `sight_center()`：最小化各帧视线到该点的**正交距离平方和** |
| `anchor` | `transforms.json` 的 `anchor_point` |

**用了哪个点，直接影响三类区间**，不是「几乎一样」：实测 10 个样本，两个点相距
0.07–0.52 m（均值 0.24），换成 `sight` 后 el 区间最大差 11.2°（均值 6.5°）、
az 跨度差 −38.9°–+17.9°（均值 15.1°）、ρ/R 区间差 0.1–0.45 m。

**判据与实测**（垂距 RMS = 各帧视线到该点的垂距均方根；夹角 = 视线方向与 `(点 − C)`
方向的夹角）：

| 指标 | `sight` | `anchor_point` |
|---|---|---|
| 垂距 RMS（均值，越小越贴合） | **0.081 m** | 0.205 m |
| 视线夹角（均值，越小越正对） | **3.46°** | 9.54° |
| 视线夹角（最差样本） | 5.72° | 16.27° |
| 正规矩阵条件数 cond | 1.9–2.8（良态） | — |

**10/10 个样本都是 `sight` 更贴合**，所以默认改成它。`anchor_point` 只是采集包自带的
标称锚点，并不严格是视线汇聚处（成因未确认：可能是会话锚点、也可能是拍摄中主体移动过）。
图上同时画出两个点（球心 = 实心十字，对照 = 空心叉 + 虚线 + Δ 距离），`VL_MARK_ALT=0` 可关。

**三个实现要点**：

1. **正交距离对视线方向的符号不敏感** —— 前向取 `[0,0,−1]` 还是 `[0,0,+1]` 对解出的点
   **毫无影响**（到一条直线取垂距，反向还是同一条直线）。所以那版移植代码里的 `dir_cam`
   符号之争是伪问题。
2. **真正要小心的是 c2w / w2c**：Remy 的 `transform_matrix` 已是 c2w，前向 = `−R[:,2]`。
   若误当成 w2c 又转置一次（等价于取 `R` 的第 3 行），残差从 0.205 m 劣化到 0.782 m
   ——**但不报错、不崩，仍然产出一个看起来合理的点**，是典型的静默失败。
3. **条件数可以用来判「解可信吗」**：`A = Σ(I − dᵢdᵢᵀ)` 的最小特征值小 = 视线方向分布太窄
   = 沿某方向的深度几乎无约束，此时残差再小也不能当准。本批样本 cond 1.9–2.8，良态。
   附带结论：静止帧（实测最小帧间步长 0.0004 m）的重复计权在这里**没有实际影响**
   ——按步长加权后残差 0.0820 vs 等权 0.0812，可以忽略，所以没引入加权。


### 右列贴身裁剪：区间两端各收 20%，不越 frame 0（VL_TRIM · 2026-09-28）

四联图**右列**画出来/标出来的 az/el 区间，在「实测 min/max ± 余量」之上两端各再收掉
20% 跨度：

```
lo' = min(lo + span·k, init)          k       = VL_TRIM（默认 0.2）
hi' = max(hi − span·k, init)          init    = frame 0 的实测角（az / el 各自）
                                      span    = hi − lo（含余量后的完整跨度）
```

两端都用 `init` 兜底。`init` 本来就在带宽内（`lo = min − pad ≤ el[0] ≤ hi`），
所以 `hi' ≥ init ≥ lo'` 恒成立，**区间不会翻转**。

**为什么要 init 兜底**：约束是给「用户在采集范围内游走」用的。收一点更贴身，但收过头会
穿帮 —— 用户一开场（frame 0）就落在约束外。实测 10 个样本：

- **el**：frame 0 的仰角**全部落在区间下端 0–28% 的位置**（10 个样本里 6 个恰等于实测最小值）。
  于是下限基本收不动：4 个样本走 `init` 兜底（10e3ec62 / 13a8ecad / 28e7da4d / 7fc5f4de），
  其余 6 个由 20% 裁剪决定。上限一律是 20% 裁剪决定（`init` 远低于 `hi`）。
- **az**：frame 0 落在区间 37%–58%（中部），**两侧都由 20% 裁剪决定**，`init` 不介入。

**为什么单独出一个 `vt` 对象、而不是直接在 `compute_view_limit` 里改主字段**：
四联图四块**共用取景**，而等轴测取景 `_limit_iso_extent()` 正是拿 az/el 的 8 个角点撑出来的。
主字段直接变窄 → 左下 ③ 等轴测跟着缩比例尺 → 左列就不再是「原始采集轨迹」原图了。约定：

| 对象 | 用途 |
|---|---|
| `vl`（原样） | 取景、② ④ 之外的其它风格、UWA 对照值 |
| `vt = trim_view_limit(vl)` | **仅**右列约束层（② `_lim_angle` / ④ `_lim_shell`）与图例数值行 |

**整圈不裁 az**：360° 收 20% 会变成 288°，语义从「整圈」滑向「扇块」，而 `full_az` 标志还
留着，图上自相矛盾。这种情况 az 保持原样，el 照裁。

实测 10 样本（含余量的完整区间 → 裁剪后，单位 °）：

| 样本 | az 完整 → 裁剪后 | el 完整 → 裁剪后 |
|---|---|---|
| 10e3ec62 | −36.6→204.2 → 11.6→156.1 | 15.6→26.8 → 17.8→24.6 |
| 13a8ecad | −204.7→−25.7 → −168.9→−61.5 | 7.4→40.1 → 10.4→33.6 |
| 28e7da4d | −228.9→19.6 → −179.2→−30.1 | 29.5→45.4 → 32.5→42.2 |
| 3fe06043 | −193.2→8.3 → −152.9→−32.0 | 13.9→27.3 → 16.6→24.6 |
| 47bae38d | −183.9→8.5 → −145.5→−30.0 | 15.1→26.1 → 17.3→23.9 |
| 67cda18d | −150.2→39.2 → −112.4→1.3 | 18.7→32.9 → 21.5→30.1 |
| 6e34ca22 | −184.3→5.2 → −146.4→−32.7 | 13.0→27.6 → 15.9→24.7 |
| 7fc5f4de | −200.1→5.7 → −158.9→−35.5 | 21.0→37.8 → 24.1→34.5 |
| 8d0d11d3 | −198.7→13.6 → −156.3→−28.9 | 8.1→19.7 → 10.4→17.4 |
| 9a7ffcda | −183.0→14.2 → −143.5→−25.2 | 19.1→32.3 → 21.7→29.7 |

**两个要留意的副作用**：

1. **裁剪后区间小于实测采集范围** —— 这跟本文档前面「余量给小的理由」一节的要求
   （约束不应小于采集范围，否则用户走到采过的位置反而被拦住）方向相反，是**按需求有意为之**：
   宁可约束更紧，也不放太大的自由度。要回到完整区间用 `VL_TRIM=0`。
2. **el 跨度掉到 8° 以下时球壳不再填色**（`_lim_shell` 的既有阈值：el 很窄时球壳本身就是
   薄带，再叠填充会在投影重叠处积成突兀的深色块）。10 个样本里 3 个落到阈值下：
   10e3ec62（6.8°）、47bae38d（6.6°）、8d0d11d3（7.0°）→ ④ 退化成细网格带。

### 四联对照图（quad · 2026-09-28 定稿，现为默认风格）

`STYLE=quad` 把「原始采集轨迹」与「视角约束」并排在同一张 2×2 图里，左列原始、右列约束：

| | 左列（原始采集轨迹） | 右列（视角约束） |
|---|---|---|
| 上行 | ① 俯视图（世界系 XZ，帧序渐变） | ② 俯视图 + 角度标注（`lim_angle`） |
| 下行 | ③ 等轴测（含世界 Y 高度） | ④ 等轴测 + 3D 球壳扇块（`lim_shell`） |

三条设计决定：

1. **四张图共用同一取景与比例尺**。这是这张图存在的意义 —— 左右逐点对照「约束落在轨迹的
   哪一段」。各自取景会让同一段轨迹在两张俯视图里大小不同，对照反而失真。取景 = 相机活动
   范围 ∪ 锚点±均值半径 ∪ 约束范围 ∪ 对照球心（俯视撑到 `ρ_hi` 且两个球心都入框；等轴测
   撑到球壳 8 个角点，即 `_limit_iso_extent`）。代价：左列比 `combo` 单看时小一圈 ——
   实测约束范围是相机范围的 1.2–1.9 倍（俯视）/ 1.06–1.26 倍（等轴测，10 个样本）。
2. **右列约束层改用青绿**（`QUAD_INK` / `QUAD_FILL`），不用 vlimit 的紫 `LIM_INK`。左列轨迹是
   按帧序的蓝→紫→玫红渐变，其中段正是紫色，紫虚线约束压上去会和轨迹糊成一片、分不清谁是谁。
   `vlimit` 系列风格保持原紫色不变。
3. **右列省掉三样**：均值半径环（与约束区间圈都是球心周围的参考圈，同时画既冗余又
   互相干扰）、ρ 的图内数值文字（该位置被 anchor_point / frame 0 / 世界原点三个底图标签
   占着，而面板下方数值行本来就写了 ρ 区间）、「张角」标注（`span_text=False`，那个位置
   常正撞轨迹起点的 `frame 0` 标签，而面板下方的「跨度」数值本来就有）。
4. **右列球心与 `anchor_point` 对照画在一起**：球心（青绿实心十字）+ `anchor_point`
   （中性深灰空心叉）+ 虚线 + `Δ` 距离值，图例区再给出两者的拟合质量对照（垂距 RMS /
   夹角 / 条件数）。见「球心取哪个点」。`VL_MARK_ALT=0` 可关掉对照点。

踩过的两个坑（都不是「数差一点」，是坐标约定接错）：

- **`_fit_view` 的 panel 参数是 `(x, y, w, h)`**，不是 `(x0, y0, x1, y1)`。第一版按后者传，
  两处调用点的 `w/h` 被当成 `x1/y1` → 长宽比算错（左列 840/570、右列 1712/570），同一份
  数据在右列被扩张成两倍比例尺、内容偏位。参数已改名 `panel_xywh` 并在 docstring 里加了 ⚠️。
  它与 `clip_rect` 的 `(x, y, w, h)` 同构 —— 同一份代码里两套像素框约定并存，最容易接错。
- **`clipPath` 的 id 必须带面板后缀**。整张 SVG 内 id 重复时 `url(#id)` 只解析到第一个，
  后面的面板会被第一个的裁剪框裁掉（单独看每张都正常，拼进一页才暴露）。

排版常量：单块绘图区 800×450（复用 combo 的排法）、画布 1752×1328、列间距 72、图例带 112、
行间距 56。轴指示右列用 `axis_hints(..., flip=True)` —— 右列右边距只剩画布边距 40px，
框外的 `+X` 标签会飘到画布外。

另：`_lim_shell` 的球壳填充按方位角切成 24 个四边形小片逐片填（整块多边形在 az 跨度大时会
自相交）。相邻片共享边，抗锯齿各画一半会留一道更浅的缝、叠出「扇骨」感 —— 每片补一道同色
细描边（`fill-opacity` / `stroke-opacity` 各 0.07）把缝填掉，看起来才是连续曲面。
约束层一律画在轨迹**下层**，所以 `lim_edges` 对原有轨迹观感的影响最小。

**三个踩过的坑（都已修）**：

- `draw_top_geometry` / `draw_iso_geometry` 的 `clip_rect` 契约是 **`(x, y, w, h)`**，不是
  `(x0, y0, x1, y1)`。按后者传时等轴测的轴三叉会按 `w=840/h=528` 算，直接画到画布外。
- `View` 是「等比例尺 + 居中」，`view.rect()` 返回的是**数据框**而不是预留框。拿它当面板
  会把甩到外弧之外的数值标注裁掉（第一版 `lim_angle` 的「张角 269.5°」就被切了）。
  新增 `_fit_view()` 按面板长宽比扩张较短的一维，让内容填满预留框。
- `vlimit` 总览把 6 张图内联进**同一个 HTML 文档**，`clipPath` 的 id 必须带 mode ——
  同一文档里重复 id 时 `url(#id)` 只解析到第一个，第 2~6 张会被第一张的裁剪框裁掉
  （单张看都正常，拼进总览才暴露）。

**加球心对照时新踩的三个坑（都已修）**：

- **两个球心标签按固定方位放必然重叠**。原打算「主点右上、对照点左下」拉开，但两点
  实测只相距 0.52 m，比例尺 110 px/m → 才 56 px，两个标签直接印成一团。现在四个候选
  方位里挑第一个不与「对方标记 / 对方标签 / Δ 文字」相交的（`label_box` + `boxes_hit`）。
  **Δ 文字也要进避让表** —— 它正好夹在两个标记中间，不避它就会被压成「Δ sight_center」。
- **对照点的颜色不能太浅**。最初用 `#94A3B8`，压在同为灰色的 `pcd.ply` 点云上直接消失
  （标记明明画了、SVG 里也在，肉眼就是看不见）。改 `#475569` 并给圆圈加背景填充。
- **图例带里插一行会把比例尺顶到脚注上**。拟合质量行插进 `_limit_foot` 后图例整体下移
  10px，右下角右对齐的长脚注就和比例尺叠了（「0.5 m」压着「Theta = 90 − pitch」）。
  脚注改成左对齐并缩短到 3 段，右下的位置留给比例尺。


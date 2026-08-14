# PDF-GS Human — 真实环绕拍摄序列 → 抗微动人体三维高斯重建

用 [PDF-GS (Progressive Distractor Filtering, CVPR 2026 Findings)](https://github.com/kangrnin/PDF-GS) 从一组**真实环绕拍摄的人物图像**重建三维高斯：逐 phase 用 DINOv3 特征把跨视图不一致的像素（呼吸 / 头发 / 衣物飘动等**真实微动**）识别为 distractor 并从 loss 里滤除，得到干净的静态人体重建。前置：SAM2 掩码把每张图的人物抠到白底（rembg 定位人物 → SAM2 bbox-prompted 抠干净边缘，多视图集），Pi3 前馈出位姿 + COLMAP 场景。

> 本目录与 [`wan22_rotate/`](../wan22_rotate/) 并列但**范式不同**：wan22_rotate 是"Wan2.2 生成旋转视频 → 3DGS 重建"（合成视频，吃 wan22 输出）；本目录是"**真实拍摄序列 → PDF-GS 抗微动重建**"（吃真实照片，不碰 Wan2.2）。两者不共用 env（pdfgs 用 torch 2.5.1+cu121，wan22_rotate 用 torch 2.6.0+cu124）。

## 为什么用真实拍摄序列（而不是 Wan2.2 旋转视频）

PDF-GS 的 distractor 模型有一个**核心假设**：干扰像素是**稀疏离群**，散布在大量静态一致的像素中。这个假设决定了输入该选什么：

1. **真实拍摄序列**：人体微动（呼吸、头发飘动、衣物晃动、重心微移）在多视图间是**真实物理不一致**，且是**稀疏、局部**的——身体主体和背景是静态的，只有少数区域动。这**正好满足 PDF-GS 的稀疏离群假设**，是其设计目标场景。每个 phase 算"上一 phase 渲染图 vs GT"的 DINOv3 cosine 相似度：静态身体各 phase 渲染收敛到 GT → clean_mask≈1 保留；微动区域永远对不齐 → 标 distractor → 从 L1+SSIM loss 排除 → 干净静态身体重建。重建出的也是**真实人体**（而非模型想象），这才是人体数字化的目标。

2. **Wan2.2 旋转视频**（wan22_rotate 的产物）：视频是扩散模型生成的**合成内容**，"微动"本质是视频扩散的**弥散性时间不一致**——每一帧都是模型"猜"的，没有真值 3D 锚点。这种不一致是**弥散的**（遍布全身细节），不满足稀疏离群假设；PDF-GS 的滤除可能误伤合理身体区域，或干脆无效。合成视频本身较干净（白底 + 居中 + 干净 360° 相机路径），干扰本就少，05/05a（2DGS/GOF）已经够用——PDF-GS 在那里价值有限。

**结论**：真实微动是稀疏 distractor → 用真实序列；合成视频的弥散不一致不是稀疏 distractor → 不用。所以本目录吃真实拍摄序列，不复用 wan22_rotate 的 Wan2.2 视频。

## 常用命令

> 假设已进入容器（脚本自动激活 `pdfgs` env）；`GPU=0` 按需换卡。首次跑前先做下方「首次准备」。
> **铁律：每条命令都必须显式写出输入路径、输出路径，不能全靠脚本里的默认值。** 用具体路径，不要用 `...` 占位。

```bash
# ── 分步 ──
# 1) 分割所有拍摄图 → 白底人物多视图集
#    两套等价方案, 输出布局相同 (segmented_frames/<rel>.png), step 02 都能直接吃:
#    a) 01a (推荐, 质量最好): 复用 wan22_rotate 的 ViTDet + SAM2 (sam-3d-body),
#       跑在 wan22_rotate conda env 里 — 不用单建分割环境。需要先做过 wan22_rotate 的首次准备。
GPU=0 INPUT_DIR=../Reconstruction/dataset/B003_Human_Data_w_pose/test_task_id_3a8b3cc746304f49b9e3275e36aa9374 \
  RESULTS_DIR=../../output/pdfgs_human_results \
  bash pdfgs_human/01a_segment_all.sh
#    b) 01 (自含于 pdfgs env): rembg 定位 → SAM2 bbox-prompted 抠边; 自动/rembg 兜底。
#       没配过 wan22 env、只想用 pdfgs env 时用它。
# GPU=0 INPUT_DIR=../Reconstruction/dataset/B003_Human_Data_w_pose/test_task_id_3a8b3cc746304f49b9e3275e36aa9374 \
#   RESULTS_DIR=../../output/pdfgs_human_results \
#   bash pdfgs_human/01_segment_all.sh

# 输出：<RESULTS_DIR>/segmented_frames/<rel>.png   (白底人物, 保留输入相对子路径)

# 2) Pi3 位姿估计 + COLMAP 导出 (喂上一步的白底图集)
GPU=0 \
  RESULTS_DIR=../../output/pdfgs_human_results \
  bash pdfgs_human/02_pi3_colmap.sh

# 输出：<RESULTS_DIR>/orbit/pi3/
#   source/images/*.png                 # COLMAP 训练图 (= 分割图 copy)
#   source/sparse/0/{cameras,images,points3D}.txt
#   predictions.npz / dense_cloud.ply / poses.json   # Pi3 原始输出 (可看)

# 3) PDF-GS 训练 (progressive distractor filtering) + 渲染
#    4 phase × 10000 iter, sim_thr 0.6/0.7/0.8 (逐 phase 收紧)
GPU=0 \
  RESULTS_DIR=../../output/pdfgs_human_results \
  bash pdfgs_human/03_train_pdfgs.sh

# 输出：<RESULTS_DIR>/orbit/model_pdfgs/
#   phase_1/ ... phase_4/                              # 每个 phase 的高斯
#     point_cloud/iteration_10000/point_cloud.ply     # 该 phase 末的高斯
#   cfg_args                                          # 训练参数 (3DGS 标准)
# phase_4 (final) 还有:
#   train/ours_10000/renders/*.png                    # 渲染图 (vs GT 看重建质量)
#   train/ours_10000/gt/*.png                         # 对应 GT
# ⚠️ v1 不出网格: PDF-GS 只含 train.py / render.py / metrics.py, 无 extract_mesh。
#    要网格走 wan22_rotate step 05/05a/05b (或在本高斯上加 TSDF 步骤, future)。

# 4) 渲染转盘/环绕展示视频 (环绕人物的新视角轨道 → mp4)
#    03 只渲输入视角的重建图; 这步生成一条绕人物转一圈的新相机路径, 出旋转展示 mp4。
#    相机中心/竖轴/半径从高斯点云自动估算 (PCA), FoV 取 COLMAP 内参。
GPU=0 \
  RESULTS_DIR=../../output/pdfgs_human_results \
  bash pdfgs_human/04_render_orbit.sh
# 输出：<RESULTS_DIR>/orbit/model_pdfgs/orbit_render/orbit_orbit.mp4  (+ frames/*.png)
# 构图不理想时调: ORBIT_RADIUS_MULT=3.0 (近) | 4.0 (远), UP_AXIS=y (强制竖轴), ORBIT_HEIGHT=0.1 (抬高)

# 可选) 跑指标 (PSNR/SSIM/LPIPS) — 默认跳过 (无 held-out test, PSNR 只是 train fit)
GPU=0 SKIP_METRICS=0 \
  RESULTS_DIR=../../output/pdfgs_human_results \
  bash pdfgs_human/03_train_pdfgs.sh
```

- 结果：白底多视图集 → `segmented_frames/*.png`；COLMAP 场景 → `orbit/pi3/source/`；高斯 → `orbit/model_pdfgs/phase_4/point_cloud/iteration_10000/point_cloud.ply`；重建渲染 → `orbit/model_pdfgs/phase_4/train/ours_10000/renders/*.png`；转盘展示视频 → `orbit/model_pdfgs/orbit_render/orbit_orbit.mp4`。

### 方案 B — 原始图像（不分割，无白底）

> 白底分割可致 3DGS 伪影：人物边缘放射状白色"翅膀"+ 全身模糊（平白底非真实 3D 几何，Gaussian 无法重建）。直接用原始图像（真实背景有正确多视图几何），PDF-GS 的 distractor filtering 照常过滤微动。
```bash
# 一键 (02 Pi3 → 03 训练 → 04 转盘), 跳过分割, WHITE_BG=0
GPU=0 INPUT_DIR=../Reconstruction/dataset/B003_Human_Data_w_pose/test_task_id_3a8b3cc746304f49b9e3275e36aa9374 \
  RESULTS_DIR=../../output/pdfgs_human_results \
  bash pdfgs_human/05_run_raw.sh
# 输出在 $RESULTS_DIR/orbit_raw/ (与分割流程 orbit/ 分开)
```
- 结果：COLMAP 场景 → `orbit_raw/pi3/source/`；高斯 → `orbit_raw/model_pdfgs/phase_4/point_cloud/iteration_10000/point_cloud.ply`；转盘视频 → `orbit_raw/model_pdfgs/orbit_render/orbit_raw_orbit.mp4`。

## 首次准备

本流程建一份独立的 `pdfgs` env（**CPython 3.10**，torch 2.5.1+cu121，按 PDF-GS 官方 environment.yml；与 wan22_rotate 的 torch 2.6.0+cu124 不兼容，故独立）。装：PDF-GS + 两个 CUDA 扩展（diff-gaussian-rasterization + simple-knn）、SAM2（分割用）、rembg（兜底）、Pi3、transformers/torchmetrics 等，并下 DINOv3（gated）+ Pi3 权重。

```bash
cd <your-code-dir>            # e.g. /data_3d/<uid>/code
git -c http.sslVerify=false clone https://github.com/CrescentVelvet/media_code.git
cd media_code && cp proxy.env.example proxy.env   # 填 http_proxy / https_proxy
# ⚠️ 确认 proxy.env 中 http_proxy / https_proxy 两行已取消注释并填好地址，
#    否则 pip 装依赖会报 "Network is unreachable"

# 建 pdfgs env + 装依赖 + 编 CUDA 扩展 + 下权重 (一次性)
# 公司代理做 TLS 拦截、代理根 CA 不在系统 bundle 里 → SSL 验证失败。
# SSL_VERIFY=false 一键解决: _env.sh unset 掉 CA bundle 环境变量 + PYTHONHTTPSVERIFY=0,
# 00 往 pdfgs env 的 site-packages 注入 sitecustomize.py (ssl._create_unverified_context),
# 让 conda/pip/git/hf/requests/urllib 全部跳过证书验证。不用再手动 unset 任何变量。
SSL_VERIFY=false INSTALL_DEPS=1 bash pdfgs_human/00_setup_env.sh
#
# DINOv3 有两种拿法 (二选一):
#   A) 本地放一份到 $MODEL_DIR/dinov3-vitl16-pretrain-lvd1689m/  (ViT-L/16, 免 token)
#      —— 00 会自动探测并 patch train.py 指向它, 跳过 gated 下载。推荐, 省事。
#   B) 用 PDF-GS 默认的 facebook/dinov3-vitb16-pretrain-lvd1689m (ViT-B/16, GATED):
#      先到 https://huggingface.co/facebook/dinov3-vitb16-pretrain-lvd1689m 点 "Request access",
#      通过后用 HF_TOKEN 跑 00:
#      SSL_VERIFY=false HF_TOKEN=hf_xxx INSTALL_DEPS=1 bash pdfgs_human/00_setup_env.sh
#      (有本地 vitl16 时可省掉 HF_TOKEN, 直接用上面那条命令)
```

需系统有 CUDA toolkit 12.x（`CUDA_HOME=/usr/local/cuda`，nvcc 可用；若 `/usr/local/cuda` 指向 11.8，00 会自动找 `cuda-12.x`）。

### 权重目录布局

```
$MODEL_DIR/                         # 默认 ../../model (code-dir 上一级, 各算法共享)
  wheels/                          # torch + nvidia 依赖 wheel 缓存 (00 首跑 pip download, 后续直接本地装, 免重下 ~2-3GB; 跨项目共享, 不同版本文件名不同不冲突)
  dinov3-vitl16-pretrain-lvd1689m/  # DINOv3 ViT-L/16 (本地放这, 00 自动用, 免 gated 下载; PDF-GS 默认是 vitb16, vitl16 架构兼容能跑)
  u2net/                           # rembg 的 u2net 模型 (00 自动 wget 下载 u2net.onnx + symlink 到 ~/.local/share/.u2net; SAM2 不可用时 step 01 兜底用)
  Pi3/
    model.safetensors               # Pi3 checkpoint (~1GB, 公开免 token)
  hf_home/                          # HuggingFace cache 根 (仅当走 B) gated vitb16 时用)
    hub/
      models--facebook--dinov3-vitb16-pretrain-lvd1689m/   # DINOv3 vitb16 (gated, 00 用 HF_TOKEN 下)
```

外部 clone 的官方代码（00 自动 clone，sibling of media_code）：
```
<code-dir>/
  media_code/pdfgs_human/            # 本目录 (编排脚本)
  PDF-GS/                            # 官方代码 + 子模块 (diff-gaussian-rasterization, simple-knn)
  sam2/                              # SAM2 官方代码 + checkpoints/sam2.1_hiera_large.pt
  Pi3/                               # Pi3 官方代码 (step 02 用)
```

---

以下为详细参考（流程原理 / 各步骤参数 / 排错 / 目录布局）。

## Pipeline（流程详解）

```
INPUT_DIR/image/               (环绕人物拍摄的多张真实图像)
    │
    ▼
[01] 人物分割 → 白底 (pdfgs env)  — 参考 wan22_rotate "person bbox → SAM2"
    │  ├─ 遍历所有图 (保留全部视图, 不只最佳正面)
    │  ├─ rembg 粗掩码 → 人物 bbox (轻量定位, 替 ViTDet)
    │  ├─ SAM2 image predictor + bbox prompt → 干净人物掩码 (无黑边)
    │  │   (兜底: SAM2 自动掩码 → rembg alpha)
    │  └─ 掩码应用: 人物像素保留, 背景 → 白 (255,255,255)
    ▼
$RESULTS_DIR/segmented_frames/<rel>.png   (白底人物多视图集)
    │
    ▼
[02] Pi3 位姿 + COLMAP 导出 (pdfgs env, 调 pi3_3dgs/pi3_recon.py, 不带 --no_colmap)
    │  ├─ Pi3 前向 (1 次推理 → 所有视角 c2w 位姿 + 稠密点云 + 置信度)
    │  └─ COLMAP 文本格式导出:
    │       cameras.txt   PINHOLE, fx=fy=max(W,H), cx=W/2, cy=H/2
    │       images.txt    c2w→w2c 四元数+平移
    │       points3D.txt  稠密点云 + RGB (体素下采样到 ≤100k, 3DGS 初始化)
    ▼
$RESULTS_DIR/orbit/pi3/source/{images, sparse/0/}   (COLMAP 场景)
    │
    ▼
[03] PDF-GS 训练 (progressive distractor filtering)  (pdfgs env)
    │  ├─ phase 1: 无过滤 (纯 3DGS 训练, 建初始高斯)
    │  ├─ phase 2..N: DINOv3FeatureExtractor 算
    │       cosine(上一 phase 渲染图特征, GT 图特征)
    │       → 低于 sim_thr 的像素 = distractor (微动)
    │       → 从 L1+SSIM loss 里 mask 掉 (clean_mask)
    │       → 阈值逐 phase 升高 (0.6→0.7→0.8), 过滤渐严
    │  └─ 每 phase 末存高斯到 $GAUSSIAN_DIR/phase_<N>/point_cloud/...
    ▼
$RESULTS_DIR/orbit/model_pdfgs/phase_<N>/point_cloud/iteration_<iter>/point_cloud.ply
$RESULTS_DIR/orbit/model_pdfgs/phase_4/train/ours_10000/renders/*.png   (重建渲染图, final phase; vs GT)
$RESULTS_DIR/orbit/model_pdfgs/phase_4/train/ours_10000/gt/*.png        (GT, final phase)
    │
    ▼
[04] 转盘/环绕展示视频 (pdfgs env) — 03 只渲输入视角; 这步生成新视角轨道 → mp4
    │  ├─ 加载 final-phase 高斯 → 质心=轨道中心, PCA=人物竖轴, 水平半幅=半径
    │  ├─ FoV/画幅取 COLMAP cameras.txt
    │  └─ 每角度 MiniCam + render() → mp4 + frames/
    ▼
$RESULTS_DIR/orbit/model_pdfgs/orbit_render/orbit_orbit.mp4   (环绕展示视频)
$RESULTS_DIR/orbit/model_pdfgs/orbit_render/frames/*.png      (逐帧)
```

### Step 01 — 分割所有图（两套等价方案，输出布局相同）

对环绕拍摄的**每一张**图分割人物到白底（不像 wan22_rotate step 01 只选最佳正面）。PDF-GS 需要多视图集来三角化身体 + 跨视图对齐 DINOv3 特征识别微动——视图越多，三角化越稳，distractor 过滤越准。两套脚本都输出 `segmented_frames/<rel>.png`，step 02 都能直接吃。

**方案 a — `01a_segment_all.sh`（推荐，质量最好）**：复用 `wan22_rotate` 的 ViTDet + SAM2（sam-3d-body `HumanDetector`/`HumanSegmentor`），就是 wan22_rotate step 01c 那套"person bbox → SAM2"——边缘最干净。脚本 source `wan22_rotate/_env.sh`，跑在 `wan22_rotate` conda env 里（已有 detectron2 + sam-3d-body + SAM2 + ViTDet 权重），不用为分割单建环境。它本质是 01c 的近拷贝，但设 `SEGMENT_ALL=1` 让 `pick_and_segment_mediapipe.py` 跳过 MediaPipe 正面评分、对**每张**图跑同一套 ViTDet 检测 + SAM2 分割（不是只选一张）。前提：先做过 wan22_rotate 的首次准备（建好 wan22 env + sam-3d-body/ViTDet 权重）。Steps 02/03 仍用 pdfgs env；只有 step 01 借 wan22 env。

**方案 b — `01_segment_all.sh`（自含于 pdfgs env）**：rembg 粗掩码 → 人物 bbox → **SAM2 image predictor + bbox prompt** → 干净人物掩码（边缘贴合真实轮廓，无黑边）→ 兜底 SAM2 自动掩码（取面积最大 salient）→ rembg alpha。这条路径**参考 wan22_rotate step 01c** 的"person bbox → SAM2"思路：那里用 ViTDet 出 bbox，这里没有 detectron2，用 rembg 当轻量人物定位器（纯 pip，无 GATED 权重），SAM2 那侧完全一致——bbox prompt 约束 SAM2 只抠框内人物，边缘自然干净。`REMBG_ALPHA_THRESH` 默认 128（旧值 16 会保留半透明边缘带 → 白底上显成黑边，已修）。没配过 wan22 env、只想用 pdfgs env 时用它。设 `SEGMENTOR=sam2`（仅 SAM2 自动掩码，无 rembg 定位）或 `rembg`（仅 rembg alpha）强制单方法；默认 `auto`。

> 01a 质量更稳（ViTDet 人物检测比 rembg 准，SAM2 prompt 一致），优先用；01 作不依赖 wan22 env 的备选。

### Step 02 — Pi3 位姿 + COLMAP 导出 (`02_pi3_colmap.sh` → 调 `pi3_3dgs/pi3_recon.py`)

复用 sibling `pi3_3dgs/pi3_recon.py`（共享 helper），**不带 `--no_colmap`**（与 wan22_rotate step 04 的 `--no_colmap` 区别——这里要导 COLMAP 场景给 PDF-GS）。Pi3 一次前向出所有视角位姿 + 稠密点云，再转 COLMAP 文本格式（`cameras.txt` / `images.txt` / `points3D.txt`），PDF-GS `train.py -s` 直接读。

**为何用 Pi3 不用 COLMAP SfM**：真实人像低纹理（皮肤）+ 白底 + 重复衣物花纹，COLMAP 特征匹配容易失败或漂。Pi3 是单前馈模型，一次出位姿 + 稠密点云，无需特征匹配，鲁棒。位姿和点云同源，互相一致；PDF-GS 只固定位姿+内参优化高斯，即使内参近似（fx=fy=max(W,H)）也能重建出合理结果（高斯自适应）。

### Step 03 — PDF-GS 训练 (`03_train_pdfgs.sh`)

在 pdfgs env 里跑 PDF-GS 的 `train.py`（`cd $PDFGS_DIR` 内跑，保证相对 import）。核心是 `compute_clean_mask`（见 PDF-GS `train.py`）：

- **phase 1**：`prev_feat = None` → `clean_mask = 全 1`（无过滤，纯 3DGS 训练建初始高斯）。
- **phase 2..N**：`feature_extractor = DINOv3FeatureExtractor(...)`；对每个训练视角，算 `cosine_similarity(gt_feat, prev_feat)`（GT 的 DINOv3 特征 vs 上一 phase 渲染图的 DINOv3 特征），低于 `sim_thr` 的像素 = distractor → `clean_mask` 置 0 → 从 `L1_loss` 和 `ssim_loss` 里乘掉。阈值 `--sim_thr 0.6 0.7 0.8` 逐 phase 升高（`sim_thr[phase-2]`），过滤渐严。`prev_mask_dict` 还和上一 phase 的 mask 相乘累积过滤。

> DINOv3：PDF-GS 默认 `facebook/dinov3-vitb16-pretrain-lvd1689m`（ViT-B/16，GATED）。00 patch 了 train.py 让它读 `DINOV3_REPO` 环境变量——`_env.sh` 自动探测：若 `$MODEL_DIR/dinov3-vitl16-pretrain-lvd1689m`（ViT-L/16）本地存在就用它（架构兼容能跑，免 gated 下载），否则用 vitb16（00 用 `HF_TOKEN` 预下到 `$HF_HOME/hub`）。`_env.sh` 设 `HF_HUB_OFFLINE=1`，本地目录或缓存离线读，避免运行时联网（公司代理拦 HF）。

**渲染**：`render.py -s SOURCE -m $GAUSSIAN_DIR/phase_$NUM_PHASES --iteration $ITER_PER_PHASE`。加载 final phase（phase_4）的高斯 + source 的相机，渲染所有训练视角到 `phase_4/train/ours_10000/{renders,gt}/`。没开 `--eval` → 无 held-out test split → `scene.getTestCameras()` 为空，"test" 集自动跳过，只渲 train 集。

**v1 不出网格**：PDF-GS 仓库只有 `train.py` / `render.py` / `metrics.py`，无 `extract_mesh`。出高斯点云 + 渲染图 + 可选指标。要网格走 wan22_rotate step 05/05a/05b（或在本高斯上加 TSDF-on-depth 步骤，future）。

### Step 04 — 转盘/环绕展示视频 (`04_render_orbit.sh` → `render_orbit.py`)

03 的 `render.py` 只渲**输入训练视角**（重建 vs GT 静态图），没有绕人物转一圈的展示视频。这步生成一条**新视角轨道**（相机绕人物的竖轴转一圈）渲染成 mp4 —— 人像 3DGS 重建的招牌展示输出。

做法（不读图片，轻量快）：加载 final-phase 高斯（`point_cloud.ply`）→ 质心 = 轨道中心；PCA → 人物竖轴（方差最大的方向 = 身高）；水平半幅 → 轨道半径（`× ORBIT_RADIUS_MULT`）；FoV/画幅取 COLMAP `cameras.txt`（内参不受归一化影响，和训练一致）。每个角度 θ 用 `MiniCam`（矩阵约定和 `scene.cameras.Camera` 一致：`world_view_transform = w2c.T`、`full_proj = wvt @ proj.T`）+ `gaussian_renderer.render()` 出一帧 → mp4 + `frames/*.png`。全程在 pdfgs env（要 `diff_gaussian_rasterization`，和 03 同）。

**构图调参**：人物太大/太小调 `ORBIT_RADIUS_MULT`（3.0 近 / 4.0 远）；竖轴猜错（人物歪倒）设 `UP_AXIS=y`（或 `x`/`z`）；想抬高视点设 `ORBIT_HEIGHT=0.1`；转两圈 `ORBIT_TURNS=2.0`；降分辨率出图快 `RES=1280`。

## Config (env vars, all optional)

### Paths & envs
| var | default | note |
| --- | --- | --- |
| `INPUT_DIR` | _(required for 01)_ | 环绕拍摄文件夹（含 `image/` 子夹，或直接散图） |
| `GPU` | _(unset)_ | physical GPU id, e.g. `GPU=0` |
| `CONDA_ENV` | `pdfgs` | conda env（torch 2.5.1+cu121） |
| `PDFGS_DIR` | `../PDF-GS` | PDF-GS 官方代码（00 clone） |
| `SSL_VERIFY` | `true` | `false` = 公司代理 TLS 拦截下彻底关 SSL：unset CA bundle 环境变量 + `PYTHONHTTPSVERIFY=0` + 00 往 pdfgs env 注入 `sitecustomize.py`（`ssl._create_unverified_context`）。conda 已单独 `ssl_verify=false`（不受此开关控制，默认关） |
| `PI3_DIR` | `../Pi3` | Pi3 官方代码（00 clone; 02 兜底 auto-clone） |
| `SAM2_DIR` | `../sam2` | SAM2 官方代码 + checkpoints（00 clone） |
| `MODEL_DIR` | `../../model` | 权重根（code-dir 上一级，共享） |
| `DINOV3_REPO` | _(auto)_ | DINOv3 加载源：本地有 `$MODEL_DIR/dinov3-vitl16-pretrain-lvd1689m` 就用它（vitl16，免 gated），否则 `facebook/dinov3-vitb16-pretrain-lvd1689m`（gated，需 HF_TOKEN）。可强制覆盖 |
| `U2NET_HOME` | `$MODEL_DIR/u2net` | rembg 模型目录（本地放 `u2net.onnx`，免联网下载；SAM2 不可用时 step 01 兜底） |
| `RESULTS_DIR` | `../pdfgs_human_results` | 输出目录 |
| `OUTPUT_NAME` | `orbit` | 基名（影响 02/03 的默认子目录） |

### Step 01 params
| var | default | note |
| --- | --- | --- |
| `SEGMENTED_DIR` | `$RESULTS_DIR/segmented_frames` | 输出（白底多视图集） |
| `SEGMENTOR` | `auto` | `auto`（rembg 定位 → SAM2 bbox-prompted; 自动/rembg 兜底） \| `sam2`（仅自动掩码） \| `rembg`（仅 rembg alpha） |
| `WHITE_BG` | `1` | `1`=白底（匹配 PDF-GS `--white_background`）；`0`=黑底 |
| `MIN_MASK_FRAC` | `0.02` | 掩码面积 < 图像 2% 视为错误对象，丢弃 |
| `REMBG_ALPHA_THRESH` | `128` | rembg alpha 阈值（旧 16 → 半透明边缘带显成黑边；128 修复。调高更紧） |
| `BBOX_PADDING` | `0.05` | rembg 掩码外扩比例，喂给 SAM2 的 bbox 含住头发/四肢 |
| `DEVICE` | `cuda` | SAM2 设备；`cpu` 可用但慢 |
| `SAM2_CHECKPOINT` | `$SAM2_DIR/checkpoints/sam2.1_hiera_large.pt` | SAM2 权重 |
| `SAM2_CONFIG` | `configs/sam2.1/sam2.1_hiera_large.yaml` | SAM2 配置（包内相对路径） |

### Step 02 params
| var | default | note |
| --- | --- | --- |
| `INPUT` | `$RESULTS_DIR/segmented_frames` | 输入图集（step 01 输出） |
| `PI3_OUTPUT_DIR` | `$RESULTS_DIR/<name>/pi3` | Pi3 输出 |
| `SOURCE_DIR` | `$PI3_OUTPUT_DIR/source` | COLMAP 场景（PDF-GS `train.py -s` 读） |
| `PI3_CKPT` | `$MODEL_DIR/Pi3/model.safetensors` | Pi3 checkpoint |
| `FRAME_FPS` | `10` | 视频抽帧 fps（图集输入时忽略，copy 全部） |
| `FRAME_MAX` | `60` | 最大帧数（Pi3 显存随 N 线性增长） |
| `CONF_THRES` | `0.1` | sigmoid-conf 阈值，过滤低置信初始化点 |
| `SKIP_PI3` | `0` | `1` = 复用已有 `source/` |

### Step 03 params
| var | default | note |
| --- | --- | --- |
| `SOURCE_DIR` | `$RESULTS_DIR/<name>/pi3/source` | COLMAP 场景 |
| `GAUSSIAN_DIR` | `$RESULTS_DIR/<name>/model_pdfgs` | 高斯输出（⚠️ 不是 `$MODEL_DIR`，那是权重根） |
| `NUM_PHASES` | `4` | progressive filtering phase 数 |
| `ITER_PER_PHASE` | `10000` | 每 phase 迭代数（总 = `NUM_PHASES × ITER_PER_PHASE`） |
| `SIM_THR` | `0.6 0.7 0.8` | 每 phase 过渡的 distractor 阈值（长度 = `NUM_PHASES-1`，或单值全 phase） |
| `COLOR_UPDATE_INTERVAL` | `30` | SH 颜色更新间隔（非末 phase） |
| `WHITE_BG` | `1` | `1` = 光栅化白底（匹配分割输入） |
| `RES` | _(unset)_ | `--resolution` 因子；**不设 = 全分辨率**（人像用全分辨率；README 的 `-r 8` 是 RobustSplat benchmark 降采样，别照搬） |
| `SKIP_TRAIN` | `0` | `1` = 复用已有 `model_pdfgs/` |
| `SKIP_RENDER` | `0` | `1` = 跳过渲染 |
| `SKIP_METRICS` | `1` | `1` = 跳过 PSNR/SSIM/LPIPS（无 held-out test → PSNR 只是 train fit；`0` 跑） |
| `TRAIN_EXTRA_ARGS` | _(empty)_ | 透传给 `train.py` 的额外参数 |

### Step 04 params
| var | default | note |
| --- | --- | --- |
| `SOURCE_DIR` | `$RESULTS_DIR/<name>/pi3/source` | COLMAP 场景（取 cameras.txt 的 FoV/画幅） |
| `GAUSSIAN_DIR` | `$RESULTS_DIR/<name>/model_pdfgs` | 高斯根（= step 03 `--model_path`；⚠️ 不是 `$MODEL_DIR`，那是权重根） |
| `PHASE` | `4` | 渲哪个 phase（final = step 03 的 `NUM_PHASES`） |
| `ITER` | _(auto)_ | 加载的 iteration（空 = `phase_<PHASE>/point_cloud/` 下最大 `iteration_*`） |
| `ORBIT_FRAMES` | `120` | 轨道视频帧数 |
| `ORBIT_TURNS` | `1.0` | 转几圈（1.0 = 360°，2.0 = 720°） |
| `ORBIT_RADIUS_MULT` | `3.5` | 轨道半径 = `MULT × 人物水平半幅`（调构图：3.0 近 / 4.0 远） |
| `ORBIT_HEIGHT` | `0.0` | 相机沿竖轴抬高量 = `值 × 人物半高`（0.1 ≈ 抬高到胸口上方） |
| `UP_AXIS` | _(auto)_ | 人物竖轴：空 = PCA 自动（最大方差方向）；猜错时设 `x`/`y`/`z` 强制 |
| `WHITE_BG` | `1` | `1` = 白底（匹配分割输入）；`0` = 黑底 |
| `FPS` | `30` | 输出 mp4 帧率 |
| `RES` | _(unset)_ | 限最大边像素（如 `1280`）；不设 = COLMAP 原始画幅 |
| `SH_DEGREE` | `3` | PDF-GS 训练的 SH 阶（ModelParams 默认 3；`cfg_args` 里看 `sh_degree`） |
| `DEVICE` | `cuda` | 光栅化设备（PDF-GS 需 CUDA） |

## 可能遇到的问题

**0. SSL 验证失败（conda / pip / hf / requests 都报证书错）**
公司代理做 TLS 拦截，代理根 CA 不在系统 bundle 里。手动 `unset SSL_CERT_FILE REQUESTS_CA_BUNDLE` 没用——`_env.sh` 被 source 时会重新 export 它们。用开关：
```bash
SSL_VERIFY=false INSTALL_DEPS=1 bash pdfgs_human/00_setup_env.sh
```
`SSL_VERIFY=false` 时：`_env.sh` unset 掉所有 CA bundle 环境变量 + `PYTHONHTTPSVERIFY=0`；00 往 pdfgs env 的 `site-packages/sitecustomize.py` 注入 `ssl._create_default_https_context = ssl._create_unverified_context`，让所有 Python 进程（含 huggingface_hub）跳过证书验证。conda 单独由 `_conda_disable_ssl` 设 `ssl_verify false`（默认关，不受此开关控制）。

**1. `step 03` 报 DINOv3 加载失败 / `HF_HUB_OFFLINE` 下找不到权重**
DINOv3 两种拿法（见「首次准备」）：
- **有本地 vitl16**：把 DINOv3 ViT-L/16 放到 `$MODEL_DIR/dinov3-vitl16-pretrain-lvd1689m/`（含 `config.json` + 权重），`_env.sh` 自动探测、00 patch train.py 指向它，免 token。
- **走 gated vitb16**：先到 [HF 模型页](https://huggingface.co/facebook/dinov3-vitb16-pretrain-lvd1689m) 点 "Request access"，通过后重跑 `HF_TOKEN=hf_xxx INSTALL_DEPS=1 bash pdfgs_human/00_setup_env.sh`；确认 `$MODEL_DIR/hf_home/hub/models--facebook--dinov3-vitb16-pretrain-lvd1689m/` 有 `snapshots/` 子夹。
若报 train.py 没读到 `DINOV3_REPO`（03 会 warn），重跑 `INSTALL_DEPS=1 bash pdfgs_human/00_setup_env.sh` 让 00 重新 patch。

**2. `step 03` 报 `import diff_gaussian_rasterization` / `import simple_knn` 失败 / CUDA 扩展没编成**
三个根因，00 都已自动处理，但首次若失败需手动兜底：

- **diff-gaussian-rasterization 分支错（main 而非 dr_aa）**：PDF-GS `.gitmodules` 指定 `branch=dr_aa`（antialiased 3DGS rasterizer），但 `git clone` 默认 checkout `main`（原版 3DGS，无 antialiasing）。00 会在 clone 后自动 `git checkout dr_aa` + 清除旧 build 缓存重编。若仍失败，手动：
  ```bash
  cd $PDFGS_DIR/submodules/diff-gaussian-rasterization
  git checkout dr_aa
  # dr_aa 分支 third_party/glm 为空 → symlink 到 PDF-GS 根的 GLM
  rm -rf third_party/glm && ln -sf $PDFGS_DIR/third_party/glm third_party/glm
  # 清 build 缓存重编
  rm -rf build dist *.egg-info
  INSTALL_DEPS=1 bash pdfgs_human/00_setup_env.sh
  ```

- **simple-knn 源码缺失**：PDF-GS 的 simple-knn 子模块指向 `gitlab.inria.fr/bkerbl/simple-knn`（公司代理封了 gitlab.inria.fr）。00 会按序尝试 gitlab.inria.fr → GitHub 镜像（`yindaheng98/simple-knn`、`jteng2127/simple-knn`）clone。若全失败，手动：
  ```bash
  cd $PDFGS_DIR/submodules && rm -rf simple-knn
  git clone https://github.com/yindaheng98/simple-knn.git simple-knn
  INSTALL_DEPS=1 bash pdfgs_human/00_setup_env.sh   # 重新编译
  ```

- **conda gcc 没装**：CUDA 扩展编译需要 gcc 12（系统 gcc 太老编不过 CUDA 12.x rasterizer）。00 先 `conda install --no-update-deps gxx_linux-64=12 python=3.10`，失败则去掉 `--no-update-deps` 重试。若仍失败：
  ```bash
  conda install -y -c conda-forge gxx_linux-64=12 python=3.10
  python -c "import platform; print(platform.python_implementation())"  # 必须 CPython
  INSTALL_DEPS=1 bash pdfgs_human/00_setup_env.sh
  ```

- **nvcc 找不到**：确认系统有 CUDA 12.x toolkit：`ls -d /usr/local/cuda-12*`。若 `/usr/local/cuda` 指向 11.8，00 会自动找 `cuda-12.x` 并设 `CUDA_HOME`；手动：
  ```bash
  export CUDA_HOME=/usr/local/cuda-12.4   # ⚠️ 不是 /usr/local/cuda
  export PATH=$CUDA_HOME/bin:$PATH
  INSTALL_DEPS=1 bash pdfgs_human/00_setup_env.sh
  ```
  GLM 缺失（`glm/glm.hpp: No such file`）时：`cd $PDFGS_DIR && git clone https://github.com/g-truc/glm.git third_party/glm`。

**3. `step 01` 人物边缘有黑边 / 掩码全是矩形 / SAM2 没出掩码**
- **黑边**：旧版 rembg alpha 阈值 16 会保留半透明边缘带（多为暗背景）→ 白底上显成黑边。现默认 `auto` 走 rembg→bbox→SAM2 predictor（边缘干净），且 `REMBG_ALPHA_THRESH` 默认 128。若仍残留，调高 `REMBG_ALPHA_THRESH=180`，或确认走的是 `auto`（不是 `rembg`）。
- **SAM2 没出掩码 / 矩形掩码**：SAM2 没装好或 checkpoint 没下。重跑 `INSTALL_DEPS=1 bash pdfgs_human/00_setup_env.sh`；或临时 `SEGMENTOR=rembg` 用 rembg 兜底（质量略降）。多人物图取面积最大掩码（离相机最近者）。

**3b. `step 01` rembg 报 SSL / u2net 模型下载失败 / 找不到 u2net.onnx**
rembg 用 pooch+requests 下载 u2net 模型，requests 用 certifi 的 CA bundle（**不遵守** `sitecustomize.py` 的 ssl hack），公司代理 TLS 拦截会直接断掉下载。且 rembg 的 `u2net_home()` 在 `U2NET_HOME` 未设时 fallback 到 `$XDG_DATA_HOME/.u2net`（如 `~/.local/share/.u2net`），而非 `_env.sh` 设的 `$MODEL_DIR/u2net`。

00 自动处理：① `wget --no-check-certificate` 下载 `u2net.onnx` 到 `$MODEL_DIR/u2net/`；② symlink 到 `~/.local/share/.u2net/` 和 `~/.u2net/`（覆盖 rembg fallback 路径）；③ `_env.sh` 设 `U2NET_HOME` + `MODEL_CHECKSUM_DISABLED`。若仍失败（如 GitHub 也被封），手动：
```bash
# 1. 手动下 u2net.onnx (从能联网的机器下，拷到容器里)
wget --no-check-certificate -O $MODEL_DIR/u2net/u2net.onnx \
  https://github.com/danielgatis/rembg/releases/download/v0.0.0/u2net.onnx

# 2. symlink 到 rembg fallback 路径 (~/.local/share/.u2net 和 ~/.u2net)
mkdir -p ~/.local/share/.u2net ~/.u2net
ln -sf $MODEL_DIR/u2net/u2net.onnx ~/.local/share/.u2net/u2net.onnx
ln -sf $MODEL_DIR/u2net/u2net.onnx ~/.u2net/u2net.onnx

# 3. 验证 (在 pdfgs env 里, source 了 _env.sh 后)
python -c "from rembg import new_session; s=new_session('u2net'); print('✅ rembg ok')"
```

**4. `step 02` Pi3 OOM**
Pi3 显存随帧数线性增长。降 `FRAME_MAX=30` 或抽稀 `segmented_frames/`（保留环绕均匀分布的子集）。图集输入时 `FRAME_FPS` 被忽略。

**5. `step 03` 渲染报找不到 `cfg_args` / 找不到 point_cloud**
`train.py` 把 `cfg_args` 写在 `$GAUSSIAN_DIR/`（model 根目录），但 `render.py` / `metrics.py` 用 `-m $PHASE_MODEL`（phase 子目录）调用，`get_combined_args()` 在 `$PHASE_MODEL/cfg_args` 找不到。脚本已在训练后自动 `cp $GAUSSIAN_DIR/cfg_args $PHASE_MODEL/cfg_args`。若仍报错，手动复制：
```bash
cp $GAUSSIAN_DIR/cfg_args $GAUSSIAN_DIR/phase_4/cfg_args
```

**6. 跑 `.sh` 报 `syntax error near unexpected token '('`**
CRLF 行尾污染。`find pdfgs_human -name '*.sh' -exec sed -i 's/\r$//' {} +` 或 `git checkout -- pdfgs_human/*.sh`（`.gitattributes` 强制 LF）。

**7. NumPy 坏了 / `import numpy` 报 ABI 不兼容 / python 变成了 GraalPy**
根因：往 pdfgs env `conda install` 任何包没加 `--no-update-deps` 时，conda 求解器会把 python 实现掉包成 GraalPy。00 的 gxx 步骤已加 `--no-update-deps` + `python=3.10` pin + 建完校验 CPython。若已中招，重建：
```bash
python -c "import platform; print(platform.python_implementation())"   # GraalPy 即中招
conda env remove -n pdfgs
conda create -n pdfgs python=3.10 -y && conda activate pdfgs
INSTALL_DEPS=1 bash pdfgs_human/00_setup_env.sh
```
> 铁律：往 pdfgs env 里 `conda install` 任何包都加 `--no-update-deps`。

## 目录布局
```
<code-dir>/
├── media_code/                    # 本仓
│   ├── proxy.env                  # 代理 + 覆盖项, gitignored
│   ├── pi3_3dgs/                   # Pi3+COLMAP 共享 helper (step 02 调 pi3_recon.py)
│   └── pdfgs_human/               # ← 本目录（编排脚本）
│       ├── _env.sh
│       ├── 00_setup_env.sh
│       ├── 01_segment_all.sh
│       ├── 01a_segment_all.sh
│       ├── 02_pi3_colmap.sh
│       ├── 03_train_pdfgs.sh
│       ├── 04_render_orbit.sh
│       ├── segment_all.py
│       └── render_orbit.py
├── PDF-GS/                         # PDF-GS 官方代码 + 子模块 (00 clone)
│   ├── submodules/
│   │   ├── diff-gaussian-rasterization/   # 3DGS 光栅化器 (CUDA 扩展)
│   │   └── simple-knn/                   # KNN (CUDA 扩展)
│   └── train.py / render.py / metrics.py
├── sam2/                           # SAM2 官方代码 + checkpoints (00 clone, step 01 分割用)
│   └── checkpoints/sam2.1_hiera_large.pt
├── Pi3/                            # Pi3 官方代码 (00 clone, step 02 用)
├── model/                          # 权重根 (code-dir 上一级, 共享)
│   ├── wheels/                    # torch wheel 缓存 (00 首跑 pip download; 跨项目共享)
│   ├── dinov3-vitl16-pretrain-lvd1689m/  # DINOv3 ViT-L/16 本地 (放这, 00 自动用, 免 gated)
│   ├── Pi3/model.safetensors
│   └── hf_home/hub/                # DINOv3 离线缓存 (仅走 gated vitb16 时, 00 用 HF_TOKEN 下)
└── pdfgs_human_results/            # 输出 (repo 外)
    ├── segmented_frames/           # step 01: 白底人物多视图集
    │   └── <rel>.png
    └── orbit/                      # OUTPUT_NAME=orbit
        ├── pi3/                    # step 02: Pi3 + COLMAP
        │   ├── source/
        │   │   ├── images/         # COLMAP 训练图 (= 分割图 copy)
        │   │   └── sparse/0/{cameras,images,points3D}.txt
        │   ├── predictions.npz
        │   ├── dense_cloud.ply
        │   └── poses.json
        └── model_pdfgs/            # step 03: PDF-GS 高斯
            ├── phase_1/ ... phase_4/
            │   └── point_cloud/iteration_10000/point_cloud.ply
            ├── cfg_args
            ├── phase_4/train/ours_10000/   # final phase 渲染 (重建 vs GT)
            │   ├── renders/*.png   # 渲染图 (vs GT 看重建质量)
            │   └── gt/*.png        # GT
            └── orbit_render/               # step 04: 转盘展示视频
                ├── orbit_orbit.mp4         # 环绕人物的新视角轨道视频
                └── frames/*.png            # 逐帧 png
```

## Notes
- Official code & weights follow their own licenses (PDF-GS = MIT-style research license; Pi3 / SAM2 / DINOv3 = their respective licenses). This folder only orchestrates; no official code is copied.
- `.gitattributes` (repo root) forces LF so Windows-pushed scripts run cleanly on Ubuntu.
- `proxy.env` (proxy creds / env overrides) is gitignored — never committed.

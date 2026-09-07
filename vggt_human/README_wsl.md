# VGGT-Human (WSL Ubuntu 24.04) — 本地复现指南

本文件是 [`README.md`](README.md) 的 **WSL 本地复现版**。照着做即可从零跑完全流程。

> 两个 README 共存：
> - [`README.md`](README.md) — 服务器流程（`doll` env clone + 公司代理 + `/usr/local/cuda`）
> - [`README_wsl.md`](README_wsl.md) — 本文件（WSL 从零建 env + 无代理 + conda 装 CUDA toolkit）

## 与服务器版的核心差异

| | 服务器 (`00_setup_env.sh`) | WSL (`00a_setup_env.sh`) |
|---|---|---|
| conda env | cp -a clone `doll`（含 torch） | conda create -n vggt_human python=3.10 + pip install torch cu121 |
| CUDA toolkit (nvcc) | 系统 `/usr/local/cuda-12.x` | conda install cuda-nvcc（env 内，无需 sudo） |
| gcc | conda gxx_linux-64=12 | 同（conda gxx_linux-64=12） |
| 代理 / CA | proxy.env 填公司代理密码 + CA bundle | 不需要（直连互联网） |
| pip 源 | 默认 PyPI | 清华镜像（`-i https://pypi.tuna.tsinghua.edu.cn/simple`） |
| HF 源 | 直连 huggingface.co | hf-mirror.com 镜像（`HF_ENDPOINT`） |
| 仓库位置 | `/mnt/c/code/`（与 media_code 同级） | `~/repos/`（Linux 文件系统，编译快） |
| 权重 / 输出 | `/mnt/c/code/model/` | `~/model/`、`~/output/vggt_human_results/`（Linux fs，训练 I/O 快） |

> **为什么仓库 / 权重放 Linux 文件系统？** `/mnt/c` `/mnt/d` 是 Windows drvfs（9p 协议），文件 I/O 慢 5-10 倍。编译 CUDA 扩展 + 训练时大量写文件，放 Linux fs 避免超时和性能问题。路径通过 `proxy.env` 自动覆盖，脚本 01-08 不需改。

## Windows 路径 → WSL 路径

WSL 自动把 Windows 盘挂到 `/mnt/`，命令里用正斜杠：

| Windows | WSL |
|---|---|
| `C:\code\media_code` | `/mnt/c/code/media_code` |
| `D:\dataset\sample` | `/mnt/d/dataset/sample` |
| `D:\output` | `/mnt/d/output` |

**路径策略**（哪些放 Linux fs，哪些放 Windows 盘）：

| 内容 | 位置 | 原因 |
|---|---|---|
| conda env + 仓库 + CUDA 扩展 | `~/` (Linux fs) | 编译必须放 Linux fs，drvfs 上 symlink 会坏 |
| 权重 (VGGT-Omega 4.6GB) | `~/model/` (Linux fs) | 读一次加载到 GPU，Linux fs 读大文件比 drvfs 快 5-10x |
| 输入图像 | `/mnt/d/...` (Windows fs) | 只读一次，慢一点无所谓 |
| 训练输出 (RESULTS_DIR) | `~/output/` (Linux fs) | step 04 训练大量写文件，放 drvfs 慢到影响训练 |
| 最终结果备份 | `/mnt/d/output/` (Windows fs) | 训练完后用 step 08 剪切过去，释放 Linux fs 空间 |

## 前提条件

1. **WSL Ubuntu 24.04** 已安装并运行
   ```powershell
   wsl -d Ubuntu2404
   ```
2. **NVIDIA 驱动** 在 Windows 上装好（WSL 内 `nvidia-smi` 能看到 GPU）
3. **Miniconda** 已安装：
   ```bash
   # 如果还没装 miniconda：
   curl -fsSL https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -o /tmp/miniconda.sh
   bash /tmp/miniconda.sh -b -p ~/miniconda3
   ```
4. **磁盘空间** ≥ 30GB（torch + CUDA toolkit + 权重 + 仓库）

## 首次准备（一次性）

### 步骤 0：安装环境

```bash
# 进入 WSL
wsl -d Ubuntu2404

# 进入 media_code 目录
cd /mnt/c/code/media_code

# 一键安装（创建 env + 装 torch + CUDA toolkit + gcc + 依赖 + clone 仓 + 编 CUDA 扩展）
bash vggt_human/00a_setup_env.sh
```

> 如果 `pip install` 很慢（默认 PyPI 在国内慢），手动用清华镜像装剩余包：
> ```bash
> conda activate vggt_human
> pip install -i https://pypi.tuna.tsinghua.edu.cn/simple opencv-python huggingface_hub \
>     plyfile tqdm torchmetrics lpips scipy trimesh matplotlib mediapipe==0.10.14 \
>     diffusers==0.32.2 transformers==4.49.0 peft==0.14.0 omegaconf kornia accelerate
> ```

安装完成后：
```bash
# 让 conda 在你的 shell 里可用
source ~/.bashrc
```

### 步骤 0b：配置 proxy.env

00a 会自动生成 `proxy.env`（含 WSL 路径覆盖）。需要手动补充两项：

```bash
nano /mnt/c/code/media_code/proxy.env
```

补充以下内容：
```bash
# HF_TOKEN（VGGT-Omega 权重是 gated 仓库）
# 1. 申请访问：https://huggingface.co/facebook/VGGT-Omega（自动审核）
# 2. 创建 token：https://huggingface.co/settings/tokens
export HF_TOKEN="hf_xxx"               # ← 填你的 token

# HF 镜像（huggingface.co 直连超时；用 hf-mirror.com 镜像）
export HF_ENDPOINT="https://hf-mirror.com"
```

### 步骤 0c：下载 VGGT-Omega 权重

```bash
cd /mnt/c/code/media_code
GPU=0 VARIANT=1b_512 bash vggt-omega/01_download_models.sh
```

> 用了 `HF_ENDPOINT=https://hf-mirror.com` 镜像，4.6GB 约 3 分钟下完。
> 权重**实际**落在 `proxy.env` 里 `MODEL_DIR` 指向的位置，本机是
> `/mnt/d/wheel/vggt_human_ms/VGGT-Omega/vggt_omega_1b_512.pt`；
> **不是** `~/model/VGGT-Omega/`（那里只有 HF 的 `.cache/`，没有 `.pt`）。
> 原因：`_env.sh` 会 source `proxy.env`，`MODEL_DIR` 非空时下载脚本的默认路径被覆盖。
> step 02 硬性检查 `$MODEL_DIR/vggt_omega_1b_512.pt`，路径写错会直接报错退出 —— 全流程命令里要显式用 D 盘这个路径。

### HYPIR 人脸增强（step 01/06）— 遗留问题

00a 默认会 clone HYPIR + 下载 SD2 base model。如果遇到问题（hf-mirror 没收录 SD2 base repo / HYPIR 的 `beauty_ppr50k` LoRA 权重未从服务器拷来），可 **跳过 step 01/06**，直接从 step 02 开始：

```bash
# 安装时跳过 HYPIR
SKIP_HYPIR=1 bash vggt_human/00a_setup_env.sh
```

准备好后再开启 HYPIR：
1. 确保 huggingface.co 可达或用镜像 `HF_ENDPOINT=https://hf-mirror.com`
2. 从服务器拷 `beauty_ppr50k` LoRA 权重到 `~/repos/HYPIR/experiments/`
3. 下载 SD2 base model：`git clone https://hf-mirror.com/stabilityai/stable-diffusion-2-base ~/model/HYPIR/sd2_base`
4. 跑 step 01 对原始图做人脸增强

## 路径布局（WSL）

```
~/                                # Linux 文件系统（WSL vhdx 100GB）
├── repos/                        # 官方仓库（00a clone，Linux fs 编译快）
│   ├── vggt-omega/               # VGGT-Omega 官方代码
│   ├── gaussian-splatting/       # 原版 3DGS
│   │   ├── submodules/
│   │   │   ├── diff-gaussian-rasterization/  # CUDA 扩展（dr_aa 分支）
│   │   │   └── simple-knn/                    # CUDA 扩展
│   │   └── third_party/glm/
│   ├── HYPIR/                    # 人脸增强（SKIP_HYPIR=1 时不 clone）
│   ├── DiffBIR/                  # 去噪（INSTALL_DENOISER=1）
│   └── SwinIR/                   # 去噪（INSTALL_DENOISER=1）
├── model/                        # 权重（Linux fs，训练 I/O 快）
│   ├── VGGT-Omega/               # vggt_omega_1b_512.pt（gated HF 下载）
│   └── HYPIR/
│       └── sd2_base/             # SD2 base model（HYPIR 启用时需要）
├── output/
│   └── vggt_human_results/       # 训练输出（跑完用 step 08 搬到 D 盘）
└── miniconda3/                   # conda 安装
    └── envs/vggt_human/          # conda env（python=3.10 + torch cu121 + nvcc + gcc）

/mnt/c/code/media_code/           # 编排脚本（Windows fs，git 仓）
├── proxy.env                     # WSL 路径覆盖 + HF_TOKEN + HF_ENDPOINT（00a 自动生成）
└── vggt_human/
    ├── 00a_setup_env.sh          # WSL 安装脚本
    ├── 01b_heic_to_jpg.sh        # HEIC→JPG 转换（iPhone 照片）
    ├── 01_face_enhance.sh        # 以下脚本与服务器共用，通过 proxy.env 自动用 WSL 路径
    ├── 02_run_inference.sh
    ├── ...
    └── 08_move_output.sh          # 训练完把结果从 Linux fs 剪切到 D:\output

/mnt/d/                           # Windows D: 盘
├── dataset/                      # 输入数据（只读，drvfs 慢但无所谓）
└── output/                       # 最终结果备份（step 08 搬过来）
    └── vggt_human_results/
```

## 全流程命令

> 假设已完成「首次准备」，输入图像在 `D:\dataset\sample\image`。
> 每条命令在 WSL 内 `/mnt/c/code/media_code` 目录下执行。
> `proxy.env` 已写入 WSL 路径覆盖，脚本自动用 `~/repos/`、`~/model/` 等路径。

```bash
cd /mnt/c/code/media_code

# ── 0) 安装环境（首次，见上方「首次准备」）──

# ── 1b) HEIC → JPG 转换（iPhone 照片是 .heic，VGGT-Omega 不认）──
#    输入：D:\dataset\sample\image（.heic 文件夹）
#    输出：D:\dataset\sample\image_jpg（.jpg 文件夹，自动创建在同级目录）
GPU=0 \
INPUT_DIR=/mnt/d/dataset/sample/image \
bash vggt_human/01b_heic_to_jpg.sh

# ── 1) 前处理人脸增强（HYPIR 遗留问题，暂跳过）──
#    跳过此步，直接用原始图像（或 step 1b 的 JPG）喂 step 02
#    准备好 HYPIR 后再启用：
# GPU=0 INPUT_DIR=/mnt/d/dataset/sample/image_jpg \
#   RESULTS_DIR=~/output/vggt_human_results \
#   bash vggt_human/01_face_enhance.sh

# ── 2) VGGT-Omega 前馈推理（图像 → 位姿+深度 → predictions.npz + scene.ply）──
#    INPUT_DIR 指向 step 1b 的 JPG 输出（如果不是 .heic 则直接用原始图夹）
GPU=0 \
INPUT_DIR=/mnt/d/dataset/测试数据sample/3fe0604320a24d66a8bde164edf18c11/image_jpg \
MODEL_DIR=/mnt/d/wheel/vggt_human_ms/VGGT-Omega \
RESULTS_DIR=~/output/vggt_human_results \
MAX_POINTS=2000000 \
bash vggt_human/02_run_inference.sh

# 输出：~/output/vggt_human_results/02_vggt/<scene>/
#   predictions.npz   # 位姿+深度+点云+置信度
#   scene.ply          # 点云（供检查）
#   frames/            # 喂给模型的图

# ── 3) npz → COLMAP 转换（自适应置信度过滤 + 体素降采样 ~200k + 坐标系对齐）──
#    ALIGN=1：把场景居中到原点（官方 train.py 直接吃这个坐标系即可收敛）
GPU=0 \
TARGET_POINTS=200000 \
ALIGN=1 \
RESULTS_DIR=~/output/vggt_human_results \
bash vggt_human/03_npz_to_colmap.sh

# 输出：~/output/vggt_human_results/03_source/
#   images/*.png              # 训练图
#   sparse/0/cameras.txt      # 内参
#   sparse/0/images.txt       # 外参 w2c
#   sparse/0/points3D.txt     # ~200k 初始点

# ── 4) 原版 3DGS 训练 + 渲染 ──
#    默认恒走官方 gaussian-splatting 的 train.py（干净官方基线）。
#    增强（已加回、默认全关，见上方各实测小节）：训练脚本切换
#      USE_DEPTH_NORMAL=1 / USE_NOISE_NEGATE=1（pose_refine 不可用）；
#    训练前预处理开关：ENABLE_DYNAMIC_MASK=1 / ENABLE_DYNAMIC_FILTER=1
#      （prompt 用 DYNAMIC_PROMPTS 覆盖，默认 "TV screen monitor" 不含 person）。
#    未加回的归档脚本见 archive/README.md。
GPU=0 \
ITERATIONS=30000 \
WHITE_BG=0 \
RESULTS_DIR=~/output/vggt_human_results \
bash vggt_human/04_train_3dgs.sh

# 输出：~/output/vggt_human_results/04_model_3dgs/
#   point_cloud/iteration_30000/point_cloud.ply    # 最终高斯
#   train/ours_30000/renders/*.png                 # 重建渲染
#   train/ours_30000/gt/*.png                      # GT

# ── 5) 渲染新视角 → 去噪 → AdaIN → 增强 COLMAP（可选）──
#    DENOISER 可选: diffbir（扩散，质量高）| swinir（前馈，快）| none（跳过去噪）
#    首次用 DiffBIR/SwinIR 需先: INSTALL_DENOISER=1 bash vggt_human/00a_setup_env.sh
#    ⚠️ 本步是 ITERATION（单数），step 04/07 是 ITERATIONS（复数），别写混。这不是小bug吗，怎么不改？
#       值必须等于 step 04 实际跑的迭代数，否则报 "3DGS checkpoint not found"。
GPU=0 \
DENOISER=diffbir \
NUM_NOVEL_VIEWS=10 \
ITERATION=30000 \
RESULTS_DIR=~/output/vggt_human_results \
bash vggt_human/05_denoise_novel.sh

# 输出：~/output/vggt_human_results/
#   05_novel_renders/*.png     # 3DGS 渲染的新视角
#   05_novel_alpha/*.png       # 覆盖度图
#   05_source_aug/            # 增强COLMAP场景（原图+去噪图）

# ── 6) 后处理人脸增强（可选，需 HYPIR 已就绪）──
#    对 05_source_aug/images/ 做人脸增强（与 step 01 同一个 face_enhance.py）
GPU=0 \
RESULTS_DIR=~/output/vggt_human_results \
bash vggt_human/06_face_enhance.sh

# 输出：~/output/vggt_human_results/06_source_aug_face/
#   images/  # 原图+去噪图，人脸区域已增强+融合
#   sparse/  # COLMAP 原样复制

# ── 7) 用增强场景训练 3DGS（可选）──
#    原图 + 去噪虚拟相机 + 人脸增强 共同监督
#    同样恒走官方 train.py（增强已归档到 archive/，需加回见 archive/README.md）
GPU=0 \
ITERATIONS=30000 \
WHITE_BG=0 \
RESULTS_DIR=~/output/vggt_human_results \
bash vggt_human/07_train_denoise.sh

# 输出：~/output/vggt_human_results/07_model_3dgs_denoise/
#   point_cloud/iteration_30000/point_cloud.ply    # 增强训练后的高斯

# ── 7b) 人脸模糊治理·方案一升级版（推荐，替代 6+7 的"重训"路线）──
#    30k 训完后从 ply 续训 finetune：人脸区互补双监督，追加不替换。
#    依赖：06_face_enhance.sh（HYPIR 增强图）+ face_masks.py（loss mask）。
GPU=0 \
RESULTS_DIR=~/output/vggt_human_verify \
bash vggt_human/06b_face_finetune.sh
#    默认: 30000→35000 iters, LR×0.1, densify OFF, FACE_WEIGHT=0.5(dual 双监督)
#    FACE_WEIGHT=1.0 → complement(增强图独占人脸监督)；FACE_SOFT=1 → 羽化软权重
#    输出: $RESULTS_DIR/06b_model_3dgs_face/point_cloud/iteration_35000/point_cloud.ply
#
#    face_masks.py 的 mask = HYPIR 实际改动区域（MediaPipe bbox×1.2 + feather
#    阈值化 + 腐蚀 2px），与 face_enhance.py 同检测器同参数，逐位对应；
#    无人脸帧（检测跳过）自动退化为纯官方监督。
#    train_face_finetune.py 是官方 train.py 副本 + 单一 hook（🧑 标注差异）：
#    ply 续训(含 exposure.json，30k 收敛为恒等阵)、LR 缩放、densification 冻结。
#
#    评估（人脸区指标，验收=LPIPS对原图防偏离 + Laplacian方差/梯度能量测锐度）：
#    先用 render_train_views.py（按原始文件名保存，便于 face_metrics 按 stem 匹配）
#    分别渲染 30k 基线与 35k finetune 的训练视角：
GS_DIR=~/repos/gaussian-splatting python vggt_human/render_train_views.py \
    -s $RESULTS_DIR/03_source -m $RESULTS_DIR/04_model_3dgs \
    --iteration 30000 --out_dir $RESULTS_DIR/render_baseline --quiet
GS_DIR=~/repos/gaussian-splatting python vggt_human/render_train_views.py \
    -s $RESULTS_DIR/03_source -m $RESULTS_DIR/06b_model_3dgs_face \
    --iteration 35000 --out_dir $RESULTS_DIR/render_face --quiet
#    再跑 face_metrics 对比：
python vggt_human/face_metrics.py \
    --render_a $RESULTS_DIR/render_baseline \
    --render_b $RESULTS_DIR/render_face \
    --ref_dir  $RESULTS_DIR/03_source/images \
    --masks_dir $RESULTS_DIR/06b_face_masks \
    --out $RESULTS_DIR/face_metrics.json

# ── 8) 训练完后：把结果从 Linux fs 剪切到 D:\output（释放 WSL 空间）──
# 预览（不实际移动，默认 DST=/mnt/d/output/vggt_human_results）：
# DRY_RUN=1 bash vggt_human/08_move_output.sh

# 实际搬运（默认 ~/output/vggt_human_results → /mnt/d/output/vggt_human_results）：
bash vggt_human/08_move_output.sh

# 若想搬到别的目录，才需要显式传 DST，例如：
# SRC=~/output/vggt_human_results \
# DST=/mnt/d/output/vggt_human_results \
# bash vggt_human/08_move_output.sh
```

- 结果：VGGT-Omega 推理 → `02_vggt/<scene>/predictions.npz`；COLMAP 场景 → `03_source/`；3DGS 高斯 → `04_model_3dgs/point_cloud/iteration_30000/point_cloud.ply`。跑完 step 08 后全部搬到 `/mnt/d/output/vggt_human_results/`。
- PLY 可拖到 [supersplat](https://playcanvas.com/supersplat/editor) 在线查看。

## 详细文档

- **基线验证实测、增强模块（depth-normal / pose_refine / BA / noise-negating / dynamic mask+filter）WSL 实测数据与结论**：见 [EXPERIMENTS.md](EXPERIMENTS.md)「WSL 基线验证与增强模块实测」。
- **WSL 专属排错**（conda not found、pip 慢、HF 镜像、nvcc、OOM、drvfs 编译慢、vhdx 压缩等 8 条）：见 [NOTES.md](NOTES.md)「可能遇到的问题（WSL 专属）」。
- 通用排错（CUDA 扩展编译、GLM 缺失、simple-knn clone 失败、CRLF 行尾等）同样在 [NOTES.md](NOTES.md)。


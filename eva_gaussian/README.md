# EVA-Gaussian — 3D 高斯实时人体新视角合成

[EVA-Gaussian](https://github.com/zhenliuZJU/EVA-Gaussian)（arXiv:2410.01425，Hu et al., HKUST）从稀疏双目视角（2 个源视角）出发，用一个端到端网络完成：深度估计 → 3D 高斯参数预测 → 可微光栅化 → 特征精修，实时合成任意新视角的人体图像。训练分两阶段：先预训练深度网络（Stage 1，用 GT depth 监督），再端到端训练完整模型（Stage 2，含高斯渲染 + feature refiner）。

本目录只含编排脚本——EVA-Gaussian 官方代码在 `../EVA-Gaussian`（00 自动 clone），无预训练权重（从头训练）。数据集需 THuman2.0 / THumansit 的 GPS-Gaussian 格式渲染数据。独立 conda env（CPython 3.10 + torch 2.5.0+cu118，含 feature-splatting CUDA 光栅化器）。

## 常用命令

> 假设已进入容器（脚本自动激活 `eva_gaussian` env）；`GPU=0` 按需换卡。首次跑前先做下方「首次准备」。训练需 ≥28 GB 显存（batch_size=1 时约 25 GB）。

```bash
# ── 一键：环境准备 → 数据检查 → 预训练深度网络 → 全模型训练 ──
GPU=0 DATA_ROOT=/path/to/thuman2.0_rendered \
  bash eva_gaussian/run_all.sh

# ── 分步 ──
# 0) 环境：clone EVA-Gaussian，装依赖，编 feature-splatting CUDA 扩展
INSTALL_DEPS=1 BUILD_CUDA=1 bash eva_gaussian/00_setup_env.sh
# 1) 数据检查 + 生成 config YAML
GPU=0 DATA_ROOT=/path/to/dataset bash eva_gaussian/01_prepare_data.sh
# 2) Stage 1：预训练深度网络（GT depth 监督）
GPU=0 DATA_ROOT=/path/to/dataset bash eva_gaussian/02_pretrain_depth.sh
# 3) Stage 2：完整 EVA-Gaussian 训练（自动找 Stage 1 的 checkpoint）
GPU=0 DATA_ROOT=/path/to/dataset bash eva_gaussian/03_train.sh

# ── 可选：anchor loss（需 landmark.json）──
# 4a) 生成 landmark（需 mmpose/mmdet/face_recognition/mediapipe）
GPU=0 INSTALL_LANDMARK_DEPS=1 DATA_ROOT=/path/to/dataset \
  bash eva_gaussian/04_gen_landmarks.sh
# 4b) 带 anchor loss 训练
GPU=0 ANCHOR=1 DATA_ROOT=/path/to/dataset bash eva_gaussian/run_all.sh

# ── 自定义 ──
# 少训练步数（快速验证）
GPU=0 NUM_STEPS=50000 DATA_ROOT=... bash eva_gaussian/run_all.sh
# 用高分辨率图片（2048×2048）
GPU=0 USE_HR_IMG=1 DATA_ROOT=... bash eva_gaussian/run_all.sh
# 覆盖源视角 / 新视角 ID
GPU=0 SOURCE_ID=0,1 TRAIN_NOVEL_ID=2,3,4 VAL_NOVEL_ID=3 DATA_ROOT=... \
  bash eva_gaussian/run_all.sh
# 从 checkpoint 续训（Stage 2）
GPU=0 RESUME_CKPT=/path/to/train_latest.pth STAGE1_CKPT=/path/to/s1.pth \
  DATA_ROOT=... bash eva_gaussian/03_train.sh
# 复用已有 cu124 本地 wheel（而非 cu118）
CUDA_TAG=cu124 TORCH_VERSION=2.6.0 TORCHVISION_VERSION=0.21.0 \
  INSTALL_DEPS=1 BUILD_CUDA=1 bash eva_gaussian/00_setup_env.sh
```

- 结果：实验产物在 `$RESULTS_DIR/experiments/<exp_name>/{ckpt,show,logs,file}/`，checkpoint 为 `*_latest.pth` / `*_final.pth`，TensorBoard 日志在 `logs/`，渲染样例在 `show/`。

## 首次准备

```bash
cd <your-code-dir>            # e.g. /data_3d/<uid>/code
git -c http.sslVerify=false clone https://github.com/CrescentVelvet/media_code.git
cd media_code && cp proxy.env.example proxy.env   # 填 http_proxy / https_proxy
# ⚠️ 确认 proxy.env 中 http_proxy / https_proxy 两行已取消注释并填好地址，
#    否则 pip 装依赖会报 "Network is unreachable"

# 1. 建 eva_gaussian env（CPython 3.10，匹配官方 torch 2.5.0+cu118）
conda create -n eva_gaussian python=3.10 -y && conda activate eva_gaussian

# 2. clone EVA-Gaussian 仓，装依赖，编 feature-splatting CUDA 扩展，验证 imports
#    INSTALL_DEPS=1 装 torch cu118 + requirements.txt + triton
#    BUILD_CUDA=1 编 feature-splatting（需 CUDA 11.8 toolkit，cu118 对应）
INSTALL_DEPS=1 BUILD_CUDA=1 bash eva_gaussian/00_setup_env.sh
```

权重目录布局（无预训练权重，从头训；但 torch wheel 可放本地）：
```
$MODEL_DIR/                       # 默认 ../../model
  torch-2.5.0+cu118-cp310-cp310-linux_x86_64.whl    # 可选：手动下 cu118 wheel 放此
  torchvision-0.20.0+cu118-cp310-cp310-linux_x86_64.whl
  # 或复用已有 cu124 wheel（CUDA_TAG=cu124 TORCH_VERSION=2.6.0）：
  torch-2.6.0+cu124-cp310-cp310-linux_x86_64.whl
  torchvision-0.21.0+cu124-cp310-cp310-linux_x86_64.whl
  triton-3.*.whl
  nvidia_*.whl  ...
```

> **公司代理封 download.pytorch.org（403）**：00 优先在 `$MODEL_DIR` 找本地 cu118 wheel；找不到则 fallback 到 pip（可能 403）。如已有 cu124 wheel（wan22_rotate / pi3_3dgs），用 `CUDA_TAG=cu124 TORCH_VERSION=2.6.0` 复用。feature-splatting 光栅化器在 torch 2.6+cu124 下也能编。

> **数据集**：EVA-Gaussian 不提供数据集，需按 [GPS-Gaussian](https://github.com/aipixel/GPS-Gaussian/blob/main/prepare_data/MAKE_DATA.md) 格式用 `prepare_data/render_data.py`（taichi_three）从 THuman2.0 扫描渲染多视角。渲染后的数据放在 `DATA_ROOT/{train,val}/`。

---

以下为详细参考（流程原理 / 各步骤参数 / 排错 / 目录布局）。

## Pipeline（流程详解）

```
DATA_ROOT/{train,val}/  (GPS-Gaussian 格式: img/mask/depth/parm)
    │
    ▼
[01] 数据检查 + 生成 config YAML
    │  ├─ 检查 train/val 下 img/mask/depth/parm 四个子目录
    │  ├─ ANCHOR=1 时检查 landmark.json（否则提示先跑 04）
    │  └─ make_config.py 写 pretrain.yaml + train.yaml 到 $EVA_DIR/config/
    ▼
[02] Stage 1: 预训练深度网络  (depthnet_pretrain.py)
    │  ├─ EVANet(with_gs_render=False) — 纯深度估计（RAFT-style 立体匹配）
    │  ├─ 输入: 2 源视角 (img0, img1) + GT depth + mask + 相机内外参
    │  ├─ loss: flow_loss (立体视差) + anchor_loss (可选)
    │  └─ 输出: experiments/pretrain_MMDD/ckpt/pretrain_latest.pth
    ▼
[03] Stage 2: 完整 EVA-Gaussian 训练  (train.py)
    │  ├─ EVANet(with_gs_render=True) — 深度 + 高斯参数预测 + 光栅化
    │  │     ├─ stereo transformer → 深度图 → 3D 点（depth2pts）
    │  │     ├─ gs_parm_network → 每个高斯的 {位置,颜色,不透明度,尺度,旋转}
    │  │     └─ feature-splatting (diff_gaussian_rasterization) → 渲染图 + 特征图
    │  ├─ feature_refiner (attention UNet) → 精修渲染特征
    │  ├─ loss: depth_loss + L2 + (1-SSIM) + opacity_reg + scale_reg + anchor_loss
    │  └─ 输出: experiments/train_EVAGaussian_MMDD/ckpt/train_EVAGaussian_latest.pth
    ▼
$RESULTS_DIR/experiments/{pretrain_*,train_EVAGaussian_*}/
    ├── ckpt/    # checkpoint (*_latest.pth, *_final.pth)
    ├── show/    # 渲染样例 .jpg (train.py only)
    ├── logs/    # TensorBoard
    └── file/    # config + 脚本备份
```

### 数据集格式（GPS-Gaussian）

数据集需按 GPS-Gaussian 的渲染格式组织（由 `prepare_data/render_data.py` 用 taichi_three 从 THuman2.0 / THumansit 扫描渲染）：

```
$DATA_ROOT/
├── train/
│   ├── img/
│   │   └── <sample_name>/         # e.g. 0004_000
│   │       ├── 0.jpg              # 源视角 0 (1024×1024)
│   │       ├── 1.jpg              # 源视角 1
│   │       ├── 2.jpg              # 新视角 2 (训练用)
│   │       ├── 3.jpg              # 新视角 3
│   │       ├── 4.jpg              # 新视角 4
│   │       └── 2_hr.jpg           # 高分辨率 (2048×2048, USE_HR_IMG=1 时用)
│   ├── mask/
│   │   └── <sample_name>/
│   │       ├── 0.png              # 二值前景 mask
│   │       └── 1.png
│   ├── depth/
│   │   └── <sample_name>/
│   │       ├── 0.png              # GT depth (uint16, value/2^15 = 米)
│   │       └── 1.png
│   └── parm/
│       └── <sample_name>/
│           ├── 0_intrinsic.npy    # 3×3 内参矩阵
│           ├── 0_extrinsic.npy    # 3×4 外参 (world-to-camera)
│           ├── 1_intrinsic.npy
│           └── 1_extrinsic.npy
├── val/                           # 同 train 结构
└── landmark.json                  # 仅 ANCHOR=1 时需要 (04 生成)
```

- **源视角**（`source_id`，默认 `[0, 1]`）：网络输入的双目立体对。
- **新视角**（`train_novel_id`，默认 `[2, 3, 4]`）：训练时合成的目标视角，用光度 loss 监督。
- **验证新视角**（`val_novel_id`，默认 `[3]`）：验证时合成的视角。
- **depth**：16 位 PNG，`cv2.imread(..., IMREAD_UNCHANGED) / 2^15` 得到米制深度。
- **parm**：`.npy` 文件，内参为 3×3 矩阵，外参为 3×4 矩阵（w2c）。

### Stage 1 — 深度网络预训练 (`depthnet_pretrain.py`)

EVANet 的核心是 stereo transformer：从 2 个源视角估计稠密深度图（RAFT-style 立体匹配 + transformer）。Stage 1 只训练深度部分（`with_gs_render=False`），用 GT depth 做监督：

- **loss**：`flow_loss`（立体视差/光流 loss）+ `(10^2) * anchor_loss`（可选，需要 landmark.json）
- **batch_size**：默认 6（源视角对数）
- **num_steps**：默认 100000
- **输出**：`experiments/pretrain_MMDD/ckpt/pretrain_latest.pth`

### Stage 2 — 完整 EVA-Gaussian 训练 (`train.py`)

Stage 2 加载 Stage 1 的 checkpoint（`stage1_ckpt`），开启高斯渲染（`with_gs_render=True`），端到端训练完整 pipeline：

1. **深度估计**：EVANet 的 stereo transformer 从 2 源视角 → 2 深度图
2. **点云生成**：`depth2pts(depth, extrinsic, intrinsic)` → 3D 点云
3. **高斯参数预测**：`gs_parm_network` → 每点的高斯属性（位置/颜色/不透明度/尺度/旋转）
4. **特征光栅化**：`feature-splatting`（修改版 diff-gaussian-rasterization）→ 渲染图 + 特征图
5. **特征精修**：`feature_refiner`（attention UNet）→ 精修渲染图

- **loss**：`depth_loss + 0.8*(L2_temp+L2) + 0.2*(1-SSIM_temp+1-SSIM) + opacity_reg + scale_reg + (10^3)*anchor_loss`
- **batch_size**：默认 2
- **num_steps**：默认 100000
- **eval_freq**：默认 2000 步验证一次
- **输出**：`experiments/train_EVAGaussian_MMDD/ckpt/train_EVAGaussian_latest.pth`

### Anchor Loss（可选）

Anchor loss 用人体关键点（面部 + 手部 landmark）正则化高斯位置，使渲染图的关键点与 GT 对齐。需额外包：mmpose 0.x、mmdet 2.x、mmcv、face_recognition、mediapipe。04 脚本自动安装并运行 `landmark_generation.py` 生成 `landmark.json`。

> **注意**：mmpose 0.x / mmdet 2.x 是旧 API，可能无法在 torch 2.5 下编译。如遇问题，建独立 env：`conda create -n eva_landmark python=3.8` + `torch==1.13.1+cu117`。

## Config (env vars, all optional)

### Paths & envs
| var | default | note |
| --- | --- | --- |
| `DATA_ROOT` | _(required)_ | 数据集根目录（GPS-Gaussian 格式，含 train/ val/） |
| `CONDA_ENV` | `eva_gaussian` | conda env（CPython 3.10 + torch cu118） |
| `GPU` | _(unset)_ | 物理 GPU id，e.g. `GPU=0` |
| `EVA_DIR` | `../EVA-Gaussian` | 官方代码目录（00 clone） |
| `MODEL_DIR` | `../../model` | 共享模型根（本地 torch wheel 放此） |
| `RESULTS_DIR` | `../eva_gaussian_results` | 输出根（experiments/ 在其下） |
| `CUDA_HOME` | `/usr/local/cuda` | CUDA toolkit 根（编 feature-splatting 用；须 cu118） |

### Torch install (00 only)
| var | default | note |
| --- | --- | --- |
| `CUDA_TAG` | `cu118` | torch CUDA build tag；`cu124` 复用已有 wheel |
| `TORCH_VERSION` | `2.5.0` | torch 版本；`2.6.0` 对应 cu124 |
| `TORCHVISION_VERSION` | `0.20.0` | 匹配 torchvision |
| `INSTALL_DEPS` | `0` | `1` = 安装 torch + requirements.txt + triton |
| `BUILD_CUDA` | `0` | `1` = 编 feature-splatting CUDA 扩展 |

### Training params (02 / 03)
| var | pretrain default | train default | note |
| --- | --- | --- | --- |
| `LR` | `0.0002` | `0.0005` | 学习率 |
| `WDECAY` | `1e-5` | `1e-5` | weight decay |
| `BATCH_SIZE` | `6` | `2` | batch size（源视角对数） |
| `NUM_STEPS` | `100000` | `100000` | 训练步数 |
| `LOSS_FREQ` | `200` | `200` | 每多少步记 loss + 存 ckpt |
| `EVAL_FREQ` | `5000` | `2000` | 每多少步验证 |
| `ANCHOR` | `0` | `0` | `1` = 启用 anchor loss（需 landmark.json） |
| `USE_HR_IMG` | `0` | `0` | `1` = 用 `_hr.jpg` 高分辨率图（2048×2048） |
| `SOURCE_ID` | `0,1` | `0,1` | 源视角 ID |
| `TRAIN_NOVEL_ID` | — | `2,3,4` | 训练新视角 ID |
| `VAL_NOVEL_ID` | — | `3` | 验证新视角 ID |
| `STAGE1_CKPT` | — | _(auto)_ | Stage 1 checkpoint 路径（不设则自动找） |
| `RESUME_CKPT` | — | — | 从 checkpoint 续训（02 或 03） |

## 可能遇到的问题

**1. `00` 报 `pip install torch` 失败（403 / Network unreachable）**
公司代理封 download.pytorch.org。两个方案：
- 手动下 cu118 wheel 放 `$MODEL_DIR/`：`torch-2.5.0+cu118-cp310-cp310-linux_x86_64.whl` + `torchvision-0.20.0+cu118-...whl`，00 自动找本地 wheel 安装。
- 复用已有 cu124 wheel：`CUDA_TAG=cu124 TORCH_VERSION=2.6.0 TORCHVISION_VERSION=0.21.0 INSTALL_DEPS=1 BUILD_CUDA=1 bash eva_gaussian/00_setup_env.sh`（feature-splatting 在 cu124 下也能编）。

**2. `00` 编 feature-splatting 报 `nvcc not found` / `CUDA_HOME` 错**
feature-splatting 是修改版 diff-gaussian-rasterization，用 `torch.utils.cpp_extension` 编 CUDA，需 `nvcc`：
- `torch==2.5.0+cu118` → 装 CUDA 11.8 toolkit，`export CUDA_HOME=/usr/local/cuda-11.8`
- `torch==2.6.0+cu124` → 装 CUDA 12.4 toolkit，`export CUDA_HOME=/usr/local/cuda-12.4`
- 验证：`$CUDA_HOME/bin/nvcc --version`
- 版本不匹配会编不过或运行时段错。

**3. `00` 编 feature-splatting 报 `error: no member named '...' in 'at::...'`**
torch 版本和 rasterizer 代码不兼容。`git -C $EVA_DIR pull` 拉最新版重编。或检查 CUDA_HOME 是否指向正确版本 toolkit。

**4. `01` 报数据集结构不完整**
确保 `$DATA_ROOT/{train,val}/{img,mask,depth,parm}/` 都存在且有子目录。数据需用 GPS-Gaussian 的 `prepare_data/render_data.py` 从 THuman2.0 扫描渲染。

**5. `02` / `03` 报 `No module named 'diff_gaussian_rasterization'`**
feature-splatting CUDA 扩展没编。`BUILD_CUDA=1 bash eva_gaussian/00_setup_env.sh`（需 `CUDA_HOME` 指向匹配的 CUDA toolkit）。

**6. `03` 报找不到 Stage 1 checkpoint**
确认 Stage 1 已完成：`find $RESULTS_DIR/experiments -name 'pretrain_*_latest.pth'`。手动指定：`STAGE1_CKPT=/path/to/pretrain_latest.pth bash eva_gaussian/03_train.sh`。

**7. 训练 OOM（显存不足）**
EVA-Gaussian 需 ~25 GB（batch_size=1）。降压：
- 减 batch：`BATCH_SIZE=1`
- 减步数：`NUM_STEPS=50000`
- 不用高分辨率：确保 `USE_HR_IMG=0`
- `export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`

**8. `04` 安装 mmpose/mmdet 失败**
mmpose 0.x / mmdet 2.x 是旧 API，可能和 torch 2.5 不兼容。建独立 env：
```bash
conda create -n eva_landmark python=3.8 -y && conda activate eva_landmark
pip install torch==1.13.1+cu117 --index-url https://download.pytorch.org/whl/cu117
pip install mmcv-full==1.7.1 mmdet==2.28.0 mmpose==0.29.0 xtcocotools face_recognition mediapipe
# 然后在 eva_landmark env 下运行 04（改 CONDA_ENV）
CONDA_ENV=eva_landmark GPU=0 INSTALL_LANDMARK_DEPS=0 DATA_ROOT=... bash eva_gaussian/04_gen_landmarks.sh
```

**9. `04` 的 landmark_generation.py 有语法错误**
官方代码 `for file in ['0.jpg', '1.jpg']` 缺冒号。04 脚本自动用 sed 修复。如仍报错，手动编辑 `$EVA_DIR/landmark_generation.py` 加冒号。

**10. 跑 `.sh` 报 `syntax error near unexpected token ('`**
CRLF 行尾污染。`find eva_gaussian -name '*.sh' -exec sed -i 's/\r$//' {} +`（`.gitattributes` 强制 LF）。

**11. NumPy 坏了 / `import numpy` 报 ABI 不兼容 / python 变成了 GraalPy**
`conda install` 任何包都加 `--no-update-deps`，否则 conda solver 可能把 python 掉包成 GraalPy。本仓 00 不用 `conda install` 装其他包（只用 pip），且建 env 时显式校验 CPython。若已坏，重建：
```bash
python -c "import platform; print(platform.python_implementation())"  # GraalPy 即中招
conda env remove -n eva_gaussian
conda create -n eva_gaussian python=3.10 -y && conda activate eva_gaussian
INSTALL_DEPS=1 BUILD_CUDA=1 bash eva_gaussian/00_setup_env.sh
```

## 目录布局
```
<code-dir>/
├── media_code/                  # 本仓
│   ├── proxy.env                # 代理 + 覆盖项, gitignored
│   └── eva_gaussian/            # ← 本目录（编排脚本）
│       ├── _env.sh
│       ├── 00_setup_env.sh
│       ├── make_config.py        #   生成 pretrain.yaml / train.yaml
│       ├── 01_prepare_data.sh
│       ├── 02_pretrain_depth.sh
│       ├── 03_train.sh
│       ├── 04_gen_landmarks.sh
│       └── run_all.sh
├── EVA-Gaussian/                # 官方代码 (00 clone)
│   ├── depthnet_pretrain.py     #   Stage 1 训练入口
│   ├── train.py                  #   Stage 2 训练入口
│   ├── landmark_generation.py    #   anchor loss landmark 生成
│   ├── config/                   #   YAML 配置 (make_config.py 写入)
│   │   ├── pretrain.yaml
│   │   └── train.yaml
│   ├── lib/                      #   模型 + 数据 + 工具
│   │   ├── network_transformer.py  # EVANet (stereo transformer + gs render)
│   │   ├── gs_parm_network.py       # 高斯参数预测网络
│   │   ├── attention_unet.py        # feature_refiner
│   │   ├── GaussianRender.py        # pts2render_feature (特征光栅化)
│   │   ├── human_depth_loader.py    # DepthHumanDataset (GPS-Gaussian 格式)
│   │   ├── loss.py                  # l1/l2/ssim/psnr/opacity/scale/anchor loss
│   │   └── train_recoder.py         # Logger + file_backup
│   ├── feature-splatting/        #   修改版 diff-gaussian-rasterization (CUDA)
│   │   ├── cuda_rasterizer/        #   CUDA 核函数
│   │   ├── diff_gaussian_rasterization/  # Python binding
│   │   └── setup.py               #   pip install -e 编译
│   └── prepare_data/             #   数据渲染工具 (taichi_three)
│       ├── render_data.py          #   从 THuman2.0 扫描渲染多视角
│       └── taichi_three/           #   taichi 3D 渲染库
└── eva_gaussian_results/         # 本流程输出
    └── experiments/
        ├── pretrain_MMDD/         #   Stage 1 产物
        │   ├── ckpt/pretrain_latest.pth
        │   ├── logs/               #   TensorBoard
        │   └── file/              #   config 备份
        └── train_EVAGaussian_MMDD/  # Stage 2 产物
            ├── ckpt/train_EVAGaussian_latest.pth
            ├── show/              #   渲染样例 .jpg
            ├── logs/              #   TensorBoard
            └── file/             #   config 备份
```

## Notes
- 官方代码遵循其 license（研究用途）。本目录只编排；未拷贝官方代码。
- 无预训练权重——EVA-Gaussian 从头训练。Stage 1 的 checkpoint 是后续 Stage 2 的初始化。
- `.gitattributes`（仓根）强制 LF，让 Windows 推送的脚本在 Ubuntu 上干净运行。
- `proxy.env`（代理凭证 + 路径 / env 覆盖）在仓内 gitignored——从不入库。
- 公司 TLS 中间人代理下：pip 用 `--trusted-host` / `PIP_CERT`；`git` 用 `GIT_SSL_CAINFO`；`curl` 用 `CURL_CA_BUNDLE`（`_env.sh` 优先用 `~/.ca-bundle.crt`）。

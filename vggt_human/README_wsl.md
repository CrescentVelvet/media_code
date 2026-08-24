# VGGT-Human (WSL Ubuntu 24.04) — 本地复现指南

本文件是 [`README.md`](README.md) 的 **WSL 本地复现补充**。服务器流程（clone `doll` env、公司代理、CA 证书）见原 README，本文件只讲 WSL 上跑的差异。

> 两个 README 共存：
> - [`README.md`](README.md) — 服务器流程（`doll` env clone + 公司代理 + `/usr/local/cuda`）
> - [`README_wsl.md`](README_wsl.md) — 本文件（WSL 从零建 env + 无代理 + conda 装 CUDA toolkit）

## 与服务器版的核心差异

| | 服务器 (`00_setup_env.sh`) | WSL (`00a_setup_env.sh`) |
|---|---|---|
| conda env | cp -a clone `doll`（含 torch） | conda create -n vggt_human python=3.10 + pip install torch cu121 |
| CUDA toolkit (nvcc) | 系统 `/usr/local/cuda-12.x` | conda install cuda-toolkit（env 内，无需 sudo） |
| gcc | conda gxx_linux-64=12 | 同（conda gxx_linux-64=12） |
| 代理 / CA | proxy.env 填公司代理密码 + CA bundle | 不需要（直连互联网） |
| 仓库位置 | `/mnt/c/code/`（与 media_code 同级） | `~/repos/`（Linux 文件系统，编译快） |
| 权重 / 输出 | `/mnt/c/code/model/` | `~/model/`、`~/output/vggt_human_results/`（Linux fs，训练 I/O 快） |

> **为什么仓库 / 权重放 Linux 文件系统？** `/mnt/c` 是 Windows drvfs（9p 协议），文件 I/O 慢 5-10 倍。编译 CUDA 扩展 + 训练时大量写文件，放 Linux fs 避免超时和性能问题。路径通过 `proxy.env` 自动覆盖，脚本 01-07 不需改。

## 前提条件

1. **WSL Ubuntu 24.04** 已安装并运行（`wsl -d Ubuntu2404`）
2. **NVIDIA 驱动** 在 Windows 上装好（WSL 内 `nvidia-smi` 能看到 GPU）
3. **Miniconda** 已安装：
   ```bash
   # 如果还没装 miniconda：
   curl -fsSL https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -o /tmp/miniconda.sh
   bash /tmp/miniconda.sh -b -p ~/miniconda3
   ```
4. **磁盘空间** ≥ 30GB（torch + CUDA toolkit + 权重 + 仓库）

## 首次准备

```bash
# 进入 WSL
wsl -d Ubuntu2404

# 进入 media_code 目录
cd /mnt/c/code/media_code

# 如果还没有 proxy.env，00a 会自动生成一个（含 WSL 路径覆盖）
# 如果已有 proxy.env（服务器用的），建议备份后让 00a 重新生成：
#   mv proxy.env proxy.env.server.bak

# 一键安装（创建 env + 装 torch + CUDA toolkit + gcc + 依赖 + clone 仓 + 编 CUDA 扩展）
bash vggt_human/00a_setup_env.sh

# 如果还需去噪模型（step 05 的 DiffBIR / SwinIR）：
# INSTALL_DENOISER=1 bash vggt_human/00a_setup_env.sh
```

安装完成后：
1. `source ~/.bashrc`（让 conda 在你的 shell 里可用）
2. 编辑 `proxy.env` 填入 `HF_TOKEN`（VGGT-Omega 权重是 gated 仓库）：
   ```bash
   # 1. 申请访问：https://huggingface.co/facebook/VGGT-Omega（自动审核）
   # 2. 创建 token：https://huggingface.co/settings/tokens
   # 3. 填入 proxy.env：
   nano /mnt/c/code/media_code/proxy.env
   # 取消注释 export HF_TOKEN="hf_xxx" 并填入你的 token
   ```
3. 下载 VGGT-Omega 权重：
   ```bash
   GPU=0 VARIANT=1b_512 bash vggt-omega/01_download_models.sh
   ```

### HYPIR 人脸增强（step 01/06）— 遗留问题

00a 默认会 clone HYPIR + 下载 SD2 base model，但这两个依赖 **huggingface.co 可达**。如果 WSL 内连不上 huggingface.co（443 超时），或 HYPIR 的 `beauty_ppr50k` LoRA 权重尚未从服务器拷过来，可以 **跳过 step 01/06**，直接从 step 02 开始：

```bash
# 安装时跳过 HYPIR（不 clone、不下 SD2 base）
SKIP_HYPIR=1 bash vggt_human/00a_setup_env.sh

# 流程跳过 step 01，直接用原始图像喂 step 02
GPU=0 \
INPUT_DIR=~/my_images \
MODEL_DIR=~/model/VGGT-Omega \
RESULTS_DIR=~/output/vggt_human_results \
bash vggt_human/02_run_inference.sh
```

准备好后再开启 HYPIR：
1. 确保 `huggingface.co` 可达（或用镜像 `HF_ENDPOINT=https://hf-mirror.com`）
2. 从服务器拷 `beauty_ppr50k` LoRA 权重到 `$HYPIR_DIR/experiments/`
3. 下载 SD2 base model：`git clone https://huggingface.co/stabilityai/stable-diffusion-2-base ~/model/HYPIR/sd2_base`
4. 跑 step 01 对原始图做人脸增强

## 路径布局（WSL）

```
~/                                # Linux 文件系统
├── repos/                        # 官方仓库（00a clone，Linux fs 编译快）
│   ├── vggt-omega/               # VGGT-Omega 官方代码
│   ├── gaussian-splatting/       # 原版 3DGS
│   │   ├── submodules/
│   │   │   ├── diff-gaussian-rasterization/  # CUDA 扩展（dr_aa 分支）
│   │   │   └── simple-knn/                    # CUDA 扩展
│   │   └── third_party/glm/
│   ├── HYPIR/                    # 人脸增强
│   ├── DiffBIR/                  # 去噪（INSTALL_DENOISER=1）
│   └── SwinIR/                   # 去噪（INSTALL_DENOISER=1）
├── model/                        # 权重（Linux fs，训练 I/O 快）
│   ├── VGGT-Omega/               # vggt_omega_1b_512.pt（gated HF 下载）
│   └── HYPIR/
│       └── sd2_base/             # SD2 base model（00a 自动 clone）
├── output/
│   └── vggt_human_results/      # 输出（训练结果、中间产物）
└── miniconda3/                   # conda 安装
    └── envs/vggt_human/          # conda env（python=3.10 + torch cu121 + CUDA toolkit）

/mnt/c/code/media_code/           # 编排脚本（Windows fs，git 仓）
├── proxy.env                     # WSL 路径覆盖 + HF_TOKEN（00a 自动生成）
└── vggt_human/
    ├── 00a_setup_env.sh          # ← WSL 安装脚本（本文件描述）
    ├── _env.sh                   # 共享环境（已加 conda fallback，兼容 WSL）
    ├── 00_setup_env.sh           # 服务器安装脚本（保留不动）
    ├── 01_face_enhance.sh        # 以下脚本与服务器共用，通过 proxy.env 自动用 WSL 路径
    └── ...
```

## 常用命令

> 与服务器版完全一致——`proxy.env` 已写入 WSL 路径覆盖，脚本 01-07 自动用 `~/repos/`、`~/model/` 等路径。
> 每条命令需显式写出输入路径、输出路径、模型路径。

```bash
# 在 WSL 内，进入 media_code 目录
cd /mnt/c/code/media_code

# 0) 一键安装（首次）
bash vggt_human/00a_setup_env.sh

# 1a) 视频 → 图像夹（可选，视频输入用）
GPU=0 \
INPUT_DIR=~/my_video.mp4 \
OUTPUT_DIR=~/output/vggt_human_results/input_frames/my_video \
VIDEO_FPS=2 \
bash vggt_human/01a_video_to_frames.sh

# 1) 前处理人脸增强
GPU=0 \
INPUT_DIR=~/my_images \
RESULTS_DIR=~/output/vggt_human_results \
bash vggt_human/01_face_enhance.sh

# 2) VGGT-Omega 前馈推理（需 HF_TOKEN + 权重已下载）
GPU=0 \
INPUT_DIR=~/output/vggt_human_results/input_face \
MODEL_DIR=~/model/VGGT-Omega \
RESULTS_DIR=~/output/vggt_human_results \
MAX_POINTS=2000000 \
bash vggt_human/02_run_inference.sh

# 3) npz → COLMAP 转换
GPU=0 \
TARGET_POINTS=200000 \
POSE_ADJUST=1 \
POSE_REFINE=1 \
RESULTS_DIR=~/output/vggt_human_results \
bash vggt_human/03_npz_to_colmap.sh

# 4) 原版 3DGS 训练 + 渲染
GPU=0 \
ITERATIONS=30000 \
WHITE_BG=0 \
POSE_ADJUST=1 \
POSE_REFINE=1 \
ENABLE_DYNAMIC_MASK=1 \
ENABLE_DYNAMIC_FILTER=1 \
ENABLE_MLP_DYNAMIC=1 \
USE_DEPTH_NORMAL=1 \
RESULTS_DIR=~/output/vggt_human_results \
bash vggt_human/04_train_3dgs.sh

# 5) 渲染新视角 → 去噪 → 增强 COLMAP（可选）
GPU=0 \
DENOISER=diffbir \
NUM_NOVEL_VIEWS=10 \
ITERATION=30000 \
RESULTS_DIR=~/output/vggt_human_results \
bash vggt_human/05_denoise_novel.sh

# 6) 后处理人脸增强（可选）
GPU=0 \
RESULTS_DIR=~/output/vggt_human_results \
bash vggt_human/06_face_enhance.sh

# 7) 增强场景训练 3DGS（可选）
GPU=0 \
ITERATION=30000 \
WHITE_BG=0 \
POSE_ADJUST=1 \
POSE_REFINE=1 \
ENABLE_DYNAMIC_MASK=1 \
ENABLE_DYNAMIC_FILTER=1 \
ENABLE_MLP_DYNAMIC=1 \
USE_DEPTH_NORMAL=1 \
RESULTS_DIR=~/output/vggt_human_results \
bash vggt_human/07_train_denoise.sh
```

## 可能遇到的问题（WSL 专属）

**1. `conda: command not found`（运行 01-07 脚本时）**
00a 末尾会执行 `conda init bash`。如果你还没 `source ~/.bashrc`，conda 不在 PATH 上。解决：
```bash
source ~/.bashrc
# 或者每次手动 source：
source ~/miniconda3/etc/profile.d/conda.sh
```
`_env.sh` 已加 fallback：如果 conda 不在 PATH，会自动找 `~/miniconda3`。但仍建议 `source ~/.bashrc` 确保万无一失。

**2. CUDA 扩展编译失败（`nvcc: command not found` 或 CUDA 版本不匹配）**
00a 通过 conda 装 `cuda-toolkit`（nvcc 在 `$CONDA_PREFIX/bin/nvcc`）。如果失败：
```bash
# 检查 nvcc 是否可用
conda activate vggt_human
which nvcc
nvcc --version

# 如果没有，手动装：
conda install -y -c nvidia/label/cuda-12.1.1 cuda-toolkit

# 如果 cu121 的 torch 不兼容你的 GPU，换 cu124：
conda install -y -c nvidia/label/cuda-12.4.0 cuda-toolkit
pip install --force-reinstall --index-url https://download.pytorch.org/whl/cu124 torch torchvision
```

**3. `torch.cuda.OutOfMemoryError`（step 02 VGGT-Omega 推理）**
RTX 3090 有 24GB 显存，但 VGGT-Omega 1B 模型对帧数敏感。降压：
```bash
RESOLUTION=256 MODE=max_size GPU=0 ... bash vggt_human/02_run_inference.sh
```

**4. `git clone` 超时 / SSL 错误**
WSL 直连互联网通常无此问题。如果遇到（网络不稳定），脚本已内置 `--ssl-verify=false` fallback。手动重试：
```bash
cd ~/repos && git clone https://github.com/graphdeco-inria/gaussian-splatting.git
```

**5. 编译极慢（超过 30 分钟）**
确认仓库在 Linux 文件系统（`~/repos/`）而非 `/mnt/c/`。如果误 clone 到 `/mnt/c/`：
```bash
mv /mnt/c/code/gaussian-splatting ~/repos/
# 更新 proxy.env 里的路径
```

**6. GPU 驱动版本太旧**
WSL 内 `nvidia-smi` 显示的 CUDA Version 是驱动支持的最高版本。torch cu121 需要驱动 ≥ 525.60。如果驱动太旧，更新 Windows 上的 NVIDIA 驱动。

**7. 想切回 cu124 / cu126**
```bash
CUDA_TOOLKIT_LABEL=nvidia/label/cuda-12.4.0 \
TORCH_INDEX_URL=https://download.pytorch.org/whl/cu124 \
bash vggt_human/00a_setup_env.sh
```

## 通用问题

其他问题（CUDA 扩展编译、GLM 缺失、simple-knn clone 失败、CRLF 行尾、mediapipe 未装等）见 [`README.md` 的「可能遇到的问题」](README.md#可能遇到的问题) 段，排错方法通用。

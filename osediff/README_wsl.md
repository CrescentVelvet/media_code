# OSEDiff (WSL Ubuntu 24.04) — 本地复现指南

本文件是 [`README.md`](README.md) 的 WSL 本地复现版。照着做即可从零跑完全流程。
OSEDiff = One-Step Effective Diffusion Network for Real-World Image Super-Resolution
（NeurIPS 2024，港理工+OPPO，[arXiv:2406.08177](https://arxiv.org/abs/2406.08177)）。

## 与服务器版的核心差异
| | 服务器 | WSL |
|---|---|---|
| conda env | 已存在（doll 等） | `conda create -n osediff` 从零建 |
| CUDA toolkit | 系统 `/usr/local/cuda` | conda `cuda-nvcc`（仅 xformers 源码编译时才需要） |
| pip 源 | 默认 PyPI | 清华镜像 `pypi.tuna.tsinghua.edu.cn` |
| HF 源 | 直连 huggingface.co | `hf-mirror.com` 镜像 |
| 仓库位置 | `/mnt/c/code/...` | `~/repos/OSEDiff`（Linux fs，I/O 快） |
| 权重位置 | `../../model/osediff` | `~/model/osediff`（Linux fs） |
| 输出 | `../osediff_results` | `~/output/osediff_results`，跑完 `08` 搬到 `/mnt/d/output/` |
| 代理 | 公司 proxy.env | 无（家用直连） |

## Windows 路径 → WSL 路径
| Windows | WSL |
|---|---|
| `D:\dataset\sample` | `/mnt/d/dataset/sample` |
| `C:\code\media_code` | `/mnt/c/code/media_code`（本仓就在这） |

## 前提条件
- WSL2 + Ubuntu 24.04（已装：`wsl -l -v` 看到 Ubuntu2404 Running）
- NVIDIA Windows 驱动（WSL 内 `nvidia-smi` 能看到 RTX 3090）
- Miniconda 装在 `~/miniconda3`（没有则见下方「首次准备」）

## 首次准备

### 0. 进 WSL + 装 Miniconda（如未装）
```bash
# 在 Windows PowerShell 里进 WSL
wsl -d Ubuntu2404
# 仓里已有 ~/miniconda3 可跳过；否则：
curl -fsSL https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -o /tmp/miniconda.sh
bash /tmp/miniconda.sh -b -p ~/miniconda3
```

### 1. 进本仓
```bash
cd /mnt/c/code/media_code   # 本仓在 Windows 盘（只读编排脚本，不在 Linux fs 也行）
```

### 2. 建环境 + clone 官方 repo + 装依赖
```bash
# 00a 会：写 proxy.env（WSL 路径覆盖 + 清华镜像 + hf-mirror）
#         -> conda create -n osediff python=3.10
#         -> pip install torch==2.0.1 (cu118) + xformers==0.0.20
#         -> pip install -r requirements.txt（清华镜像）
#         -> clone https://github.com/cswry/OSEDiff -> ~/repos/OSEDiff
bash osediff/00a_setup_env.sh
# 装完 source 一下让 conda 进当前 shell
source ~/.bashrc
conda activate osediff
```
> 首次跑约 10–15 分钟（下 torch ~2GB + 其余依赖）。若 `xformers` 预编译 wheel
> 装不上，00a 会自动 conda 装 `cuda-nvcc` + `gxx_linux-64=12` 再源码编译。

### 3. 下权重（SD2.1-Base ~5GB + RAM ~1.7GB）
```bash
bash osediff/01_download_models.sh
```
走 `hf-mirror.com` 镜像，约 6.7GB。`DAPE.pth` 从 repo 自带的 `preset/models/DAPE.pth`
复制（若推理报维度不对，再 gdown 完整版，见服务器 README「可能遇到的问题」）。

## 全流程命令

### 推理（02）
```bash
# A. 默认 preset 测试图 x4 超分（最快验证）
GPU=0 bash osediff/02_run_inference.sh
# 结果在 ~/output/osediff_results/output/

# B. 自己的图（输入放 Windows 盘，输出写 Linux fs 再搬）
GPU=0 INPUT_DIR=/mnt/d/dataset/my_lq \
  OUTPUT_DIR=~/output/osediff_results/my_out \
  bash osediff/02_run_inference.sh

# C. 人脸修复（x1）
GPU=0 MODE=face INPUT_DIR=/mnt/d/dataset/my_face \
  bash osediff/02_run_inference.sh

# 跑完搬到 D 盘
bash osediff/08_move_output.sh
```

### LoRA 训练（04，单卡 24GB）
```bash
# 1. 数据集 txt（每行一张图绝对路径）。LSDIR/FFHQ 放 Windows 盘只读：
find /mnt/d/dataset/LSDIR -type f \( -name '*.png' -o -name '*.jpg' \) > ~/lsdir.txt
# 2. 训练（bs=1 accum=4 + fp16 + xformers + checkpointing + LoRA r=4）
GPU=0 DATASET_TXT=~/lsdir.txt \
  bash osediff/04_train_lora.sh
# 训练产物在 ~/output/osediff_experiments/exp1/

# 3. 用自己训的 LoRA 推理
GPU=0 OSEDIFF_PKL=~/output/osediff_experiments/exp1/osediff.pkl \
  INPUT_DIR=/mnt/d/dataset/my_lq \
  bash osediff/02_run_inference.sh

# 4. 训练完搬到 D 盘（释放 vhdx）
bash osediff/08_move_output.sh
```

### 一键（00a + 01 + 02）
```bash
GPU=0 bash osediff/run_all.sh
```

## 可能遇到的问题（WSL 专属）

**1. `conda: command not found`（交互 shell）**
00a 跑了 `conda init bash`，但当前 shell 没生效。`source ~/.bashrc` 或重开终端。
非交互脚本（01/02/04）不受影响——`_env.sh` 有 fallback 自动 source `~/miniconda3`。

**2. `xformers` 装不上（预编译 wheel 缺）**
torch 2.0.1 + xformers 0.0.20 的 cu118 预编译 wheel 偶尔 PyPI 没有。00a 会自动
切到源码编译分支：`conda install cuda-nvcc + gxx_linux-64=12` 然后
`pip install --no-build-isolation xformers==0.0.20`。编译约 3–5 分钟。若仍失败，
推理可不装 xformers（变慢但能跑）；训练必须装，否则 OOM。

**3. `hf download` 卡 / SSL 错（hf-mirror 偶发）**
01 脚本有 curl `--insecure` fallback 逐文件拉。仍失败手动：
```bash
curl -L https://hf-mirror.com/Manojb/stable-diffusion-2-1-base/resolve/main/unet/diffusion_pytorch_model.safetensors \
  -o ~/model/osediff/sd21_base/unet/diffusion_pytorch_model.safetensors
```

**4. 训练 OOM（24GB）**
默认 `bs=1 accum=4 + fp16 + xformers + checkpointing` 在 3090 可跑。若 OOM：
- 确认 `xformers` 已装（`python -c "import xformers.ops"`）
- 确认 `--mixed_precision=fp16`（默认）
- 训练分辨率别超 512（官方默认）

**5. WSL vhdx 空间不足**
训练 checkpoint 多了 vhdx 撑爆。跑完 `08_move_output.sh` 把结果剪到
`/mnt/d/output/`。vhdx 不缩时 Windows PowerShell：`wsl --shutdown` 再
`diskpart` 的 `compact vdisk`。

**6. 在 `/mnt/c` 下跑训练慢**
`/mnt/c` 是 drvfs（9p），I/O 比 Linux fs 慢 5–10×。**仓库和权重务必在 Linux fs**
（`~/repos/`、`~/model/`、`~/output/`，00a 已自动配好）。输入图可放 `/mnt/d/`
只读无妨。

## 目录布局（WSL）
```
~/
├── repos/OSEDiff/                  # 官方代码（00a clone，Linux fs）
│   └── preset/models/osediff.pkl   # 训练权重（repo 自带）
├── model/osediff/                   # 权重（Linux fs）
│   ├── sd21_base/
│   ├── ram_swin_large_14m.pth
│   └── DAPE.pth
├── output/osediff_results/          # 推理输出（Linux fs）
└── output/osediff_experiments/      # 训练产物（Linux fs）
/mnt/d/output/osediff_results/        # 08 搬运后的最终结果（Windows 盘）
```

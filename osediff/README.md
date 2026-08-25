# OSEDiff runner — One-Step Effective Diffusion for Real-World Image Super-Resolution

一步扩散真实世界图像超分辨率（NeurIPS 2024，港理工+OPPO）。在 NVIDIA 服务器上跑
推理 / LoRA 训练。本目录只含编排脚本——官方代码自动 clone、权重从本地 `$MODEL_DIR` 加载。

> **WSL 本机复现** 见 [`README_wsl.md`](README_wsl.md)（与本文并列，不删旧）。

## 常用命令
> 假设已进入容器并 `conda activate osediff`；`GPU=0` 按需换卡。首次跑前先做下方「首次准备」。
> **铁律：每条命令都显式写出模型/输入/输出路径，不能只靠脚本默认值。**

```bash
# ── 一键（环境 + 下权重 + 推理）──
GPU=0 bash osediff/run_all.sh

# ── 推理(02) ──
# 1) 默认 preset 测试图 x4 超分
GPU=0 bash osediff/02_run_inference.sh
# 2) 自己的图
GPU=0 INPUT_DIR=/path/to/lq OUTPUT_DIR=/path/to/out bash osediff/02_run_inference.sh
# 3) 人脸修复（x1，用 osediff_face.pkl + codeformer 退化）
GPU=0 MODE=face INPUT_DIR=/path/to/face bash osediff/02_run_inference.sh

# ── 下权重(01) ──
GPU=0 bash osediff/01_download_models.sh
# 分项跳过：
GPU=0 SKIP_SD=1 bash osediff/01_download_models.sh   # 只下 RAM + DAPE
GPU=0 SKIP_RAM=1 bash osediff/01_download_models.sh  # 只下 SD2.1 + DAPE

# ── LoRA 训练(04) ──
# 1) 准备数据 txt（每行一张图的绝对路径）
find /path/to/LSDIR -type f \( -name '*.png' -o -name '*.jpg' \) > lsdir.txt
# 2) 单卡训练（bs=1 accum=4 ≈ 论文 bs=4，24GB 够）
GPU=0 DATASET_TXT=$PWD/lsdir.txt \
  bash osediff/04_train_lora.sh
# 3) 双数据集混合（LSDIR + FFHQ）
GPU=0 DATASET_TXT=$PWD/lsdir.txt DATASET_TXT2=$PWD/ffhq.txt \
  bash osediff/04_train_lora.sh
# 4) 人脸修复训练
GPU=0 MODE=face DATASET_TXT=$PWD/ffhq.txt bash osediff/04_train_lora.sh
```

- 结果：推理 → `../osediff_results/output/`；训练 → `../osediff_experiments/exp1/`。
- 训练曲线：`tensorboard --logdir ../osediff_experiments/exp1`。

## 首次准备
```bash
cd <your-code-dir>
git -c http.sslVerify=false clone https://github.com/CrescentVelvet/media_code.git
cd media_code && cp proxy.env.example proxy.env
# ⚠️ 确认 proxy.env 中 http_proxy / https_proxy 已取消注释

# 建 env + 装依赖（服务器假设已有 CUDA torch；只补 diffusers/xformers 等）
INSTALL_DEPS=1 bash osediff/00_setup_env.sh
# 00 会自动 clone 官方 repo -> ../OSEDiff

# 下权重（SD2.1-Base ~5GB + RAM ~1.7GB + DAPE 从 repo 复制）
bash osediff/01_download_models.sh
```

权重目录布局：
```
$MODEL_DIR/                        # = ../../model/osediff
├── sd21_base/                     # SD2.1-Base (HF: Manojb/stable-diffusion-2-1-base)
│   ├── model_index.json
│   ├── scheduler/
│   ├── tokenizer/
│   ├── text_encoder/{config.json,model.safetensors}
│   ├── unet/{config.json,diffusion_pytorch_model.safetensors}
│   └── vae/{config.json,diffusion_pytorch_model.safetensors}
├── ram_swin_large_14m.pth         # RAM (HF: xinyu1205/recognize-anything)
└── DAPE.pth                        # RAM fine-tuned (repo 自带 preset/models/DAPE.pth 复制)

$OSEDIFF_DIR/preset/models/        # = ../OSEDiff/preset/models（repo 自带，clone 即有）
├── osediff.pkl                    # 训练好的 SR 权重 (20MB)
└── osediff_face.pkl               # 人脸修复权重 (18MB)
```

---

以下为详细参考。

## Pipeline
```
首次:  [00] clone repo + 装依赖  ->  [01] 下 SD2.1+RAM+DAPE
推理:  输入图  ->  [02] test_osediff.py (SD2.1+RAM+DAPE+osediff.pkl)  ->  SR 图
训练:  图集 txt  ->  [04] accelerate launch train_osediff.py (VSD+LPIPS+L2+LoRA)  ->  osediff.pkl
```

## Config (env vars)
| var | default | note |
|---|---|---|
| `GPU` | (none) | 物理卡号，映射到 `CUDA_VISIBLE_DEVICES` |
| `CONDA_ENV` | `osediff` | conda env 名 |
| `OSEDIFF_DIR` | `$REPO_DIR/../OSEDiff` | 官方代码路径 |
| `MODEL_DIR` | `$REPO_DIR/../../model/osediff` | 权重根 |
| `RESULTS_DIR` | `$REPO_DIR/../osediff_results` | 推理输出 |
| `EXPERIMENTS_DIR` | `$REPO_DIR/../osediff_experiments` | 训练产物 |
| `SD21_BASE_DIR` | `$MODEL_DIR/sd21_base` | SD2.1-Base 目录 |
| `RAM_PATH` | `$MODEL_DIR/ram_swin_large_14m.pth` | RAM 权重 |
| `DAPE_PATH` | `$MODEL_DIR/DAPE.pth` | DAPE 权重 |
| `OSEDIFF_PKL` | `$OSEDIFF_DIR/preset/models/osediff.pkl` | OSEDiff 主权重（repo 自带） |

### Inference (02)
| var | default | note |
|---|---|---|
| `MODE` | `sr` | `sr`=x4 超分；`face`=x1 人脸修复（用 osediff_face.pkl + codeformer） |
| `INPUT_DIR` | preset 测试图 | 输入低清图目录 |
| `OUTPUT_DIR` | `$RESULTS_DIR/output` | 输出目录 |
| `UPSCALE` | `4`（sr）/ `1`（face） | 放大倍数 |

### Training (04)
| var | default | note |
|---|---|---|
| `MODE` | `sr` | `face` 用 train_osediff_face.py + params_codeformer.yml |
| `DATASET_TXT` | (required) | 图集 txt，每行一张图绝对路径 |
| `DATASET_TXT2` | (none) | 第二数据集（混合训练，如 LSDIR+FFHQ） |
| `TRAIN_BATCH_SIZE` | `1` | 单卡；多卡设 4 |
| `GRAD_ACCUM` | `4` | 梯度累积，单卡补回等效 batch |
| `LORA_RANK` | `4` | LoRA 秩（论文值） |
| `LEARNING_RATE` | `5e-5` | |
| `MIXED_PRECISION` | `fp16` | |
| `N_TRAIN_GPU` | (auto) | >1 时走多卡 + 自动端口 |
| `DEG_FILE` | `params_realesrgan.yml`（sr）/ `params_codeformer.yml`（face） | 退化模型配置 |
| `OUTPUT_DIR` | `$EXPERIMENTS_DIR/exp1` | checkpoint 输出 |

## 可能遇到的问题

**1. `import xformers` 失败 / 训练 OOM**
OSEDiff `requirements.txt` pin `xformers==0.0.20`，对应 `torch==2.0.1`。预编译 wheel
通常匹配 `torch 2.0.1+cu118`。若装不上：WSL 走 00a 的源码编译分支（conda 装
`cuda-nvcc` + `gxx_linux-64=12`）；服务器用系统 nvcc。训练时务必带
`--enable_xformers_memory_efficient_attention --mixed_precision=fp16`。

**2. DAPE 权重不对 / 推理报维度不匹配**
repo 自带的 `preset/models/DAPE.pth` 仅 ~7MB，可能是精简版。若推理报 RAM/DAPE 加载
错误，下载完整版（Google Drive file id `1KIV6VewwO2eDC9g4Gcvgm-a0LDI7Lmwm`）：
```bash
pip install gdown
gdown 1KIV6VewwO2eDC9g4Gcvgm-a0LDI7Lmwm -O $MODEL_DIR/DAPE.pth
```

**3. SD2.1-Base 下载失败（HF 限流 / SSL）**
`Manojb/stable-diffusion-2-1-base` 是公开镜像。若 HF 直连慢，WSL 走
`HF_ENDPOINT=https://hf-mirror.com`（00a 已写入 proxy.env）。仍失败用 01 脚本的
curl fallback，或换 `stabilityai/stable-diffusion-2-1-base`（可能需 HF_TOKEN）。

**4. 训练显存不足（24GB）**
默认 `bs=1 accum=4 + fp16 + xformers + checkpointing + LoRA(r=4)` 在 RTX 3090 24GB
可跑。若仍 OOM：降 `--train_batch_size` 到 1（已默认）、确认 xformers 已装、
或减小训练分辨率（官方默认 512）。

**5. `test_osediff.py` 报 `ModuleNotFoundError: ram` / `models`**
脚本从 `$OSEDIFF_DIR` 目录内运行（`cd` 后调），保证相对 import 生效。若手动调
官方脚本，需 `export PYTHONPATH=$OSEDIFF_DIR:$PYTHONPATH`。

## 目录布局
```
<code-dir>/
├── media_code/
│   ├── proxy.env
│   └── osediff/
├── OSEDiff/                        # 官方代码（00 clone）
│   └── preset/models/osediff.pkl   # 训练权重（repo 自带）
└── ../../model/osediff/            # 权重（共享根的子目录）
    ├── sd21_base/
    ├── ram_swin_large_14m.pth
    └── DAPE.pth
```

## Notes
- Official code & weights follow their own license (Apache 2.0); see
  [cswry/OSEDiff](https://github.com/cswry/OSEDiff) and [arXiv:2406.08177](https://arxiv.org/abs/2406.08177).
- `.gitattributes` forces LF for `*.sh` / `*.py` — Windows pushes run fine on Ubuntu.
- `proxy.env` is gitignored; never commit proxy credentials.

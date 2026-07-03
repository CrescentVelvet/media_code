# HYPIR runner

One-click orchestration to run [HYPIR](https://github.com/XPixelGroup/HYPIR) (SIGGRAPH 2025, "Harnessing Diffusion-Yielded Score Priors for Image Restoration") **inference**, **dataset construction**, and **LoRA training** on an Ubuntu + NVIDIA GPU server.
This folder holds **only orchestration scripts** — no official code, no weights. The official repo is cloned automatically; weights are downloaded from HuggingFace.

Compared with `triposplat/`, this set adds a **dataset-build** step (`03_build_dataset.sh`) and a **LoRA training** step (`04_train.sh`) — the full reproduce path: env → weights → inference → your-data → train.

## Design
- **Reuses an existing conda env** (default name `doll`) that already has a CUDA-enabled torch — no separate venv. Run `INSTALL_DEPS=1 bash hypir/00_setup_env.sh` once to install the official `requirements.txt`.
  - ⚠️ HYPIR pins `diffusers==0.32.2` / `transformers==4.49.0` / `peft==0.14.0` — these **conflict** with other algos in this repo (e.g. `hunyuanvideo-1.5` wants `diffusers==0.35.0` / `transformers==4.57.1`). **Use a dedicated env**: `conda create -n hypir python=3.10 -y` then `CONDA_ENV=hypir ...`. Set `SKIP_TORCH=1` if you don't want the `torch==2.6.0` pin to disturb an existing torch.
- Code and weights live outside this repo (see Layout).

## Layout (when this repo is cloned under your code dir)
```
<code-dir>/
├── media_code/                  # this repo
│   ├── proxy.env                # proxy + optional overrides, gitignored
│   └── hypir/
│       ├── _env.sh                  # shared: proxy + CA bundle + conda activate
│       ├── 00_setup_env.sh          # activate env + verify torch (INSTALL_DEPS=1 -> requirements.txt)
│       ├── 01_download_models.sh    # hf download SD2-base + HYPIR_sd2.pth (lora)
│       ├── 02_run_inference.sh      # batch test.py -> restored PNGs
│       ├── 03_build_dataset.sh      # image folder -> training parquet (+ optional 512 crop)
│       ├── build_dataset.py         #   parquet builder (polars)
│       ├── 04_train.sh              # accelerate launch train.py on your parquet (LoRA)
│       ├── gen_train_config.py      #   fills TODOs in configs/sd2_train.yaml (no fork)
│       ├── run_all.sh               # one-click: clone -> 00 -> 01 -> 02
│       ├── setup_ca_bundle.sh       # one-time: extract proxy CA -> ~/.ca-bundle.crt
│       ├── _extract_ca.py           #   helper used by setup_ca_bundle.sh
│       └── _hf_download.py          #   snapshot_download with SSL verify off (01 fallback)
├── HYPIR/                       # official code (auto-cloned to ../HYPIR)
└── ../../model/                 # weights live one dir above <code-dir> (shared by all algos)
    └── HYPIR/
        ├── sd2_base/            # Manojb/stable-diffusion-2-1-base (public mirror; diffusers: scheduler/tokenizer/text_encoder/unet/vae)
        └── HYPIR_sd2.pth        # LoRA weights (lxq007/HYPIR)
```
Defaults: official code at `../HYPIR`, weights at `../../model/HYPIR` (relative to this repo). Override with `HYPIR_DIR` / `MODEL_DIR`.

## Prerequisites
- Ubuntu, NVIDIA driver (CUDA 12.x), `git`, `conda`
- A conda env with a CUDA-enabled torch already installed. Recommended **dedicated** env:
  ```bash
  conda create -n hypir python=3.10 -y && conda activate hypir
  pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
  ```
- NVIDIA GPU: a free T4 is enough for inference (per the official colab); A100/H100/RTX-4090 for training.

## Setup (on the server)
```bash
cd <your-code-dir>   # e.g. /data_3d/<uid>/code
git -c http.sslVerify=false clone https://github.com/CrescentVelvet/media_code.git
cd media_code
cp proxy.env.example proxy.env
# edit proxy.env: http_proxy / https_proxy (and CONDA_ENV if your env isn't 'doll')
# Dedicated env recommended (HYPIR's pins conflict with other algos):
CONDA_ENV=hypir INSTALL_DEPS=1 bash hypir/run_all.sh
# run_all.sh: clone official repo -> activate env + install requirements -> download weights -> run example restoration.
```

## Step-by-step
```bash
sudo docker exec -it <container> /bin/bash
conda activate hypir   # or 'doll' if you share it
INSTALL_DEPS=1 bash hypir/00_setup_env.sh          # activate + verify torch + pip install -r requirements.txt
HF_DISABLE_SSL=1 bash hypir/01_download_models.sh  # hf download sd2-base (public Manojb mirror) + HYPIR_sd2.pth
# Inference on the bundled examples (6 LQ images + prompts):
GPU=0 bash hypir/02_run_inference.sh
# On your own images (no prompts -> empty caption):
GPU=0 LQ_DIR=/path/to/images bash hypir/02_run_inference.sh
# Build a training parquet from your HQ image folder (crops to 512x512 by default):
DATA_DIR=/data/LSDIR bash hypir/03_build_dataset.sh
# LoRA fine-tune:
PARQUET_PATH=/data/LSDIR/hypir_train.parquet GPU=0 bash hypir/04_train.sh
```
Missing a package? Just `pip install <pkg>` in the conda env and rerun the failed step.

## Inference (02 — image restoration / super-resolution)
`02_run_inference.sh` runs the official `test.py`, which loads the SD2 base + LoRA **once** and loops over every image in `LQ_DIR` (walked recursively). For each image it writes:
- `OUTPUT_DIR/result/<same-relative-path>.png` — restored image
- `OUTPUT_DIR/prompt/<same-relative-path>.txt` — prompt used

Prompts: pass `TXT_DIR` (a folder mirroring `LQ_DIR`'s structure, one `.txt` per image) to use per-image captions. Without `TXT_DIR`, `--captioner empty` is used (null-text restoration — works well per the paper).
```bash
# 4x upscale, default example prompts:
GPU=0 bash hypir/02_run_inference.sh
# Your images, empty prompts, 2x upscale:
GPU=0 LQ_DIR=/path/to/lq UPSCALE=2 bash hypir/02_run_inference.sh
# Scale to a target longest side instead of a fixed factor:
GPU=0 LQ_DIR=/path/to/lq SCALE_BY=longest_side TARGET_LONGEST_SIDE=1920 bash hypir/02_run_inference.sh
# Use a trained LoRA checkpoint instead of the released one:
GPU=0 WEIGHT_PATH=/path/to/checkpoint-N/state_dict.pth bash hypir/02_run_inference.sh
```
The LoRA module list / rank (256) match the official HYPIR-SD2 config; override via `LORA_MODULES` / `LORA_RANK` only if you trained a different config.

## Pipeline（推理流程详解）
对应官方代码 `HYPIR/enhancer/base.py::enhance` + `HYPIR/enhancer/sd2.py::forward_generator` + `HYPIR/utils/common.py`。一张 LQ 图像从输入到输出经过 5 步：

1. **上采样（bicubic 插值）** — `F.interpolate(lq, scale_factor=upscale, mode="bicubic")`。先把 LQ 双三次插值放大到目标分辨率（`scale_by=factor` 时按固定倍数，`longest_side` 时按长边到固定尺寸）。这一步的输出同时存为 `ref`（参考图），供第 5 步小波融合用。若短边 ≤ `patch_size`(512)，还会把短边 resize 到至少 512，保证 VAE 有足够大的画布（此 resize 只作用于送入 VAE 的 `lq`，不改动 `ref`）。
2. **VAE 编码（分块，patch=512）** — `lq` 归一化到 `[-1,1]`、pad 到 8 的倍数（`vae_scale_factor=8`），再用 `make_tiled_fn` 分块送入 `vae.encode(...).latent_dist.sample()`。tile 在**像素空间**大小为 `patch_size`(512)、stride=`stride`(256)，下采样到**潜空间** 64（512/8）。重叠区用高斯权重平滑拼接，避免接缝。注意 `.sample()` 从 VAE 编码分布里采样，带轻微随机性——这就是推理要设 `--seed` 的原因。
3. **UNet 一步去噪（LoRA + SD2，t=200）** — 见下方「一步去噪」。加载了 LoRA 的 SD2 UNet 以 LQ 潜变量为输入、在 timestep=200 做一次前向 + 一次 DDPM 反推，得到复原的 HR 潜变量。同样分块（潜空间 64 / stride 32）。
4. **VAE 解码（分块）** — `make_tiled_fn(vae.decode(tile).sample, scale_type="up", scale=8, channel=3)` 把复原潜变量解码回像素空间（潜 64 → 像素 512 的 tile）。然后裁掉 padding、`(x+1)/2` 回到 `[0,1]`、bicubic 缩回 `ref` 的尺寸 `(h0,w0)`。
5. **小波融合** — `wavelet_reconstruction(x, ref)`：把第 4 步解码输出（`content`）与第 1 步上采样参考图（`style=ref`）做多尺度融合。见下方「小波融合」。

### 一步去噪（one-step denoising）怎么回事
标准 DDPM 生成要迭代 ~1000 步（DDIM ~50 步）逐步去噪。**HYPIR 不迭代，只走一步**，关键在于把「LQ 潜变量」直接当作部分加噪的 `x_t`：

```
z_in = z_lq * vae.scaling_factor          # LQ 潜变量当 x_t（不再加随机噪声！）
eps  = UNet_lora(z_in, t=200, text_embed)  # UNet 预测"把 x_0 变成 x_t 的噪声"
z0   = scheduler.step(eps, coeff_t=200, z_in).pred_original_sample   # 一步反推 x_0
```

- DDPM 有闭式关系：`x_0 = (x_t − √(1−ᾱ_t)·eps) / √(ᾱ_t)`。给定 `x_t`（=LQ 潜变量）和 UNet 预测的噪声 `eps`，**一次代数运算**就能解出估计的干净潜变量 `x_0`，不需要迭代。
- 为什么一步就够？因为 LoRA 是**专门为 t=200 这一步训练**的：训练时同样把 GT 的 LQ 潜变量当 `x_t`、UNet 预测 `eps`、一步反推 `x_0`，再用 `x_0` 解码出的图像与 GT HR 算 L2+LPIPS+GAN 损失（见 `HYPIR/trainer/base.py::optimize_generator`）。也就是说 LoRA 学到的就是「在 t=200 这一步、从 LQ 潜变量一步反推出 HR 潜变量」这个映射，推理时自然一步即出。
- `model_t=200` 是喂给 UNet 的时间步标签（UNet 以为自己正在 t=200 去噪）；`coeff_t=200` 是反推 `x_0` 时用的噪声调度系数所在时间步。两者在此都取 200。
- 这就是论文标题里的 "score prior"：扩散 UNet 的噪声预测（score）在单个时间步上提供了一个强先验，把退化图直接拉回干净图——快（一次前向）且借用了 SD2 在海量图像上学到的先验。

### 小波融合（wavelet reconstruction）怎么回事
解码出的 HR 图像纹理清晰，但扩散模型可能引入色偏/结构幻觉；而上采样 LQ 颜色和整体结构是可靠的，只是模糊（缺高频）。小波融合取两者之长：

- `wavelet_decomposition(img, levels=5)` 用逐级放大的高斯模糊（dilation 半径 1,2,4,8,16）把图像拆成 **低频**（`low`：颜色、光照、大尺度结构）和 **高频**（`high`：边缘、纹理、细节）。
- `wavelet_reconstruction(content=x_decoded, style=ref)` 的实际计算是：
  ```
  result = content_high + style_low
         = 解码输出的高频 + 上采样LQ的低频
  ```
  即**高频（纹理/边缘）来自扩散解码输出，低频（颜色/结构）来自上采样 LQ**。
- 效果：最终图保留 LQ 原本正确的颜色与构图（低频），同时注入扩散模型生成的锐利纹理（高频），避免色偏和结构漂移，又实现了超分。这是 SUPIR/CCSR 一脉相承的经典 trick。

> 总结一句：**上采样定颜色结构 → VAE 编码进潜空间 → LoRA-UNet 一步反推干净潜变量 → VAE 解码回像素 → 与上采样原图小波融合（取扩散的高频 + LQ 的低频）**。

## Dataset construction (03 — image folder → parquet)
HYPIR's `RealESRGANDataset` (default `crop_type=none`, `out_size=512`) **asserts every GT image is exactly 512×512**. `03_build_dataset.sh` handles this:
- `CROP=1` (default): slices each image into 512×512 patches (non-overlapping; set `CROP_STRIDE < CROP_SIZE` for overlap) saved under `<parquet_dir>/patches`, and the parquet points at the patches. This mirrors the README's "crop LSDIR into 512×512 patches".
- `CROP=0`: uses images as-is (they must already be 512×512, or pass `CROP_TYPE=random` to `04_train.sh`).

Output: a parquet (`image_path`[absolute], `prompt`) consumed by `configs/sd2_train.yaml`'s `file_meta`.
```bash
DATA_DIR=/data/LSDIR bash hypir/03_build_dataset.sh
# -> /data/LSDIR/hypir_train.parquet  (+ /data/LSDIR/patches/*.png when CROP=1)

# Custom: 256 stride (overlap), a fixed caption, output elsewhere:
DATA_DIR=/data/LSDIR CROP_STRIDE=256 PROMPT="high quality, highly detailed" \
  PARQUET_OUT=/data/hypir.parquet bash hypir/03_build_dataset.sh
```

## Training (04 — LoRA fine-tune)
`04_train.sh` generates a **derived** config (`gen_train_config.py` fills the TODOs in `configs/sd2_train.yaml` — no official file is modified), patches in your hyperparams, and runs `accelerate launch train.py --config <filled>`. The trainer trains LoRA adapters on the SD2 UNet with a RealESRGAN degradation pipeline + GAN (ConvNeXt discriminator) + LPIPS + L2 losses.
```bash
# Build the parquet first (see 03), then:
PARQUET_PATH=/data/LSDIR/hypir_train.parquet GPU=0 bash hypir/04_train.sh

# Let 04 build the parquet for you from DATA_DIR, then train:
DATA_DIR=/data/LSDIR GPU=0 bash hypir/04_train.sh

# Your GT images are >512 and you did NOT pre-crop? use random crop in-config:
PARQUET_PATH=/data/big.parquet CROP_TYPE=random OUT_SIZE=512 GPU=0 bash hypir/04_train.sh

# Shorter run / custom batch / resume:
PARQUET_PATH=/data/x.parquet MAX_TRAIN_STEPS=5000 BATCH_SIZE=4 LR_G=1e-5 \
  RESUME=/path/to/checkpoint-500 GPU=0 bash hypir/04_train.sh

# Multi-GPU (run `accelerate config` once first; leave GPU unset):
PARQUET_PATH=/data/x.parquet N_TRAIN_GPU=8 bash hypir/04_train.sh
```
Checkpoints land in `OUTPUT_DIR/checkpoint-<step>/state_dict.pth` (LoRA params only). Use one for inference:
```bash
WEIGHT_PATH=$OUTPUT_DIR/checkpoint-N/state_dict.pth GPU=0 bash hypir/02_run_inference.sh
```
> The official trainer also saves `ema_state_dict.pth` alongside each checkpoint. To use EMA weights, point `WEIGHT_PATH` at the EMA file (same key set) — the loader in `SD2Enhancer.init_generator` accepts either.

## Paired face fine-tuning on REAL LQ+HQ (03b / 04-paired) — `crop_faces_paired.py` output → HYPIR

The official `03`/`04` path trains on **HQ only** and *synthesizes* LQ via
RealESRGAN degradation. To train on **real** degradation (e.g. the 360p camera
LQ + RAW-decoded HQ face pairs from `face_crop/crop_faces_paired.py`), use the
paired plug-ins — **no official file is modified**: the derived config points
`target:` at plug-in classes in `paired_face_plugin.py` (importable via
PYTHONPATH), and `train_paired.py` subclasses `SD2Trainer` to warm-start the
LoRA from the released `HYPIR_sd2.pth`.

> 中文说明：官方 `03/04` 只用 HQ、LQ 是现场用 RealESRGAN 合成退化的（盲复原配方）。
> 你用 `face_crop/crop_faces_paired.py` 建的是「真实 360p 相机 LQ + RAW 解码 HQ」配对，
> 想直接喂真实退化，就用这套配对插件。**不改任何官方文件**：填好的配置把 `target:`
> 指向 `paired_face_plugin.py` 里的插件类（靠 `04_train_paired.sh` 设的 `PYTHONPATH`
> 导入），`train_paired.py` 再子类化 `SD2Trainer` 实现从发布权重暖启动。
>
> 核心思想：7k 张配对从零训 LoRA 不够（发布模型是 bs1024 大数据训的），所以默认
> **暖启动**——把发布的 `HYPIR_sd2.pth` 当 LoRA 初始化，再在你的人脸数据上继续练，
> 等于把通用复原模型「适配」到人脸域。数据流见下图，训练循环(一步去噪+L2/LPIPS/GAN)
> 与官方完全一致，只是 LQ 来自真实配对而非合成。

```
HQ folder (.../hq/<stem>_faceN.png)  ┐  paired by filename
LQ folder (.../lq/<stem>_faceN.png)  ┘
   └─ 03b_build_paired_dataset.py  -> parquet(hq_path, lq_path, prompt)
        └─ PairedFaceDataset        : load HQ+LQ, resize/paired-crop to 512,
                                      same flip/rot -> {hq, lq, txt}
        └─ PairedFaceBatchTransform : USM-sharpen HQ, rename -> {GT, LQ, txt}
        └─ FineTuneSD2Trainer       : warm-start LoRA from HYPIR_sd2.pth, then
                                      the unchanged one-step + L2/LPIPS/GAN loop
```

### Usage
```bash
# Dataset already uploaded as .../ppr10k_faces_20260703/{hq,lq} (defaults):
GPU=0 bash hypir/04_train_paired.sh

# Build the parquet only (no train):
HQ_DIR=.../hq LQ_DIR=.../lq bash hypir/03b_build_paired_dataset.sh

# Custom everything:
GPU=0 DATASET_ROOT=.../ppr10k_faces_20260703 MAX_TRAIN_STEPS=20000 \
    LR_G=5e-6 BATCH_SIZE=8 bash hypir/04_train_paired.sh

# Train from scratch (no warm-start) instead of fine-tuning the released LoRA:
LORA_WEIGHT_PATH= GPU=0 bash hypir/04_train_paired.sh
```
Inference with your checkpoint (same as 02):
```bash
WEIGHT_PATH=$OUTPUT_DIR/checkpoint-N/state_dict.pth GPU=0 bash hypir/02_run_inference.sh
```

### Why fine-tune (warm-start) instead of from scratch
The released `HYPIR_sd2.pth` was trained on a large dataset at batch 1024. With
only ~7k face pairs you cannot relearn the one-step restoration from scratch —
so `04_train_paired.sh` **warm-starts** from `HYPIR_sd2.pth` by default
(`LORA_WEIGHT_PATH`) and adapts it to your face domain. Set `LORA_WEIGHT_PATH=`
to disable (gaussian-init from scratch, official behaviour).

### Training parameters (for ~7k paired face crops)
| var | default | note |
| --- | --- | --- |
| `DATASET_ROOT` | `/data_3d/w00950754/code/HYPIR/dataset/ppr10k_faces_20260703` | expects `hq/` and `lq/` subdirs |
| `LORA_WEIGHT_PATH` | `$MODEL_DIR/HYPIR_sd2.pth` | warm-start; `""` = from scratch |
| `OUT_SIZE` | `512` | HQ & LQ both resized to this (HYPIR's VAE patch size) |
| `CROP_TYPE` | `none` | resize whole face; `random` for paired random-patch aug |
| `MAX_TRAIN_STEPS` | `15000` | ~2 epochs over 7k at bs=6; try 10k–30k |
| `BATCH_SIZE` | `6` | per-GPU; raise to 8–12 on A100/H100 |
| `LR_G` / `LR_D` | `1e-5` / `1e-5` | use `5e-6` for gentler adaptation |
| `GRAD_ACCUM` | `1` | increase if a bigger effective batch is wanted |
| `CHECKPOINTING_STEPS` | `500` | saves `checkpoint-N/state_dict.pth` |
| `N_TRAIN_GPU` | _(unset)_ | `>1` → multi-GPU (run `accelerate config` first) |
| `RESUME` | _(unset)_ | a `checkpoint-N` dir to resume from |

> The HQ/LQ scale of these pairs is ~10× (full RAW decode); both are rendered at
> `OUT_SIZE=512`, so the model learns real 360p-camera → clean face restoration.
> `use_sharpener=true` (USM on HQ) matches the preprocessing the released LoRA
> was trained with — keep it on when warm-starting.

## 配对人脸微调全流程

这篇给第一次用的人：从「数据已上传」到「跑完训练、看到还原结果」，每步都有可复制的命令。命令都在服务器上执行。

### 你需要先有的
- 一台 Ubuntu + NVIDIA GPU 服务器（训练建议 A100 / H100 / 4090；只推理 T4 也行）。
- 已用 `face_crop/crop_faces_paired.py` 建好的**配对人脸数据集**（`hq/` 与 `lq/` 同名 PNG），并上传到默认路径：
  `/data_3d/w00950754/code/HYPIR/dataset/ppr10k_faces_20260703/{hq,lq}`
- 服务器上已装好 `conda` 和 `git`。

### 训练到底在干什么（一句话版）
把「模糊的 360p LQ 人脸」和「清晰的 HQ 人脸」**成对**喂给模型，让它学会把模糊脸还原成清晰脸。我们不是从零训练，而是在官方已发布的 `HYPIR_sd2.pth` 上**继续练（暖启动）**，让它专精你的人脸数据——所以 7 千张就够、收敛也快。

### 整体流程
1. 拉本仓 + 配代理 → 2. 建专用 conda 环境 + 装依赖 → 3. 下官方权重 → 4. 确认数据在位 → 5. 一条命令开训 → 6. 看进度/曲线 → 7. 用 checkpoint 推理看图。

---

### Step 1 — 拉仓库、配代理（一次性）
```bash
cd /data_3d/w00950754/code
git -c http.sslVerify=false clone https://github.com/CrescentVelvet/media_code.git
cd media_code
cp proxy.env.example proxy.env
# 编辑 proxy.env：填 http_proxy / https_proxy（公司代理用）；自用网络可跳过
```

### Step 2 — 建专用环境 + 装依赖（一次性）
HYPIR 的依赖和本仓别的算法冲突，**务必用独立环境**：
```bash
conda create -n hypir python=3.10 -y
conda activate hypir
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
CONDA_ENV=hypir INSTALL_DEPS=1 bash hypir/00_setup_env.sh   # 装官方 requirements.txt
pip install polars pillow                                    # 配对数据集要用
```

### Step 3 — 下官方权重（一次性）
```bash
conda activate hypir
bash hypir/setup_ca_bundle.sh          # 公司代理才需要：建 CA 包；自用网络可跳过
HF_DISABLE_SSL=1 bash hypir/01_download_models.sh
# 下完在 ../../model/HYPIR/ 下：sd2_base/（基座）和 HYPIR_sd2.pth（暖启动用的 LoRA）
```

### Step 4 — 确认数据集在位
```bash
# Windows
scp -r D:\模型数据集\ppr10k_faces_20260703 w00950754@xx.xx.xxx.xxx:/data_3d/w00950754/code/HYPIR/dataset
# Ubuntu
DATA=/data_3d/w00950754/code/HYPIR/dataset/ppr10k_faces_20260703
ls $DATA/hq | head        # 应能看到 0_0_face1.png 之类
ls $DATA/lq | head        # hq/ 与 lq/ 文件名必须一一相同
```

### Step 5 — 开始训练（一条命令）
```bash
conda activate hypir
GPU=4,5,6,7 bash hypir/04_train_paired.sh
```
这一条命令会自动：按文件名建配对 parquet → 填配置 → 从 `HYPIR_sd2.pth` 暖启动 LoRA → `accelerate launch` 开训。
默认 15000 步、bs=6、lr=1e-5，每 500 步存一个 checkpoint 到：
`../HYPIR/experiments/ppr10k_faces_paired/checkpoint-N/state_dict.pth`

### Step 6 — 看训练进度
- 终端每 100 步打印一次 loss。
- TensorBoard 看曲线：
```bash
tensorboard --logdir ../HYPIR/experiments/ppr10k_faces_paired --port 6006
# 本地浏览器开 http://<服务器IP>:6006
```
- 看已存的 checkpoint：
```bash
ls ../HYPIR/experiments/ppr10k_faces_paired/
```
中途想停没问题。续训：
```bash
RESUME=../HYPIR/experiments/ppr10k_faces_paired/checkpoint-5000 GPU=0 bash hypir/04_train_paired.sh
```

### Step 7 — 用训好的 checkpoint 还原人脸、看效果
```bash
conda activate hypir
CKPT=../HYPIR/experiments/ppr10k_faces_paired/checkpoint-15000/state_dict.pth
LQ=/data_3d/w00950754/code/HYPIR/dataset/ppr10k_faces_20260703/lq
WEIGHT_PATH=$CKPT GPU=0 LQ_DIR=$LQ \
  SCALE_BY=longest_side TARGET_LONGEST_SIDE=512 \
  bash hypir/02_run_inference.sh
```
还原结果在 `../HYPIR/results/lq/result/*.png`。下载下来和原 LQ 对比，脸应更清晰。
> 想用别的 checkpoint，把 `CKPT` 路径里的步数换掉即可（如 `checkpoint-10000`）。

### 调参速查（按需改）
- 显存不够：`BATCH_SIZE=4`，或 `GRAD_ACCUM=2`（等效翻倍 batch）。
- 想更稳、别把发布模型「练歪」：`LR_G=5e-6 LR_D=5e-6`。
- 想多练/少练：`MAX_TRAIN_STEPS=30000`（更多）或 `10000`（更快）。
- 多卡：先跑一次 `accelerate config` 选 multi-GPU，再 `N_TRAIN_GPU=8 bash hypir/04_train_paired.sh`（此时**不要**设 `GPU=`）。
- 跑报错了：看下面「可能遇到的问题」对应条目。

## 可能遇到的问题

公司代理做 HTTPS 中间人解密，下面按流水线阶段列出常见报错与修法（命令在服务器上、conda 环境已激活时执行）。

**1. clone 本仓 / 官方仓报 SSL / 认证**
公开仓免认证，加 `-c http.sslVerify=false` 即可。`run_all.sh` 已对官方仓做了一次 `sslVerify=false` 兜底。

**2. `pip install -r requirements.txt` 报 `SSL:CERTIFICATE_VERIFY_FAILED` / 超时**
```bash
pip config set global.trusted-host "pypi.org pypi.python.org files.pythonhosted.org download.pytorch.org"
INSTALL_DEPS=1 bash hypir/00_setup_env.sh
```
仍超时（大文件 torch==2.6.0）：加 `--timeout 600 --retries 10`，或先单独装 torch：`pip install --timeout 600 --retries 10 torch==2.6.0` 再重跑。不想动现有 torch 就 `SKIP_TORCH=1 INSTALL_DEPS=1 bash hypir/00_setup_env.sh`。

**3. `hf download` 报 `CAS service error : ReqwestMiddleware`**
HF 的 Xet/Rust 通道不认代理。`_env.sh` 已设 `HF_HUB_DISABLE_XET=1`；仍报则彻底卸载：
```bash
pip uninstall -y hf_xet
bash hypir/01_download_models.sh
```

**4. `hf download` 报 `SSLCertVerificationError`**
代理根 CA 不在系统证书包。先一次性建包，`_env.sh` 会自动用 `~/.ca-bundle.crt`：
```bash
bash hypir/setup_ca_bundle.sh    # 抓代理证书链 -> ~/.ca-bundle.crt，并自检
bash hypir/01_download_models.sh
```
- 自检 `[OK]` → 直接重跑 `01`。
- 自检 `[FAIL]` → 把公司根 CA 追加到 `~/.ca-bundle.crt` 后重跑（公司根 CA 常见于 `/usr/local/share/ca-certificates/`，脚本已自动并入）。
- 仍报 SSL（CDN 端点用了不同的 MITM 证书）→ `01` 自动回退到禁用 SSL 校验的下载器（`_hf_download.py`）；或直接 `HF_DISABLE_SSL=1 bash hypir/01_download_models.sh`。

**5. `hf download` 报 `repository not found for url .../stable-diffusion-2-1-base`**
原始 `stabilityai/stable-diffusion-2-1-base` 已从 HuggingFace 下架。脚本默认改用公开镜像 `Manojb/stable-diffusion-2-1-base`（完整 diffusers 格式，非 gated，无需 token）。若你仍指向旧的 `stabilityai/...`，会报此错；改成默认源即可：
```bash
unset HF_BASE_REPO   # 用默认 Manojb/stable-diffusion-2-1-base
bash hypir/01_download_models.sh
```
若你换用的镜像确为 gated 仓库（HF 对未认证账号返回 "not found" 实为 401），则需：1) 在该仓库页面接受许可证；2) 建 read token；3) `HF_TOKEN=<token> bash hypir/01_download_models.sh`（脚本会自动把 token 透传给 `hf download` 和 SSL 兜底下载器）。

**6. 推理 找不到diffusion_pytorch_model.bin**
是HYPIR/HYPIR/enhancer/sd2.py里强制要求bin格式，但下载到模型是safetensor格式，修改use_safetensors=True即可。

**7. 推理 OOM（显存不足）**
降 `PATCH_SIZE`（512→256，并相应降 `STRIDE` 到 128），或降 `UPSCALE`。免费 T4 可跑默认 512 patch（见官方 colab）。仍紧张时设 `export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`。

**8. 训练时 `open_clip` / `lpips` 下载权重失败**
- 判别器 `ImageOpenCLIPConvNext` 在初始化时下载 `convnext_xxlarge`（laion2b_s34b_b82k_augreg_soup，open_clip 走 HuggingFace）。`_env.sh` 的 CA bundle + `HF_HUB_DISABLE_XET` 通常能覆盖；仍失败时手动放到 open_clip 缓存或 `HF_DISABLE_SSL=1` 后重试。
- `lpips.LPIPS(net="vgg")` 从作者 URL 下载 VGG 权重（走 `torch.hub`）。代理下若失败：先 `bash hypir/setup_ca_bundle.sh`，或预先把 `vgg.pth` 放进 `~/.cache/torch/hub/checkpoints/`。

**9. 训练报 `assert image.height == self.out_size`**
你的 GT 不是 512×512 且 `crop_type=none`。要么用 `03_build_dataset.sh` 的 `CROP=1` 预切 512 patch，要么 `CROP_TYPE=random bash hypir/04_train.sh`。

**10. 训练多卡 `accelerate launch` 只用一卡**
没配 accelerate 多进程。先 `accelerate config`（选 multi-GPU），再 `N_TRAIN_GPU=8 bash hypir/04_train.sh`（脚本会加 `--num_processes`）。单卡可忽略。

> 通用：`proxy.env`（代理凭证）在仓内 gitignored，`~/.ca-bundle.crt` 在家目录，都不入库；切勿把凭证写进脚本。

## Config (env vars, all optional)
| var | default | note |
|---|---|---|
| `CONDA_ENV` | `doll` | conda env to activate (dedicated `hypir` recommended — pins conflict with other algos) |
| `GPU` | _(unset)_ | physical GPU id to pin, e.g. `GPU=0`; leave unset for multi-GPU training |
| `HYPIR_DIR` | `../HYPIR` | official code path |
| `MODEL_DIR` | `../../model/HYPIR` | weights path (sd2_base/ + HYPIR_sd2.pth) |
| `HYPIR_REPO` | official GitHub URL | clone source |
| `INSTALL_DEPS` | `0` (run_all: `1`) | set `1` to `pip install -r requirements.txt` |
| `SKIP_TORCH` | `0` | `1` = filter torch/torchvision pins out of requirements (keep existing torch) |
| `HF_HUB_DISABLE_XET` | `1` | disable HF Xet/CAS Rust path (proxy-unfriendly) |
| `HF_DISABLE_SSL` | `0` | set `1` to download weights with SSL verification disabled |
| `HF_TOKEN` | _(unset)_ | only needed if you point `HF_BASE_REPO` at a gated repo; the default mirror is public |
| `HF_BASE_REPO` | `Manojb/stable-diffusion-2-1-base` | base diffusers model repo (public mirror; original `stabilityai/...` was removed) |
| `HF_LORA_REPO` | `lxq007/HYPIR` | LoRA weights repo |
| `LORA_FILE` | `HYPIR_sd2.pth` | LoRA file name inside HF_LORA_REPO |
| `BASE_MODEL_DIR` | `$MODEL_DIR/sd2_base` | local base model dir (passed as `--base_model_path`) |
| `WEIGHT_PATH` | `$MODEL_DIR/HYPIR_sd2.pth` | LoRA weight file (passed as `--weight_path`) |

### Inference (02)
| var | default | note |
|---|---|---|
| `LQ_DIR` | `../HYPIR/examples/lq` | folder of low-quality images (walked recursively) |
| `TXT_DIR` | `../HYPIR/examples/prompt` | matching prompt folder (.txt per image); empty → `--captioner empty` |
| `OUTPUT_DIR` | `../HYPIR/results/<lq_folder_name>` | writes `result/` + `prompt/` under here |
| `SCALE_BY` | `factor` | `factor` \| `longest_side` |
| `UPSCALE` | `4` | upscaling factor (when `SCALE_BY=factor`) |
| `TARGET_LONGEST_SIDE` | _(unset)_ | required when `SCALE_BY=longest_side` |
| `PATCH_SIZE` / `STRIDE` | `512` / `256` | tiled processing size / stride |
| `SEED` | `231` | |
| `LORA_RANK` / `LORA_MODULES` | `256` / official list | only override if you trained a different config |

### Dataset (03)
| var | default | note |
|---|---|---|
| `DATA_DIR` | _(required)_ | folder of high-quality images |
| `PARQUET_OUT` | `$DATA_DIR/hypir_train.parquet` | output parquet path |
| `PROMPT` | `""` | caption for every image (null-text training) |
| `CROP` | `1` | `1` = slice into 512 patches first; `0` = use images as-is |
| `CROP_SIZE` / `CROP_STRIDE` | `512` / `=CROP_SIZE` | patch size / stride (smaller ⇒ overlap) |
| `CROP_OUT` | `<parquet_dir>/patches` | where patches are saved |

### Training (04)
| var | default | note |
|---|---|---|
| `PARQUET_PATH` | _(required)_ | parquet from 03 (or set `DATA_DIR` to build it first) |
| `OUTPUT_DIR` | `../HYPIR/experiments/exp1` | checkpoints + logs dir |
| `CROP_TYPE` | `none` | `none`\|`random`\|`center` (use `random` if GTs aren't 512×512 and not pre-cropped) |
| `OUT_SIZE` | `512` | must match your GT image size when `CROP_TYPE=none` |
| `MAX_TRAIN_STEPS` | `30000` | |
| `BATCH_SIZE` | `6` | per-GPU |
| `LR_G` / `LR_D` | `1e-5` / `1e-5` | generator / discriminator learning rates |
| `GRAD_ACCUM` | `1` | gradient accumulation steps |
| `SEED` | `231` | |
| `CHECKPOINTING_STEPS` | `500` | save a checkpoint every N steps |
| `LOG_IMAGE_STEPS` / `LOG_GRAD_STEPS` | `100` / `100` | |
| `N_TRAIN_GPU` | _(unset)_ | `>1` → `accelerate launch --num_processes N` (run `accelerate config` first) |
| `MIXED_PRECISION` | _(unset; config=bf16)_ | override accelerate mixed precision |
| `RESUME` | _(unset)_ | `checkpoint-N` dir to resume from |

## Outputs
- **02 inference**: `OUTPUT_DIR/result/<rel>.png` (restored) + `OUTPUT_DIR/prompt/<rel>.txt`.
- **03 dataset**: `PARQUET_OUT` (+ `patches/*.png` when `CROP=1`).
- **04 training**: `OUTPUT_DIR/checkpoint-<step>/{state_dict.pth, ema_state_dict.pth, ...}`. Point `02`'s `WEIGHT_PATH` at `state_dict.pth` to run your fine-tuned model.

## Notes
- Official code & weights follow their own license (HYPIR = non-commercial use only — see the repo). This folder only orchestrates; no official code is copied.
- `.gitattributes` (repo root) forces LF so Windows-pushed scripts run cleanly on Ubuntu.
- `proxy.env` (proxy creds / env overrides) is gitignored — never committed. Don't put credentials in scripts.
- SSL behind a TLS-intercepting corporate proxy: pip uses `--trusted-host`; `hf`/`git` use the CA bundle (`_env.sh` prefers `~/.ca-bundle.crt`, built by `setup_ca_bundle.sh`).

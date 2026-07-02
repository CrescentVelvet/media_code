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
        ├── sd2_base/            # stabilityai/stable-diffusion-2-1-base (diffusers: scheduler/tokenizer/text_encoder/unet/vae)
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
HF_DISABLE_SSL=1 bash hypir/01_download_models.sh  # hf download sd2-base + HYPIR_sd2.pth
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

**5. 推理 OOM（显存不足）**
降 `PATCH_SIZE`（512→256，并相应降 `STRIDE` 到 128），或降 `UPSCALE`。免费 T4 可跑默认 512 patch（见官方 colab）。仍紧张时设 `export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`。

**6. 训练时 `open_clip` / `lpips` 下载权重失败**
- 判别器 `ImageOpenCLIPConvNext` 在初始化时下载 `convnext_xxlarge`（laion2b_s34b_b82k_augreg_soup，open_clip 走 HuggingFace）。`_env.sh` 的 CA bundle + `HF_HUB_DISABLE_XET` 通常能覆盖；仍失败时手动放到 open_clip 缓存或 `HF_DISABLE_SSL=1` 后重试。
- `lpips.LPIPS(net="vgg")` 从作者 URL 下载 VGG 权重（走 `torch.hub`）。代理下若失败：先 `bash hypir/setup_ca_bundle.sh`，或预先把 `vgg.pth` 放进 `~/.cache/torch/hub/checkpoints/`。

**7. 训练报 `assert image.height == self.out_size`**
你的 GT 不是 512×512 且 `crop_type=none`。要么用 `03_build_dataset.sh` 的 `CROP=1` 预切 512 patch，要么 `CROP_TYPE=random bash hypir/04_train.sh`。

**8. 训练多卡 `accelerate launch` 只用一卡**
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
| `HF_BASE_REPO` | `stabilityai/stable-diffusion-2-1-base` | base diffusers model repo |
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

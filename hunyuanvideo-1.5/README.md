# HunyuanVideo-1.5 runner

One-click orchestration to run [HunyuanVideo-1.5](https://github.com/Tencent-Hunyuan/HunyuanVideo-1.5) **inference** and **training** on an Ubuntu + NVIDIA GPU server.
This folder holds **only orchestration scripts** — no official code, no weights. The official repo is cloned automatically; weights are downloaded from HuggingFace / ModelScope.

Compared with `triposplat/`, this set adds a **training** script (`03_train.sh`) that points at your own dataset and forwards all official training parameters.

## Design
- **Reuses an existing conda env** (default name `doll`) that already has a CUDA-enabled torch — no torch download, no venv creation. Run `INSTALL_DEPS=1 bash hunyuanvideo-1.5/00_setup_env.sh` once to install the official `requirements.txt` (this upgrades torch to **>=2.6 + CUDA 12.x** as the repo requires).
- Runtime/attention extras (flash-attn, SageAttention, flex-block-attn) are **optional** and installed by you — the default first run needs none of them.
- Code and weights live outside this repo (see Layout).

## Layout (when this repo is cloned under your code dir)
```
<code-dir>/
├── media_code/                  # this repo
│   ├── proxy.env                # proxy + optional overrides, gitignored
│   └── hunyuanvideo-1.5/
│       ├── _env.sh                  # shared: proxy + CA bundle + conda activate
│       ├── 00_setup_env.sh          # activate env + verify torch (INSTALL_DEPS=1 -> requirements.txt)
│       ├── 01_download_models.sh    # hf download DiT/VAE/text/vision encoders + ckpts symlink
│       ├── 02_run_inference.sh      # torchrun generate.py (T2V / I2V) -> .mp4
│       ├── 03_train.sh              # torchrun train_run.py on your dataset (NEW vs triposplat)
│       ├── train_run.py             #   injects a real dataloader into official train.py (no fork)
│       ├── train_dataset.py         #   video/image + caption dataset (reads DATA_DIR)
│       ├── run_all.sh               # one-click: clone -> 00 -> 01 -> 02
│       ├── setup_ca_bundle.sh       # one-time: extract proxy CA -> ~/.ca-bundle.crt
│       ├── _extract_ca.py           #   helper used by setup_ca_bundle.sh
│       └── _hf_download.py          #   snapshot_download with SSL verify off (01 fallback)
├── HunyuanVideo-1.5/            # official code (auto-cloned to ../HunyuanVideo-1.5)
│   └── ckpts -> ../model/HunyuanVideo-1.5   # symlink to shared weights
└── model/
    └── HunyuanVideo-1.5/        # weights (hf / modelscope download)
        ├── transformer/         #   480p/720p t2v & i2v (+ distilled / step-distill / sr) variants
        ├── vae/  text_encoder/  #   Qwen2.5-VL-7B (llm), byt5-small, Glyph-SDXL-v2
        └── vision_encoder/siglip
```
Defaults: official code at `../HunyuanVideo-1.5`, weights at `../model/HunyuanVideo-1.5` (relative to this repo). Override with `HYVIDEO_DIR` / `MODEL_DIR`.

## Prerequisites
- Ubuntu, NVIDIA driver (CUDA 12.x), `git`, `conda`
- A conda env with a CUDA-enabled torch already installed (default env name `doll`). HunyuanVideo-1.5 needs **torch>=2.6**; `INSTALL_DEPS=1` upgrades it in place:
  ```bash
  conda create -n doll python=3.10 -y && conda activate doll
  pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
  ```
- NVIDIA GPU: **14 GB** VRAM minimum (with CPU offloading) for 480p. A100/H100/RTX-4090 recommended; 8x H800 for the official speed config.

## Setup (on the server)
```bash
cd <your-code-dir>   # e.g. /data_3d/<uid>/code
git -c http.sslVerify=false clone https://github.com/CrescentVelvet/media_code.git
cd media_code
cp proxy.env.example proxy.env
# edit proxy.env: http_proxy / https_proxy (and CONDA_ENV if your env isn't 'doll')
INSTALL_DEPS=1 bash hunyuanvideo-1.5/run_all.sh
# run_all.sh: clone official repo -> activate env + install requirements -> download weights -> run a sample T2V generation.
```

## Step-by-step
```bash
sudo docker exec -it <container> /bin/bash
conda activate doll
INSTALL_DEPS=1 bash hunyuanvideo-1.5/00_setup_env.sh          # activate + verify torch + pip install -r requirements.txt
HF_DISABLE_SSL=1 bash hunyuanvideo-1.5/01_download_models.sh  # hf download + ckpts symlink
# Inference (T2V default):
GPU=0 bash hunyuanvideo-1.5/02_run_inference.sh
# Inference (I2V — needs the gated siglip weights, see 01):
GPU=0 IMAGE_PATH=/path/to/first_frame.png bash hunyuanvideo-1.5/02_run_inference.sh
# Training on your dataset (NEW):
DATA_DIR=/data/my_videos N_TRAIN_GPU=8 SP_SIZE=8 \
  LEARNING_RATE=1e-5 MAX_STEPS=10000 USE_LORA=true \
  bash hunyuanvideo-1.5/03_train.sh
```
Missing a package? Just `pip install <pkg>` in the conda env and rerun the failed step.

## Inference (02 — text/image → video)
`02_run_inference.sh` runs the official `generate.py` via `torchrun`. One prompt per run → one `.mp4`.
Defaults are a **light, dependency-free first run**: 480p, T2V, base model (non-distilled), CPU offloading on, no SR/rewrite/cache/sage. Optimize with env vars:
```bash
# Faster (cfg-distilled, 2x; needs the *_distilled transformer that 01 already pulled):
GPU=0 CFG_DISTILLED=true bash hunyuanvideo-1.5/02_run_inference.sh
# 480p I2V in ~8 steps (step-distill; ~75% faster on a 4090):
GPU=0 IMAGE_PATH=frame.png ENABLE_STEP_DISTILL=true NUM_INFERENCE_STEPS=8 bash hunyuanvideo-1.5/02_run_inference.sh
# 720p with super-resolution:
GPU=0 RESOLUTION=720p ENABLE_SR=true bash hunyuanvideo-1.5/02_run_inference.sh
# Multi-GPU (leave GPU unset so all visible cards are used):
N_INFERENCE_GPU=8 CFG_DISTILLED=true ENABLE_CACHE=true CACHE_TYPE=deepcache bash hunyuanvideo-1.5/02_run_inference.sh
# Prompt rewriting (needs a vLLM endpoint; see official README for T2V_REWRITE_*/I2V_REWRITE_*):
REWRITE=true PROMPT='...' bash hunyuanvideo-1.5/02_run_inference.sh
```
OOM on a >=14 GB card? `export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:128` or `--offloading true` (default) / `--overlap_group_offloading false`.

## Training (03 — your dataset)
`03_train.sh` runs the official `train.py` via `torchrun`, but with a **real dataloader** (`train_dataset.HunyuanVideoDataset`) injected in place of the placeholder `create_dummy_dataloader()` — **no official code is modified**. `train_run.py` does the injection; all official flags pass through unchanged.

### Dataset layout (`DATA_DIR`)
```
my_videos/
├── captions.jsonl          # preferred: one JSON per line
│   # {"file": "videos/clip001.mp4", "text": "a red panda climbing a tree"}
│   # {"file": "images/pic002.jpg",  "text": "a bowl of ramen, top view"}
├── videos/*.mp4            # video clips (data_type="video")
├── images/*.{png,jpg,jpeg,webp,bmp}   # stills (data_type="image")
└── prompts/<stem>.txt      # per-clip caption (used when no captions.jsonl)
```
Caption resolution: `captions.jsonl` → `prompts/<stem>.txt` → the file stem (underscores→spaces).
`data_type` is inferred from the extension: video ext → `"video"` (the trainer randomly picks t2v/i2v by `--i2v_prob`), image ext → `"image"` (always t2v).

### Sample format (what the dataset returns; matches `train.py`)
- `"pixel_values"`: `Tensor[3, F, H, W]` (video) or `Tensor[3, H, W]` (image), **range [-1, 1]**, float32. **F must be 4n+1** (default 41).
- `"text"`: the caption string.
- `"data_type"`: `"video"` | `"image"`.

### Run
```bash
# Full fine-tune, 8 GPUs, Muon optimizer (recommended by the official repo):
DATA_DIR=/data/my_videos N_TRAIN_GPU=8 SP_SIZE=8 \
  LEARNING_RATE=1e-5 BATCH_SIZE=1 MAX_STEPS=10000 WARMUP_STEPS=500 \
  bash hunyuanvideo-1.5/03_train.sh

# LoRA fine-tune (much lighter; single GPU OK):
DATA_DIR=/data/my_videos N_TRAIN_GPU=1 \
  USE_LORA=true LORA_R=8 LORA_ALPHA=16 LEARNING_RATE=1e-4 \
  bash hunyuanvideo-1.5/03_train.sh

# Resume:
RESUME=/path/to/outputs_train/checkpoint-1000 DATA_DIR=/data/my_videos N_TRAIN_GPU=8 bash hunyuanvideo-1.5/03_train.sh
```
Checkpoints land in `OUTPUT_DIR` (default `../HunyuanVideo-1.5/outputs_train/`) as `checkpoint-<step>/{transformer,optimizer,training_state.pt}` (+ `lora/<adapter>/` for LoRA). To use a trained checkpoint for inference, point `generate.py`/`02` at it (see official README).

> The official `train.py`'s `validate()` is a no-op stub; set `VALIDATION_INTERVAL` large or implement your own in a fork if you want sampled videos during training.

## 可能遇到的问题

公司代理做 HTTPS 中间人解密，下面按流水线阶段列出常见报错与修法（命令在服务器上、`doll` 环境已激活时执行）。

**1. clone 本仓 / 官方仓报 SSL / 认证**
公开仓免认证，加 `-c http.sslVerify=false` 即可。`run_all.sh` 已对官方仓做了一次 `sslVerify=false` 兜底。

**2. `pip install -r requirements.txt` 报 `SSL:CERTIFICATE_VERIFY_FAILED` / 超时**
```bash
pip config set global.trusted-host "pypi.org pypi.python.org files.pythonhosted.org download.pytorch.org mirrors.tencent.com"
INSTALL_DEPS=1 bash hunyuanvideo-1.5/00_setup_env.sh
```
仍超时（大文件）：加 `--timeout 600 --retries 10`，或先单独装 torch：`pip install --timeout 600 --retries 10 torch>=2.6.0` 再重跑。

**3. `hf download` 报 `CAS service error : ReqwestMiddleware`**
HF 的 Xet/Rust 通道不认代理。`_env.sh` 已设 `HF_HUB_DISABLE_XET=1`；仍报则彻底卸载：
```bash
pip uninstall -y hf_xet
bash hunyuanvideo-1.5/01_download_models.sh
```

**4. `hf download` 报 `SSLCertVerificationError`**
代理根 CA 不在系统证书包。先一次性建包，`_env.sh` 会自动用 `~/.ca-bundle.crt`：
```bash
bash hunyuanvideo-1.5/setup_ca_bundle.sh    # 抓代理证书链 -> ~/.ca-bundle.crt，并自检
bash hunyuanvideo-1.5/01_download_models.sh
```
- 自检 `[OK]` → 直接重跑 `01`。
- 自检 `[FAIL]` → 把公司根 CA 追加到 `~/.ca-bundle.crt` 后重跑（公司根 CA 常见于 `/usr/local/share/ca-certificates/`，脚本已自动并入）。
- 仍报 SSL（CDN 端点用了不同的 MITM 证书）→ `01` 自动回退到禁用 SSL 校验的下载器（`_hf_download.py`）；或直接 `HF_DISABLE_SSL=1 bash 01_download_models.sh`。

**5. ModelScope（Glyph-SDXL-v2 / byT5 权重）下载失败**
`01` 会先尝试 `pip install modelscope` 再 `modelscope download`。代理下仍失败时手动下载：打开 https://modelscope.cn/models/AI-ModelScope/Glyph-SDXL-v2/files ，把 `checkpoints/byt5_model.pt`（及 `assets/`）放进 `../model/HunyuanVideo-1.5/text_encoder/Glyph-SDXL-v2/` 后重跑。

**6. FLUX.1-Redux-dev（siglip 视觉编码器）需授权**
仅 I2V 需要。去 https://huggingface.co/black-forest-labs/FLUX.1-Redux-dev 申请访问，通过后：
```bash
HF_TOKEN=<your_token> bash hunyuanvideo-1.5/01_download_models.sh
```
只做 T2V 可忽略此步。

**7. flash-attn / SageAttention / flex-block-attn 安装失败**
这些都是**可选**加速内核，默认推理不依赖它们。失败时不影响 480p 基线推理；需要时按官方 README 单独编译（flash-attn：`pip install flash-attn --no-build-isolation`）。

**8. 训练报 `sp_size ... must divide world_size` / `cannot be greater than world_size`**
`SP_SIZE` 必须整除 `N_TRAIN_GPU`。单卡训练用 `SP_SIZE=1`（默认）；8 卡可用 `SP_SIZE=8`（或 1/2/4）。

**9. 训练 OOM**
降 `TRAIN_VIDEO_LENGTH`（保持 4n+1，如 41→21→9）、用 `TRAIN_RESOLUTION=480p`、`USE_LORA=true`、`ENABLE_GRAD_CKPT=true`，或多卡 + `SP_SIZE` 增大。

> 通用：`proxy.env`（代理凭证）在仓内 gitignored，`~/.ca-bundle.crt` 在家目录，都不入库；切勿把凭证写进脚本。

## Config (env vars, all optional)
| var | default | note |
|---|---|---|
| `CONDA_ENV` | `doll` | conda env to activate (must have torch>=2.6 after INSTALL_DEPS) |
| `GPU` | _(unset)_ | physical GPU id to pin, e.g. `GPU=0`; for multi-GPU leave unset |
| `HYVIDEO_DIR` | `../HunyuanVideo-1.5` | official code path |
| `MODEL_DIR` | `../model/HunyuanVideo-1.5` | weights path (ckpts symlinks here) |
| `HYVIDEO_REPO` | official GitHub URL | clone source |
| `INSTALL_DEPS` | `0` (run_all: `1`) | set `1` to `pip install -r requirements.txt` |
| `HF_HUB_DISABLE_XET` | `1` | disable HF Xet/CAS Rust path (proxy-unfriendly) |
| `HF_DISABLE_SSL` | `0` | set `1` to download weights with SSL verification disabled |
| `HF_TOKEN` | _(unset)_ | needed only for the gated FLUX.1-Redux-dev (I2V) |

### Inference (02)
| var | default | note |
|---|---|---|
| `PROMPT` | _(builtin sample)_ | text prompt |
| `IMAGE_PATH` | `none` | `none`=T2V; a path=I2V (needs siglip) |
| `RESOLUTION` | `480p` | `480p` \| `720p` |
| `ASPECT_RATIO` | `16:9` | |
| `VIDEO_LENGTH` | `121` | frames |
| `NUM_INFERENCE_STEPS` | `50` | (step-distill: 8/12) |
| `SEED` | `1` | |
| `N_INFERENCE_GPU` | `1` | torchrun `--nproc_per_node` |
| `DTYPE` | `bf16` | |
| `OFFLOADING` | `true` | CPU offloading (~14 GB VRAM) |
| `OVERLAP_GROUP_OFFLOADING` | `true` | faster but uses more CPU RAM |
| `CFG_DISTILLED` | `false` | use cfg-distilled transformer (2x) |
| `ENABLE_STEP_DISTILL` | `false` | 480p I2V step-distill (~75% faster) |
| `SPARSE_ATTN` | `false` | needs flex-block-attn (720p only) |
| `SAGE_ATTN` | `false` | needs SageAttention built |
| `ENABLE_CACHE` | `false` | `CACHE_TYPE`=deepcache\|teacache\|taylorcache |
| `ENABLE_SR` | `false` | super-resolution to 720p/1080p |
| `REWRITE` | `false` | prompt rewrite (needs vLLM endpoint) |
| `OUTPUT_PATH` | _(auto)_ | explicit mp4 path |

### Training (03)
| var | default | note |
|---|---|---|
| `DATA_DIR` | _(required)_ | dataset root (see layout above) |
| `N_TRAIN_GPU` | `1` | torchrun `--nproc_per_node` |
| `PRETRAINED_TRANSFORMER_VERSION` | `480p_t2v` | which transformer variant to train |
| `LEARNING_RATE` | `1e-5` | (LoRA: try `1e-4`) |
| `BATCH_SIZE` | `1` | per-GPU |
| `MAX_STEPS` | `10000` | |
| `WARMUP_STEPS` | `500` | |
| `GRAD_ACCUM` | `1` | gradient accumulation |
| `I2V_PROB` | `0.3` | prob. of i2v task for video samples |
| `USE_MUON` | `true` | Muon optimizer (official recommendation) |
| `ENABLE_FSDP` | `true` | FSDP2 (no-op when world_size=1) |
| `ENABLE_GRAD_CKPT` | `true` | gradient checkpointing |
| `SP_SIZE` | `1` | sequence-parallel size (must divide `N_TRAIN_GPU`) |
| `USE_LORA` | `false` | LoRA fine-tune |
| `LORA_R` / `LORA_ALPHA` | `8` / `16` | LoRA rank / alpha |
| `TRAIN_VIDEO_LENGTH` | `41` | **must be 4n+1** |
| `TRAIN_RESOLUTION` | `480p` | `480p`→(480,848) `720p`→(720,1280); or `TRAIN_HEIGHT`/`TRAIN_WIDTH` (÷16) |
| `OUTPUT_DIR` | `../HunyuanVideo-1.5/outputs_train` | checkpoints dir |
| `SAVE_INTERVAL` / `LOG_INTERVAL` / `VALIDATION_INTERVAL` | `1000` / `10` / `100` | |
| `RESUME` | _(unset)_ | checkpoint dir to resume from |

## Outputs
- **02 inference**: a single `.mp4` at `OUTPUT_PATH` (default `../HunyuanVideo-1.5/outputs/<res>_<t2v|image>_seed<N>.mp4`).
- **03 training**: `OUTPUT_DIR/checkpoint-<step>/{transformer,optimizer,training_state.pt}` (+ `lora/<adapter>/` for LoRA). Use the checkpoint dir to resume or to drive inference.

## Notes
- Official code & weights follow their own license (HunyuanVideo-1.5 = TENCENT HUNYUAN COMMUNITY LICENSE). This folder only orchestrates; no official code is copied.
- `.gitattributes` (repo root) forces LF so Windows-pushed scripts run cleanly on Ubuntu.
- `proxy.env` (proxy creds / env overrides) is gitignored — never committed. Don't put credentials in scripts.
- SSL behind a TLS-intercepting corporate proxy: pip uses `--trusted-host`; `hf`/`git`/`modelscope` use the CA bundle (`_env.sh` prefers `~/.ca-bundle.crt`, built by `setup_ca_bundle.sh`).

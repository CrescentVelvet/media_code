# MiniMax-H3 (WSL Ubuntu 24.04) — 本地复现指南

本文件是 [`README.md`](README.md) 的 **WSL 本地复现补充**。服务器流程（clone `doll` env、公司代理、CA 证书、`/usr/local/cuda`）见原 README，本文件只讲 WSL 上跑的差异。

> 两个 README 共存：
> - [`README.md`](README.md) — 服务器流程（`doll` env clone + 公司代理 + `/usr/local/cuda`）
> - [`README_wsl.md`](README_wsl.md) — 本文件（WSL 从零建 env + 无代理 + conda 装 CUDA toolkit）

> 📖 WSL2 + Ubuntu 24.04 + NVIDIA 驱动 + Miniconda 的基础环境搭建见 [`wsl/README.md`](../wsl/README.md)（本文件假设已按它配好 WSL + Miniconda）。

## 当前状态（已配置完成，直接跑 06c）

> 本机 RTX 3090 (24GB) + 64GB RAM + D 盘 9.3TB。环境已配好，权重已下完。

| 组件 | 状态 | 路径 |
|---|---|---|
| conda env `minimax_h3` | ✅ 已建（python 3.11 + torch 2.6 cu124 + diffusers 0.40 + torchao 0.15 + transformers 5.15） | `~/miniconda3/envs/minimax_h3` |
| gcc 12（triton JIT 编译用） | ✅ 已装（conda gxx_linux-64=12） | env 内 |
| MiniMax-H3 权重（60 文件 135GB） | ✅ 全部下完 | `/mnt/d/wheel/minimaxh3_ms` |
| WSL 内存 | ✅ `.wslconfig` memory=56GB + swap=32GB | vhdx 27GB（C 盘） |

**直接跑 06c 常驻服务**（模型在 D 盘，输出也写 D 盘，不吃 C 盘空间）：
```bash
# 启动服务（从 D 盘加载 ~20-40min，然后监听 :8000）
cd /mnt/c/code/media_code
GPU=0 MODEL_PATH=/mnt/d/wheel/minimaxh3_ms \
  DEVICE=cuda:0 MAX_PIXELS=133120 \
  bash minimax_h3/06c_int8_serve.sh

# 另一个终端发请求：
curl -X POST http://localhost:8000/generate \
  -H 'Content-Type: application/json' \
  -d '{"prompt":"a drone shot over mountains","seed":42}'

# 带首帧图的 FL2VA：
curl -X POST http://localhost:8000/generate \
  -H 'Content-Type: application/json' \
  -d '{"prompt":"integrated_multimodal_description: ...","first_frame":"/mnt/d/your_image.jpg","seed":42,"output_name":"rotate.mp4"}'

# 健康检查 / 关闭
curl http://localhost:8000/health
curl -X POST http://localhost:8000/shutdown
```

> 06a int8 虽然注释说"单卡 80GB 常驻"，实际代码用了 `enable_group_offload(block_level)`——transformer/text_encoder 按 block offload 到 CPU，GPU 只占当前 block (~2-3GB) + VAE (~3GB) + 激活。RTX 3090 24GB **够用**，但需要 ~56GB CPU RAM 装 int8 后的 ~62GB 权重（部分走 swap）。

---

## ⚠️ 显存现实检查（先看这个）

MiniMax-H3 是 **33B Omni-Transformer + 62GB Qwen3-VL-32B 编码器** ≈ **120GB bf16**。本地复现前先对照你的卡：

| 硬件 | 02 serve（SGLang 多卡） | 06（diffusers bf16） | 06a/06c（int8 + block offload） | 06b（Turbo LoRA 4 步） | 07 FlashVSR |
|---|---|---|---|---|---|
| **4× A100 80GB**（服务器） | ✅ FSDP 容量配方 | ✅ 两卡分拆 | ✅ 单卡常驻 | ✅ 最快 | ✅ |
| **2× RTX 5090 32GB** | ⚠️ 逐层 offload（慢） | ❌ 放不下 | ✅ block offload | ⚠️ offload | ✅ |
| **单 RTX 3090 24GB**（WSL 常见） | ❌ 放不下 | ❌ 放不下 | ✅ **block offload**（~56GB CPU RAM） | ⚠️ offload | ✅ |

> **RTX 3090 单卡用户**：用 **06c int8 常驻服务**（加载一次 D 盘模型，多次请求不重新加载）。06a/06c 的代码用了 `enable_group_offload(block_level)`——transformer 按 block 搬到 CPU，GPU 只占当前 block (~2-3GB) + VAE (~3GB)，24GB 够用。需 ~56GB CPU RAM（`.wslconfig` `memory=56GB`）。
>
> 06b Turbo LoRA 需 ~124GB CPU RAM（bf16 全量 offload），单机 64GB 跑不动，不推荐。

## 与服务器版的核心差异

| | 服务器 (`00_setup_env.sh`) | WSL (`00a_setup_env.sh`) |
|---|---|---|
| conda env | `conda create -n minimax_h3 --clone doll`（本地复制不走 conda 通道，绕开 TUNA 403 + 代理 SSL） | `conda create -n minimax_h3 python=3.11 -y` + `pip install torch cu124`（无 `doll` env，直连 PyPI 无 SSL 问题） |
| CUDA toolkit (nvcc) | 系统 `/usr/local/cuda-12.x`（运维装好） | conda install `nvidia/label/cuda-12.4.0`（env 内，无需 sudo；仅 07 FlashVSR BSA kernel 编译需要，可 `SKIP_CUDA_TOOLKIT=1` 跳过） |
| gcc | conda `gxx_linux-64=12` | 同（pin `python=3.11` 防止 GraalPy 降级） |
| 代理 / CA | `proxy.env` 填公司代理密码 + `~/.ca-bundle.crt` 抓代理根 CA | 不需要（直连互联网；`proxy.env` 只放路径覆盖 + HF_TOKEN） |
| 仓库位置 | `/mnt/c/code/`（与 media_code 同级，drvfs 9p 协议） | `~/repos/`（Linux 文件系统，编译 BSA CUDA kernel + SGLang JIT 快 5-10×） |
| 权重 / 输出 | `/mnt/c/code/model/`（Windows fs） | `~/model/`、`~/output/minimaxh3_rotate_results/`（Linux fs，模型加载快） |
| SGLang / diffusers 源码 | `/tmp/sglang-src`、`/tmp/diffusers-src`（重启清空，每次重跑 06/00 都要重 clone） | `~/repos/sglang-src`、`~/repos/diffusers-src`（持久化，editable install 跨重启保留） |
| 07 FlashVSR / BSA 仓 | 运行时 clone 到 `/mnt/c/code/` 同级 | 00a 预 clone 到 `~/repos/`（Linux fs，编译 BSA CUDA kernel 快） |

> **为什么仓库 / 权重放 Linux 文件系统？** `/mnt/c` 是 Windows drvfs（9p 协议），文件 I/O 慢 5-10 倍。SGLang 加载 120GB 权重、BSA 编译 CUDA kernel、diffusers editable install 都吃 I/O，放 Linux fs 避免超时和性能问题。路径通过 `proxy.env` 自动覆盖，脚本 01-08 不需改。

## 前提条件

1. **WSL Ubuntu 24.04** 已按 [`wsl/README.md`](../wsl/README.md) 装好（含 GPU 直通、Miniconda）。验证：
   ```bash
   nvidia-smi                 # 看到 GPU + 驱动版本
   conda --version            # conda 在 PATH 上（已 source ~/.bashrc）
   ```
2. **磁盘空间**（Linux fs，即 vhdx）：
   - env（torch cu124 + sglang[all] + cuda-toolkit + diffusers）≈ **15GB**
   - MiniMax-H3 FL2VA 权重 ≈ **120GB**（bf16 33B + 32B）
   - MiniMax-H3-Turbo LoRA ≈ **2GB**
   - FlashVSR 权重 ≈ **10GB**（可选，仅 07）
   - 输出视频 ≈ 每分钟 ~200MB
   - **合计 ≥ 150GB**。`wsl/README.md` 默认 `vhdxSize=100GB` 不够，编辑 `C:\Users\<你>\.wslconfig` 调到 `vhdxSize=256GB`（或更大），`wsl --shutdown` 后重启。
   - 权重也可放 `/mnt/c/...` 或 `/mnt/d/...`（Windows / 机械盘），但加载慢；改 `MODEL_DIR=/mnt/c/...` 覆盖 `proxy.env` 默认。
3. **HuggingFace 账号**：MiniMax-H3 是 Community License，可能 gated。先去 https://huggingface.co/MiniMaxAI/MiniMax-H3 接受协议、建 read token。

## 首次准备

```bash
# 进入 WSL（如果还没进）
wsl -d Ubuntu2404
cd ~                                  # 避免 /mnt/c/Windows/system32 启动坑（见 wsl/README.md 踩坑 9）

# 进入 media_code 目录
cd /mnt/c/code/media_code

# 一键安装（建 env + 装 torch cu124 + nvcc + gcc12 + SGLang[all] + 热修 + diffusers + 预 clone FlashVSR/BSA）
bash minimax_h3/00a_setup_env.sh

# 选项：
#   SKIP_CUDA_TOOLKIT=1 bash minimax_h3/00a_setup_env.sh   # 跳过 nvcc（不跑 07 可省 ~3GB + 装时）
#   SKIP_FLASHVSR=1     bash minimax_h3/00a_setup_env.sh   # 跳过 FlashVSR/BSA clone（不跑 07 可省时）

source ~/.bashrc                      # 让 conda 在交互 shell 里可用

# 填 HF_TOKEN（MiniMax-H3 可能 gated）
nano /mnt/c/code/media_code/proxy.env
# 取消注释 `export HF_TOKEN="hf_xxx"` 并填入你的 token

# 下权重（FL2VA ~120GB；HF_ENDPOINT 镜像已设在 proxy.env）
HF_TOKEN=hf_xxx bash minimax_h3/01_download_models.sh

# （可选）下 Turbo LoRA（06b 要，~2GB）
hf download lightx2v/Minimax-h3-Turbo \
  minimax_h3_fl2v_turbo_4step_v1.0_768p_bf16.safetensors \
  --local-dir ~/model/MiniMax-H3-Turbo
```

安装完成后验证 SGLang Diffusion 参数：
```bash
conda activate minimax_h3
sglang serve --help | grep -E -- '--model-variant|--performance-mode'
# 期望: 看到 --model-variant / --performance-mode 两行（Diffusion serving ok）
```

### 07 FlashVSR 依赖（仅跑 07 才需）

07 复用 minimax_h3 env（不建独立 env）。00a 已预 clone FlashVSR + BSA 到 `~/repos/`。还需编译 BSA CUDA kernel + 装 diffsynth：
```bash
# 已 conda activate minimax_h3（同 06/06a/06b）
INSTALL_DEPS=1 bash minimax_h3/07_flashvsr_sr.sh    # 编 BSA kernel（需 nvcc + 足够内存；00a 已装 nvcc）
# 权重（v1.1 推荐）下到 ~/model/FlashVSR/：
#   https://huggingface.co/JunhaoZhuang/FlashVSR-v1.1
#   diffusion_pytorch_model_streaming_dmd.safetensors / Wan2.1_VAE.pth / LQ_proj_in.ckpt / TCDecoder.ckpt
```
> ⚠️ Block-Sparse-Attention 编译吃内存（多 ninja 并发易 OOM）。WSL 内存配额见 `wsl/README.md` 第 5 步 `.wslconfig` 的 `memory=` 字段，建议 ≥ 16GB。
> 权重放 `~/model/FlashVSR/`（`FLASHVSR_MODEL_DIR` 默认指向它，`proxy.env` 已覆盖）。

## 路径布局（WSL）

```
~/                                # Linux 文件系统（vhdx，I/O 快）
├── repos/                        # 官方仓库 + 源码（00a clone，Linux fs 编译快）
│   ├── sglang-src/               # SGLang git main（00a clone + editable install + 热修）
│   ├── diffusers-src/            # diffusers git main（06/06a/06b 首次跑时 clone，持久化避免重 clone）
│   ├── FlashVSR/                 # 07 仓（diffsynth + examples/WanVSR/utils）
│   └── Block-Sparse-Attention/   # 07 仓（LCSA CUDA kernel，需 nvcc 编译）
├── model/                        # 权重（Linux fs，加载快）
│   ├── MiniMax-H3/               # FL2VA/ + Ref2VA/（HF 快照，SGLang --model-path 指它）
│   ├── MiniMax-H3-Turbo/         # 06b Turbo LoRA（~2GB）
│   └── FlashVSR/                 # 07 权重（v1.1，~10GB）
├── output/
│   └── minimaxh3_rotate_results/  # 输出（视频 + 中间产物）
└── miniconda3/                   # conda 安装（wsl/README.md step 7）
    └── envs/minimax_h3/          # conda env（python=3.11 + torch cu124 + CUDA toolkit + gcc12 + sglang[all]）

/mnt/c/code/media_code/           # 编排脚本（Windows fs，git 仓）
├── proxy.env                     # WSL 路径覆盖 + HF_TOKEN + HF_ENDPOINT（00a 自动生成）
└── minimax_h3/
    ├── 00a_setup_env.sh          # ← WSL 安装脚本（本文件描述）
    ├── _env.sh                   # 共享环境（已加 conda fallback，兼容 WSL 无 conda init 的 shell）
    ├── 00_setup_env.sh           # 服务器安装脚本（保留不动；00a 内部复用它装 SGLang）
    ├── 01_download_models.sh    # 以下脚本与服务器共用，通过 proxy.env 自动用 WSL 路径
    ├── 02_serve.sh               # SGLang serve（多卡；单卡 WSL 不推荐）
    ├── 03_generate.sh            # 发请求给 02 服务
    ├── 06_diffusers_inference.sh # diffusers 直接推理（bf16，需多卡 offload）
    ├── 06a_diffusers_inference.sh # int8 量化（单卡 80GB 常驻；WSL 单 3090 放不下）
    ├── 06b_turbo_lora_inference.sh # ← WSL 单卡推荐：Turbo LoRA 4 步 + auto offload
    ├── 07_flashvsr_sr.sh         # FlashVSR 4× 超分
    └── 08_context_ir.sh         # H3-Context-IR API（非开源，需 MiniMax API token）
```

## 常用命令

> 与服务器版完全一致——`proxy.env` 已写入 WSL 路径覆盖，脚本 01-08 自动用 `~/model/`、`~/repos/`、`~/output/` 等路径。
> 每条命令需显式写出输入路径、输出路径、模型路径（铁律，不全靠脚本默认值）。

```bash
# 在 WSL 内，进入 media_code 目录
cd /mnt/c/code/media_code

# 0) 一键安装（首次）
bash minimax_h3/00a_setup_env.sh

# 1) 下权重（FL2VA ~120GB；需 HF_TOKEN）
HF_TOKEN=hf_xxx bash minimax_h3/01_download_models.sh

# 6b) Turbo LoRA 4 步推理（单卡 bf16 + auto offload，⚠️ WSL 单卡唯一可行路径）
# 先下 LoRA（~2GB）：
#   hf download lightx2v/Minimax-h3-Turbo \
#     minimax_h3_fl2v_turbo_4step_v1.0_768p_bf16.safetensors \
#     --local-dir ~/model/MiniMax-H3-Turbo
GPU=0 \
MODEL_PATH=~/model/MiniMax-H3 \
LORA_PATH=~/model/MiniMax-H3-Turbo/minimax_h3_fl2v_turbo_4step_v1.0_768p_bf16.safetensors \
DEVICE=cuda:0 \
MAX_PIXELS=1032192 \
FPS=24 \
NUM_FRAMES=124 \
SEED=42 \
TASK=fl2va \
FIRST_FRAME=~/my_images/subject.jpg \
PROMPT="integrated_multimodal_description: [Shot 1] Cinematic medium-wide shot. The subject shown in the first frame stands perfectly centered in frame, stock-still and frozen in place — absolutely no body movement. The camera is mounted rigidly on a perfectly horizontal circular ring track at fixed height, gliding along the track in a smooth, perfectly level, constant-speed 360-degree orbit around the subject. [Shot 2] At 00:06.000, the camera completes the full 360-degree revolution and stops exactly at the starting front-facing angle, the subject still perfectly frozen in its original pose.\noverall_soundscape: A near-silent, steady room tone with only a faint, constant ambient hum.\nnon_diegetic_music: A single sustained ambient synth drone, unchanging and continuous throughout." \
OUTPUT_DIR=~/output/minimaxh3_rotate_results/results_turbo \
OUTPUT_NAME=rotate_360_turbo.mp4 \
bash minimax_h3/06b_turbo_lora_inference.sh

# 7) FlashVSR 4× 超分（可选；把 06b 低分辨率输出超分到高清）
# 首次需 INSTALL_DEPS=1 编 BSA kernel（00a 已装 nvcc + 预 clone 仓）
GPU=0 \
MUX_AUDIO=0 \
INPUT=~/output/minimaxh3_rotate_results/results_turbo/rotate_360_turbo.mp4 \
FLASHVSR_MODEL_DIR=~/model/FlashVSR \
RESULTS_DIR=~/output/minimaxh3_rotate_results/results_sr \
OUTPUT_NAME=rotate_360_turbo_sr.mp4 \
bash minimax_h3/07_flashvsr_sr.sh

# 8) H3-Context-IR API（可选；短 prompt → 长描述，效果比短 prompt 好）
# ⚠️ 需 MiniMax API token；FIRST_FRAME 必须是 http URL
MINIMAX_API_KEY=xxx \
PROMPT="视频中的人物保持绝对静止，相机围绕画面中心水平旋转一圈 360°" \
FIRST_FRAME=https://raw.githubusercontent.com/CrescentVelvet/media_code/main/minimax_h3/examples/681533632532078.jpg \
bash minimax_h3/08_context_ir.sh
```

## 多卡 serve（02 — 仅多 GPU WSL 用，单卡跳过）

单 RTX 3090 (24GB) 跑不了 02 serve（120GB 权重放不下）。多卡 WSL 可按服务器配方跑，路径换成 `~/model/MiniMax-H3`：

```bash
# 2× RTX 5090 32GB 逐层 offload（慢但能塞下；64GB 共需靠 offload 搬运）
GPU=0,1 NUM_GPUS=2 MODEL_PATH=~/model/MiniMax-H3 \
EXTRA_SGLANG_FLAGS="--performance-mode memory --layerwise-offload-components dit,text_encoder,vae --dit-layerwise-resident-layers 20" \
bash minimax_h3/02_serve.sh

# 4× A100 80GB（如 WSL 机器有 4 卡）— FSDP 容量配方（最稳）
GPU=0,1,2,3 NUM_GPUS=4 ULYSSES_DEGREE=4 USE_FSDP=1 \
  MODEL_PATH=~/model/MiniMax-H3 bash minimax_h3/02_serve.sh
# 起好服务后发请求：
GPU=0,1,2,3 SERVER_URL=http://localhost:30010 \
  TASK=fl2va FIRST_FRAME=~/my_images/first.png DURATION=8 \
  PROMPT="..." \
  OUTPUT_DIR=~/output/minimaxh3_rotate_results/results/fl2va OUTPUT_NAME=fl2va.mp4 \
  bash minimax_h3/03_generate.sh
```

其他 02 配方（TP2+Ulysses2、offload 选项、BG 后台模式等）见 [`README.md` 的「Serving」段](README.md#serving-02--sglang-起-h3-base-服务)。

## 可能遇到的问题（WSL 专属）

**1. `conda: command not found`（运行 01-08 脚本时）**
00a 末尾会执行 `conda init bash`。如果你还没 `source ~/.bashrc`，conda 不在 PATH 上。解决：
```bash
source ~/.bashrc
# 或者每次手动 source：
source ~/miniconda3/etc/profile.d/conda.sh
```
`_env.sh` 已加 fallback：如果 conda 不在 PATH，会自动找 `~/miniconda3`。但仍建议 `source ~/.bashrc` 确保万无一失。

**2. `torch.cuda.OutOfMemoryError`（06b Turbo LoRA）**
RTX 3090 只有 24GB，06b 用 auto CPU offload 把 60GB 权重搬CPU↔GPU，但仍可能 OOM。降压：
```bash
# 降分辨率上限（默认 1344×768；544p 配方 960×544 更省）
MAX_PIXELS=522240 \
LORA_PATH=~/model/MiniMax-H3-Turbo/minimax_h3_fl2v_turbo_4step_v0.1.safetensors \
  ... bash minimax_h3/06b_turbo_lora_inference.sh
```
> ⚠️ 544p checkpoint 配方不同：`VIDEO_SHIFT=12 LORA_ALPHA=8`（不是 768p 的 `6/128`）。脚本默认按 768p；跑 544p 要同时设 `VIDEO_SHIFT=12 LORA_ALPHA=8`。详见 [`README.md` Turbo LoRA 参数表](README.md#turbo-lora-06b)。

**3. `06b` 跑得极慢（10-30 分钟一段）**
正常——auto CPU offload 在 24GB 卡上搬运 60GB 权重，每步去噪都过一次。要快：
- 上多卡（02 serve + 03 generate，走 SGLang 多卡并行）
- 或换更大显存卡（≥ 80GB 才能常驻）

**4. SGLang JIT kernel 编译失败（`nvcc: command not found` 或 `ninja exited with status 1`）**
00a 通过 conda 装 `gxx_linux-64=12` + `cuda-toolkit`（nvcc）。如果 JIT 仍失败：
```bash
conda activate minimax_h3
which nvcc             # 应为 ~/miniconda3/envs/minimax_h3/bin/nvcc
which g++              # 应为 .../x86_64-conda-linux-gnu-g++
nvcc --version

# 如果 g++ 不在，手动装：
conda install -y -c conda-forge gxx_linux-64=12 python=3.11

# 清旧 JIT 失败缓存后重启 02：
CLEAR_JIT_CACHE=1 bash minimax_h3/02_serve.sh
```

**5. `git clone` 超时 / SSL 错误**
WSL 直连互联网通常无此问题。如果遇到（网络不稳定），脚本已内置 `git -c http.sslVerify=false` fallback。手动重试：
```bash
cd ~/repos && git clone --depth 1 https://github.com/OpenImagingLab/FlashVSR.git
```

**6. `hf download` 超时 / SSL 错误**
`proxy.env` 已设 `HF_ENDPOINT=https://hf-mirror.com` 镜像加速。仍失败：
```bash
# 临时关 SSL 验证（HF_DISABLE_SSL=1 走兜底下载器）
HF_DISABLE_SSL=1 HF_TOKEN=hf_xxx bash minimax_h3/01_download_models.sh
```

**7. `hf download` 报 401 / `repository not found`（MiniMax-H3 gated）**
HF 上 MiniMax-H3 是 Community License，需接受协议。去 https://huggingface.co/MiniMaxAI/MiniMax-H3 点 Accept，建 read token，再：
```bash
HF_TOKEN=hf_xxx bash minimax_h3/01_download_models.sh
```

**8. `vhdx` 满了（磁盘空间不足）**
MiniMax-H3 权重 ~120GB，`wsl/README.md` 默认 `vhdxSize=100GB` 不够。编辑 `C:\Users\<你>\.wslconfig`：
```ini
[wsl2]
memory=24GB
processors=8
swap=16GB
vhdxSize=256GB         # ← 调大到 256GB+
```
然后 PowerShell `wsl --shutdown` + 重启。已用的空间不会自动回收，按 `wsl/README.md` 第 8 步 `diskpart compact` 压缩。

**9. 编译 BSA CUDA kernel 极慢（超过 30 分钟）**
确认 BSA_DIR 指向 `~/repos/Block-Sparse-Attention`（Linux fs），不是 `/mnt/c/`。如果误 clone 到 `/mnt/c/`：
```bash
mv /mnt/c/code/Block-Sparse-Attention ~/repos/
# proxy.env 已设 BSA_DIR=~/repos/Block-Sparse-Attention，自动生效
```
内存不足导致 ninja 并发 OOM：调大 `.wslconfig` 的 `memory=` 字段，或限制 ninja 并发：
```bash
MAX_JOBS=2 INSTALL_DEPS=1 bash minimax_h3/07_flashvsr_sr.sh
```

**10. GPU 驱动版本太旧**
WSL 内 `nvidia-smi` 显示的 CUDA Version 是驱动支持的最高版本。torch cu124 需要驱动 ≥ 525.60。如果驱动太旧，更新 Windows 上的 NVIDIA 驱动（WSL 不装显卡驱动，走 Windows 那套）。

**11. 想切回 cu121 / cu126**
```bash
CUDA_TOOLKIT_LABEL=nvidia/label/cuda-12.1.1 \
TORCH_INDEX_URL=https://download.pytorch.org/whl/cu121 \
bash minimax_h3/00a_setup_env.sh
```

**12. 跑 `.sh` 报 `syntax error near unexpected token ('（CRLF 行尾）**
Windows 编辑器（非 git checkout）保存的 .sh 带 CRLF 行尾。本仓 `.gitattributes` 强制 LF，但只有 `git checkout/pull` 才落 LF，非 git 传输不会转。
```bash
file minimax_h3/00a_setup_env.sh     # 出现 "CRLF line terminators" 即中招
find minimax_h3 -name '*.sh' -exec sed -i 's/\r$//' {} +    # 一次性修所有 .sh
# 或 git checkout -- minimax_h3/
```

## 通用问题

其他问题（SGLang Diffusion 参数缺失、`--model-variant` 不识别、`qwen3_asr` 重复注册、`only_qv` 参数不兼容、02 OOM、generate 报 5xx、Ref2VA `file://` 路径、Full 2K workflow 调 API 等）见 [`README.md` 的「可能遇到的问题」](README.md#可能遇到的问题) 段，排错方法通用（00a 复用 00_setup_env.sh 的全部 SGLang 热修逻辑）。

## Config (env vars, WSL 相关)

| var | WSL 默认 | note |
|---|---|---|
| `CONDA_ENV` | `minimax_h3` | 专用 env（SGLang pin 与其他算法冲突） |
| `MODEL_DIR` / `MODEL_PATH` | `~/model/MiniMax-H3` | HF 权重快照（00a 写入 proxy.env） |
| `RESULTS_DIR` | `~/output/minimaxh3_rotate_results` | 输出（00a 写入 proxy.env） |
| `FLASHVSR_DIR` | `~/repos/FlashVSR` | 07 官方仓（00a 预 clone） |
| `BSA_DIR` | `~/repos/Block-Sparse-Attention` | 07 BSA 仓（00a 预 clone） |
| `FLASHVSR_MODEL_DIR` | `~/model/FlashVSR` | 07 权重 |
| `SGLANG_SRC` | `~/repos/sglang-src` | SGLang git main（00a 持久化路径） |
| `DIFFUSERS_SRC` | `~/repos/diffusers-src` | diffusers git main（06 持久化路径） |
| `HF_ENDPOINT` | `https://hf-mirror.com` | HF 镜像（00a 写入 proxy.env） |
| `HF_TOKEN` | _(unset，需手填)_ | MiniMax-H3 gated 仓才需（00a 在 proxy.env 留占位） |
| `HF_DISABLE_SSL` | `0` | `1` = SSL 免校验下载（HF 镜像仍失败时兜底） |
| `SKIP_CUDA_TOOLKIT` | `0` | `1` = 跳过 conda 装 nvcc（不跑 07 可省） |
| `SKIP_FLASHVSR` | `0` | `1` = 跳过 FlashVSR/BSA clone（不跑 07 可省） |
| `TORCH_INDEX_URL` | `https://download.pytorch.org/whl/cu124` | 切 cu121/cu126 用 |
| `CUDA_TOOLKIT_LABEL` | `nvidia/label/cuda-12.4.0` | conda 装 nvcc 的通道标签 |

其他 env vars（`GPU`、`TASK`、`PROMPT`、`FIRST_FRAME`、`NUM_FRAMES`、`SEED`、`NUM_INFERENCE_STEPS`、`VIDEO_SHIFT`、`LORA_ALPHA`、`LORA_PATH`、`MAX_PIXELS`、`PIPELINE`、`SCALE`、`MUX_AUDIO`、`INSTALL_DEPS` 等）见 [`README.md` 的 Config 段](README.md#config-env-vars-all-optional)，与服务器完全一致。

# Wan2.2-TI2V-5B runner

在 Ubuntu + NVIDIA 服务器上跑 [DiffSynth-Studio](https://github.com/modelscope/DiffSynth-Studio) 的 **Wan2.2-TI2V-5B**（文/图生视频）**推理 / 数据集构建 / LoRA 训练**。本目录只含编排脚本——官方代码自动 clone、权重从本地 `$MODEL_DIR` 加载（已下载好）。

## 常用命令

> 假设已进入容器并 `conda activate wan22`；`GPU=0` 按需换卡。首次跑前先做下方「首次准备」。

```bash
# ── 推理(02) ──
# 1) 文生视频(T2V) —— 默认 prompt
GPU=0 bash wan22/02_run_inference.sh
# 2) 文生视频 —— 自定义 prompt + 分辨率 + 帧数
GPU=0 PROMPT="一只小狗在草地上奔跑，阳光明媚。" HEIGHT=480 WIDTH=832 NUM_FRAMES=49 \
  OUTPUT_NAME=my_t2v bash wan22/02_run_inference.sh
# 3) 图生视频(I2V) —— 指定输入图片
GPU=0 PROMPT="猫咪在拳击台上搏斗" INPUT_IMAGE=/path/to/cat.jpg \
  OUTPUT_NAME=my_i2v bash wan22/02_run_inference.sh
# 4) 低显存推理(磁盘 offload，慢但小卡也能跑)
GPU=0 LOW_VRAM=1 bash wan22/02_run_inference.sh
# 5) 用自己训的 LoRA 推理
GPU=0 WEIGHT_PATH=../wan22_experiments/exp1/epoch-4.safetensors \
  PROMPT="..." bash wan22/02_run_inference.sh

# ── 数据集构建(03) ──
# 6) 从视频文件夹建 metadata.csv(配对 .txt prompt 文件夹可选)
DATA_DIR=/data_3d/w00xxxxxx/code/wan22_dataset \
  PROMPT="高质量视频" bash wan22/03_build_dataset.sh
# 7) 逐视频 prompt(与视频同名 .txt，放 TXT_DIR)
DATA_DIR=/data_3d/w00xxxxxx/code/wan22_dataset \
  TXT_DIR=/data_3d/w00xxxxxx/code/wan22_dataset/prompts \
  bash wan22/03_build_dataset.sh

# ── LoRA 训练(04) ──
# 8) 开始训练(默认 lr=1e-4, 5 epochs, rank=32, 480x832x49f)
GPU=0 DATASET_BASE_PATH=/data_3d/w00xxxxxx/code/wan22_dataset \
  bash wan22/04_train_lora.sh
# 9) 低显存训练(CPU offload，消费级 GPU 也能训)
GPU=0 LOW_VRAM_TRAIN=1 DATASET_BASE_PATH=... bash wan22/04_train_lora.sh
# 10) 多卡训练(先跑一次 accelerate config 选 multi-GPU)
N_TRAIN_GPU=8 DATASET_BASE_PATH=... bash wan22/04_train_lora.sh
# 11) 自定义超参
GPU=0 DATASET_BASE_PATH=... LEARNING_RATE=5e-5 NUM_EPOCHS=10 \
  LORA_RANK=64 HEIGHT=480 WIDTH=832 NUM_FRAMES=49 \
  SAVE_STEPS=500 bash wan22/04_train_lora.sh
```

- 结果：训练 → `../wan22_experiments/<exp>/epoch-*.safetensors`（LoRA 权重）；推理 → `../wan22_results/<name>.mp4`。
- 想看训练曲线：`tensorboard --logdir ../wan22_experiments/exp1 --port 6006`（需 `ENABLE_TENSORBOARD=1` 训练时开启）。
- prompt 默认中文示例（两只橘猫拳击）；`NEGATIVE_PROMPT` 默认官方负向提示词。

## 首次准备
```bash
cd <your-code-dir>            # e.g. /data_3d/<uid>/code
git -c http.sslVerify=false clone https://github.com/CrescentVelvet/media_code.git
cd media_code && cp proxy.env.example proxy.env   # 填 http_proxy / https_proxy
conda create -n wan22 python=3.10 -y && conda activate wan22
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
INSTALL_DEPS=1 bash wan22/00_setup_env.sh          # clone DiffSynth-Studio + pip install -e .
bash wan22/01_verify_models.sh                    # 确认权重在位(你已下载到 $MODEL_DIR)
```
模型需已在 `$MODEL_DIR`（默认 `/data_3d/w00xxxxxx/model`）下，结构如下（两种皆可，01 脚本自动建符号链接兼容）：
```
$MODEL_DIR/
  Wan-AI/Wan2.2-TI2V-5B/           # 或 Wan2.2-TI2V-5B/（无 org 前缀，01 自动 link）
    diffusion_pytorch_model*.safetensors   # DiT 主模型(可能分片)
    models_t5_umt5-xxl-enc-bf16.pth        # T5 文本编码器
    Wan2.2_VAE.pth                         # VAE
  Wan-AI/Wan2.1-T2V-1.3B/           # tokenizer 来自这个 repo
    google/umt5-xxl/                       # UMT5-XXL tokenizer 文件夹
```
下载命令（如需补下）：
```bash
modelscope download Wan-AI/Wan2.2-TI2V-5B --local_dir $MODEL_DIR/Wan-AI/Wan2.2-TI2V-5B
modelscope download Wan-AI/Wan2.1-T2V-1.3B --local_dir $MODEL_DIR/Wan-AI/Wan2.1-T2V-1.3B --include google/umt5-xxl/
```

---

以下为详细参考（流程原理 / 各脚本参数 / 排错 / 目录布局）。

## Inference (02 — 更多用法)
`02_run_inference.sh` 调 `run_inference.py`：加载 `WanVideoPipeline`（DiffSynth-Studio），生成一段视频并保存为 mp4，打印模型加载 + 生成耗时 + 输出路径。

### 模型加载机制
DiffSynth-Studio 的 `ModelConfig` 有两种模式：
- `ModelConfig(model_id="Wan-AI/Wan2.2-TI2V-5B", origin_file_pattern="...")` —— 从 ModelScope/HF 下载到 `local_model_path/model_id/`
- `ModelConfig(path="/abs/path/to/file")` —— 直接用本地路径，不下载

本仓设了两个环境变量（`_env.sh`），让 `model_id` 模式也能走本地、不下载：
- `DIFFSYNTH_MODEL_BASE_PATH=$MODEL_DIR` —— 告诉 ModelConfig 在 `$MODEL_DIR/<model_id>/` 下找文件
- `DIFFSYNTH_SKIP_DOWNLOAD=True` —— 跳过下载，只用本地已有文件

所以 Python 代码和官方示例完全一致（`ModelConfig(model_id=..., origin_file_pattern=...)`），靠环境变量重定向到本地。`01_verify_models.sh` 确保目录结构匹配（org 前缀 `Wan-AI/` 有无都自动兼容）。

### T2V vs I2V
- **T2V（文生视频）**：只传 `PROMPT`，不传 `INPUT_IMAGE`。模型从纯噪声 + 文本生成视频。
- **I2V（图生视频）**：传 `PROMPT` + `INPUT_IMAGE=/path/to/img`。图片被 resize 到 `WIDTH×HEIGHT`，作为首帧条件，模型生成后续帧。Wan2.2-TI2V-5B 的 "TI2V" = Text+Image-to-Video，两种模式都支持。

### LoRA 加载
训练产物是 `epoch-N.safetensors`（仅 LoRA 参数）。推理时 `pipe.load_lora(pipe.dit, weight_path, alpha=1)` 把 LoRA 叠加到 DiT 上。`alpha=1` 表示不缩放（与训练一致）。

### 低显存推理
`LOW_VRAM=1` 启用磁盘 offload：模型权重按层在 disk→CPU→GPU 之间搬运，大幅降低显存峰值（但慢很多）。VRAM 限制默认取 `free_vram - 2 GB`，可用 `VRAM_LIMIT=10` 指定。

## Dataset construction (03 — video folder → metadata.csv)
DiffSynth-Studio 的 `UnifiedDataset` 读 `metadata.csv`（`--dataset_metadata_path`），视频路径相对于 `--dataset_base_path`。`03_build_dataset.sh` + `build_dataset.py` 做的就是：
1. 递归扫描 `DATA_DIR` 下的视频文件（mp4/mov/avi/webm/mkv/flv）。
2. 每个视频配一个 prompt：从 `TXT_DIR` 找同名 `.txt`（优先按相对路径匹配，回退到文件名），找不到则用固定 `PROMPT`。
3. 写 `metadata.csv`，两列：`video`（相对 `DATASET_BASE_PATH` 的路径）、`prompt`（文本）。

> TI2V 训练不需要单独的图片列——`--extra_inputs "input_image"` 让训练器自动取每个训练视频的**第一帧**作为图片条件。所以你只需准备「视频 + prompt」。

```bash
# 固定 prompt(所有视频同一句)
DATA_DIR=/data/videos PROMPT="高质量视频" bash wan22/03_build_dataset.sh
# 逐视频 prompt(DATA_DIR/prompts/<name>.txt 与视频同名)
DATA_DIR=/data/videos TXT_DIR=/data/videos/prompts bash wan22/03_build_dataset.sh
# 自定义输出路径
DATA_DIR=/data/videos METADATA_OUT=/data/my_metadata.csv bash wan22/03_build_dataset.sh
```

## Training (04 — LoRA fine-tune)
`04_train_lora.sh` 调官方 `accelerate launch examples/wanvideo/model_training/train.py`，对 DiT 加 LoRA 微调。核心参数（与官方 `Wan2.2-TI2V-5B.sh` 一致）：
- `--lora_base_model "dit"` —— LoRA 加到 DiT
- `--lora_target_modules "q,k,v,o,ffn.0,ffn.2"` —— DiT 的 attention + FFN 层
- `--lora_rank 32` —— LoRA 秩
- `--extra_inputs "input_image"` —— TI2V 模式：每视频第一帧作图片条件
- `--remove_prefix_in_ckpt "pipe.dit."` —— checkpoint 去前缀
- `--use_gradient_checkpointing` —— 省显存（强制开启，关了会 OOM）

### 训练流程
```
metadata.csv (video + prompt)
    └─ UnifiedDataset : LoadVideo(num_frames) + ImageCropAndResize(H, W)
         └─ WanTrainingModule : WanVideoPipeline (DiT + T5 + VAE)
              ├─ DiT 上加 LoRA (q,k,v,o,ffn.0,ffn.2, rank=32)
              ├─ 第一帧 → input_image (extra_inputs)
              ├─ 视频 → VAE encode → 潜变量
              ├─ 加噪 → DiT 预测 → FlowMatch SFT loss
              └─ 仅 LoRA 参数有梯度，反传更新
```

每个 epoch 存一个 `epoch-N.safetensors`（LoRA 权重）。设 `SAVE_STEPS=500` 改为每 500 步存。

### 训练参数
| var | default | note |
| --- | --- | --- |
| `DATASET_BASE_PATH` | _(required)_ | 数据集根目录（含 metadata.csv） |
| `METADATA_PATH` | `$DATASET_BASE_PATH/metadata.csv` | metadata 路径 |
| `OUTPUT_DIR` | `../wan22_experiments/exp1` | checkpoint + 日志目录 |
| `HEIGHT` / `WIDTH` | `480` / `832` | 训练分辨率（小 = 省显存） |
| `NUM_FRAMES` | `49` | 训练帧数（4k+1，Wan 的帧数约束） |
| `DATASET_REPEAT` | `100` | 每 epoch 重复数据集次数 |
| `LEARNING_RATE` | `1e-4` | LoRA 学习率 |
| `NUM_EPOCHS` | `5` | 训练轮数 |
| `LORA_RANK` | `32` | LoRA 秩 |
| `LORA_TARGET_MODULES` | `q,k,v,o,ffn.0,ffn.2` | LoRA 目标层 |
| `GRAD_ACCUM` | `1` | 梯度累积步数 |
| `SAVE_STEPS` | _(空=每epoch)_ | 每 N 步存一次 checkpoint |
| `LOW_VRAM_TRAIN` | `0` | `1` = CPU offload 训练（小卡可用） |
| `N_TRAIN_GPU` | _(unset)_ | `>1` → 多卡（先 `accelerate config`） |
| `ENABLE_TENSORBOARD` | `0` | `1` = 开 TensorBoard 日志 |

## Config (env vars, all optional)
| var | default | note |
|---|---|---|
| `CONDA_ENV` | `doll` | conda env to activate (dedicated `wan22` recommended) |
| `GPU` | _(unset)_ | physical GPU id to pin, e.g. `GPU=0`; leave unset for multi-GPU training |
| `DIFFSYNTH_DIR` | `../DiffSynth-Studio` | official code path |
| `MODEL_DIR` | `../../model` | weights path (model root; expects `Wan-AI/Wan2.2-TI2V-5B/` under it) |
| `DIFFSYNTH_REPO` | official GitHub URL | clone source |
| `INSTALL_DEPS` | `0` (run_all: `1`) | set `1` to `pip install -e .` the diffsynth package |
| `DIFFSYNTH_SKIP_DOWNLOAD` | `True` | skip model download, load from local `$MODEL_DIR` |
| `DIFFSYNTH_DOWNLOAD_SOURCE` | `modelscope` | `modelscope` or `HuggingFace` |
| `DIFFSYNTH_MODEL_BASE_PATH` | `$MODEL_DIR` | where ModelConfig looks for model files |

### Inference (02)
| var | default | note |
|---|---|---|
| `PROMPT` | 橘猫拳击示例 | 生成提示词 |
| `NEGATIVE_PROMPT` | 官方负向提示词 | 负向提示词 |
| `INPUT_IMAGE` | _(unset)_ | 设为图片路径 = I2V；不设 = T2V |
| `WEIGHT_PATH` | _(unset)_ | 训练好的 LoRA `.safetensors`；不设 = 原生模型 |
| `OUTPUT_DIR` | `../wan22_results` | 视频输出目录 |
| `OUTPUT_NAME` | `video` | 输出文件名（不含扩展名） |
| `HEIGHT` / `WIDTH` | `704` / `1248` | 视频分辨率 |
| `NUM_FRAMES` | `121` | 生成帧数 |
| `SEED` | `0` | 随机种子 |
| `TILED` | `1` | `1` = 分块 VAE（大帧数防 OOM） |
| `FPS` / `QUALITY` | `15` / `5` | 输出 mp4 帧率 / 质量 |
| `LOW_VRAM` | `0` | `1` = 磁盘 offload（慢但省显存） |
| `VRAM_LIMIT` | `free - 2GB` | 低显存模式的 VRAM 上限 (GB) |

## 可能遇到的问题

**1. `import diffsynth` 报 ModuleNotFoundError**
DiffSynth-Studio 没装。跑 `INSTALL_DEPS=1 bash wan22/00_setup_env.sh`（会 `pip install -e .` editable 安装）。

**2. 推理报 model files not found / 下载失败**
`01_verify_models.sh` 检查模型是否在位。模型目录结构需有 `Wan-AI/` org 前缀（modelscope 默认），或在 `Wan-AI/Wan2.2-TI2V-5B/` 下有 `diffusion_pytorch_model*.safetensors`、`models_t5_umt5-xxl-enc-bf16.pth`、`Wan2.2_VAE.pth`。tokenizer 在 `Wan-AI/Wan2.1-T2V-1.3B/google/umt5-xxl/`。如果你的目录没有 `Wan-AI/` 前缀，01 脚本会自动建符号链接兼容。

**3. 推理 OOM（显存不足）**
- 降分辨率 / 帧数：`HEIGHT=480 WIDTH=832 NUM_FRAMES=49`。
- 开磁盘 offload：`LOW_VRAM=1`（慢但能跑）。
- 确认 `TILED=1`（分块 VAE，默认已开）。
- 设 `export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`。

**4. 训练 OOM**
- 降 `HEIGHT`/`WIDTH`/`NUM_FRAMES`（如 `480 832 49`，官方示例就是这套）。
- 开 `LOW_VRAM_TRAIN=1`（CPU offload，消费级 GPU 也能训，但慢）。
- 增 `GRAD_ACCUM=2`（等效翻倍 batch，不增显存）。

**5. 训练多卡 `accelerate launch` 只用一卡**
没配 accelerate 多进程。先 `accelerate config`（选 multi-GPU），再 `N_TRAIN_GPU=8 bash wan22/04_train_lora.sh`（此时**不要**设 `GPU=`）。

**6. 跑 `.sh` 报 `syntax error near unexpected token ('`（CRLF 行尾）**
脚本被 CRLF 行尾污染（Windows→服务器 scp 同步）。修法：
```bash
find wan22 -name '*.sh' -exec sed -i 's/\r$//' {} +
git checkout -- wan22/*.sh    # git 同步的：.gitattributes 还原 LF
```

**7. modelscope / pip 下载报 SSL 证书错误**
公司代理做 HTTPS 中间人解密。先建 CA 包（`bash hypir/setup_ca_bundle.sh`），`_env.sh` 会自动用 `~/.ca-bundle.crt`。或 `HF_DISABLE_SSL=1` 跳过 SSL 校验。

## 目录布局
```
<code-dir>/
├── media_code/                  # 本仓
│   ├── proxy.env                # 代理 + 覆盖项, gitignored
│   └── wan22/                   # 编排脚本(本目录)
├── DiffSynth-Studio/            # 官方代码(自动 clone 到 ../DiffSynth-Studio)
└── ../../model/                 # 权重(在 <code-dir> 上一级, 各算法共享)
    └── Wan-AI/
        ├── Wan2.2-TI2V-5B/      # DiT + T5 + VAE
        └── Wan2.1-T2V-1.3B/     # tokenizer (google/umt5-xxl/)
```
默认：官方代码 `../DiffSynth-Studio`、权重 `../../model`（相对本目录）；用 `DIFFSYNTH_DIR` / `MODEL_DIR` 覆盖。复用现有 conda env（默认 `doll`），但建议专用 env（`CONDA_ENV=wan22`）。

## Notes
- Official code & weights follow their own license (Wan2.2 = Apache 2.0). This folder only orchestrates; no official code is copied.
- `.gitattributes` (repo root) forces LF so Windows-pushed scripts run cleanly on Ubuntu.
- `proxy.env` (proxy creds / env overrides) is gitignored — never committed.

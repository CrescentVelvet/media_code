# ArtiFixer — 3D 重建增强与扩展

在 Ubuntu + NVIDIA A100 上跑 [ArtiFixer](https://github.com/nv-tlabs/ArtiFixer)（SIGGRAPH 2026）的**推理全流程**：COLMAP/DL3DV 场景 → 3DGRUT 稀疏重建 → ArtiFixer 扩散修正 → ArtiFixer3D 蒸馏 → ArtiFixer3D+ 再修正。本目录只含编排脚本——官方代码自动 clone、权重从本地 `$MODEL_DIR` 加载。

**模型选择：1.3B**（基于 Wan2.1-T2V-1.3B-Diffusers，~1.68B 参数）。官方明确说"fits comfortably on a single 80 GB GPU for all workflows"。14B（~16.9B）需要更多显存，可在 `_env.sh` 中改 `MODEL_VARIANT=14b`。A100 上代码自动用 cuDNN SDPA attention，**不需要 flash-attn 3/4**（Hopper/Blackwell 专用）。

## 常用命令

> 假设已进入容器；首次跑前先做下方「首次准备」。**铁律：每条命令都显式写出模型地址、输入路径、输出路径。**

```bash
# ── 一键全流程（DL3DV demo 场景）──
GPU=0 \
  ARTIFIXER_CHECKPOINT=../../model/artifixer/artifixer-1.3b.pt \
  WAN_MODEL_ID=../../model/Wan-AI/Wan2.1-T2V-1.3B-Diffusers \
  RESULTS_DIR=../../output/artifixer_results \
  bash artifixer/run_all.sh

# ── 分步 ──
# 0) clone 官方仓 + 装依赖
INSTALL_DEPS=1 bash artifixer/00_setup_env.sh

# 1) 下权重（1.3B checkpoint + Wan2.1 base + MoGe + Qwen3-VL）
bash artifixer/01_download_models.sh

# 2) 准备场景数据（DL3DV demo → COLMAP → 3DGRUT 重建 → 渲染 → MoGe 对齐 → Qwen3-VL caption）
GPU=0 \
  PREP_ROOT=../../output/artifixer_results/prep \
  bash artifixer/02_prepare_data.sh

# 3) ArtiFixer 推理（扩散修正 3DGRUT 渲染帧）
GPU=0 \
  ARTIFIXER_CHECKPOINT=../../model/artifixer/artifixer-1.3b.pt \
  WAN_MODEL_ID=../../model/Wan-AI/Wan2.1-T2V-1.3B-Diffusers \
  SCENE_ROOT=../../output/artifixer_results/prep/my_scene \
  SAVE_DIR=../../output/artifixer_results/corrected \
  bash artifixer/03_run_inference.sh

# 4) ArtiFixer3D（3DGRUT 蒸馏，用 ArtiFixer 预测帧训练新 3DGRUT）
GPU=0 \
  SCENE_ROOT=../../output/artifixer_results/prep/my_scene \
  ARTIFIXER_SAVE_DIR=../../output/artifixer_results/corrected \
  bash artifixer/04_run_artifixer3d.sh

# 5) ArtiFixer3D+（对蒸馏后渲染再次修正）
GPU=0 \
  ARTIFIXER_CHECKPOINT=../../model/artifixer/artifixer-1.3b.pt \
  WAN_MODEL_ID=../../model/Wan-AI/Wan2.1-T2V-1.3B-Diffusers \
  SCENE_ROOT=../../output/artifixer_results/prep/my_scene \
  SAVE_3D_PLUS_DIR=../../output/artifixer_results/artifixer3d_plus \
  bash artifixer/05_run_artifixer3d_plus.sh

# ── 用自己的 COLMAP 场景 ──
GPU=0 \
  COLMAP_DIR=/path/to/colmap_scene \
  SCENE_NAME=my_scene \
  PREP_ROOT=../../output/artifixer_results/prep \
  bash artifixer/02_prepare_data.sh
# 然后 03/04/05 同上，SCENE_NAME=my_scene
```

## 首次准备
```bash
cd <code-dir>            # e.g. /data_3d/<uid>/code
git clone https://github.com/CrescentVelvet/media_code.git
cd media_code && cp proxy.env.example proxy.env   # 填 http_proxy / https_proxy

# 建 conda env + 装 torch
conda create -n artifixer python=3.12 -y && conda activate artifixer
pip install torch torchvision   # CUDA build (from PyPI if behind proxy)

# clone ArtiFixer + 装依赖
INSTALL_DEPS=1 bash artifixer/00_setup_env.sh

# 下权重
bash artifixer/01_download_models.sh
```

权重目录布局：
```
$MODEL_DIR/                           # 默认 ../../model
  artifixer/
    artifixer-1.3b.pt                 # ArtiFixer transformer 权重
  Wan-AI/Wan2.1-T2V-1.3B-Diffusers/   # 基础模型 (transformer config + VAE + scheduler + tokenizer + text_encoder)
  moge/                               # MoGe v2 单目深度权重
  hf_cache/                           # HF 缓存 (Qwen3-VL-30B 通过 repo id 加载)
    hub/models--Qwen--Qwen3-VL-30B-A3B-Instruct/
```

---

以下为详细参考。

## Pipeline（流程详解）

```
COLMAP 场景 (images/ + sparse/0/*.bin)
  │
  ├─ [02] prepare_colmap_artifixer_inputs
  │    ├─ prepare   — symlink 图片, transforms.json, selected_indices
  │    ├─ reconstruct — 3DGRUT MCMC 训练 (10k iters)
  │    ├─ render    — 3DGRUT 沿源相机渲染
  │    ├─ scale     — MoGe 单目深度 → metric scale 对齐
  │    └─ caption   — Qwen3-VL-30B 生成描述 → UMT5 文本编码 → caption.h5
  │         └─ split.json (推理入口)
  │
  ├─ [03] ArtiFixer (model_eval.run_inference)
  │    └─ 扩散模型修正 3DGRUT 渲染帧 → pred/*.png
  │
  ├─ [04] ArtiFixer3D (data_processing.run_artifixer3d)
  │    ├─ distill — 用真实锚点 + ArtiFixer 预测帧训练新 3DGRUT (30k iters)
  │    ├─ render  — 渲染蒸馏后的 3DGRUT
  │    └─ prepare_artifixer3d_plus — 生成 split_artifixer3d_plus.json
  │
  └─ [05] ArtiFixer3D+ (model_eval.run_inference)
       └─ 对蒸馏后渲染再次修正 → 最终结果
```

### 为什么要三阶段？
- **ArtiFixer (03)**：直接修正 3DGRUT 的渲染伪影（floaters, 边缘模糊等），但只是逐帧修，帧间不连贯。
- **ArtiFixer3D (04)**：用修正后的帧重新训练 3DGRUT，得到一个更干净的 3D 表示。
- **ArtiFixer3D+ (05)**：对蒸馏后的 3DGRUT 渲染再修一次，修正蒸馏引入的新伪影。

## 1.3B vs 14B

| | 1.3B | 14B |
|---|---|---|
| 参数量 | ~1.68B | ~16.9B |
| 基础模型 | Wan2.1-T2V-1.3B-Diffusers | Wan2.1-T2V-14B-Diffusers |
| 单 80GB GPU | ✅ 全流程舒适运行 | ⚠️ 可能 OOM（含 3DGRUT+扩散+MoGe） |
| 2× 80GB GPU | ✅ | ✅ 可运行 |
| 质量 | 好 | 最高 |
| 速度 | 快 | 慢 |

**本仓默认 1.3B**。切 14B：在 `run_all.sh` 或命令行设 `MODEL_VARIANT=14b`，脚本自动选对应的 checkpoint 和 base model。

### A100 Attention 机制
ArtiFixer 的 `model_training/net/transformer.py` 根据 GPU 架构自动选择 attention 后端：
- **A100 (sm_80)**：cuDNN SDPA（标准 PyTorch，无需额外包）
- **H100 (sm_90)**：FA3（Dao-AILab Hopper kernels）
- **GB200 (sm_100+)**：FA4 CuTeDSL

A100 上 `_select_attention_config()` 返回 `None`，diffusers 自动用 cuDNN。`_patch_flash_3_backend()` 的 FA3 import 是 `try/except`，不装 FA3 也不影响。

## Config (env vars)

### 全局 (_env.sh)
| var | default | note |
|---|---|---|
| `CONDA_ENV` | `artifixer` | conda env 名 |
| `GPU` | _(unset)_ | 物理卡号，如 `GPU=0` |
| `MODEL_VARIANT` | `1.3b` | `1.3b` 或 `14b` |
| `ARTIFIXER_DIR` | `../ArtiFixer` | 官方代码路径 |
| `MODEL_DIR` | `../../model` | 权重根 |
| `RESULTS_DIR` | `../artifixer_results` | 输出根 |
| `ARTIFIXER_CHECKPOINT` | `$MODEL_DIR/artifixer/artifixer-{variant}.pt` | checkpoint 路径 |
| `WAN_MODEL_ID` | `$MODEL_DIR/Wan-AI/Wan2.1-T2V-{variant}-Diffusers` | 基础模型本地路径（作为 --model_id 传给 diffusers） |
| `HF_MODEL_ID` | `Wan-AI/Wan2.1-T2V-{variant}-Diffusers` | HF repo id（仅下载用） |
| `HF_HUB_OFFLINE` | `1` | 推理时禁止联网；下载数据准备时设 0 |
| `HF_HOME` | `$MODEL_DIR/hf_cache` | HF 缓存目录（Qwen3-VL 通过 repo id 加载） |
| `MOGE_MODEL_PATH` | `$MODEL_DIR/moge` | MoGe 权重目录 |
| `CAPTIONING_MODEL_ID` | `Qwen/Qwen3-VL-30B-A3B-Instruct` | captioning 模型 repo id（硬编码在官方代码中） |

### 02 — 数据准备
| var | default | note |
|---|---|---|
| `COLMAP_DIR` | _(unset)_ | 用户自己的 COLMAP 场景路径；不设则下载 DL3DV demo |
| `DL3DV_ROOT` | `../DL3DV-ALL-960P` | DL3DV 数据集根 |
| `DL3DV_SCENE_ID` | _(README 中的 demo 场景)_ | DL3DV 场景 ID |
| `DL3DV_SUBDIR` | `8K` | DL3DV 子目录 |
| `PREP_ROOT` | `$RESULTS_DIR/prep` | 准备输出根 |
| `SCENE_NAME` | `my_scene` | 场景名（输出子目录名） |
| `RECON_STEPS` | `10000` | 3DGRUT 训练迭代数 |
| `PHASES` | `prepare,reconstruct,render,scale,caption` | 要运行的阶段 |
| `METRIC_SCALE` | _(unset)_ | 已知 metric scale 则跳过 MoGe 对齐 |
| `SELECTED_IMAGE_NAMES_FILE` | _(unset)_ | 训练视角图片名列表文件 |
| `TRAJECTORY_PATH` | _(unset)_ | 自定义相机轨迹 JSON（transforms-style） |

### 03/05 — 推理
| var | default | note |
|---|---|---|
| `SCENE_ROOT` | `$RESULTS_DIR/prep/my_scene` | 02 的输出目录 |
| `SAVE_DIR` / `SAVE_3D_PLUS_DIR` | `$RESULTS_DIR/corrected` / `$RESULTS_DIR/artifixer3d_plus` | 推理输出目录 |
| `SPLIT_PATH` | `$SCENE_ROOT/split.json` (03) / `split_artifixer3d_plus.json` (05) | split JSON 路径 |
| `RENDER_TRAJECTORY` | `all_frames` | `val_frames` / `all_frames` / `trajectory` |
| `INFERENCE_PIPELINE` | `kv_cache` | `kv_cache`（块级局部注意力）或 `bidirectional`（全序列） |
| `NUM_INFERENCE_STEPS` | `4` | 扩散去噪步数 |
| `FRAMES_PER_BLOCK` | `7` | 每块帧数 |
| `LOCAL_ATTN_SIZE` | `21` | 局部注意力窗口 |
| `SINK_SIZE` | `7` | sink attention 大小 |
| `OUTPUT_FPS` | `15` | 输出视频帧率 |
| `NUM_VIEWS` | _(auto for reconstructed_colmap)_ | 邻居视角数 |
| `MAX_NEIGHBORS_PER_ENCODE` | _(unset=all at once)_ | 显存不够时设 1 |
| `NUM_GPUS` | `1` | >1 用 torchrun 分布式 |

### 04 — ArtiFixer3D
| var | default | note |
|---|---|---|
| `ARTIFIXER3D_STEPS` | `30000` | 3DGRUT 蒸馏迭代数 |
| `PHASES` | `distill,render,prepare_artifixer3d_plus` | 阶段 |
| `ARTIFIXER_FRAMES_DIR` | _(auto-find)_ | 03 的 pred 目录；自动搜索 |
| `USE_WANDB` | `0` | `1` 开 W&B 日志 |

## 可能遇到的问题

**1. `import threedgrut` 失败**
3DGRUT-ArtiFixer 子模块没装。跑 `INSTALL_DEPS=1 bash 00_setup_env.sh`。需要 `slangc` 编译器（由 `scripts/install_slangc.sh` 自动安装）和 `slangtorch==1.3.4`。

**2. 3DGRUT 训练 OOM（02 的 reconstruct 阶段）**
- 降 `RECON_STEPS=5000`（减少训练时间，不影响显存峰值但减少总时间）。
- 确保没有其他 GPU 进程占用。
- DL3DV 8K 场景分辨率较高；用自己的 COLMAP 时降采样图片。

**3. Qwen3-VL captioning OOM（02 的 caption 阶段）**
- Qwen3-VL-30B-A3B 是 MoE：~60 GB bf16 加载，3B active。A100 80GB 可跑但紧。
- 用 `device_map="auto"` 自动分配（代码已内置）。
- 如果仍 OOM，跳过 caption 阶段（`PHASES=prepare,reconstruct,render,scale`），手动写 caption.h5。

**4. ArtiFixer 推理 OOM（03/05）**
- 设 `MAX_NEIGHBORS_PER_ENCODE=1`（逐帧 VAE 编码邻居，省显存但慢）。
- 1.3B 在 80GB A100 上不应该 OOM；如仍 OOM 检查是否有残留 GPU 进程。
- 设 `export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`。

**5. `from_pretrained` 报 model not found**
- 检查 `$WAN_MODEL_ID` 目录是否存在且非空。
- 推理时确保 `HF_HUB_OFFLINE=1`（`_env.sh` 默认设了）。
- `--model_id` 传的是本地路径（不是 HF repo id），diffusers 直接读本地文件。

**6. MoGe 下载/加载失败（02 的 scale 阶段）**
- 设 `MOGE_MODEL_PATH` 指向本地 MoGe 权重目录。
- 或设 `METRIC_SCALE=1.0`（已知 scale）跳过 MoGe 对齐。

**7. DL3DV 场景没有 COLMAP sparse 重建**
DL3DV zip 可能只含图片不含 COLMAP。需先跑 COLMAP：
```bash
colmap feature_extractor --image_path <scene>/images --database_path <scene>/db.db
colmap exhaustive_matcher --database_path <scene>/db.db
colmap mapper --image_path <scene>/images --database_path <scene>/db.db --output_path <scene>/sparse
mkdir -p <scene>/sparse/0 && mv <scene>/sparse/0/* <scene>/sparse/0/  # 确保在 sparse/0/
```
然后设 `COLMAP_DIR=<scene>` 跑 02。

**8. 脚本 CRLF 行尾问题**
```bash
find artifixer -name '*.sh' -exec sed -i 's/\r$//' {} +
git checkout -- artifixer/*.sh
```

## 目录布局
```
<code-dir>/
├── media_code/                    # 本仓
│   ├── proxy.env                 # 代理 + env 覆盖, gitignored
│   └── artifixer/                # 编排脚本
├── ArtiFixer/                     # 官方代码 (自动 clone)
│   └── thirdparty/3DGRUT-ArtiFixer/  # 3DGRUT 子模块
├── ../../model/                   # 权重根
│   ├── artifixer/
│   ├── Wan-AI/Wan2.1-T2V-1.3B-Diffusers/
│   ├── moge/
│   └── hf_cache/
└── artifixer_results/             # 输出
    ├── prep/my_scene/             # 02 的输出
    │   ├── split.json
    │   ├── 3dgrut_input/
    │   ├── recon_results/
    │   ├── captions/
    │   ├── metric_alignment/
    │   └── artifixer3d/           # 04 的输出
    ├── corrected/                 # 03 的输出
    └── artifixer3d_plus/         # 05 的输出
```

## Notes
- Official code & weights follow Apache 2.0 license. This folder only orchestrates; no official code is copied.
- `.gitattributes` forces LF so Windows-pushed scripts run cleanly on Ubuntu.
- `proxy.env` is gitignored — never committed.
- ArtiFixer uses standard diffusers `from_pretrained()` — passing a local path as `--model_id` works directly (no env var redirect needed unlike DiffSynth-Studio).

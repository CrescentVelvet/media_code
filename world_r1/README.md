# World-R1 — 用 3D 约束强化学习微调 Wan2.1 文生视频（ICML 2026）

World-R1 通过 Flow-GRPO 强化学习对 Wan2.1-T2V 文生视频模型做后训练，用 3D 几何约束（深度重建一致性 + 相机轨迹对齐 + meta-view 评估）+ 美学奖励来改善视频生成的几何一致性，同时保持视觉质量和运动多样性。不改基础模型架构——通过 camera-aware latent initialization（噪声包裹）实现隐式相机条件，用 periodic decoupled training（周期性动态训练）防止过拟合到静态场景。

> **官方代码**: [microsoft/World-R1](https://github.com/microsoft/World-R1)（clone 到 `../World-R1`）
> **权重来源**: Wan2.1-T2V-14B-Diffusers (HF) + Depth Anything 3 (HF) + Qwen3-VL-4B-Instruct (HF)

## 常用命令

> 假设已进入容器（脚本自动激活 `world_r1` env）。首次跑前先做下方「首次准备」。
> **铁律：每条命令都显式写出模型地址、输入路径、输出路径。**

```bash
# ── 一键全流程（验证环境 → 下权重 → 训练 → 推理）──
# 默认 8 GPU: 2 server + 6 train
bash world_r1/run_all.sh

# 4 GPU 示例（2 server + 2 train）
SERVER_VISIBLE_DEVICES=0,1 TRAIN_VISIBLE_DEVICES=2,3 NUM_PROCESSES=2 \
  bash world_r1/run_all.sh

# ── 分步 ──
# 1) RL 训练（一体化模式：自动启 reward server + 训练）
#    训练产物（LoRA checkpoint + 日志）→ ../../output/world_r1_experiments/
MODEL_PATH=../../model/Wan2.1-T2V-14B-Diffusers \
  SERVER_VISIBLE_DEVICES=0,1 TRAIN_VISIBLE_DEVICES=2,3,4,5,6,7 NUM_PROCESSES=6 \
  OUTPUT_ROOT=../../output/world_r1_experiments \
  bash world_r1/03_run_training.sh

# 1b) 用 1.3B 小模型（显存不够时）
MODEL_FAMILY=wan_small \
  SERVER_VISIBLE_DEVICES=0,1 TRAIN_VISIBLE_DEVICES=2,3 NUM_PROCESSES=2 \
  OUTPUT_ROOT=../../output/world_r1_experiments \
  bash world_r1/03_run_training.sh

# 2) 推理（基础模型 + 训练好的 LoRA → 视频）
#    视频输出 → ../../output/world_r1_results/
GPU=0 MODEL_FAMILY=wan_large \
  LORA_PATH=../../output/world_r1_experiments/world_r1_large_run1/checkpoint-60 \
  PROMPT="A camera slowly orbits around a detailed ceramic vase on a wooden table, soft natural lighting" \
  HEIGHT=480 WIDTH=832 NUM_FRAMES=81 NUM_STEPS=20 GUIDANCE_SCALE=5.0 FPS=12 SEED=42 \
  RESULTS_DIR=../../output/world_r1_results \
  bash world_r1/04_run_inference.sh

# 2b) 多 GPU 并行推理（按 prompt 分配到各卡）
MODEL_FAMILY=wan_large \
  LORA_PATH=../../output/world_r1_experiments/world_r1_large_run1/checkpoint-60 \
  PROMPT_FILE=../../output/world_r1_results/prompts.txt \
  DEVICES=cuda:0,cuda:1 \
  RESULTS_DIR=../../output/world_r1_results \
  bash world_r1/04_run_inference.sh

# 3) 只跑 reward server（调试 / 多节点训练用，Ctrl+C 停）
#    然后在另一个终端用 EXTERNAL_REWARD=1 跑训练
SERVER_VISIBLE_DEVICES=0,1 \
  bash world_r1/02_start_reward_servers.sh
# 另一个终端:
EXTERNAL_REWARD=1 MODEL_PATH=../../model/Wan2.1-T2V-14B-Diffusers \
  TRAIN_VISIBLE_DEVICES=2,3,4,5,6,7 NUM_PROCESSES=6 \
  OUTPUT_ROOT=../../output/world_r1_experiments \
  bash world_r1/03_run_training.sh
```

- 结果：LoRA checkpoint → `../../output/world_r1_experiments/<run_name>/checkpoint-*/`；视频 → `../../output/world_r1_results/*.mp4`；reward server 日志 → `../../output/world_r1_experiments/reward_servers/*.log`。

## 首次准备

```bash
cd <code-dir>            # e.g. /data_3d/<uid>/code
git clone https://github.com/CrescentVelvet/media_code.git
cd media_code && cp proxy.env.example proxy.env   # 填 http_proxy / https_proxy
# ⚠️ 确认 proxy.env 中 http_proxy / https_proxy 已取消注释并填好

# 1. clone World-R1 官方仓 + 建 conda env + 装全部依赖 + 验证
#    (Python 3.10 CPython; PyTorch cu124; flow_grpo + reward_server;
#     gsplat 等 3D reward 依赖; gxx 12 编 CUDA 扩展)
INSTALL_DEPS=1 bash world_r1/00_setup_env.sh

# 2. 下权重（Wan2.1-T2V-14B + DA3-GIANT + Qwen3-VL-4B）
bash world_r1/01_download_models.sh
# 显存不够用小模型: MODEL_FAMILY=wan_small bash world_r1/01_download_models.sh
```

权重目录布局：
```
$MODEL_DIR/                              # 默认 ../../model
  Wan2.1-T2V-14B-Diffusers/              # 基础视频模型 (DiT + T5 + VAE, ~28GB)
    model_index.json, config.json, ...
  Wan2.1-T2V-1.3B-Diffusers/              # 1.3B 版 (MODEL_FAMILY=wan_small)
  huggingface/                            # HF cache (DA3 + Qwen3-VL, 离线加载)
    hub/
      models--depth-anything--DA3-GIANT/  # Depth Anything 3 (3D reward 重建)
      models--Qwen--Qwen3-VL-4B-Instruct/ # Qwen3-VL (reward scorer)
```

---

以下为详细参考。

## Pipeline（流程详解）

```
dataset/enhanced/ (prompt-only, 官方仓自带)
    │
    ▼
[03] Flow-GRPO RL 训练 (torchrun, world_r1 env)
    │  ├─ camera-aware latent init: prompt 中的相机指令 → 轨迹 → 噪声包裹
    │  ├─ rollout: Wan2.1 生成 G 个视频 (num_image_per_prompt=2)
    │  ├─ 3D reward server (端口 18089):
    │  │    ├─ Depth Anything 3: 视频 → 3D 重建 → GS 渲染视频 + meta view
    │  │    ├─ Qwen3-VL: 评估 GS 视频 (S_recon) + meta view (S_meta)
    │  │    └─ 轨迹对齐: 预测 vs 目标相机轨迹 (S_traj)
    │  │    R_3D = S_meta + S_recon + S_traj, each ∈ [0,1]
    │  ├─ general reward server (端口 18090):
    │  │    └─ HPSv2: 首帧美学评分 (R_general ∈ [0,1])
    │  ├─ advantage 计算: per-prompt 归一化 (per_prompt_stat_tracking)
    │  ├─ policy gradient: clip_range=1e-3, beta=0.004 (KL)
    │  └─ periodic decoupled: 100 步主训练 (相机 prompt) + 50 步动态训练 (无相机 prompt)
    ▼
$EXPERIMENTS_DIR/<run_name>/checkpoint-*/  (LoRA 权重, save_freq=60)
    │
    ▼
[04] 推理 (Wan2.1 + LoRA → MP4)
    │  ├─ 加载 WanPipeline (DiT + T5 + VAE)
    │  ├─ 加载 LoRA (PeftModel.from_pretrained)
    │  └─ 生成: prompt → video (480×832×81, 20 steps, cfg=5.0)
    ▼
$RESULTS_DIR/*.mp4
```

### Step 03 — RL 训练 (`03_run_training.sh`)

调用官方 `scripts/run_single_node.sh`（自动启 reward server + 训练）或 `scripts/run_training.sh`（仅训练，`EXTERNAL_REWARD=1`）。

**默认配置** (`config/world_r1.py:world_r1_large`)：
- 分辨率: 480×832, 81 帧
- 采样: 50 步, guidance_scale=5.0, noise_level=0.7
- batch: train_batch_size=1, num_image_per_prompt=2, num_batches_per_epoch=24
- gradient_accumulation_steps=24 (= 2×1×24//2)
- 学习率: 1e-4, clip_range=1e-3, beta=0.004
- wrap_strength=0.4 (camera-aware noise wrap 强度)
- dynamic_training: main_steps=100, dynamic_steps=50
- mixed_precision=bf16, LoRA, save_freq=60, eval_freq=30

**GPU 分配**：
- reward server: `SERVER_VISIBLE_DEVICES` (默认 0,1; DA3 + Qwen3-VL 各需 ~10GB)
- 训练: `TRAIN_VISIBLE_DEVICES` (默认 2,3,4,5,6,7; 14B LoRA bf16 每卡 ~20GB)
- 14B 模型建议 ≥6 训练 GPU；1.3B 模型 (`wan_small`) 2-4 GPU 即可

### Step 04 — 推理 (`04_run_inference.sh`)

调用官方 `scripts/infer_wan_lora.py`。支持单 prompt (`PROMPT=`) 或 prompt 文件 (`PROMPT_FILE=`, 每行一个)。不设 `LORA_PATH` 则只跑基础模型（baseline 对比）。

## Config (env vars)

### Paths & envs
| var | default | note |
| --- | --- | --- |
| `MODEL_FAMILY` | `wan_large` | `wan_large` (14B) \| `wan_small` (1.3B) \| `cogvideox` (5B) |
| `WAN_MODEL_PATH` | `$MODEL_DIR/Wan2.1-T2V-14B-Diffusers` | 基础视频模型路径（按 MODEL_FAMILY 自动切换） |
| `CONDA_ENV` | `world_r1` | conda env |
| `WORLD_R1_DIR` | `../World-R1` | 官方代码 |
| `MODEL_DIR` | `../../model` | 权重根 |
| `RESULTS_DIR` | `../world_r1_results` | 推理输出 |
| `EXPERIMENTS_DIR` | `../world_r1_experiments` | 训练产物（checkpoint + 日志） |
| `HF_HOME` | `$MODEL_DIR/huggingface` | HF cache（DA3 + Qwen3-VL 离线加载） |

### Step 03 params (训练)
| var | default | note |
| --- | --- | --- |
| `SERVER_VISIBLE_DEVICES` | `0,1` | reward server GPU |
| `TRAIN_VISIBLE_DEVICES` | `2,3,4,5,6,7` | 训练 GPU |
| `NUM_PROCESSES` | _(auto)_ | torchrun 进程数 (= train GPU 数) |
| `OUTPUT_ROOT` | `$EXPERIMENTS_DIR` | 训练输出根目录 |
| `TRAIN_CONFIG` | `config/world_r1.py:world_r1_<family>` | 训练配置 |
| `EXTERNAL_REWARD` | `0` | `1` = 不启动 reward server（已在跑，用 02） |
| `TRAIN_HEIGHT` / `TRAIN_WIDTH` / `TRAIN_FRAMES` | 480 / 832 / 81 | 视频分辨率 |
| `TRAIN_NUM_STEPS` | 50 | 采样步数 |
| `TRAIN_BATCH_SIZE` | 1 | 每卡 batch size |
| `TRAIN_WRAP_STRENGTH` | 0.4 | noise wrap 强度 |
| `TRAIN_LORA_PATH` | _(unset)_ | 从已有 LoRA 继续训练 |
| `SERVER_PORT` | 18089 | 3D reward server 端口 |
| `GENERAL_REWARD_PORT` | 18090 | general reward server 端口 |

### Step 04 params (推理)
| var | default | note |
| --- | --- | --- |
| `LORA_PATH` | _(unset)_ | LoRA 目录；不设则只跑基础模型 |
| `PROMPT` | _(default 5 prompts)_ | 单 prompt |
| `PROMPT_FILE` | _(unset)_ | prompt 文件（每行一个；优先于 PROMPT） |
| `HEIGHT` / `WIDTH` / `NUM_FRAMES` | 480 / 832 / 81 | 视频分辨率 |
| `NUM_STEPS` | 20 | 推理步数（比训练少，快） |
| `GUIDANCE_SCALE` | 5.0 | CFG |
| `FPS` | 12 | 输出 mp4 帧率 |
| `SEED` | 42 | 随机种子 |
| `GPU` | _(unset)_ | 单卡推理（如 `GPU=0`） |
| `DEVICES` | _(unset)_ | 多卡并行（如 `cuda:0,cuda:1`） |
| `DTYPE` | `bf16` | 推理精度 |

## 可能遇到的问题

**1. 训练 OOM（显存不够）**
- 换 1.3B 模型：`MODEL_FAMILY=wan_small`
- 降分辨率 / 帧数：`TRAIN_HEIGHT=320 TRAIN_WIDTH=576 TRAIN_FRAMES=49`
- 减 num_image_per_prompt：`TRAIN_NUM_IMAGE_PER_PROMPT=1`
- 减少训练 GPU 但保持 reward server 2 GPU

**2. reward server 启动失败（DA3 / Qwen3-VL 加载失败）**
- 模型未下载全：`bash world_r1/01_download_models.sh`
- HF cache 路径不对：确认 `HF_HOME` 指向 `$MODEL_DIR/huggingface`
- `_env.sh` 设了 `HF_HUB_OFFLINE=1`，模型必须在 cache 里（01 已下载则没问题）

**3. HF 下载失败（代理封 huggingface.co）**
手动下载：
```bash
# Wan2.1 (用 huggingface-cli 或浏览器)
huggingface-cli download Wan-AI/Wan2.1-T2V-14B-Diffusers \
  --local-dir ../../model/Wan2.1-T2V-14B-Diffusers
# DA3 + Qwen3-VL (下到 HF cache)
huggingface-cli download depth-anything/DA3-GIANT \
  --cache-dir ../../model/huggingface/hub
huggingface-cli download Qwen/Qwen3-VL-4B-Instruct \
  --cache-dir ../../model/huggingface/hub
```

**4. gsplat 编译失败（CUDA 版本不匹配 / nvcc 找不到）**
- 找 CUDA 12.x 路径：`ls -d /usr/local/cuda-12*`
- 设 `CUDA_HOME=/usr/local/cuda-12.4 PATH=$CUDA_HOME/bin:$PATH`
- 重装：`pip install --no-build-isolation --force-reinstall gsplat`

**5. LPIPS / HPSv2 权重自动下载失败（代理拦截）**
reward server 首次启动时会下载 LPIPS (alex net) 和 HPSv2 权重。若代理拦：
- LPIPS: 权重在 `~/.cache/torch/hub/checkpoints/`，手动下 `alex.pth` 放进去
- HPSv2: 权重在 HF cache，`huggingface-cli download yuvalkirstain/HPSv2 --cache-dir $HF_HOME/hub`

**6. 跑 `.sh` 报 `syntax error near unexpected token ('`**
CRLF 行尾污染。`find world_r1 -name '*.sh' -exec sed -i 's/\r$//' {} +`（`.gitattributes` 强制 LF，但 Windows 编辑可能引入 CRLF）。

**7. 多节点训练**
启动外部 reward server（02），然后提供 `MASTER_ADDR` / `MASTER_PORT` / `NNODES` / `NODE_RANK`：
```bash
EXTERNAL_REWARD=1 \
  REWARD_3D_SERVER_URL=http://<server-node>:18089 \
  GENERAL_REWARD_SERVER_URL=http://<server-node>:18090 \
  MASTER_ADDR=<main-node-ip> MASTER_PORT=29511 NNODES=2 NODE_RANK=0 \
  bash world_r1/03_run_training.sh
```

## 目录布局
```
<code-dir>/
├── media_code/
│   └── world_r1/                # ← 本目录（编排脚本）
├── World-R1/                    # 官方代码 (00 clone)
│   ├── flow_grpo/               # Flow-GRPO RL 训练框架
│   ├── reward_server/           # reward server (DA3 + Qwen3-VL + HPSv2)
│   │   └── depth_anything_3/    #   bundled DA3 代码
│   ├── config/                   # 训练配置 (world_r1.py)
│   ├── dataset/                  # prompt-only 数据 (enhanced/ + final/)
│   └── scripts/                  # 官方脚本 (train/infer/serve)
├── model/                        # 权重根 (共享)
│   ├── Wan2.1-T2V-14B-Diffusers/
│   └── huggingface/hub/          # HF cache (DA3 + Qwen3-VL)
├── world_r1_experiments/         # 训练产物 (LoRA checkpoint + 日志)
└── world_r1_results/             # 推理输出 (*.mp4)
```

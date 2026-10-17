# FlashWorld — 单图/文本 → 3DGS，秒级 3D 场景生成

腾讯 + 厦大 FlashWorld（ICLR 2026 Oral）。3D-oriented 多视图扩散：去噪过程中直接回归
3D Gaussian，而不是先生成多视图再重建。官方仓只发**推理**（训练代码需填表申请）。
权重来自 HF（`imlixinyang/FlashWorld` 19.5 GB）+ Wan2.2-TI2V-5B-Diffusers 基座（34.2 GB）。

> ⚠️ 官方**只支持单图或文本**输入，不支持多图（依据见 [NOTES.md](NOTES.md) 第 6 节）。
> ⚠️ 代码许可 CC BY-NC-SA 4.0，仅限学术研究。

## 常用命令
> 假设已进入容器；首次跑前先做下方「首次准备」。
> 24 GB 卡（3090/4090）默认就带 `--offload_t5 --offload_transformer_during_vae`。

```bash
# 一键：单图 → 3DGS（PLY）
GPU=0 IMAGE=<code-dir>/data/scene.jpg \
  RESULTS_DIR=<code-dir>/flashworld_results \
  bash flashworld/run_all.sh

# 分步 1) 单图 → 输入 json（相机轨迹）
GPU=0 IMAGE=<code-dir>/data/scene.jpg \
  OUTPUT=<code-dir>/flashworld_results/02_input_json \
  TRAJ=orbit YAW_SPAN=40 FOV=60 \
  bash flashworld/02_make_input.sh

# 分步 2) json → 3DGS（PLY + 环绕视频）
GPU=0 INPUT_DIR=<code-dir>/flashworld_results/02_input_json \
  OUT_DIR=<code-dir>/flashworld_results/03_gaussians \
  PLY=1 VIDEO=1 \
  bash flashworld/02_run_inference.sh
```

## 首次准备
```bash
# clone 本仓 + proxy.env
cd <code-dir>
git clone <本仓> && cd media_code && cp proxy.env.example proxy.env
# ⚠️ 确认 proxy.env 中 http_proxy / https_proxy 已取消注释

# 建 env + clone 官方仓（含三个 pin 住的 git 依赖）+ 编译 gsplat/spz
INSTALL_DEPS=1 bash flashworld/00_setup_env.sh

# 下权重（54 GB；慢就按 download_urls.md 用迅雷，再跑 01b 挂载）
bash flashworld/01_download_models.sh
# 或者：
# bash flashworld/01b_link_local_weights.sh
```

权重目录布局（`$MODEL_DIR` 下，HF cache 格式）：
```
$MODEL_DIR/
├── hf_cache/hub/models--Wan-AI--Wan2.2-TI2V-5B-Diffusers/snapshots/<sha>/
│   ├── vae/            diffusion_pytorch_model.safetensors  2.8 GB
│   ├── transformer/    5 个分片                           20.0 GB
│   ├── text_encoder/   3 个分片                           11.4 GB
│   ├── tokenizer/  scheduler/  model_index.json
└── hf_cache/hub/models--imlixinyang--FlashWorld/snapshots/<sha>/model.ckpt   19.5 GB
```

---

以下为详细参考。机制/排错见 [NOTES.md](NOTES.md)，实测数据见 EXPERIMENTS.md。

## Config (env vars)

| var | default | note |
|---|---|---|
| `GPU` | 全部 | 映射到 `CUDA_VISIBLE_DEVICES` |
| `IMAGE` | — | 单张图或图目录（02/run_all 必填） |
| `TEXT` | `""` | 文本提示；官方实际送进模型的是 `"[Static] " + TEXT` |
| `N_FRAME` | `24` | 帧数；模型按 24 帧训练，改它只是插值 |
| `HEIGHT` / `WIDTH` | `480` / `704` | 官方只给 480×704（横）或 704×480（竖） |
| `FOV` | `60` | 竖直 FOV；单图无内参真值，按观感调 |
| `TRAJ` | `orbit` | `orbit` / `official_orbit`(±15°) / `forward` |
| `YAW_SPAN` | `40` | 环绕总角度（度） |
| `PITCH` | `0` | 环绕仰角（度，限 ±80） |
| `RADIUS` | `1.0` | 环绕半径（世界尺度，导出时乘回 T_norm） |
| `DOLLY` | `0.5` | `forward` 模式推进距离 |
| `IMAGE_INDEX` | `0` | 输入图落在哪一帧 |
| `PLY` / `SPZ` / `VIDEO` | `1` / `0` / `0` | 产物开关；SPZ 需 spz 包，VIDEO 很吃显存 |
| `REALLY_LOW_VRAM` | `0` | `1` → 加 `--offload_vae`（<10 GB，但慢到分钟级） |
| `INPUT_DIR` / `OUT_DIR` | `$RESULTS_DIR/02_input_json` / `.../03_gaussians` | 推理输入输出 |
| `FLASHWORLD_DIR` | `<code-dir>/FlashWorld` | 官方仓（clone 到本仓 sibling） |
| `FLASHWORLD_DEPS` | `<code-dir>/_flashworld_deps` | gsplat / spz / diffusers 三个 pin 住的 clone |
| `MODEL_DIR` | `<code-dir>/../model/flashworld` | 权重根（内含 `hf_cache/`） |
| `RESULTS_DIR` | `<code-dir>/flashworld_results` | 输出 |

## 脚本

| 脚本 | 作用 |
|---|---|
| `00_setup_env.sh` | 服务器：建 env + clone 官方仓与 git 依赖 + 编译 |
| `00a_setup_env.sh` | WSL：clone `minimax_h3` 复用 torch 2.6.0+cu124 + 装 nvcc/gxx + 编译 |
| `01_download_models.sh` / `.py` | 下 54 GB 权重到 HF cache（断点续传 + 完整性校验） |
| `01b_link_local_weights.sh` | 迅雷下好的目录 → 软链接成 HF cache 布局 |
| `02_make_input.py` / `.sh` | 单图/图目录 → 输入 json（相机轨迹见 NOTES 第 3 节） |
| `02_run_inference.sh` | 调官方 `cli.py` → PLY / SPZ / MP4 |
| `08_move_output.sh` | WSL：结果搬到 `/mnt/d/output/flashworld_results` |
| `run_all.sh` | 02 + 02 一键 |

## 目录布局
```
<code-dir>/
├── media_code/flashworld/     # 本目录（编排脚本）
├── FlashWorld/                # 官方仓（00 自动 clone）
├── _flashworld_deps/          # gsplat / spz / diffusers 三个 pin 住的 clone
├── model/flashworld/hf_cache/ # 权重（HF cache 布局）
└── flashworld_results/        # 输出（02_input_json / 03_gaussians）
```

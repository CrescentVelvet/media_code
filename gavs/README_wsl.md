# GaVS (WSL Ubuntu 24.04) — 本地复现指南

本文件是 [`README.md`](README.md) 的 WSL 实操版。硬件：单卡 RTX 3090 24GB。

## ⚡ 迅雷下载清单（峰哥手动）

本机 huggingface.co 直连被重置、hf-mirror.com 不稳定（2026-10-10 实测），
以下文件用迅雷下载到 **`D:\wheel\gavs_ms\`**（WSL 路径 `/mnt/d/wheel/gavs_ms/`），
然后跑 `01_download_models.sh` 自动归位。

### 必需（3 个，共 ~2.1 GB）

| 文件 | 大小 | 迅雷链接 | 下载后文件名要求 |
|---|---|---|---|
| Flash3D 预训练权重（RealEstate10K） | 690 MB | `https://huggingface.co/datasets/sinoyou/gavs-data/resolve/main/re10k_v2_checkpoints/model_re10k_v2.pth` | `model_re10k_v2.pth`（放根目录） |
| UniDepth v1 ViT-L14 权重 | ~1.3 GB | `https://huggingface.co/lpiccinelli/unidepth-v1-vitl14/resolve/main/pytorch_model.bin` | 改名 `unidepth-v1-vitl14-pytorch_model.bin`（放根目录）⚠️ |
| ResNet50 ImageNet 权重（torchvision V2） | 98 MB | `https://download.pytorch.org/models/resnet50-11ad3fa6.pth` | `resnet50-11ad3fa6.pth`（放根目录，**不要改名**——文件名含 sha256 前缀，torch 要校验）⚠️ |

> ResNet50 这一项官方 README 没提：`models/encoder/resnet_encoder.py` 构建 backbone 时
> torchvision 会联网下载它，而 download.pytorch.org 在本机"通而不稳"（连接建立后无数据，
> urllib 无超时直接挂死训练初始化，2026-10-12 实测卡死 8 分钟定位）。

### 场景数据（至少下 1 个；推荐先下 dynamic_dance）

放 `D:\wheel\gavs_ms\dataset\` 子目录。先跑通一个场景（dynamic_dance 是论文主展示场景，
455 帧含动态人体），其余按需再下。

| 场景 | 大小 | 链接 |
|---|---|---|
| **dynamic_dance（推荐先下）** | 919 MB | `https://huggingface.co/datasets/sinoyou/gavs-data/resolve/main/dataset/dynamic_dance.zip` |
| dynamic_downstairs | 1.09 GB | `.../dataset/dynamic_downstairs.zip` |
| dynamic_fountain | 898 MB | `.../dataset/dynamic_fountain.zip` |
| dynamic_vehicle | 736 MB | `.../dataset/dynamic_vehicle.zip` |
| dynamic_walk | 1.16 GB | `.../dataset/dynamic_walk.zip` |
| intense_forward | 998 MB | `.../dataset/intense_forward.zip` |
| intense_forward2 | 1.30 GB | `.../dataset/intense_forward2.zip` |
| intense_forward3 | 926 MB | `.../dataset/intense_forward3.zip` |
| intense_roadside | 1.21 GB | `.../dataset/intense_roadside.zip` |
| intense_slerp | 810 MB | `.../dataset/intense_slerp.zip` |
| mild_building | 1.18 GB | `.../dataset/mild_building.zip` |
| mild_bush | 1.60 GB | `.../dataset/mild_bush.zip` |
| mild_rotation | 1.93 GB | `.../dataset/mild_rotation.zip` |
| mild_rotation2 | 1.98 GB | `.../dataset/mild_rotation2.zip` |
| mild_walk | 1.86 GB | `.../dataset/mild_walk.zip` |

（链接前缀统一为 `https://huggingface.co/datasets/sinoyou/gavs-data/resolve/main/`。
若迅雷吃不动 huggingface.co，可把前缀换成 `https://hf-mirror.com/datasets/sinoyou/gavs-data/resolve/main/` 碰碰运气 —— 但该镜像 2026-10-10 直连超时。）

> 各方法对比结果（`result_and_comparison/`，每个 25-97 MB）复现**不需要**，暂不下。

### 下载完成后的暂存区布局

```
D:\wheel\gavs_ms\
├── model_re10k_v2.pth
├── unidepth-v1-vitl14-pytorch_model.bin
├── resnet50-11ad3fa6.pth
└── dataset\
    └── dynamic_dance.zip   （+ 其他场景可选；也接受已解压的 dynamic_dance\ 目录）
```

## 首次准备（一次性，~30 min 含下载）

```bash
wsl -d Ubuntu2404
cd /mnt/c/code/media_code

# 1) 建环境：conda env gavs + torch 2.2.2(cu121) + nvcc/gcc(conda) + 编译光栅化扩展
bash gavs/00a_setup_env.sh

# 2) 迅雷文件归位（搬权重、种 HF 缓存、解压场景）
bash gavs/01_download_models.sh

# 3) 微调 + 评估（dynamic_dance，3 epoch 微调后自动评估）
SCENE=dynamic_dance bash gavs/02_run_finetune.sh
```

产物在 `~/output/gavs_results/dynamic_dance/`：
- `checkpoints/` — 微调后的模型
- `visualization/numerical_results.json` — D/S/GC-S/GC-D 指标
- `visualization/*.mp4` — 不稳定输入 vs 稳定输出对比视频

换稳定强度重渲染（微调不用重跑）：

```bash
SCENE=dynamic_dance STABILITY=8 bash gavs/03_run_eval.sh
```

## 环境决策（与官方 README 的偏离及理由）

| 项 | 官方 | 本机 | 理由 |
|---|---|---|---|
| torch CUDA | cu126 | **cu121** | xformers==0.0.25.post1 是 UniDepthV1 decoder（NystromBlock）的**运行时硬依赖**，其 PyPI wheel 按 torch 2.2.2+cu121 编译；混 cu126 会在 `import xformers.ops` 时 CUDA 扩展加载失败 |
| CUDA toolkit | 系统 CUDA 12.6 | conda `nvidia/label/cuda-12.1.1` | 与 torch cu121 对齐；且 `thirdparty/diff-gaussian-rasterization-w-pose/setup.py` 硬编码 `-D__CUDACC_VER_MAJOR__=12 -D__CUDACC_VER_MINOR__=1` |
| gcc | — | conda `gxx_linux-64=12` | WSL 无系统 gcc，免 sudo |
| ffmpeg | 系统 | conda conda-forge | `evaluate.py` 用 `os.system("ffmpeg ...")` 合成 mp4 |
| UniDepth 代码 | torch.hub 在线拉 GitHub | 预置 `~/.cache/torch/hub/lpiccinelli-eth_UniDepth_main` | 代码 `force_reload=False`，缓存命中即零网络 |
| UniDepth 权重 | hf_hub_download 在线拉 | 01 脚本种入 HF 缓存 + `HF_HUB_OFFLINE=1` | huggingface.co 直连被 RST |
| wandb | 在线 | `WANDB_MODE=disabled` | wandb.ai 不可达；train.py 的 init/log 全部 no-op |

## 机制要点（读源码确认，排错先看这里）

- **cwd 敏感**：`train.py` 不 chdir，`checkpoints/`、`depth_scale.json`、`visualization/`
  全落在运行目录；且训练结束后 `train.py` 用 `os.system("python evaluate.py")`（相对路径）
  自动评估 —— 所以 `02` 固定 `cd $GAVS_DIR` 运行，跑完把产物搬到 `~/output/gavs_results/<scene>/`。
- **防跨场景误 resume**：`train.py` 发现 `checkpoints/` 存在会续训。`02` 开跑前检测残留，
  存在即报错（`FORCE=1` 强制删除重跑）。
- **显存**：3090 24GB vs 论文 RTX 5000 Ada 32GB。训练 320×576、batch=5 帧，理论可放下；
  若 OOM，先试 `data_loader.batch_size=3`（命令行追加覆盖）。
- **评估分辨率** 480×864、外扩 pad 96px（`layered_gavs_eval.yaml`），数据集 zip 里已含
  ProPainter 外推帧/GLOMAP 位姿/动态掩码，**不需要**自己跑预处理五件套。
- **场景清单**：15 个 = dynamic×5 + intense×5 + mild×5（DeepFused 改造）。

## 常见问题

**Q: `hf_hub_download` 卡住或报网络错？**
`_env.sh` 已设 `HF_HUB_OFFLINE=1`，权重由 01 种入缓存。若报找不到文件，检查
`~/.cache/huggingface/hub/models--lpiccinelli--unidepth-v1-vitl14/snapshots/local/pytorch_model.bin` 是否存在。

**Q: `import xformers.ops` 失败？**
说明 torch 被换成了非 cu121 版本（比如某依赖连带升级）。恢复：
`pip install --index-url https://download.pytorch.org/whl/cu121 torch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2`

**Q: 训练 OOM？**
追加 `data_loader.batch_size=3` 再试；仍不行改 `data_loader.num_workers=2`。

**Q: 官方 README 的 GLOMAP 链接是错的（指向 ProPainter）？**
已知 upstream 笔误（README.md:67,168），复现用官方预处理好的数据，不受影响。

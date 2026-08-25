# 4DAnyone — 单目视频 → 多视角同步视频（4DGS 重建前置）

[4DAnyone](https://github.com/ant-research/4DAnyone)（ant-research）把一段普通单目视频变成多个同步目标视角的视频，可直接喂给下游 4DGS 重建。它内部用 [GVHMR](https://github.com/zju3dv/GVHMR)（浙大）做运动恢复，再用 Wan2.2 系生成器渲染目标视角。

本目录只含编排脚本——官方代码在 `../4DAnyone`（00 自动 clone + 初始化 GVHMR 子模块），权重在 `../../model/4danyone`。

> ⚠️ **显存硬门槛**：6-view 推理峰值 **~43 GiB**（见 `docs/inference_performance.md`），RTX 3090（24 GiB）**跑不了**，A100-40GB 也不够。必须在服务器上用 **≥48 GiB** 的卡（A6000/A100-80GB/H100/H200）。README 的 Todo 标注 "Low-memory inference (<32 GB)" 尚未实现。

## 常用命令

> 假设已进入容器（脚本自动激活 `4danyone` env）；`GPU=0` 按需换卡。首次跑前先做下方「首次准备」。

```bash
# ── 一键：环境 → 下权重 → 推理 ──
# VIDEO_PATH: 9:16 竖屏、≥720p、≥121 帧的人像视频
GPU=0 VIDEO_PATH=../4DAnyone/data/source/pexels/2785536-uhd_2160_3840_25fps.mp4 \
  bash 4danyone/run_all.sh

# ── 分步 ──
# 0) 环境准备：clone 4DAnyone + GVHMR 子模块，建 4danyone env，装依赖
INSTALL_DEPS=1 bash 4danyone/00_setup_env.sh
# 1) 下权重（HF 模型 + BiRefNet；SMPL-X 见下）
bash 4danyone/01_download_models.sh
# 2) 推理（6-view 最低显存，~43 GiB）
GPU=0 VIDEO_PATH=../4DAnyone/data/source/pexels/2785536-uhd_2160_3840_25fps.mp4 \
  RESULTS_DIR=../4danyone_results \
  bash 4danyone/02_run_inference.sh

# ── 视角配置（VIEWS_PER_LAYER 必须被 4 或 6 整除）──
# 6-view 全轨道（最低显存，初始测试用）
GPU=0 VIDEO_PATH=... VIEWS_PER_LAYER=6 bash 4danyone/02_run_inference.sh
# 24-view 全轨道（适合 4DGS 重建）
GPU=0 VIDEO_PATH=... VIEWS_PER_LAYER=24 bash 4danyone/02_run_inference.sh
# 48-view 三层 pitch（自由视角 4DGS）
GPU=0 VIDEO_PATH=... VIEWS_PER_LAYER=16 LAYER_PITCHES='[-10,15,35]' \
  bash 4danyone/02_run_inference.sh
# 8-view 正面 180° 弧（只需前侧视角）
GPU=0 VIDEO_PATH=... VIEWS_PER_LAYER=8 START_YAW=-90 YAW_SPAN=180 \
  bash 4danyone/02_run_inference.sh
```

- 结果布局（`DATA_DIR` 默认 `$RESULTS_DIR`）：
  ```
  gvhmr/results/<clip>/          # 可复用的运动恢复结果
  fdanyone/<clip>/
  ├── metadata.json             # 运行设置/耗时/资源
  ├── cameras.json              # 最终 N 相机阵列
  ├── skeletons/00.mp4 ...       # 每视角骨架视频
  └── videos/
      ├── sparse/{00,04,09,12,14,19}.mp4  # 默认 24-view RCP 提案
      └── dense/00.mp4 ...                # 生成的目标视角视频
  ```

## 首次准备

```bash
cd <your-code-dir>            # e.g. /data_3d/<uid>/code
git -c http.sslVerify=false clone https://github.com/CrescentVelvet/media_code.git
cd media_code && cp proxy.env.example proxy.env   # 填 http_proxy / https_proxy
# ⚠️ 确认 proxy.env 中 http_proxy / https_proxy 已取消注释并填好，否则 pip 报 "Network is unreachable"

# 1. clone 4DAnyone + GVHMR 子模块，建 4danyone env（CPython 3.11），装依赖
#    torch 2.8 从 PyPI 装（PyPI wheel 自带 cu12x，不需要 download.pytorch.org，
#    公司代理封那个 host 也不影响）
INSTALL_DEPS=1 bash 4danyone/00_setup_env.sh

# 2. 下 HF 权重（公开，免 token；umt5-xxl encoder ~5GB，整体 ~10-15GB）
bash 4danyone/01_download_models.sh

# 3. （可选）下示例视频
EXAMPLE=1 bash 4danyone/01_download_models.sh
```

### SMPL-X（单独授权，不在 HF 上）

GVHMR 需要 `SMPLX_NEUTRAL.npz`，但它受 Max Planck 许可保护，**不在 Hugging Face 上**：

1. 注册账号并接受许可：https://smpl-x.is.tue.mpg.de/
2. 下载 `models_smplx_v1_1.zip`
3. 用 zip 路径重跑 01：
   ```bash
   SMPLX_ARCHIVE=/path/to/models_smplx_v1_1.zip bash 4danyone/01_download_models.sh
   ```
   01 会调用官方 `scripts/download_smplx.py --archive_path` 自动解压安装到 `$MODEL_DIR/body_models/smplx/SMPLX_NEUTRAL.npz` 并建 GVHMR 兼容链接。

## 权重目录布局

```
$MODEL_DIR/                          # ../../model/4danyone
├── 4danyone/
│   ├── model.safetensors            # 4DAnyone 主 checkpoint
│   ├── smplx_to_goliath70.pt        # SMPL-X → MHR70 回归器
│   ├── Wan2.2_VAE.pth              # Wan2.2 VAE
│   ├── models_t5_umt5-xxl-enc-bf16.pth  # UMT5-XXL 文本编码器 (~5GB)
│   └── umt5-xxl/                   # tokenizer (4 files)
├── gvhmr/
│   ├── gvhmr_siga24_release.ckpt   # GVHMR
│   ├── epoch=10-step=25000.ckpt    # HMR2
│   ├── vitpose-h-multi-coco.pth    # ViTPose
│   └── yolov8x.pt                  # YOLOv8x (人物检测)
├── birefnet/                        # 前景分割模型 (ZhengPeng7/BiRefNet)
│   └── model.safetensors
├── perceptual/
│   └── imagenet-vgg-verydeep-19-conv.safetensors
└── body_models/smplx/
    └── SMPLX_NEUTRAL.npz           # ← 需手动下载（见上）
```

---

以下为详细参考。

## Pipeline（流程详解）

```
单目视频
  │
  ├─ [GVHMR] 跟踪 + ViTPose + 图像特征提取 → 全局运动预测
  │     (输出: gvhmr/results/<clip>/ 可复用)
  │
  ├─ [骨架提取] 目标视角骨架条件渲染
  │     (输出: fdanyone/<clip>/skeletons/*.mp4)
  │
  └─ [去噪/生成] Wan2.2 系生成器渲染 N 个目标视角
        (输出: fdanyone/<clip>/videos/dense/*.mp4 + cameras.json)
              ↑
   可选加速: FlashAttention-3 (Hopper) / SageAttention
```

## Config (env vars)

| var | default | note |
|---|---|---|
| `CONDA_ENV` | `4danyone` | conda env 名（python=3.11, torch 2.8） |
| `GPU` | (unset) | 选卡（0-indexed）；设 `CUDA_VISIBLE_DEVICES` |
| `VIDEO_PATH` | (必填) | 输入视频；≥720p、9:16 竖屏、≥121 帧、单人全身/上半身、相机轻微运动 |
| `VIEWS_PER_LAYER` | `6` | 每层视角数；须被 4 或 6 整除；总视角 = `VIEWS_PER_LAYER × len(LAYER_PITCHES)` |
| `LAYER_PITCHES` | `[15]` | 每层 pitch 角（度），JSON 列表，如 `[-10,15,35]`；正值俯视，范围 [-15, 45] |
| `START_YAW` | `0` | 第一视角水平角（度）；0 = 正面 |
| `YAW_SPAN` | `360` | 每层覆盖的水平范围（度），1-360 |
| `DATA_DIR` | `$RESULTS_DIR` | GVHMR 运动 + 4DAnyone 输出根目录 |
| `DEVICE` | `cuda:0` | CUDA 设备（遵守 `CUDA_VISIBLE_DEVICES`） |
| `FDANYONE_DIR` | `$REPO_DIR/../4DAnyone` | 官方代码目录 |
| `GVHMR_DIR` | `$FDANYONE_DIR/third_party/GVHMR` | GVHMR 子模块 |
| `MODEL_DIR` | `$REPO_DIR/../../model/4danyone` | 权重根 |
| `RESULTS_DIR` | `$REPO_DIR/../4danyone_results` | 输出目录 |
| `SMPLX_ARCHIVE` | (unset) | 手动下的 SMPL-X zip 路径；装 SMPL-X 用 |

## 可能遇到的问题

1. **显存不足（OOM）**：6-view 最低也需 ~43 GiB。RTX 3090(24G)/A100-40G 都不够。换 ≥48G 卡（A6000/A100-80G/H100/H200）。`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` 已在 02 脚本默认设置，减少碎片但不能突破总量。低显存版（<32G）官方 Todo 未实现。
2. **SMPL-X 下载**：受许可保护，不在 HF。必须去 https://smpl-x.is.tue.mpg.de/ 注册+接受许可，下 `models_smplx_v1_1.zip`，再 `SMPLX_ARCHIVE=... bash 01_download_models.sh`。不装推理会在 02 检查时直接退出。
3. **HuggingFace 连不上**：在 `proxy.env` 设 `export HF_ENDPOINT="https://hf-mirror.com"`（国内镜像），01 的 `snapshot_download` 会自动走镜像。`_env.sh` 已设 `HF_HUB_DISABLE_XET=1`（避免 xet 协议在某些代理下失败）。
4. **GVHMR 子模块为空**：00 用 `git submodule update --init third_party/GVHMR`；公司代理下加 `git -c http.sslVerify=false`。若仍失败，手动 `cd $FDANYONE_DIR && git -c http.sslVerify=false submodule update --init third_party/GVHMR`。
5. **torch 装不上**：requirements pin `torch>=2.8,<2.9`。从 PyPI 默认源装（PyPI 的 torch wheel 自带 cu12x）。公司代理封 `download.pytorch.org` 不影响（4DAnyone 不用那个 index）。若 PyPI 也被限速，配清华镜像 `-i https://pypi.tuna.tsinghua.edu.cn/simple`。
6. **pip 冲突**：`smplx==0.1.28`、`timm==0.9.12`、`ultralytics==8.2.42` 是精确 pin，别让其他包升它们。`numpy>=1.26,<2`（numpy 2.x 与部分依赖不兼容）。

## 目录布局

```
<code-dir>/
├── media_code/4danyone/          # 本编排脚本
├── 4DAnyone/                      # 官方代码（00 自动 clone）
│   └── third_party/GVHMR/         # GVHMR 子模块
└── model/4danyone/                # 权重（共享 model 根下）
<code-dir>/../4danyone_results/    # 输出（media_code 的 sibling）
```

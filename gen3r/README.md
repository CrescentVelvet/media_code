# Gen3R — 单图/双图 → 3D 场景（点云 + 相机 + 视频）

复现 [Gen3R](https://github.com/JaceyHuang/Gen3R)（浙大 + 字节，arXiv 2601.04090）：
用 VGGT 的几何先验接进 Wan2.1-Fun 视频扩散，一次前向同时生成 **RGB 视频 + 相机位姿 +
深度图 + 全局点云**。三条任务共用一套权重：

| task | 输入 | 用途 |
|---|---|---|
| `1view` | 单张图 | 单图"脑补"环绕视角（我们要试的主线） |
| `2view` | 首帧 + 末帧（可给相机轨迹） | 两端约束中间插值 |
| `allview` | 视频 / 帧目录 | 纯重建（生成分量为零，等价 VGGT 重建） |

> **本仓定位**：Gen3R 只出点云，不出 Gaussian。`03_to_colmap.sh` 把它的
> (视频帧 + w2c 相机 + 点云) 转成 COLMAP 场景，交给 `vggt_human/04_train_3dgs.sh`
> 训练 3DGS——这就是"怎么利用"的落地接口。评估与取舍见 [NOTES.md](NOTES.md)。

## 常用命令

```bash
# 一键（首次跑前先做下方「首次准备」）
GPU=0 \
TASK=1view \
FRAME_PATH=$GEN3R_DIR/examples/1-view/knossos.png \
PROMPTS="A ruined Minoan palace courtyard with stone walls" \
CAMERAS=zoom_out \
OUTPUT_DIR=~/output/gen3r_results/02_infer \
bash gen3r/02_run_inference.sh

# 分步
# 1) 权重（37.7GB，HF 走 hf-mirror 镜像）
HF_ENDPOINT=https://hf-mirror.com \
MODEL_DIR=~/model/gen3r \
WORKERS=6 \
bash gen3r/01_download_models.sh
# 2) 推理
GPU=0 TASK=2view \
FRAME_PATH="examples/2-view/bedroom/first.png examples/2-view/bedroom/last.png" \
PROMPTS="A cozy bedroom with a large bed and warm lighting" \
CAMERAS=$GEN3R_DIR/examples/2-view/bedroom/cameras.json \
OUTPUT_DIR=~/output/gen3r_results/02_infer \
bash gen3r/02_run_inference.sh
# 3) 转 COLMAP（给 3DGS 训练）
GEN3R_OUT=~/output/gen3r_results/02_infer/1view/<时间戳> \
OUT_DIR=~/output/gen3r_results/03_colmap/knossos \
bash gen3r/03_to_colmap.sh
# 4) 训 3DGS（用 vggt_human 的原始 3DGS 链路）
GPU=0 SOURCE_DIR=~/output/gen3r_results/03_colmap/knossos \
GAUSSIAN_DIR=~/output/gen3r_results/04_model_3dgs/knossos \
bash vggt_human/04_train_3dgs.sh
```

## 首次准备

```bash
# WSL 本机：clone 官方仓到 ~/repos + 建 conda env + 装依赖
INSTALL_DEPS=1 bash gen3r/00a_setup_env.sh

# 服务器：clone 到 media_code 同级 + 复用已有 env
INSTALL_DEPS=1 bash gen3r/00_setup_env.sh

# 权重（37.7GB）
bash gen3r/01_download_models.sh
```

权重目录布局：

```
$MODEL_DIR/                      # ~/model/gen3r
  model_index.json
  transformer/                   # Wan2.1-Fun 1.3B DiT（Gen3R 微调）6.0 GB
  geo_adapter/                   # VGGT token ↔ Wan latent 适配器 1.2 GB
  vggt/                          # VGGT（相机/深度/点云头）4.4 GB
  wan_vae/                       # Wan2.1 VAE 0.5 GB
  text_encoder/                  # umt5-xxl，fp32 三片共 22.7 GB
  clip_image_encoder/            # CLIP 图像编码器 4.4 GB
  tokenizer/ scheduler/
```

## 产物（每次运行一个时间戳目录）

```
$OUTPUT_DIR/<task>/<YYYY-MM-DD-HH-MM-00>/
  rgb.mp4         # 49 帧 560x560，3D 一致的生成视频（3DGS 的训练图像源）
  pcds.ply        # 全局点云（世界系，带 RGB；体素 0.005 降采样 + 统计离群过滤）
  cameras.json    # extrinsics [F,4,4] w2c(OpenCV) + intrinsics [F,3,3]
  prompts.txt
```

`03_to_colmap.sh` 再生成：

```
$OUT_DIR/
  images/0000.png ... 0048.png
  sparse/0/cameras.txt     # PINHOLE，逐帧内参
  sparse/0/images.txt      # w2c 四元数（Gen3R 已是 w2c，不取逆）
  sparse/0/points3D.txt    # 点云，作为 3DGS 初始化
```

## Config (env vars)

| var | default | note |
|---|---|---|
| `GEN3R_DIR` | `~/repos/Gen3R` | 官方仓（WSL）；服务器为 `media_code` 同级 |
| `MODEL_DIR` | `~/model/gen3r` | 权重 |
| `RESULTS_DIR` | `~/output/gen3r_results` | 输出根 |
| `TASK` | `1view` | `1view` / `2view` / `allview` |
| `FRAME_PATH` | — | 1view 单图；2view 空格分隔两图；allview 视频/目录 |
| `PROMPTS` | — | 英文场景描述，`infer.py` 取 txt 第一行 |
| `CAMERAS` | `free` | `free` / `zoom_in|zoom_out|arc_left|arc_right|translate_up|translate_down` / cameras.json 路径 |
| `REMOVE_FAR_POINTS` | `0` | 1 = 截最远 20% 点、体素 0.003（近景主体场景建议开） |
| `WORKERS` | `6` | 下载并发，>8 会被 hf-mirror 429 |
| `MAX_POINTS` | `200000` | 转 COLMAP 时点云上限 |

## 目录布局

```
<code-dir>/
├── media_code/gen3r/            # 本目录（编排脚本）
├── Gen3R/                       # 官方仓（WSL: ~/repos/Gen3R）
└── gen3r_results/               # 输出（WSL: ~/output/gen3r_results）
```

## 已知限制

- 官方只发布推理代码与权重，训练代码未发布（README TODO）。
- 输出是**生成**内容，不是观测：单图任务的几何是"脑补"的，人物/人脸类主体
  会被场景先验拉偏，详见 NOTES.md 的实测。
- 默认分辨率 560×560（模型固定），更高分辨率要走后续超分。

排错见 [NOTES.md](NOTES.md)。

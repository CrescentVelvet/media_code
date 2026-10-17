# Gen3R (WSL Ubuntu 24.04) — 本地复现指南

本文件是 [`README.md`](README.md) 的 WSL 本地复现版。

## 与服务器版的核心差异

| | 服务器 | WSL 本机 |
|---|---|---|
| conda env | `doll` 等已有 env 或新建 | 新建 `gen3r`（python 3.11.2） |
| torch | 官方 `2.5.1+cu124` | **本地 wheel `2.8.0+cu128`**（见下方坑点） |
| 官方仓 | `media_code` 同级 | `~/repos/Gen3R`（Linux fs） |
| 权重 | HF 直连或镜像 | `hf-mirror.com`（**当前不稳定，见坑点**） |
| 输出 | `<algo>_results` | `~/output/gen3r_results`，跑完用 `08_move_output.sh` 搬到 D 盘 |

## Windows 路径 → WSL 路径

| Windows | WSL |
|---|---|
| `D:\dataset\测试数据sample\<id>` | `/mnt/d/dataset/测试数据sample/<id>` |
| `C:\code\media_code` | `/mnt/c/code/media_code` |

## 首次准备

```bash
# 1) clone 官方仓 + 建 env + 装依赖（torch 走本地 wheel）
cd /mnt/c/code/media_code
TORCH_VERSION=2.8.0 TV_VERSION=0.23.0 INSTALL_DEPS=1 \
  bash gen3r/00a_setup_env.sh

# 2) 权重 37.7GB（hf-mirror；断了就重跑，脚本自带断点续传）
MODEL_DIR=~/model/gen3r WORKERS=4 \
  bash gen3r/01_download_models.sh
```

## 全流程命令

```bash
cd /mnt/c/code/media_code

# ── 1) 推理（官方示例，验证链路）──
#    输入：官方自带的 knossos 单图
#    输出：~/output/gen3r_results/02_infer/1view/<时间戳>/
GPU=0 \
TASK=1view \
FRAME_PATH=/home/velvet/repos/Gen3R/examples/1-view/knossos.png \
PROMPTS="A ruined Minoan palace courtyard, stone walls, warm sunlight" \
CAMERAS=zoom_out \
OUTPUT_DIR=~/output/gen3r_results/02_infer \
bash gen3r/02_run_inference.sh

# 输出：~/output/gen3r_results/02_infer/1view/<ts>/
#   rgb.mp4       # 49 帧 560x560 生成视频
#   pcds.ply      # 世界系点云（带 RGB）
#   cameras.json  # w2c 外参 + 内参
#   prompts.txt

# ── 2) 用我们自己的采集包首帧（单图 → 环绕）──
#    输入：Remy 采集包（取第一张 jpg 当作唯一条件图）
GPU=0 \
TASK=1view \
FRAME_PATH=/mnt/d/dataset/测试数据sample/10e3ec6291c04bedaad735309e4bc43b/image_jpg/00000.jpg \
PROMPTS="A person standing in a room, full body, natural lighting" \
CAMERAS=arc_left \
OUTPUT_DIR=~/output/gen3r_results/02_infer \
bash gen3r/02_run_inference.sh

# ── 3) 转 COLMAP（给 3DGS 训练）──
#    输入：上面任一次运行的产物目录
#    输出：~/output/gen3r_results/03_colmap/<name>/{images,sparse/0}
GEN3R_OUT=~/output/gen3r_results/02_infer/1view/<时间戳> \
OUT_DIR=~/output/gen3r_results/03_colmap/knossos \
bash gen3r/03_to_colmap.sh

# ── 4) 训 3DGS（复用 vggt_human 的原始 3DGS）──
GPU=0 \
SOURCE_DIR=~/output/gen3r_results/03_colmap/knossos \
GAUSSIAN_DIR=~/output/gen3r_results/04_model_3dgs/knossos \
ITERATIONS=30000 \
bash vggt_human/04_train_3dgs.sh

# ── 5) 搬回 D 盘 ──
bash vggt_human/08_move_output.sh
```

## 可能遇到的问题（WSL 专属）

- **torch 版本**：本机联网装 `2.5.1+cu124` 要 ~1.5h（阿里源单线程 ~0.4MB/s），
  直接用 `/mnt/d/wheel` 的 `torch-2.8.0+cu128` + `torchvision-0.23.0` 本地 wheel，
  2 分钟装完，实测 3090 可用。Gen3R 代码未用到 2.5 专有 API，兼容。
- **本地 wheel 装 torch 必须先装纯 py 依赖**：`--no-index` 一并装 torch 会因
  `filelock/sympy/numpy` 缺失直接失败，顺序见 `00a_setup_env.sh`。
- **open3d 报 `libgomp.so.1: cannot open shared object file`**：系统没装 libgomp，
  `conda install -n gen3r -c conda-forge libgomp` + 运行时把 `$CONDA_PREFIX/lib`
  加进 `LD_LIBRARY_PATH`（`_env.sh` 已处理）。
- **opencv 版本**：`opencv-python-headless` 新版（5.x）会把 numpy 顶到 2.x，
  与本仓 `numpy==1.26.0` 冲突 → 固定 `4.10.0.84`。
- **hf-mirror 抖动**：2026-10-17 实测大文件经常 SSL 握手超时 / read timeout，
  4 并发约 0.2–2 MB/s 且不稳定。下载器支持断点续传，反复重跑直到 21 个文件齐全。
  并发 >8 会被 429。
- **`pkill -f download_models.py` 会杀掉自己**：`bash -lc` 的命令行本身含该字符串，
  要用 `[d]ownload_models` 这类写法或先 `pgrep` 再 kill。

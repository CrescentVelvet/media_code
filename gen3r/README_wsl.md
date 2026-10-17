# Gen3R (WSL Ubuntu 24.04) — 本地复现指南

本文件是 [`README.md`](README.md) 的 WSL 本地复现版。

## 与服务器版的核心差异

| | 服务器 | WSL 本机 |
|---|---|---|
| conda env | `doll` 等已有 env 或新建 | 新建 `gen3r`（python 3.11.2） |
| torch | 官方 `2.5.1+cu124` | **本地 wheel `2.8.0+cu128`**（见下方坑点） |
| attention | 官方 flash-attn 2.7.0 | **预编译 wheel flash_attn 2.8.3.post2**（cu12+torch2.8+cp311+abiTRUE，在 `/mnt/d/wheel/`） |
| 官方仓 | `media_code` 同级 | `~/repos/Gen3R`（Linux fs，**00a 会自动打本仓补丁**） |
| 权重 | HF 直连或镜像 | `aifasthub.com`（**hf-mirror 已废，见坑点**） |
| 输出 | `<algo>_results` | `~/output/gen3r_results`，跑完用 `08_move_output.sh` 搬到 D 盘 |

> ⚠️ **24GB 卡必读**：原版脚本在 3090 上不可用——它先按 fp32 把 37.7GB 权重推上显存再转 bf16，
> 直接溢出 21.6GB 到主机内存，单步 >20min。本仓两个补丁（`patch_infer_bf16.py` +
> `patch_pipeline_vram.py`）治好后单步 **9.5s**、显存 16.5GB。`00a_setup_env.sh` 装完依赖会自动打补丁。

## Windows 路径 → WSL 路径

| Windows | WSL |
|---|---|
| `D:\dataset\测试数据sample\<id>` | `/mnt/d/dataset/测试数据sample/<id>` |
| `C:\code\media_code` | `/mnt/c/code/media_code` |

## 首次准备

```bash
# 1) clone 官方仓 + 建 env + 装依赖 + 打 24GB 补丁（torch/flash-attn 走本地 wheel）
cd /mnt/c/code/media_code
INSTALL_DEPS=1 bash gen3r/00a_setup_env.sh

# 2) 权重 37.7GB。两条路二选一：
#    a. 脚本自动下（aifasthub，~2MB/s，断点续传）：
bash gen3r/01_download_models.sh
#    b. 迅雷批量下（推荐，几分钟）：全选复制 gen3r/download_urls.txt 到迅雷，
#       目录设 D:\wheel\gen3r_ms（子目录不用管），下完：
bash gen3r/01b_link_local_weights.sh
```

常用 env（推理时）：
- `GEN3R_STEPS=10`：快速烟测（默认 50，1view 全程约 13min：加载1 + VGGT4.5 + 采样8）
- `GEN3R_OFFLOAD_ENCODERS=0`：关掉编码器显存卸载（默认开，24GB 必须开）
- `GEN3R_DTYPE=fp32`：回退官方原版行为（24GB 卡上别用）

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
- **hf-mirror 已废**：2026-10-17 实测大文件 SSL 握手超时 / read timeout，并发 >8 还会被 429。
  改用 `--endpoint https://aifasthub.com`（`download_models.py` 已伪装 UA，aifasthub 会拦
  `Python-urllib`）。已验证下载 URL 302 到 `us.aws.cdn.hf.co`（Xet CDN），迅雷吃满带宽。
- **flash-attn 必须装**：没有它 SDPA 走 math 后端，24GB 卡 49 帧×560² 单步 >4min；
  装上 2.8.3.post2 预编译 wheel 后单步 9.5s。wheel 已在 `/mnt/d/wheel/`，
  或从 GitHub releases 下（搜 `flash_attn-2.8.3.post2+cu12torch2.8cxx11abiTRUE-cp311`）。
- **显存溢出到主机内存的症状**：GPU util 100% 但功率 ~180W、单步 20min+ 不出进度。
  Windows 侧查 `Get-Counter '\GPU Adapter Memory(*)\Shared Usage'`，
  >0 GB 就是 WSL 在把显存溢到 PCIe 上跑。
- **`pkill -f download_models.py` 会杀掉自己**：`bash -lc` 的命令行本身含该字符串，
  要用 `[d]ownload_models` 这类写法或先 `pgrep` 再 kill。

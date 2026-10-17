# FlashWorld (WSL Ubuntu 24.04) — 本地复现指南

本文件是 [`README.md`](README.md) 的 WSL 本地复现版。照着做即可从零跑完全流程。
原理、相机约定、排错见 [NOTES.md](NOTES.md)；权重下载清单见 [download_urls.md](download_urls.md)。

## 与服务器版的核心差异

| | 服务器 | WSL |
|---|---|---|
| conda env | 从零建 `flashworld`（python 3.11 + torch 2.6.0） | **clone `minimax_h3`**（已含 torch 2.6.0+cu124 + triton 3.2.0）→ 省 4 GB 下载 |
| CUDA toolkit | 系统 `/usr/local/cuda` | conda `cuda-nvcc` 12.4（无 sudo、无系统 nvcc） |
| gcc/g++ | 系统自带 | conda `gxx_linux-64=12`（本机**没有**系统 gcc） |
| pip 源 | 阿里云镜像 | 阿里云镜像 + `--find-links /mnt/d/wheel` 先查本地 wheel |
| 仓库位置 | `<code-dir>/FlashWorld`（与 media_code 同级） | `~/repos/FlashWorld`（Linux fs，编译快） |
| 权重位置 | `<code-dir>/../model/flashworld` | `/mnt/d/wheel/flashworld_ms`（D 盘，读一次） |
| 输出 | `<code-dir>/flashworld_results` | `~/output/flashworld_results`，跑完 08 搬 D 盘 |
| 网络 | 直连 | github clone 通；huggingface.co **RST**；hf-mirror ~1.9 MB/s；modelscope ~2.3 MB/s |

## 前提条件

- WSL2 + NVIDIA 驱动（本机 3090 24 GB，驱动 595.95）
- Miniconda 在 `~/miniconda3`
- 已有 conda env `minimax_h3`（python 3.11 + torch 2.6.0+cu124 + triton 3.2.0）
  - 没有就 `SKIP_ENV_CLONE=1 bash flashworld/00a_setup_env.sh` 从零建（会多下 ~4 GB）

## 首次准备

```bash
cd /mnt/c/code/media_code

# 建 env（clone minimax_h3 → flashworld）+ 装 nvcc/gxx + clone 官方仓 + 编译 gsplat/spz
# 约 10-20 min（gsplat 编译占大头）
bash flashworld/00a_setup_env.sh

# 下权重：54 GB。两条路选一条
#   A) 自动（hf-mirror ~1.9 MB/s，约 7 h，可断点续传，建议后台跑）
nohup bash flashworld/01_download_models.sh > ~/fw_download.log 2>&1 &
#   B) 迅雷手动下（快）：按 download_urls.md 下到 /mnt/d/wheel/flashworld_ms/，然后
# bash flashworld/01b_link_local_weights.sh

# 校验
CHECK_ONLY=1 bash flashworld/01_download_models.sh
```

## 全流程命令

```bash
cd /mnt/c/code/media_code

# ── 1) 单图 → 输入 json（相机轨迹在这里定）──
#    输入：/mnt/d/dataset/scene.jpg（单张，或换成目录批量）
#    输出：~/output/flashworld_results/02_input_json/scene.json
#    可选开关：TRAJ=orbit(默认)/official_orbit(官方±15°)/forward；YAW_SPAN=40；FOV=60
GPU=0 \
IMAGE=/mnt/d/dataset/scene.jpg \
OUTPUT=~/output/flashworld_results/02_input_json \
TRAJ=orbit YAW_SPAN=40 FOV=60 \
bash flashworld/02_make_input.sh

# 输出：~/output/flashworld_results/02_input_json/
#   scene.json     # 24 个相机（c2w，四元数 [w,x,y,z]）+ 分辨率 + 图片绝对路径

# ── 2) 推理 → 3DGS ──
#    输入：上一步的 json 目录
#    输出：~/output/flashworld_results/03_gaussians/<json 同名>/
#    可选开关：VIDEO=1 出环绕视频（吃显存）；SPZ=1 出压缩格式；REALLY_LOW_VRAM=1（<10 GB 但极慢）
GPU=0 \
INPUT_DIR=~/output/flashworld_results/02_input_json \
OUT_DIR=~/output/flashworld_results/03_gaussians \
PLY=1 VIDEO=0 \
bash flashworld/02_run_inference.sh

# 输出：~/output/flashworld_results/03_gaussians/scene/
#   gaussians.ply    # 3DGS（世界尺度已按轨迹半径还原）
#   video.mp4        # 仅 VIDEO=1 时

# ── 3) 搬到 Windows 盘（vhdx 空间有限）──
KEEP_SRC=0 bash flashworld/08_move_output.sh
# 输出：/mnt/d/output/flashworld_results/
```

一键版（2 + 1 合并）：

```bash
GPU=0 IMAGE=/mnt/d/dataset/scene.jpg VIDEO=0 bash flashworld/run_all.sh
```

## 可能遇到的问题（WSL 专属）

| 现象 | 处理 |
|---|---|
| `conda: command not found` | `source ~/miniconda3/etc/profile.d/conda.sh`；`_env.sh` 有 fallback，但 `00a` 之外手动跑命令时要自己 source |
| gsplat 编译报 `nvcc not found` | 00a 装的是 conda 版 nvcc（`$CONDA_PREFIX/bin/nvcc`）；`_env.sh` 会自动设 `CUDA_HOME`。手动跑时先 `source flashworld/_env.sh` |
| 编译卡在 `/mnt/d` | 不要在任何 `/mnt/*` 目录里编译（9p 文件系统，symlink 会坏）；代码固定放 `~/repos/` |
| `pip install` 慢 | 已在 00a 里配阿里云镜像 + `--find-links /mnt/d/wheel`；大包走手动 wheel 放 `/mnt/d/wheel` |
| `huggingface.co` RST / hf-mirror 慢 | 用迅雷 + `01b_link_local_weights.sh`；或 modelscope 域名替换（见 download_urls.md） |
| CUDA OOM | `REALLY_LOW_VRAM=1`；或 `N_FRAME=16`（偏离训练分布，画质会掉） |
| vhdx 变大 | `08_move_output.sh` 搬完 → PowerShell `wsl --shutdown` + `diskpart compact vdisk` |

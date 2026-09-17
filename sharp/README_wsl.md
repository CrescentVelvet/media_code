# SHARP (WSL Ubuntu 24.04) — 本地复现指南

[SHARP](https://github.com/apple-aiml-research/ml-sharp)（Apple，ICLR 2026，arXiv 2512.10685）：
**单张照片 → 一次性前馈回归出 ~120 万个 3D Gaussian**，亚秒级完成，输出带绝对尺度的
3DGS `.ply`，可用任意公开 3DGS 渲染器实时渲染。

> **本仓只有 WSL 链路**（本机 `C:\code\media_code` 在 WSL Ubuntu 24.04 下跑）。
> 脚本命名/路径策略沿用 `vggt_human/` 的同一套约定（见 [`../AGENTS.md`](../AGENTS.md)）。

## ⚠️ 先看两条硬约束

1. **官方仓只有推理代码**（`pyproject.toml` 的 description 原文：*"Inference/Network/Model
   code for SHARP view synthesis model."*）。**训练无法复现** —— 论文的训练数据/流程未开源。
   本链路复现的是「单图 → 3DGS + 环绕视频」的推理端到端流程。
2. **许可证只允许研究用途**（`LICENSE_MODEL` 原文）：*"Research Purposes means non-commercial
   scientific research and academic development activities… does not include any commercial
   exploitation, product development or use in any commercial product or service."*
   商用/产品化被明确排除。

## 与 vggt_human 的差异

| | vggt_human | sharp |
|---|---|---|
| conda env | `vggt_human`（python 3.10 + torch cu121） | `sharp`（python 3.13 + torch 2.8.0 cu128） |
| 官方仓 | VGGT-Omega + gaussian-splatting（两个） | `ml-sharp`（一个，自包含） |
| CUDA 扩展 | diff-gaussian-rasterization + simple-knn（**要编译**） | gsplat 1.5.3 纯 Python wheel（**首次运行时 JIT 编译**） |
| 权重 | VGGT-Omega 4.6GB（HF **gated**，要 token） | 1 个文件 2.62GB（**非 gated，免 token**） |
| 输出 | COLMAP 场景 → 训练 3DGS | 直接出 3DGS `.ply` + `.mp4` |

## 前提条件

1. **WSL Ubuntu 24.04** 已安装并运行（本机：`wsl -d Ubuntu2404`）
2. **NVIDIA 驱动**（本机实测：Windows 侧 595.95，WSL 内 `nvidia-smi` 可见
   `RTX 3090 24576 MiB` —— 支持 CUDA 12.8，无需升级）
3. **Miniconda**：`~/miniconda3`（本机已有）
4. **磁盘**：WSL vhdx 需 ≥ 15GB 余量（torch+cu128 全家桶约 4-5GB，CUDA toolkit 约 3GB，
   权重 2.6GB）。本机当前 `/` 884GB 可用。

## 首次准备（一次性）

```bash
wsl -d Ubuntu2404
cd /mnt/c/code/media_code

# 建 env + 装 torch/cuda-toolkit/gcc + clone 官方仓 + 装依赖
bash sharp/00a_setup_env.sh

# 下权重（2.62GB）
bash sharp/01_download_models.sh
```

`00a` 会做 6 件事：conda 建 `sharp`(py3.13) → pip 装 torch 2.8.0+torchvision 0.23.0 →
conda 装 `cuda-toolkit 12.8` + `gxx_linux-64=12`（**不需要 sudo**）→ clone
`~/repos/ml-sharp` → `pip install -r requirements.txt`（含 `-e .`，产出 `sharp` 命令）→
把 `SHARP_*` 路径追加进 `proxy.env`。

> **为什么必须装 nvcc + gcc？** `gsplat==1.5.3` 在 PyPI 上是 `py3-none-any` 纯 Python wheel，
> 里面**没有**编译好的 CUDA 光栅化核 —— 首次用到渲染时通过 torch 的 JIT 现场编译。
> 本机没有系统 gcc/g++，所以两者都装进 conda env。编译产物缓存在
> `~/.cache/torch_extensions/`，只编译一次。

## 模型下载（迅雷）

**只需要一个文件**，我已在 WSL 内用 HTTP HEAD 实测过大小（`content-length`）：

| 用途 | 直链 | 实测大小 | 说明 |
|---|---|---|---|
| **模型权重（唯一必需）** | `https://ml-site.cdn-apple.com/models/sharp/sharp_2572gikvuh.pt` | **2,809,738,232 B ≈ 2.62 GiB** | Apple 官方 CDN（落地节点 hkhkg，即香港），`accept-ranges: bytes` → **支持迅雷断点续传/多线程** |
| 备用镜像 | `https://huggingface.co/apple/Sharp/resolve/main/sharp_2572gikvuh.pt` | 同 | HF repo `apple/Sharp`，**非 gated，无需 token**；国内可换 `hf-mirror.com` |
| torch wheel（可选） | `https://download.pytorch.org/whl/cu128/torch-2.8.0%2Bcu128-cp313-cp313-manylinux_2_28_x86_64.whl` | 889,052,836 B ≈ 848 MiB | pip 慢时才用；下完 `pip install <本地whl>` |
| torchvision wheel（可选） | `https://download.pytorch.org/whl/cu128/torchvision-0.23.0%2Bcu128-cp313-cp313-manylinux_2_28_x86_64.whl` | 8,627,708 B | 同上 |

迅雷下完后的两种接法（任选）：

```bash
# A. 直接拷到权重目录（脚本会校验字节数，对不上会拒绝）
mkdir -p /mnt/d/wheel/sharp_ms
cp /mnt/d/downloads/sharp_2572gikvuh.pt /mnt/d/wheel/sharp_ms/

# B. 让 01 脚本替你搬（同样会校验大小）
SRC=/mnt/d/downloads/sharp_2572gikvuh.pt bash sharp/01_download_models.sh
```

> **不要**只下模型就以为齐了：torch + CUDA toolkit + gsplat 的 nvidia-* 依赖包（pip 会自动拉
> cudnn/cublas 等约 3GB）仍需从 PyPI 装。走 `proxy.env` 里的清华镜像即可。

## 路径布局（WSL）

```
~/repos/ml-sharp/                 # 官方仓（Linux fs，JIT 编译快）
~/output/sharp_results/           # 推理输出（跑完用 08 搬到 D 盘）
~/miniconda3/envs/sharp/          # conda env（py3.13 + torch 2.8 cu128 + nvcc + gcc12）

/mnt/d/wheel/sharp_ms/            # 权重：sharp_2572gikvuh.pt（D 盘，读一次即可）
/mnt/c/code/media_code/sharp/     # 本目录（编排脚本，git 仓）
/mnt/d/output/sharp_results/      # 最终归档（step 08 搬过来）
```

路径通过 `proxy.env` 的 `SHARP_MODEL_DIR` / `SHARP_RESULTS_DIR` 覆盖，脚本不用改。

## 全流程命令

```bash
cd /mnt/c/code/media_code

# ── 1) 下载权重（首次，2.62GB）──
bash sharp/01_download_models.sh

# ── 2) 推理：单图或目录 → 3DGS .ply ──
GPU=0 INPUT=/mnt/d/dataset/sample/a.jpg bash sharp/02_run_inference.sh
#    目录也行（递归匹配所有支持的图像格式）：
#    GPU=0 INPUT=/mnt/d/dataset/sample bash sharp/02_run_inference.sh
# 输出：~/output/sharp_results/02_gaussians/<stem>.ply

# ── 2b) 加渲染：额外输出环绕轨迹 .mp4（仅 CUDA，首次会 JIT 编译 gsplat）──
GPU=0 INPUT=/mnt/d/dataset/sample RENDER=1 bash sharp/02_run_inference.sh

# ── 3) 归档到 D 盘（释放 WSL 空间）──
DRY_RUN=1 bash sharp/08_move_output.sh     # 预览
bash sharp/08_move_output.sh               # 实际搬
```

`.ply` 可直接拖进 [SuperSplat](https://playcanvas.com/supersplat/editor) 在线查看。
坐标系遵循 OpenCV 约定（x 右 / y 下 / z 前），场景中心大致在 `(0, 0, +z)`；
导入第三方渲染器时需自行缩放旋转居中。

## 已知风险与排错

| 症状 | 原因 / 处理 |
|---|---|
| `--render` 报 CUDA 编译错 | nvcc/g++ 不可用。查：`nvcc --version`、`which x86_64-conda-linux-gnu-g++`。修：`conda activate sharp && conda install -c conda-forge gxx_linux-64=12` |
| JIT 编译卡很久 | 正常（分钟级）。设了 `TORCH_CUDA_ARCH_LIST=8.6`（`_env.sh` 里针对 3090）可显著缩短；换卡要覆盖该变量 |
| `sharp: command not found` | `conda activate sharp && pip install -e ~/repos/ml-sharp` |
| pip 装 torch 慢 | 清华镜像已配（`proxy.env` 的 `PIP_INDEX_URL`）；或迅雷下 wheel 后本地装 |
| 显存不足 | 3090 24GB 对 1536×1536 输入绰绰有余；若报 OOM 先试单张图 |
| `conda install` 把 python 换掉 | 脚本里已显式 pin `python=3.13`，与 vggt_human 同款处理 |

## 参考

- 论文：arXiv [2512.10685](https://arxiv.org/abs/2512.10685)（ICLR 2026）
- 官方仓：<https://github.com/apple-aiml-research/ml-sharp>（`apple/ml-sharp` 会重定向到此）
- 效果页：<https://apple.github.io/ml-sharp/>（含与相关工作的视频对比）

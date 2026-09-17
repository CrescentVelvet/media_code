# sharp — Apple SHARP 单图 → 3DGS 前馈推理（ICLR 2026）本地复现

给定**一张照片**，用一次前馈网络回归出约 120 万个 3D Gaussian 的参数（<1s，标准 GPU），
输出带**绝对度量尺度**的 3DGS `.ply`，并可实时渲染周边视角（支持环绕轨迹视频）。
官方代码：<https://github.com/apple-aiml-research/ml-sharp>（唯一权重
`sharp_2572gikvuh.pt`，2.62 GiB，非 gated）。

> ⚠️ 官方仓**只开源推理代码**，训练流程未开源，无法复现训练。
> ⚠️ 模型许可证限**研究用途**，明确排除商用（见 `LICENSE_MODEL`）。

> 🖥️ 本机（WSL Ubuntu 24.04 + RTX 3090）的完整复现步骤看
> **[`README_wsl.md`](README_wsl.md)** —— 含迅雷下载链接、路径布局、排错表。

## 常用命令

> 在 WSL 内、`/mnt/c/code/media_code` 目录下执行。首次跑前先做下方「首次准备」。

```bash
# 一键（安装 + 下载 + 推理）
bash sharp/00a_setup_env.sh
bash sharp/01_download_models.sh
GPU=0 INPUT=/mnt/d/dataset/sample \
  RESULTS_DIR=~/output/sharp_results \
  bash sharp/02_run_inference.sh

# 分步
# 1) 权重（2.62GB；迅雷下好的用 SRC= 导入）
bash sharp/01_download_models.sh
# 2) 推理 → 3DGS .ply
GPU=0 INPUT=/mnt/d/dataset/sample \
  SHARP_CKPT=/mnt/d/wheel/sharp_ms/sharp_2572gikvuh.pt \
  RESULTS_DIR=~/output/sharp_results \
  bash sharp/02_run_inference.sh
# 2b) 附带环绕轨迹视频（仅 CUDA；首次会 JIT 编译 gsplat）
GPU=0 INPUT=/mnt/d/dataset/sample RENDER=1 \
  RESULTS_DIR=~/output/sharp_results \
  bash sharp/02_run_inference.sh
# 3) 归档到 D 盘
bash sharp/08_move_output.sh
```

## 首次准备

```bash
wsl -d Ubuntu2404 && cd /mnt/c/code/media_code
bash sharp/00a_setup_env.sh     # conda env sharp(py3.13) + torch 2.8 cu128 + cuda-toolkit + gcc12 + clone
bash sharp/01_download_models.sh
```

权重目录布局：

```
/mnt/d/wheel/sharp_ms/
  sharp_2572gikvuh.pt     # 2,809,738,232 B（模型唯一权重，含编码器）
```

## Config (env vars)

| var | default | note |
|---|---|---|
| `GPU` | 空 | `CUDA_VISIBLE_DEVICES`，如 `GPU=0` |
| `INPUT` | 必填（02） | 单张图或图片目录（递归匹配） |
| `RENDER` | `0` | `1` = 额外渲染环绕轨迹 `.mp4`（仅 CUDA） |
| `DEVICE` | `default` | 传给 CLI 的 `--device`（`cuda`/`cpu`/`mps`） |
| `SHARP_CKPT` | `$SHARP_MODEL_DIR/sharp_2572gikvuh.pt` | 权重路径 |
| `SHARP_DIR` | `~/repos/ml-sharp` | 官方仓（WSL） |
| `SHARP_MODEL_DIR` | `/mnt/d/wheel/sharp_ms` | 权重根（写在 `proxy.env`） |
| `SHARP_RESULTS_DIR` | `~/output/sharp_results` | 输出根（写在 `proxy.env`） |
| `TORCH_CUDA_ARCH_LIST` | `8.6` | 3090 = sm_86；换卡要覆盖 |
| `SRC` | 空（01） | 导入外部已下载的权重文件 |

## 目录布局

```
media_code/sharp/                 # 本目录（编排脚本）
~/repos/ml-sharp/                 # 官方代码（00a clone）
/mnt/d/wheel/sharp_ms/            # 权重
~/output/sharp_results/           # 输出（08 搬到 /mnt/d/output/sharp_results）
~/miniconda3/envs/sharp/          # conda env
```

## Notes

- **实测（3090）**：单图 02 脚本端到端 42 秒（首次）/ 26 秒（JIT 缓存后），其中权重加载约 30 秒；
  产出 `.ply` 66 MB / 1,179,648 个高斯。论文给的 <1s 是 A100 上的纯前馈推理时间，不含权重加载。
- 输出 `.ply` 遵循 OpenCV 坐标系（x 右 / y 下 / z 前），中心在 `(0,0,+z)`；
  第三方渲染器需自行缩放旋转居中。有 EXIF 焦距的照片才有准确度量尺度（无则按 30mm 兜底）。
- `gsplat` 是纯 Python wheel，CUDA 核**首次运行时 JIT 编译**（实测 113 秒）→ 必须装 nvcc + gcc，
  且需把 `$CONDA_PREFIX/targets/x86_64-linux/include` 补进 `$CONDA_PREFIX/include`
  （`00a` 已自动处理，详见 [`README_wsl.md`](README_wsl.md) 排错表）。

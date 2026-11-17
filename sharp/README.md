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

# ── 机制诊断（纯推理，不训练，不占训练资源）────────────────────
# 4) 层数消融：一份前馈输出按 layer-major 拆两层，full/layer0/layer1 各自渲染
GPU=0 INPUT=/mnt/d/dataset/sample/xxx.heic \
  OUT_DIR=~/output/sharp_results/03_layer_ablation \
  bash sharp/03_layer_ablation.sh
# 5) 受控干预：复制/抖动/缩放/改opacity/换色 + 双底色 + 几何错位诊断
GPU=0 INPUT=/mnt/d/dataset/sample/xxx.heic \
  OUT_DIR=~/output/sharp_results/04_controlled \
  RENDER_SCALE=0.5 JITTER_PX=0.5 \
  bash sharp/04_controlled_ablation.sh
# 04 支持 `--bg black,white`（默认两者都跑）；单层 PSNR 对底色极敏感，见 EXPERIMENTS.md
# 产出 layer_ablation.json / controlled_ablation.json，含 layer_gap、d1_geometry、H3 分档
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
| `OUT_DIR` | `$RESULTS_DIR/03_layer_ablation`<br>`$RESULTS_DIR/04_controlled_ablation` | 03/04 的输出目录 |
| `RENDER_SCALE` | `0.5` | 04 的渲染分辨率缩放（0.5 → 960×720，速度×4） |
| `JITTER_PX` | `0.5` | 04 里 `layer0_dup_jit` 的像素抖动量 |

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
  第三方渲染器需自行缩放旋转居中。焦距来自EXIF（无则按30mm兜底）；有EXIF也不保证预测尺度准确。
  **焦距不只是尺度**：`disparity_factor = f_px/width` 同时进入网络作为条件输入（5 通道视差归一化），
  焦距错 → 输入本身就错，事后乘系数只能改尺度、补不回细节。
- `gsplat` 是纯 Python wheel，CUDA 核**首次运行时 JIT 编译**（实测 113 秒）→ 必须装 nvcc + gcc，
  且需把 `$CONDA_PREFIX/targets/x86_64-linux/include` 补进 `$CONDA_PREFIX/include`
  （`00a` 已自动处理，详见 [`README_wsl.md`](README_wsl.md) 排错表）。
- 三图人像接入边界见 [NOTES.md](NOTES.md)，试跑与验收状态见 [EXPERIMENTS.md](EXPERIMENTS.md)。

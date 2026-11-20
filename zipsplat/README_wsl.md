# ZipSplat (WSL Ubuntu 24.04) — 佳佳抱枕复现

在 RTX 3090 24GB 上从多视角 PNG/HEIC 采集包前馈重建高斯，不需要输入相机。
本机实测使用 Python 3.11 + torch 2.8.0+cu128；不是官方基准环境的逐项复刻。

## 常用命令

```bash
cd /mnt/c/code/media_code

# ── 0) 建独立环境 ──
# 复用 querysplat 的 torch/CUDA 栈，clone 成 zipsplat，不修改原 env。
CONDA_ENV=zipsplat ZIPSPLAT_REPO=~/repos/ZipSplat \
  bash zipsplat/00a_setup_env.sh

# ── 1) 下载官方真实图像权重（约 5.59 GiB）──
# 实体：/mnt/d/wheel/zipsplat/；消费侧：~/model/zipsplat/ 软链。
WEIGHT_PATH=~/model/zipsplat/zipsplat-da3g-504.tar \
  bash zipsplat/01_download_models.sh

# ── 2) 三视角冒烟（先用已有转换图；验证权重、前馈、gsplat 与 PLY）──
SCENE_NAME=jiajia_smoke_3v \
  IMAGES_DIR=~/output/infinisplat_v2_results/_captures/佳佳抱枕_3v/images \
  WEIGHT_PATH=~/model/zipsplat/zipsplat-da3g-504.tar \
  RESULTS_DIR=~/output/zipsplat_results \
  bash zipsplat/02a_smoke_jiajia.sh

# ── 3) 原始 82 张 HEIC → 16 张去畸变 PNG ──
# 输出：~/output/zipsplat_results/_captures/jiajia_baozhen/images。
RAW_DIR=/mnt/d/dataset/手部微动样例/01e954de1b8a4cde9315c976f4dcb964-佳佳抱枕 \
  CAP_DIR=~/output/zipsplat_results/_captures/jiajia_baozhen N_VIEWS=16 \
  bash zipsplat/01b_prepare_jiajia.sh

# ── 4) 16 视角前馈 + TokenBA + PLY/相机/渲染 ──
IMAGES_DIR=~/output/zipsplat_results/_captures/jiajia_baozhen/images \
  WEIGHT_PATH=~/model/zipsplat/zipsplat-da3g-504.tar \
  RESULTS_DIR=~/output/zipsplat_results SCENE_NAME=jiajia_16v \
  MAX_VIEWS=16 BA=1 RENDER_VIDEO=1 bash zipsplat/02_run_inference.sh

# ── 5) 构造独立留出帧（不参与 16 视角重建或 TokenBA）──
RAW_DIR=/mnt/d/dataset/手部微动样例/01e954de1b8a4cde9315c976f4dcb964-佳佳抱枕 \
  CAP_DIR=~/output/zipsplat_results/_captures/jiajia_holdout \
  FRAMES=3,14,19,36,47,62,74,79 bash zipsplat/01b_prepare_jiajia.sh

# ── 6) 源视角自洽 + 留出视角对齐评测 ──
RESULT=~/output/zipsplat_results/jiajia_16v \
  SOURCE_CAMERAS=~/output/zipsplat_results/_captures/jiajia_baozhen/cameras.npz \
  TARGET_CAMERAS=~/output/zipsplat_results/_captures/jiajia_holdout/cameras.npz \
  bash zipsplat/03_evaluate_jiajia.sh
# ── 7) 归档到 D 盘（逐文件长度校验）──
RESULTS_DIR=~/output/zipsplat_results bash zipsplat/08_move_output.sh
```

## 本机实测速查（2026-11-20）

16 视角、378×504 网络输入、248,832 高斯、峰值 allocated 9,832MiB：前馈约 2.5s、
TokenBA 10.6s。TokenBA 输入自洽 23.334dB / SSIM 0.6453，8 留出 19.995dB /
SSIM 0.4817；严格前馈分别为 20.827 / 0.5088 与 19.757 / 0.4772。协议和限制见
[`EXPERIMENTS.md`](EXPERIMENTS.md)。

`01b_prepare_jiajia.sh` 的第一个参数也可直接写采集包路径，避免跨 Windows/WSL 传中文环境变量。

## 输出与口径

- `jiajia_16v/gaussians_feedforward.ply`：官方前馈高斯；`gaussians.ply`：TokenBA 后高斯。
- `predicted_cameras.npz/json`：OpenCV c2w + 原图内参；`metrics.json`：耗时、峰值显存、alpha 覆盖率。
- `evaluation/metrics.json`：源视角自洽与 8 个独立留出视角，二者不混称。
  严格前馈另用 `SCENE_NAME=jiajia_16v_feedforward BA=0` 独立运行后评测。
- 人物与抱枕不是静态刚性物体；手持抱枕可能有微动。该包测试的是人物＋抱枕＋局部室内环境，不是纯抱枕分割重建。
- 数据包有动态/微动与环绕大基线，不能直接把论文 DL3DV 数字当作本包应达到的分数。

## 可能遇到的问题

- 跨 Windows 调 WSL：优先在 WSL shell 中直接运行仓内脚本。通用 `wsl_run.ps1` 会把脚本复制到 `/tmp`，依赖 `SCRIPT_DIR/_env.sh` 的算法脚本不能直接走该包装器。
- Windows → WSL 的环境变量不会自动继承：路径可作为脚本位置参数传入；需要环境变量时写 `wsl.exe -d Ubuntu2404 -- env KEY=value bash /mnt/c/.../script.sh`（走 PowerShell 工具，不要让 Git Bash 重写 `/mnt/...` 路径）。
- 镜像下载中断：分段临时文件不会被当作完整权重；`01` 只有在总字节数等于 6,002,349,727 且 `torch.load` 成功才通过。
- 显存不足：先跑 `02a`；16 视角 OOM 后降到 8 视角（重新转换输入），不要通过删减评测帧来掩盖失败。
- 模型权重仅非商业使用，详见官方 model card。

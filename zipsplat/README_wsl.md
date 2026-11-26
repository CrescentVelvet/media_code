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
# 输出：~/output/zipsplat_results/<dir>/
#   file_a      # 说明

# ── 7) 可选：与 DA3-GS / SplatAnything 的同三输入对照 ──
# 从原始包显式选 0/40/81；准备脚本会校验它与历史三输入逐字节一致。
FRAMES=0,40,81 CAP_DIR=~/output/zipsplat_results/_captures/jiajia_3v_compare \
  bash zipsplat/01b_prepare_jiajia.sh \
  /mnt/d/dataset/手部微动样例/01e954de1b8a4cde9315c976f4dcb964-佳佳抱枕
IMAGES_DIR=~/output/zipsplat_results/_captures/jiajia_3v_compare/images \
  WEIGHT_PATH=~/model/zipsplat/zipsplat-da3g-504.tar \
  ZIPSPLAT_RESULTS_DIR=~/output/zipsplat_results \
  SCENE_NAME=jiajia_3v_compare MAX_VIEWS=3 BA=1 RENDER_VIDEO=1 \
  bash zipsplat/02_run_inference.sh
IMAGES_DIR=~/output/zipsplat_results/_captures/jiajia_3v_compare/images \
  WEIGHT_PATH=~/model/zipsplat/zipsplat-da3g-504.tar \
  ZIPSPLAT_RESULTS_DIR=~/output/zipsplat_results \
  SCENE_NAME=jiajia_3v_compare_feedforward MAX_VIEWS=3 BA=0 RENDER_VIDEO=0 \
  bash zipsplat/02_run_inference.sh
# 三输入共同画布 512²、29 留出；分别指定 CASE_FILTER 运行，结果分开保存。
OUT_DIR=/home/velvet/output/zipsplat_results \
  CASE_FILTER=ZipSplat RESULT_NAME=ZipSplat_metrics.json \
  bash zipsplat/03b_eval_3view_compare.sh
OUT_DIR=/home/velvet/output/zipsplat_results \
  CASE_FILTER=DA3-GS RESULT_NAME=DA3-GS_metrics.json \
  bash zipsplat/03b_eval_3view_compare.sh
OUT_DIR=/home/velvet/output/zipsplat_results \
  CASE_FILTER=SplatAnything RESULT_NAME=SplatAnything_metrics.json \
  bash zipsplat/03b_eval_3view_compare.sh
# 注意：SA 为 pose 模式，ZipSplat / DA3-GS 无位姿；这不是完全等条件竞赛。
# 光度精化与最终评分均为黑底；不与旧 E13/E14 混报。

# ── 8) 三图三方法同轨迹单向 360° 环绕（不是往复摆动）──
# 复用正式三图评分链的 PLY、裁切、K、黑底及 source-only Sim(3) 精化。
# 先校验三方法的对齐数值，再渲染 144 帧；留出图不参与对齐。
OUT=~/output/zipsplat_results/jiajia_3v_orbit_360 \
  NFRAMES=144 FPS=24 CELLW=368 bash zipsplat/03c_render_3view_orbit.sh
# SMOKE=1 NFRAMES=3 OUT=~/output/zipsplat_results/jiajia_3v_orbit_smoke \
#   bash zipsplat/03c_render_3view_orbit.sh  # 仅首/中/末帧，不覆盖正式 MP4
# 输出：3 个单方法 MP4 + three_view_orbit_360.mp4、首/中/末 PNG、
# orbit_cameras.npz、protocol.json、render_metrics.json。

# ── 9) 归档到 D 盘（逐文件长度校验）──
RESULTS_DIR=~/output/zipsplat_results bash zipsplat/08_move_output.sh
```

## 本机实测速查（2026-11-20）

16 视角、378×504 网络输入、248,832 高斯、峰值 allocated 9,832MiB：前馈约 2.5s、
TokenBA 10.6s。TokenBA 输入自洽 23.334dB / SSIM 0.6453，8 留出 19.995dB /
SSIM 0.4817；严格前馈分别为 20.827 / 0.5088 与 19.757 / 0.4772。协议和限制见
[`EXPERIMENTS.md`](EXPERIMENTS.md)。

`01b_prepare_jiajia.sh` 的第一个参数也可直接写采集包路径，避免跨 Windows/WSL 传中文环境变量。

## 海思十图批处理与验收

- 每例为 704² PNG → 官方 504² 网络输入；207,360 高斯，26 属性，21,566,072 字节。
- 2026-11-26 完成 19/19：`case1..case17_nopose_recentered`、`model_man`、`model_plant`；
  全部使用 TokenBA 后高斯，官方 commit `73ab63f048fcc15ee54ada4ef9cdd79b925611a8`。
- 原始相机 JSON、其它算法 PLY 和输入图未作为 ZipSplat 条件，也未修改；不能将预测的
  首视角任意尺度坐标当作真实米制场景，也不能将输入自渲覆盖当作新视角质量结论。
- 最终报告：`D:/output/results/zipsplat_results/hisilicon_10v/final_validation.json`。

```bash
# 十图批处理（WSL shell 直接运行，勿经 Git Bash 传中文 env）
DATASET_ROOT=/mnt/d/dataset/测试数据/海思端侧重建拍摄数据 \
  ZIPSPLAT_RESULTS_DIR=/mnt/d/output/results/zipsplat_results/hisilicon_10v \
  WEIGHT_PATH=~/model/zipsplat/zipsplat-da3g-504.tar \
  MIN_FREE_MIB=14000 bash zipsplat/04_batch_hisilicon.sh

# 只跑一组验证；已有成品默认跳过，绝不静默覆盖。
# 显存不足时保留进度并停止后续批次。每组恰好十图，不读已有相机/PLY。
ONLY=model_man ZIPSPLAT_RESULTS_DIR=/mnt/d/output/results/zipsplat_results/hisilicon_10v \
  MIN_FREE_MIB=14000 bash zipsplat/04_batch_hisilicon.sh

# 独立验收：检查十图清单、PLY 点数据、必需属性、有限值、SHA-256 和 TokenBA。
python zipsplat/validate_hisilicon.py

# ── v2：先从静止段 MP4 抽十帧，再复用同一批处理（WSL shell 内执行）──
CASE=/mnt/d/dataset/测试数据/海思端侧重建拍摄数据v2/case1_nopose_recentered
~/miniconda3/envs/zipsplat/bin/python splatanything/36_extract_hisilicon.py \
  --case "$CASE" --out "$CASE"
# 逐例重复以上抽帧命令；要求恰好十段，sourceIndex 不是视频帧号。
DATASET_ROOT=/mnt/d/dataset/测试数据/海思端侧重建拍摄数据v2 \
  ZIPSPLAT_RESULTS_DIR=/mnt/d/output/results/zipsplat_results/hisilicon_v2_10v \
  MIN_FREE_MIB=14000 bash zipsplat/04_batch_hisilicon.sh
# 独立验收：显式传 v2 根与独立结果目录，不混用 v1 报告。
DATASET_ROOT=/mnt/d/dataset/测试数据/海思端侧重建拍摄数据v2 \
  RESULTS_DIR=/mnt/d/output/results/zipsplat_results/hisilicon_v2_10v \
  ~/miniconda3/envs/zipsplat/bin/python zipsplat/validate_hisilicon.py
# 2026-11-26：34/34 完成并通过，报告为该目录 final_validation.json。
```

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

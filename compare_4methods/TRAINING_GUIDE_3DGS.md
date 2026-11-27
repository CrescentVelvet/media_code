# DA3-GS、ZipSplat、InfiniSplat v2、SplatAnything 训练指南

> 依据本仓脚本、本机官方代码和已有实验记录整理。本文解释“现在可以跑什么”，不声称复现了未公开的训练器。
>
> 以下命令默认在 **WSL Ubuntu、仓库根目录 `/mnt/c/code/media_code`** 执行。示例数据路径 `my_scene` 是需要替换的占位名，不表示该数据已经存在。
>
> **只想快速定位：先看 [入口页](README.md)**（一页说明四份文档各管什么、你想做的事对应哪一节）。需要核对论文原始数字时查 [官方训练明细](OFFICIAL_TRAINING_DETAILS.md)；ZipSplat 重训的下载与后处理查 [数据准备手册](ZIPSPLAT_DATA_PREP.md)。

## 1. 先看结论：这里的“训练”不是同一件事

| 名称 | 官方跨场景网络训练 | 本仓实际可执行内容 | 会得到可泛化的新网络吗 |
|---|---|---|---|
| DA3-GS / `da3_human` | 本仓没有 DA3/GSDPT 跨场景训练器 | DA3 点云与相机初始化，再跑原版 3DGS | 否，只拟合当前场景 |
| ZipSplat / `zipsplat` | 官方仓有完整 `splatfactory` 训练代码 | 本仓脚本使用固定发布权重；另有 TokenBA、导出和评测 | 本仓推理不会；官方重训才会 |
| InfiniSplat v2 / `infinisplat_v2` | 公开包不含 Stage I/II 训练器和 learned refiner | 前馈导出 PLY，再做单场景属性精调 | 否，只更新当前 PLY 的部分属性 |
| SplatAnything / `splatanything` | 官方只发布推理包 | `09` 自研属性头训练探针；`40` 人物域短训协议 | `09` 不证明；`40` 尚待正式实验验证 |

因此不要把它们都写成“准备图片后训练神经网络”：

- **跨场景训练**：更新网络权重，目标是泛化到新场景；ZipSplat 官方仓属于这一类。
- **单场景 3DGS 优化**：拟合当前场景的高斯位置、外观等；DA3 初始化的原版 3DGS 属于这一类。
- **测试时/后处理优化**：例如 ZipSplat TokenBA、InfiniSplat v2 的 `14`，不产生跨场景网络。
- **自研训练探针**：例如 SplatAnything `09/40`，不是官方训练复现，结论受实验协议约束。

### 1.1 评测时最容易混淆的两个口径

| 口径 | 图像是否参与构建/优化 | 能说明什么 |
|---|---|---|
| source/train-view 自洽 | 是 | 输入重建、相机、导出和渲染链是否自洽；训练是否过拟合输入 |
| 独立 target / 严格 NVS | 否，从前馈或建云之前就留出 | 对未见视角的泛化能力 |

严格留出中，target 不能进入编码、点云构建、优化、Sim(3) 对齐或 checkpoint 选择。**在已经参与前馈的图片里再做 train/holdout 划分，不等于严格前馈 NVS。** 本文所有 PSNR 提升都按这两个口径分开，不跨场景、不跨协议比较。

## 2. DA3-GS：DA3 初始化 + 原版 3DGS

### 2.1 先区分两条 DA3 链路

1. **官方 GSDPT 前馈**：一次推理同时预测深度、相机和 Gaussian 属性，可导出 PLY。本仓 `02` 没有 optimizer/backward，不是训练。
2. **本仓 `04` 基线**：DA3 提供点云与相机，随后调用官方 `gaussian-splatting/train.py`。真正优化的是原版 3DGS；DA3 网络不更新。

`da3_human/06_train_da3_2dgs.py` 是另一套已终止的 anchor + 有界 offset 双分支 2DGS 外观路线，不作为入门主链。当前人物产品路线也不依赖逐场景 3DGS 精调；这里介绍它是为了复现基线。

### 2.2 输入和目录

原始输入只需同一场景的平铺多视角图片，但不要混用不同场景或未经确认的不同分辨率：

```text
/mnt/d/dataset/my_scene/images/
  0001.jpg
  0002.jpg
  ...
```

先运行 DA3 推理。`MAX_VIEWS=0` 表示全部视角；这里实际变量是 `MAX_VIEWS`，不是某些旧说明中的 `N_VIEWS`：

```bash
GPU=0 \
INPUT_DIR=/mnt/d/dataset/my_scene/images \
SCENE_NAME=my_scene \
DA3_MODEL_REPO=depth-anything/DA3NESTED-GIANT-LARGE-1.1 \
PROCESS_RES=504 MAX_VIEWS=32 \
EXPORT_FORMAT=mini_npz-colmap-depth_vis \
OUTPUT_DIR=~/output/da3_human_results/da3_inference/my_scene_32v \
bash da3_human/02_run_da3_inference.sh
```

`04` 再通过 `build_da3_colmap_scene.py` 将导出整理成原版 3DGS 可读的 COLMAP 场景：

```text
~/output/da3_human_results/colmap_scene/my_scene_32v/
  images/                 # 指回同源 RGB 的软链
  sparse/0/
    cameras.bin
    images.bin
    points3D.bin          # 初始化 XYZ/RGB，可体素降采样
```

注意 DA3 的内参与其处理分辨率对应；不能把另一套预处理下的 K、坐标或深度混进来。

### 2.3 训练循环：哪些量会更新

原版 3DGS 每个 iteration 随机选择一个训练相机，用当前高斯渲染该视角并与 GT 计算光度损失。默认可更新：

- 高斯中心、SH 系数、scale、opacity、rotation；
- 每 1000 步提升一级 SH；
- 500 步起按屏幕空间梯度 split/clone/prune，15000 步后停止 densify，默认间隔 100 步；
- 默认 opacity reset 间隔 3000 步。

**这不是冻结几何的属性精调**：位置和点数都会变化。默认总损失为：

```text
L = 0.8 * L1 + 0.2 * (1 - SSIM)
```

默认总步数为 30000；Adam 的参数学习率、SH/densify 等完整设置见官方 `train.py`、`scene/gaussian_model.py` 和 `arguments/__init__.py`。

### 2.4 可执行的 5K 短训命令

以下假设官方 3DGS 仓和 CUDA 扩展已按 `da3_human/README_wsl.md`、`vggt_human/00a_setup_env.sh` 准备好。它显式保存最终 PLY 与训练 checkpoint；`--checkpoint_iterations` 不传时，不能假定会生成 `chkpnt<N>.pth`。

```bash
GPU=0 \
GS_DIR=~/repos/gaussian-splatting GS_CONDA_ENV=vggt_human \
DA3_INFER_DIR=~/output/da3_human_results/da3_inference/my_scene_32v \
IMAGES_DIR=/mnt/d/dataset/my_scene/images \
SCENE_DIR=~/output/da3_human_results/colmap_scene/my_scene_32v \
GAUSSIAN_DIR=~/output/da3_human_results/baseline_3dgs/my_scene_32v_5k \
MAX_INIT_POINTS=1000000 ITERATIONS=5000 RES=1 \
TRAIN_EXTRA_ARGS="--save_iterations 1000 5000 --checkpoint_iterations 5000" \
bash da3_human/04_run_da3_gs_baseline.sh
```

`RES=1` 沿用原始分辨率；显存不足时再根据数据规格降低训练分辨率，不要只改一个 K。`MAX_INIT_POINTS` 超限时会体素降采样。已有输出目录应另开实验名，不覆盖旧结果。

### 2.5 输出和验收

```text
<GAUSSIAN_DIR>/
  point_cloud/iteration_5000/point_cloud.ply
  chkpnt5000.pth
  train/ours_5000/renders/       # 输入训练视角的渲染图
  train/ours_5000/gt/
  test/ours_5000 -> ../train/ours_5000
  results.json                   # 以实际 metrics.py 输出为准
  cameras.json / cfg_args / input.ply
```

脚本会渲染 **train views** 并计算 PSNR/SSIM/LPIPS。由于本仓定制 `metrics.py` 只扫描 `test/` 分割，脚本用软链让它读取 train 渲染。因此这里的指标明确是 **train-view in-sample**，不能当新视角泛化成绩。

建议查看顺序：训练终端 loss/点数日志 → `point_cloud.ply` → `train/ours_5000/renders` 与 GT 并排图 → `results.json`。若训练失败，先检查 `GS_CONDA_ENV` 中 `diff_gaussian_rasterization`、`simple_knn` 是否可导入。

### 2.6 本仓已有的资源参照

| 口径 | GPU 与配置 | 实测 | 训练峰值显存 |
|---|---|---|---|
| GSDPT **推理**，32 视角、504 | RTX 3090 | 前向 6.6 s，峰值 13.0 GB；约 610 万高斯，全 SH PLY 约 953 MB | 不适用，不是训练 |
| GSDPT **推理**，32 视角、672 | RTX 3090 | 前向 12.5 s，峰值 18.0 GB；约 1084 万高斯 | 不适用 |
| DA3-init 3DGS，125 视角、1440×1920、18,057 点初始化、5K 步 | RTX 3090 | 训练 461 s；densify 到 616,420 点；train-view PSNR 23.79 | **未知，无可靠统一记录** |
| 同类任务，459,433 点初始化、30K 步 | RTX 3090 | 约 3600 s；最终约 500 万高斯；train PSNR 25.505 / SSIM 0.8613 / LPIPS 0.1870 | **未知** |

后两项历史实验的 125 个视角全部用于训练，没有严格留出；耗时也未统一包含全部加载、导出和评测。不能据此承诺你的 32 视角、5K 短训会得到同样数字。

## 3. ZipSplat：唯一有完整官方跨场景训练器的一项

### 3.1 本仓推理与官方重训要分开

本仓 `zipsplat/02_run_inference.sh` 使用固定发布权重做前馈，可选 TokenBA；它不会更新发布网络。完整训练代码在本机官方 `~/repos/ZipSplat/splatfactory/`（核对时 HEAD 为 `73ab63f048fcc15ee54ada4ef9cdd79b925611a8`）。

官方配置不是“单卡跑几张人像微调”：252 论文配置使用 450K steps、global batch 384；论文资源为 **16×GH200，每卡 24 samples**。论文没有给出可用于换算单卡 ETA 的训练 wall-clock 和训练峰值显存；本仓 3090 的秒级记录只是推理/TokenBA。

### 3.2 网络、冻结范围和训练监督

主流程是 **prepare → compress → fuse → decode**：

1. DA3-Giant backbone 提取多视角 token；`ColorEmbedMLP` 把 RGB patch 转成颜色 token。
2. `TokenMerge` 形成 geometry token；可选 k-means 把多视角 token 压成 scene tokens。
3. cross-attention/self-attention 融合颜色与几何。
4. `GaussianHead` 把每个 scene token 解码为 G 个具有自由三维位置的高斯。
5. DA3 camera decoder 同时预测位姿与内参。

默认 DA3 backbone **不是完全冻结**：其学习率是主学习率的 0.1 倍，`cam_dec.fc` 保留完整学习率。optimizer 只收集 `requires_grad=True` 的参数。

每个训练样本包含 context/source 和独立 target：source 构建场景，target 提供相机与 RGB 新视角监督。默认 `loss_on_context=False`，即通常不渲染 context 计算训练损失，只监督 target。训练时另有 20% step 可注入 GT camera prior；这不意味着推理必须提供相机。

### 3.3 正式训练的数据契约：不能只给一个 RGB 文件夹

官方 `splatfactory` 从统一 tar shards 流式读取场景：

```text
~/repos/ZipSplat/data/<dataset>/<HxW>/
  train-scenes/           # 场景 tar shards
  test-scenes/
```

每个场景包含 images、cameras、poses、depth 等，而不是只有 RGB。各数据集的下载/转换命令见官方 `splatfactory/datasets/README.md`：

- 252 配方：RE10K/DL3DV 各 50%，每样本 2–24 个 context + 4 个 target。
- RE10K/DL3DV 原数据缺深度时，用 `extract_depth` 在 GT 相机条件下经 DA3 补齐。
- ScanNet++ 等还需要 covisibility rankings；按对应数据集脚本准备。
- 504 配方：十个数据集混合，七种长边 504 的宽高比，仍为 2–24 context + 4 target。

默认 rendered-depth L1 权重虽为 0，GeoCE 等仍需要深度与位姿。因此，“删掉 depth，只留图片”不是同一训练配方。

训练前还需下载并转换 DA3-Giant 初始化权重；以下命令在 **`~/repos/ZipSplat` 官方仓根目录**执行：

```bash
cd ~/repos/ZipSplat
huggingface-cli download depth-anything/DA3NESTED-GIANT-LARGE-1.1 \
  model.safetensors --local-dir weights/da3
python -m splatfactory.models.encoders.dav3_encoder --size giant --nested \
  --input weights/da3/model.safetensors \
  --output weights/da3/da3-giant-1.1.pth
```

官方训练依赖使用独立 `splatfactory` 环境（Python 3.12），不要默认本仓轻量推理 env 已具备训练依赖。

### 3.4 损失、优化器和步数

默认总损失可概括为：

```text
L = L1(target RGB) + 0.1 * LPIPS
  + scale_penalty + opacity_penalty + location_penalty
  + 0.05 * GeoCE + 0.01 * cone + 0.05 * pose
```

其中 `depth_weight=0`，所以默认不添加 rendered-depth L1；scale/opacity penalty 权重均为 1.0。`gaussian_head.py` 中变量名虽叫 `mse_loss`，配置 `use_l1_loss=True` 时实际是 L1。

优化设置：AdamW，主学习率 `3e-4`，weight decay `0.05`，betas `(0.9, 0.95)`，bf16，梯度裁剪 1.0；前 5% steps warmup，随后 cosine decay。一个 iteration 对应一个 optimizer step，梯度累积不重复推进训练步数。

**252 与 504 是两套配方，不能只改输入尺寸：**

| 阶段 | 输入与 G | 数据 | 步数 / 学习率 | 初始化 |
|---|---|---|---|---|
| 论文 252 模型 | 252×252，G=32 | RE10K + DL3DV | 450K / `3e-4` | DA3-Giant backbone |
| 504 发布模型第 1 阶段 | 252×252，**G=16** | 同 252 配方 | 450K / `3e-4` | DA3-Giant backbone |
| 504 发布模型第 2 阶段 | 长边 504、多宽高比，G=16 | 十数据集 | 100K / `1e-4` | `zipsplat-g16/checkpoint_best.tar` |

504 阶段启用 DA3 trunk activation checkpointing，SH dropout 为 0.5。**不能用 G=32 的 best 代替它要求的 G=16 初始化。**

### 3.5 官方训练、恢复与评测命令

以下展示官方配方，不是已在 3090 上验证的可承受配置。先准备齐全数据与权重，再从官方仓根目录执行：

```bash
cd ~/repos/ZipSplat

# 论文 252 模型；16 卡官方运行是 4 节点 × 4 GPU，这里仅展示单机入口
# 硬件较少时必须重新设计 data.batch_size，并实测显存，不能照抄 global batch 384。
torchrun --nproc_per_node=4 -m splatfactory.train \
  zipsplat --conf zipsplat train.writer=tensorboard

# 中断后恢复同一实验目录
torchrun --nproc_per_node=4 -m splatfactory.train \
  zipsplat --conf zipsplat train.writer=tensorboard --restore

# 评测训练后的 252 实验，target 是 held-out views
python -m splatfactory.eval.run dl3dv_benchmark \
  --checkpoint zipsplat eval.num_views=16
```

要复现 **504 发布模型**，需两阶段都执行：

```bash
cd ~/repos/ZipSplat
torchrun --nproc_per_node=4 -m splatfactory.train \
  zipsplat-g16 --conf zipsplat model.gaussians_per_token=16 \
  train.writer=tensorboard
torchrun --nproc_per_node=4 -m splatfactory.train \
  zipsplat-504 --conf zipsplat_ft504 train.writer=tensorboard
```

若改掉 `zipsplat-g16` 实验名，必须同步修改 `zipsplat_ft504.yaml` 的 `model.weights`。已存在训练目录默认报错，只有明确选择 `--restore` 等模式才复用；不要为了省事随手 `--clean` 或 `--overwrite`。

### 3.6 训练日志、checkpoint 和评测产物

输出根目录由官方 `settings.py` 固定：

```text
~/repos/ZipSplat/outputs/training/<experiment>/
  cli_config.yaml / raw_config.yaml / config.yaml
  log.log                         # rank 0 stdout/stderr
  checkpoint_best.tar             # 默认按 loss/total 选 best
  checkpoint_last.tar
  checkpoint_<epoch>_<iter>.tar    # 默认每 5000 iter 周期保存
  test_<epoch>/<benchmark>/       # benchmark 渲染/指标
  events.out.tfevents.*           # train.writer=tensorboard 时

~/repos/ZipSplat/outputs/results/ # 独立评测输出
```

checkpoint 包含 model、optimizer、lr_scheduler、配置、epoch 和 eval 信息；`best/last` 是轻量模型+配置版本。恢复训练用 `--restore`，不要只拿 PLY 当网络 checkpoint。日志会包含 loss、速度、ETA 和设备显存；查看 TensorBoard：

```bash
tensorboard --logdir ~/repos/ZipSplat/outputs/training/zipsplat
```

### 3.7 本仓前馈与 TokenBA：适合先跑通，不是重训

回到本仓根目录，输入只需平铺 JPG/JPEG/PNG/WebP/BMP；默认取 16 张。下面分别保存纯前馈和 TokenBA 两套结果，避免混用相机与高斯：

```bash
cd /mnt/c/code/media_code
GPU=0 ZIPSPLAT_REPO=~/repos/ZipSplat \
WEIGHT_PATH=~/model/zipsplat/zipsplat-da3g-504.tar \
IMAGES_DIR=/mnt/d/dataset/my_scene/images SCENE_NAME=my_scene \
MAX_VIEWS=16 BA=0 SAVE_FEEDFORWARD=1 RENDER_VIDEO=0 \
OUT_DIR=~/output/zipsplat_results/my_scene_ff \
bash zipsplat/02_run_inference.sh

GPU=0 ZIPSPLAT_REPO=~/repos/ZipSplat \
WEIGHT_PATH=~/model/zipsplat/zipsplat-da3g-504.tar \
IMAGES_DIR=/mnt/d/dataset/my_scene/images SCENE_NAME=my_scene \
MAX_VIEWS=16 BA=1 SAVE_FEEDFORWARD=1 RENDER_VIDEO=1 \
OUT_DIR=~/output/zipsplat_results/my_scene_ba \
bash zipsplat/02_run_inference.sh
```

本仓输出根由 `ZIPSPLAT_RESULTS_DIR` 或显式 `OUT_DIR` 决定。典型内容：

```text
<OUT_DIR>/
  gaussians_feedforward.ply       # SAVE_FEEDFORWARD=1；BA=0 时也是最终前馈 PLY
  gaussians.ply                   # 仅 BA=1：TokenBA 后
  predicted_cameras.npz / .json
  renders/
  metrics.json                    # 时间和 allocated/reserved 峰值
  orbit_wiggle.mp4                # RENDER_VIDEO=1
```

TokenBA 先用 token-patch 对应做 robust PnP 和 photometric polish，再通过**冻结的 GaussianHead** 联合优化 scene tokens 与相机。它不更新发布网络权重，也不产出跨场景训练 checkpoint。

RTX 3090、16 视角、378×504 网络输入、248,832 高斯的历史记录：模型加载约 23 s；纯前馈 2.51 s；前馈+TokenBA 为 2.46 s + 10.56 s；peak allocated 9,832.2 MiB。8 个严格留出视角上，FF PSNR 19.757、TokenBA 19.995，仅代表该协议约 +0.238 dB。**这些都不是训练耗时或训练显存。**

历史上把前馈高斯与 BA 后相机混拼的 `evaluation_feedforward/` 是误配产物，不得用于报数；FF 高斯与 FF 相机应通过独立 `BA=0` 运行成套获得。

## 4. InfiniSplat v2：已有 PLY 的单场景后处理精调

### 4.1 复现边界与输入

公开包没有 Stage I/II 训练器、learned refiner 和官方评测划分。本仓复现前馈推理，并自接 gsplat 渲染与 `14_finetune_gs.py`；这不是论文网络训练复现。

前馈输入为 JPG/JPEG/PNG/WebP 图片，可选同名 `cameras.npz`：

```text
c2w        [N,4,4]   OpenCV camera-to-world
intrinsics [N,3,3]   原始输入图像的像素内参，与图片顺序一一对应
```

无相机时默认走 DA3 pose。默认 `LONG_SIDE=518`、`MAX_VIEWS=32`。输出 PLY 和相机位于 **RDNorm 空间**，不是外部世界米制坐标；渲染外部相机时必须按本仓评测脚本做 Sim(3) 对齐，不能混用 c2w/w2c 或 OpenCV/OpenGL。

### 4.2 前馈 + 3000 步精调命令

下面路径是需替换的示例。`FINETUNE=1` 会在前馈后自动调用 `02b`：

```bash
GPU=0 WEIGHTS_DIR=~/model/infinisplat_v2 \
IMAGES_DIR=/mnt/d/dataset/my_scene/images \
CAMERAS=/mnt/d/dataset/my_scene/cameras.npz \
OUT_NAME=my_scene LONG_SIDE=518 MAX_VIEWS=32 \
FINETUNE=1 FINETUNE_ITERS=3000 \
RESULTS_DIR=~/output/infinisplat_v2_results \
bash infinisplat_v2/02_run_inference.sh
```

若没有相机，去掉 `CAMERAS=...` 并显式设置 `POSE_MODE=da3-pose`。前馈原始输出为 `<RESULT>/{scene.ply,cameras.npz,result.json}`；精调结果写入 `<RESULT>/ft/`，不覆盖 raw PLY。

也可在已有前馈结果上独立运行：

```bash
GPU=0 RESULT=~/output/infinisplat_v2_results/my_scene \
IMAGES_DIR=/mnt/d/dataset/my_scene/images \
FINETUNE_ITERS=3000 FINETUNE_EXTRA_ARGS="" \
bash infinisplat_v2/02b_finetune.sh
```

已有 `ft/finetune.json` 时默认跳过。不要为了省事随意 `FINETUNE_FORCE=1` 覆盖已有实验；消融应使用不同 `FT_SUBDIR`。

### 4.3 精调具体更新什么

高斯中心来自 DA3 深度反投影，网络 Gaussian head 预测 rotation、scale、opacity、SH0 等属性。本仓后处理进一步冻结 **means、全部相机和 DA3/InfiniSplat 网络**，不 densify、不 prune，只开放：

| 参数组 | 默认学习率 |
|---|---|
| SH0 | `2.5e-3` |
| log-scale | `5e-3` |
| opacity-logit | `5e-2` |
| quat | `1e-3` |

每步随机一个训练视角，Adam（`eps=1e-15`），末期学习率衰减至初始的 0.05 倍，并对 scale/opacity/quat 做范围约束。默认 3000 步，损失为：

```text
L = 0.8 * L1 + 0.2 * (1 - SSIM)
```

图像优先按 `result.json` 中输入文件的 SHA256 配对，命中率不足才退回文件名排序。要检查日志的配对方式，错误 GT 即使让 loss 下降也没有意义。`--holdout-every K` 只会在这批**已经参与前馈**的图像中划分精调 train/hold，不是严格前馈 NVS。

### 4.4 输出、评测与资源记录

```text
<RESULT>/ft/
  scene.ply
  cameras.npz / result.json
  finetune.json        # init/final/history、耗时、allocated 峰值、PLY 回读检查
```

先读 `finetune.json` 的 train/hold 曲线和 `ply_roundtrip.delta`，再渲染检查。`03` 默认在存在 `ft/scene.ply` 时优先评精调；下面 `EVAL_FT=0` 是为了显式评 raw 自洽：

```bash
GPU=0 RESULT=~/output/infinisplat_v2_results/my_scene \
IMAGES_DIR=/mnt/d/dataset/my_scene/images EVAL_FT=0 MODE=rerender \
bash infinisplat_v2/03_eval_render.sh
```

**原始前馈的严格留出**必须在前馈之前划分。下面示例要求原始包至少 32 张图及对应 GT 相机，使用 25 input + 7 target：

```bash
GPU=0 IMAGES_DIR=/mnt/d/dataset/my_scene/images \
CAMERAS=/mnt/d/dataset/my_scene/cameras.npz \
OUT_NAME=my_scene_raw_holdout MODE=holdout N_INPUT=25 N_TARGET=7 \
SPLIT_DIR=~/output/infinisplat_v2_results/_splits/my_scene_25_7 \
RESULTS_DIR=~/output/infinisplat_v2_results LONG_SIDE=518 \
FINETUNE=0 EVAL_FT=0 \
bash infinisplat_v2/03_eval_render.sh
```

要测精调后的严格留出，应使用另一 `OUT_NAME`，让内部前馈 `FINETUNE=1`，再评生成的 `ft/`；不能只把上述 raw 命令的 `EVAL_FT` 改成 1：

```bash
GPU=0 IMAGES_DIR=/mnt/d/dataset/my_scene/images \
CAMERAS=/mnt/d/dataset/my_scene/cameras.npz \
OUT_NAME=my_scene_ft_holdout MODE=holdout N_INPUT=25 N_TARGET=7 \
SPLIT_DIR=~/output/infinisplat_v2_results/_splits/my_scene_25_7 \
RESULTS_DIR=~/output/infinisplat_v2_results LONG_SIDE=518 \
FINETUNE=1 FINETUNE_ITERS=3000 EVAL_FT=1 \
bash infinisplat_v2/03_eval_render.sh
```

输出位于被评测结果目录的 `eval/`：

```text
metrics_holdout.json / metrics_rerender.json
compare_holdout.jpg  / compare_rerender.jpg
逐视角 render / GT 图片
```

`17_run_orbit.sh` 可从 `scene.ply` 生成轨道 MP4 供目检，但轨道视频不是独立 NVS 定量证明。

| 口径 | RTX 3090 实测 |
|---|---|
| 前馈，长边 518、佳佳抱枕 32 视角 | 约 35.5 s，峰值约 18.3 GB；raw 约 650 万高斯 |
| 同一前馈结果后处理 3000 步 | 约 161 s，allocated 峰值约 6.3 GB |
| 严格 25 输入 / 7 留出 | train PSNR 20.67 → 28.78；holdout PSNR 19.05 → 21.15，仅该场景/协议 +2.11 dB |

论文背景而非本仓复现：Stage I 冻结几何分支，训练属性 backbone 最后 10 个 Transformer blocks 与 Gaussian head；Stage II 冻结 reconstructor，训练 DPT feature projection 和 learned refiner。两阶段均 16×H20、global batch 16，Stage I 25K steps 约 48 h，Stage II 5K 约 13 h。总约 61 h 是 **16 卡 wall-clock**，不是单卡耗时；训练峰值显存未知。

## 5. SplatAnything：先跑训练探针，再决定是否进入人物短训

### 5.1 输入和可训练范围

官方发布的是推理包，无官方训练器、refiner、光栅化器或论文评测划分。本仓 `03/06/08` 自接 gsplat，`09/40` 是自研实验。

自有场景输入：

```text
/mnt/d/dataset/my_scene/
  images/               # 至少按脚本要求准备 3 张
  prior_camera.json     # POSE=1 必需；camera_id 与图片文件 stem 一致
```

c2w 使用 OpenCV 约定，K 必须对应原始输入图像像素。输出 PLY 和导出相机位于 **GSN 坐标系**，不是米制。

外部 `GeometryWorldMirror` 教师提供深度、相机和几何。本仓训练冻结教师、深度、相机、means 和 source encoder，只开放：

```text
model.gs_head
model.gs_renderer.decoder.gs_head
```

因此主要检验属性与融合适配；它不能直接承诺修复教师几何引起的双源错位。官方训练耗时与训练显存均未知。

### 5.2 `09`：source 自重建训练探针

`09_train_probe.py` 补齐可导前向、可微 merge、gsplat 渲染、loss、optimizer 和 checkpoint。可训练参数 32,934,540 个、63 个张量；合并处用 out-of-place `index_add` 代替官方 in-place 操作。

其 target 直接来自 `views["img"]`，与 source 相同；主损失是 RGB MSE，优化器是 Adam。它只能证明反传、输入过拟合、保存/恢复链可用，**不证明独立 NVS、跨身份泛化或人物质量提升**，也不自动导出正式交付 PLY。

先做只前向检查，不做参数更新：

```bash
GPU=0 SCENE=/mnt/d/dataset/my_scene/images RES=504 POSE=0 \
SPLATANYTHING_RESULTS_DIR=~/output/splatanything_results/my_scene_probe \
bash splatanything/09_run_train_probe.sh --stages fwd \
  --out ~/output/splatanything_results/my_scene_probe/fwd.json
```

通过后再做 100-step 工程探针：

```bash
GPU=0 SCENE=/mnt/d/dataset/my_scene/images RES=504 POSE=0 \
SPLATANYTHING_RESULTS_DIR=~/output/splatanything_results/my_scene_probe \
bash splatanything/09_run_train_probe.sh \
  --stages all --n-steps 100 --lr 1e-5 --log-every 10 \
  --out ~/output/splatanything_results/my_scene_probe/train_probe.json \
  --ckpt ~/output/splatanything_results/my_scene_probe/train_probe_ckpt.pt
```

⚠️ shell 注释声称默认 lr=`1e-5`，但 Python 真实默认是 `1e-3`，且 shell 没有自动传参。因此命令中必须显式写 `--lr 1e-5`，不要依赖注释。

输出为 `train_probe.json` 和 `train_probe_ckpt.pt`。checkpoint 保存可训练参数、Adam 状态和 RNG，并验证恢复后再走一步。查看 JSON 中的：

- `fwd_equiv`：可导前向与原前向的字段/RGB 一致性；
- `step1`：分支梯度、实际参数 delta、冻结教师 checksum；
- `steps`：loss 曲线、step 时间、allocated/reserved；
- `resume`：checkpoint 大小和恢复后一步是否有限。

历史 RTX 3090、3 视角、504²、RGB、100 步、显式 lr=`1e-5`：总耗时 51.6 s，稳态约 0.511 s/step，peak allocated 约 12.62 GB、reserved 约 14.93 GB，checkpoint 377 MB；MSE 0.005975 → 0.002553，source 自洽 PSNR 22.24 → 25.93。**最后两项只是 source 过拟合证据。**

### 5.3 `40`：独立 target 人物短训协议（当前不能直接开跑）

这才是本仓自研的独立训练 target 协议，不是官方训练，也不是 LoRA：

- 每步三个已知相机 source + 两个独立训练 target；每步一个 sequence，最多 2000 optimizer steps。
- held-out eval 不进入模型、损失、缓存、Sim(3) 或 checkpoint 选择。
- 冻结教师和 source encoder，只训学生属性头；不修改原 SA 表示。
- 默认 AdamW，lr=`1e-4`，weight decay=`1e-6`，betas `(0.9,0.95)`，梯度裁剪 1.0。

```text
total = rgb_full + 2.0 * rgb_person
      + 0.1 * alpha_loss + depth_weight * depth_loss
# 默认 depth_weight = 0.1
```

人物 RGB 与 alpha 监督是为了避免“把人物抹掉”被误报为重影减少。正式训练强制逐视角人工 `--review`，不能用开关绕过。

**截至已核对的 2026-11-27 状态，UE53 尚未启动真实人物 100-step 训练。** 当前 27 个身份、53 个 sequence 的人工 review 仍 pending；合成/真实零更新门通过不等于训练或质量通过。因此这里不给一条可直接绕过审核的训练命令，正式操作应严格按 [`splatanything/README_H100.md`](../splatanything/README_H100.md) 的顺序：

1. `44_test_portrait.py`：离线契约测试；
2. `43_smoke_portrait.py`：合成 GPU 零更新冒烟；
3. `40_train_portrait.py --gate-only`：真实数据反传，但不更新参数；
4. 逐视角人工审核，且 H100 门通过后才跑 100-step；
5. 读 `run.json` 的 `measured_mean_step_s`，再决定是否续到 2000 步。

`ada_env17` 504² 零更新门在本机峰值分配约 12.318 GiB；4.852 s 是 source cache 后的单次 gate 区间，**不是训练单步 ETA**，不能乘以 2000 作为训练预报。

正式训练产物：

```text
<TRAIN>/
  B0.pt / step_0100.pt / ...
  run.json / metrics.jsonl / metrics.csv
```

独立 val/test 身份用 `42_eval_portrait.py`：先用 val 选 checkpoint，再对 test 评一次。输出 `per_view_metrics.csv`、`checkpoint_eval.json`、`summary.json`、`*_B0_vs_checkpoints.png`，以及 RGB/alpha/ED 诊断图。现有两测试身份只支持探索性信号，不能外推为跨身份泛化定论。

## 6. 汇总：第一次应该跑哪条

| 目的 | 推荐入口 | 理由 |
|---|---|---|
| 学会“单场景高斯训练”的完整流程 | DA3 `02` → `04`，先 5K | 有标准 PLY、densify、渲染与 train-view 指标 |
| 使用 ZipSplat 发布模型处理自有图片 | `zipsplat/02`，分别跑 BA=0/1 | 几分钟内可检查 PLY、相机、渲染和资源 |
| 真正重训 ZipSplat 网络 | 官方 `splatfactory` | 数据、算力和多阶段配方成本最高，先不要做未经设计的小数据全参微调 |
| 在已有 InfiniSplat PLY 上改善光度 | `02b` / `14` | 可快速对照 raw/ft；但要另做严格 NVS |
| 验证 SplatAnything 属性能否反传 | `09 --stages fwd`，再 100-step | 先验工程链，不把自重建误当 NVS |
| 做 SplatAnything 人物域适配 | `44 → 43 → gate-only → review → 100-step` | 当前仍有未通过的实验门，不能直接宣称改善 |

## 7. 通用验收清单与常见坑

1. **先确认训练类型**：终端出现 loss/optimizer step 不代表在训练跨场景网络；检查实际 `requires_grad=True` 的参数。
2. **记录完整口径**：GPU、输入分辨率、视角数、步数、batch、是否含加载/导出/JIT。区分 `torch.cuda.max_memory_allocated`、reserved 与 `nvidia-smi`；推理显存不能写成训练显存。
3. **先划分 target，再前馈/重建**：留出集不得参与输入编码、优化、对齐或模型选择。
4. **检查图像—相机一一对应**：K 必须匹配原始或明确指定的像素尺寸；OpenCV/OpenGL、c2w/w2c、坐标归一化不能混用。
5. **PLY 不是网络 checkpoint**：PLY 通常是当前高斯表示；恢复网络训练需要模型及优化器状态。
6. **保存图和数值一起验收**：PSNR 上升仍可能伴随人物消失、低 alpha 晕圈、几何重影或错误坐标；至少并排看 RGB、GT、alpha/覆盖。
7. **不要覆盖旧实验**：换实验名/输出目录；恢复训练必须使用对应实现支持的 checkpoint 和恢复参数。
8. **多卡配置不换算成单卡承诺**：论文的 16 卡 wall-clock 只能用于说明原实验规模，未知项就保持“未知”。

## 8. 证据索引：继续读哪份代码

| 主题 | 主要文件 |
|---|---|
| DA3 推理与 3DGS 基线 | `da3_human/02_run_da3_inference.sh`、`04_run_da3_gs_baseline.sh`、`_train_3dgs.sh`、`build_da3_colmap_scene.py` |
| DA3 历史资源与路线 | `da3_human/EXPERIMENTS.md`、`EXPERIMENT_PLAN.md`；官方 `~/repos/gaussian-splatting/train.py` |
| ZipSplat 本机推理/TokenBA | `zipsplat/02_run_inference.sh`、`run_inference.py`、`README_wsl.md`、`EXPERIMENTS.md` |
| ZipSplat 正式训练 | 官方 `~/repos/ZipSplat/splatfactory/configs/{zipsplat,zipsplat_ft504}.yaml`、`models/networks/zipsplat.py`、`models/decoders/gaussian_head.py`、`trainer.py` |
| ZipSplat 数据与输出 | 官方 `splatfactory/datasets/README.md`、`settings.py`、`utils/experiments.py`、`train.py` |
| InfiniSplat v2 前馈/精调/评测 | `infinisplat_v2/02_run_inference.sh`、`02b_finetune.sh`、`14_finetune_gs.py`、`03_eval_render.sh`、`README_wsl.md` |
| SplatAnything 探针与人物协议 | `splatanything/09_train_probe.py`、`09_run_train_probe.sh`、`39_portrait_common.py`、`40_train_portrait.py`、`42_eval_portrait.py`、`README_H100.md` |

本文是跨项目入门与口径说明，具体依赖安装和最新实验状态仍以各目录 README/EXPERIMENTS 为准。

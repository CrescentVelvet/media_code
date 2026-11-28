# compare_4methods 怎么读：四种方法训练入口

> 目录里现在有 4 份文档，职责不同。**先看本页第 1 节选路径，再去对应小节**，不要通读。
>
> 核对日期：2026-11-27。

## 1. 我该读哪一份

| 你要做的事 | 读哪份 | 直接跳 |
|---|---|---|
| 用发布权重跑推理、看 PLY / 渲染效果 | 各算法自己的 `README_wsl.md` | `da3_human/`、`zipsplat/`、`infinisplat_v2/`、`splatanything/` |
| **单场景训练**：DA3 出点云后跑原版 3DGS | [训练指南](TRAINING_GUIDE_3DGS.md) | §2 |
| **重训 ZipSplat 网络**（唯一有官方跨场景训练器） | 训练指南 + [自训手册](ZIPSPLAT_DATA_PREP.md)（要多少数据 / 数据形态 / shard 结构 / 代码坑 / 参数表） | §3 / 全文 |
| 在已有 PLY 上做**属性精调**（InfiniSplat `14`、SplatAnything `09`） | 训练指南 | §4 / §5 |
| 核对官方**超参、数据集规模、显卡与耗时** | [官方训练明细](OFFICIAL_TRAINING_DETAILS.md) | 表 1–11 |
| 找**下载链接、体积、许可** | [数据集清单](TRAINING_DATASETS.md) | §2–§6 |

四份文档的定位，一句话版：

- **训练指南**（570 行）= 每个算法“训练代码怎么走、命令怎么敲、产物在哪、坑在哪”。唯一主文档。
- **官方训练明细**（17 张表）= 论文/官方代码的原始数字，用来核对，不教操作。
- **数据集清单** = 全量数据目录、体积、许可，用来决定“要不要下”。
- **ZipSplat 自训手册** = 只服务 ZipSplat 重训：前半“自己造数据去训”（数据量 / 数据形态 / shard 结构 / 代码坑 / 参数表），后半“下载官方数据集复现”。

## 2. 四种“训练”到底在训什么

| 算法 | 谁在更新 | 代码在哪 | 可训练参数 | 损失 | 能泛化到新场景吗 |
|---|---|---|---|---|---|
| **DA3-GS** | 原版 3DGS 的全部高斯属性 | 本仓 `04` → 官方 `gaussian-splatting/train.py` | 位置、SH、scale、opacity、rotation（点数还会 split/clone/prune） | `0.8·L1 + 0.2·(1−SSIM)` | **否**，只拟合当前场景 |
| **ZipSplat** | 跨场景网络（backbone + 融合 + GaussianHead） | 官方 `splatfactory/train.py` → `trainer.py` | 网络全参；DA3 backbone 用 0.1× 学习率（**非冻结**） | `L1(target RGB) + 0.1·LPIPS + 正则项 + GeoCE/cone/pose` | **是**，这是唯一真正的跨场景训练 |
| **InfiniSplat v2** | 已有 PLY 的 4 组属性 | 本仓 `14_finetune_gs.py` | SH0 / log-scale / opacity-logit / quat（冻结 means、相机、网络） | `0.8·L1 + 0.2·(1−SSIM)` | 否，后处理 |
| **SplatAnything** | `09`：属性头探针；`40`：学生属性头 | 本仓 `09_train_probe.py`、`40_train_portrait.py` | `09` 32,934,540 参数 / 63 张量；`40` 只训学生头，冻结教师与 encoder | `09` RGB MSE；`40` 人物加权 RGB + alpha + depth | 否；`09` 只证明反传，`40` 尚未正式开跑 |

记住一句：**只有 ZipSplat 的“训练”会改网络权重并泛化到新场景**，其余三种都是“针对当前场景/当前 PLY 的优化”。

## 3. 代码走读（每个五行）

### 3.1 DA3-GS：DA3 出初始化，3DGS 负责训

```text
02_run_da3_inference.sh      DA3 前馈：预测深度、相机、Gaussian → 导出 npz/PLY
  └ build_da3_colmap_scene.py  整理成 COLMAP：images/ + sparse/0/{cameras,images,points3D}.bin
04_run_da3_gs_baseline.sh    调官方 train.py（DA3 网络此时不再参与）
  └ 官方 train.py            每步随机 1 个训练相机 → 渲染 → L1+SSIM → 反向
```

关键点：初始化点数由 `MAX_INIT_POINTS` 控制（超限体素降采样）；`RES=1` 沿用原分辨率；5K 步在 3090、125 视角 1440×1920 上实测 **461 s**，densify 到 61.6 万点。30K 步的历史实验（459k 点初始化）约 3600 s，只是量级参考，不是承诺。

### 3.2 ZipSplat：prepare → compress → fuse → decode

```text
DA3-Giant backbone    多视角 token（+ ColorEmbedMLP 把 RGB patch 变颜色 token）
  └ TokenMerge        几何 token，可选 k-means 压成 scene tokens
  └ attention 融合    颜色 × 几何
  └ GaussianHead      每个 scene token 解码 G 个自由位置高斯（252 论文 G=32，发布链路 G=16）
  └ cam decoder       同时预测位姿与内参
```

一次样本 = N 个 context（建表示）+ 4 个 target（算损失），`loss_on_context=False`，即不监督 context 自身。默认 `depth_weight=0`，但有 GeoCE 等几何项仍需要 depth 与位姿。

### 3.3 InfiniSplat v2：前馈出 PLY，`14` 只动四组属性

```text
02_run_inference.sh   前馈：RGB(+可选 cameras.npz) → scene.ply（RDNorm 空间）
02b_finetune.sh       → 14_finetune_gs.py
                       冻结 means / 全部相机 / 网络；只训 SH0、log-scale、opacity-logit、quat
                       每步 1 个训练视角，Adam(eps=1e-15)，末期 lr 衰减到 0.05×
```

图像靠 `result.json` 的 SHA256 配对，配对错了 loss 降也没意义；3000 步在 3090、32 视角 518 上约 **161 s**，峰值分配 6.3 GB。

### 3.4 SplatAnything：先证反传，再谈人物

```text
09_train_probe.py     补齐可导前向 + 可微 merge + gsplat 渲染 + loss + Adam
                      target 就是 source 自己 → 只能证明“能训、能存、能恢复”
40_train_portrait.py  3 source + 2 独立 target，冻结教师/source encoder，只训学生属性头
                      需先过 44 契约测试 → 43 合成冒烟 → --gate-only → 人工 review
```

`09` 在 3090、3 视角 504²、100 步：51.6 s，0.511 s/step，peak allocated 12.62 GB，MSE 0.005975 → 0.002553。**这是 source 过拟合，不是 NVS 提升。**

## 4. 自己训练要准备什么

| 算法 | 数据 | 权重 / 依赖 | 显存参考 | 时间参考 |
|---|---|---|---|---|
| **DA3-GS** | 一个场景的平铺图片（同分辨率、同场景） | DA3 权重 + 官方 3DGS 的 CUDA 扩展（`diff_gaussian_rasterization`、`simple_knn`） | 训练峰值未统一记录；初始化 PLY 与分辨率是主因 | 5K ≈ 461 s（125 视角 1440×1920，3090） |
| **ZipSplat** | **tar shards**，不是图片文件夹（images/cameras/poses/depth） | DA3-Giant 转好的 `.pth` + `splatfactory` env（Py3.12、gsplat） | 未公开；单卡必须重设 `data.batch_size` | 官方 16×GH200 未报 wall-clock；单卡无参考 |
| **InfiniSplat v2** | 图片 + 可选 `cameras.npz` | 官方权重 + gsplat | 精调 peak allocated 6.3 GB | 3000 步 ≈ 161 s |
| **SplatAnything** | 图片（≥3）+ 可选 `prior_camera.json`（OpenCV c2w） | 官方权重 + gsplat | `09` peak allocated 12.62 GB | `09` 100 步 ≈ 51.6 s |

**要买/渲染数据做人像 LoRA**：组数与身份多样性的采购判断见 [`splatanything/EXPERIMENTS.md`](../splatanything/EXPERIMENTS.md) 的 2026-11-28 条（400 组甜点区、100 组先验方向）。

## 5. 四条最短命令

```bash
cd /mnt/c/code/media_code

# 1) DA3 初始化 + 3DGS 5K 短训（完整参数见指南 §2.4）
GPU=0 GS_DIR=~/repos/gaussian-splatting GS_CONDA_ENV=vggt_human \
DA3_INFER_DIR=~/output/da3_human_results/da3_inference/my_scene_32v \
IMAGES_DIR=/mnt/d/dataset/my_scene/images \
SCENE_DIR=~/output/da3_human_results/colmap_scene/my_scene_32v \
GAUSSIAN_DIR=~/output/da3_human_results/baseline_3dgs/my_scene_32v_5k \
MAX_INIT_POINTS=1000000 ITERATIONS=5000 RES=1 \
TRAIN_EXTRA_ARGS="--save_iterations 1000 5000 --checkpoint_iterations 5000" \
bash da3_human/04_run_da3_gs_baseline.sh

# 2) ZipSplat 前馈 + TokenBA 推理（不是训练；完整参数见指南 §3.7）
GPU=0 ZIPSPLAT_REPO=~/repos/ZipSplat WEIGHT_PATH=~/model/zipsplat/zipsplat-da3g-504.tar \
IMAGES_DIR=/mnt/d/dataset/my_scene/images SCENE_NAME=my_scene \
MAX_VIEWS=16 BA=1 SAVE_FEEDFORWARD=1 RENDER_VIDEO=1 \
OUT_DIR=~/output/zipsplat_results/my_scene_ba bash zipsplat/02_run_inference.sh

# 3) InfiniSplat 前馈 + 3000 步属性精调（完整参数见指南 §4.2）
GPU=0 WEIGHTS_DIR=~/model/infinisplat_v2 IMAGES_DIR=/mnt/d/dataset/my_scene/images \
CAMERAS=/mnt/d/dataset/my_scene/cameras.npz OUT_NAME=my_scene LONG_SIDE=518 MAX_VIEWS=32 \
FINETUNE=1 FINETUNE_ITERS=3000 RESULTS_DIR=~/output/infinisplat_v2_results \
bash infinisplat_v2/02_run_inference.sh

# 4) SplatAnything 训练探针（先 fwd，再 100 步；注意 --lr 必须显式给）
GPU=0 SCENE=/mnt/d/dataset/my_scene/images RES=504 POSE=0 \
SPLATANYTHING_RESULTS_DIR=~/output/splatanything_results/my_scene_probe \
bash splatanything/09_run_train_probe.sh --stages fwd \
  --out ~/output/splatanything_results/my_scene_probe/fwd.json
```

ZipSplat 的**真正训练**不在本仓，在 `~/repos/ZipSplat`：先按[自训手册](ZIPSPLAT_DATA_PREP.md)造 shards，再 `torchrun -m splatfactory.train`，命令见指南 §3.5。

## 6. 三个最容易搞混的地方

1. **“跑出 loss”不等于“在训网络”**。查 `requires_grad=True` 的参数，别只看终端有没有 loss。
2. **推理显存不是训练显存**。ZipSplat 前馈 9.8 GB、InfiniSplat 前馈 18.3 GB 都是推理；训练峰值大多未公开。
3. **PLY 不是 checkpoint**。`point_cloud.ply` / `scene.ply` 是当前表示；恢复网络训练要模型 + 优化器状态（ZipSplat 的 `.tar`）。

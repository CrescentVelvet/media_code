# InfiniSplat v2 — 前馈多视角 3DGS 重建（WSL 复现）

从多张 RGB 图（可选位姿）前馈重建 3D Gaussian 场景。官方实现把几何基座
DA3（Depth-Anything-3）**冻结**、位置全程钉死在深度反投影上，只学 Gaussian 属性
（rotation/scale/opacity/SH0），再走体素合并 + opacity<0.01 剪枝。

- 论文：<https://arxiv.org/abs/2610.20341>（浙大 ZJU3DV）
- 官方仓：<https://github.com/zju3dv/InfiniSplat>（v2 在 `InfiniSplat_v2/`）
- 权重：`Edisoneh/InfiniSplat-v2`（HF，公开免登录，6.6 GB，含 DA3 backbone）

## ⚠️ 复现边界（先读这段，别拿本仓数字去和论文表 1 比）

官方**只开源推理包**，训练代码、refiner、官方评测划分都没有。本仓复现的是推理链路，
不是论文训练流程。

| 项 | 状态 | 说明 |
| --- | --- | --- |
| 前馈推理（表 1 "Ours" 行） | ✅ 可复现 | 代码+权重齐全，`02_run_inference.sh` |
| 三种 pose 模式 | ✅ 可复现 | provided / colmap / da3-pose（+ pose-free） |
| 高斯的 opacity 剪枝 / 体素合并 | ✅ 可复现 | 契约硬校验：voxel=median_scale×1.4、τ_α=0.01 |
| 三步 learned refinement（"Ours*" 行） | ❌ 不可得 | 发布包无 refiner 代码/权重，`import` 里都没有 |
| 训练（Stage I/II） | ❌ 不可复现 | 无训练代码；数据含私有 Raw3D；16×H20 约 61h |
| 论文表 1 的精确数值 | ⚠️ 不可直接对标 | 官方划分未公开；本仓用自建 holdout（等间隔抽目标视角）给同量级参考 |
| 渲染/评测 | ⚠️ 需自接 | 发布包刻意不带 gsplat，`03_eval_render.sh` 自己接 |

结论：**能复现"一个前馈模型吐出可渲染 3DGS 场景"这条能力链，不能复现论文的
25.8%/32.9% 提升数字**——那个数字来自训练侧三项手术（冻结几何 + GT 位姿对齐 +
大角度监督），训练没开源，无法复算。

## 实测结果（本机 3090，长边 518）

| 场景 | 输入 | 口径 | PSNR↑ | SSIM↑ |
| --- | --- | --- | --- | --- |
| hypersim_provided | 50 视角（GT 位姿） | 自洽 re-render | **27.05** | **0.918** |
| hypersim_da3_pose | 32 视角（DA3 预测位姿） | 自洽 re-render | 23.29 | 0.829 |
| dtu_provided | 16 视角（GT 位姿） | 自洽 re-render | 20.30 | 0.726 |
| dtu_provided | 16 视角 | **NVS holdout（4 目标）** | 13.10 | 0.585 |

推理耗时：16 视角 22 s、32 视角 28 s、50 视角 31 s（含 6.6 GB 权重 SHA256 校验），
产出 2.7 M / 5.3 M / 6.1 M 高斯，峰值显存约 11 GB。

### 三视角自有数据集实测（`D:\dataset\物体三图数据`，8 个物体）

| 项 | 实测 |
| --- | --- |
| 端到端（冷启动，含加载） | 20.9–25.2 s，均值 **21.8 s** |
| 常驻复用单次（warm） | **0.81 s** |
| 显存峰值（`nvidia-smi`） | 9.72–9.87 GB（基线 1.55 GB，净增 **8.2 GB**） |
| `torch` allocated 峰值 | 7.09 GB |
| 高斯数 / PLY | 22–31 万 / 15–21 MB |
| 自洽 PSNR（3 图全输入） | 15.9–24.5，均值 21.5 |
| **真实 NVS（2 入 → 1 出）** | **11.3–16.1，均值 13.1** |

⚠️ 三视角有两个硬坑，动手前必读：
1. **DA3 位姿会静默塌缩**。榨汁机（透明）/ 电饭煲（浅色圆柱）被估成三张同一视角
   （两两夹角 0.2–0.9°），推理照样 exit 0 输出 PLY，但几何是废的。跑完先执行
   `python diag_pose.py` 看夹角/基线，或用文件名帧号交叉验证。`pose-free` 不能救，
   COLMAP 在这批数据上也建不起来（SIFT 太少，无初始像对）。
2. **自洽 ≠ NVS，差 8.4 dB**。留出视角里物体位置对但未见区域是空洞（alpha 0.47–0.81）。

细节与逐物体数据见 `EXPERIMENTS.md` 的 E-07~E-11。

⚠️ **自洽 ≠ NVS**：自洽时每个高斯中心就是从该视角反投影来的，只证明"属性学对了 + 渲染器
接对了"；能对标论文的只有 holdout 那一行。且官方划分未公开、分辨率与目标视角选取都不同，
**本表与论文表 1 不可直接比较**。推导与逐视角数据见 `EXPERIMENTS.md`。

## 常用命令

> 以下都在 WSL 里、`cd` 到本目录（`/mnt/c/code/media_code/infinisplat_v2`）执行。

```bash
# 一键（建环境 → 下权重 → 跑 dtu_provided → holdout 评测）
bash run_all.sh

# 分步
# 0) 建 conda env + torch + infinisplat + DA3 + gsplat
CONDA_ENV=infinisplat_v2 PY_VER=3.11 TORCH_VER=2.7.1 bash 00_setup_env.sh

# 1) 下权重（6.6 GB，走 aifasthub 镜像，8 路分块）+ 官方示例（90 MB）
HF_BASE_URL=https://aifasthub.com DOWNLOAD_JOBS=8 bash 01_download_models.sh
SKIP_EXAMPLES=1 bash 01_download_models.sh     # 只下权重

# 2) 推理：官方三示例
EXAMPLE=dtu_provided        LONG_SIDE=518 MAX_VIEWS=32 bash 02_run_inference.sh
EXAMPLE=hypersim_provided   LONG_SIDE=518 MAX_VIEWS=50 bash 02_run_inference.sh
EXAMPLE=hypersim_da3_pose   LONG_SIDE=518 MAX_VIEWS=32 bash 02_run_inference.sh

# 2') 自有图像
IMAGES_DIR=/mnt/d/data/myscene/images CAMERAS=/mnt/d/data/myscene/cameras.npz \
  OUT_NAME=myscene bash 02_run_inference.sh
IMAGES_DIR=/mnt/d/data/myscene/images POSE_MODE=da3-pose OUT_NAME=myscene \
  bash 02_run_inference.sh                       # 无相机，DA3 先测位姿

# 3) 评测（自接 gsplat 渲染）
RESULT=$HOME/output/infinisplat_v2_results/dtu_provided \
  IMAGES_DIR=$HOME/repos/InfiniSplat/InfiniSplat_v2/examples/data/dtu_provided/images \
  MODE=rerender bash 03_eval_render.sh           # A: 输入视角自洽
EXAMPLE=dtu_provided MODE=holdout N_INPUT=8 N_TARGET=4 bash 03_eval_render.sh   # B: NVS

# 4) 自有数据集批量（每子目录一个物体）+ 记录显存/耗时
DATASET_ROOT=/mnt/d/dataset/物体三图数据 bash 04_run_dataset.sh
DATASET_ROOT=... LONG_SIDE=1022 DO_EVAL=0 bash 04_run_dataset.sh

# 5) 少视角真实 NVS（2 输入 → 1 留出，位姿同源）
DATASET_ROOT=... CASES="手表 鼠标" N_INPUT=2 N_TARGET=1 bash 06_nvs_holdout.sh

# 6) 位姿自检（判断 DA3 是否塌缩）
python diag_pose.py --results ~/output/infinisplat_v2_results --prefix threeview

# 7) 拆耗时段 / 测常驻复用成本
python bench_stages.py --model ~/model/infinisplat_v2/da3 \
  --images-dir /mnt/d/dataset/物体三图数据/手表 --repeat 6 --out /tmp/stages.json

# 8) COLMAP 供位姿（DA3 塌缩时的备选；本数据集上因纹理太少未能重建）
IMAGES_DIR=/mnt/d/data/imgs COLMAP_LONG_SIDE=1024 bash 05_pose_colmap.sh
```

## 首次准备

```bash
# 官方仓 + DA3（00 会自动 clone；此处为手动版本）
mkdir -p ~/repos && cd ~/repos
git clone --depth 1 https://github.com/zju3dv/InfiniSplat.git
git clone https://github.com/ByteDance-Seed/Depth-Anything-3.git
git -C Depth-Anything-3 checkout 41736238f5bced4debf3f2a12375d2466874866d

# 环境
cd /mnt/c/code/media_code/infinisplat_v2
bash 00_setup_env.sh
bash 01_download_models.sh
```

权重目录布局：

```
~/model/infinisplat_v2/
└── da3/
    ├── config.json          # 含 weights_sha256 与融合契约（voxel_multiplier=1.4 等）
    └── model.safetensors    # 6.6 GB，SHA256 6d23f9ce…a66e8
```

## Config (env vars)

| var | default | note |
| --- | --- | --- |
| `CONDA_ENV` | `infinisplat_v2` | 独立 env，别混用已有的 |
| `PY_VER` | `3.11` | 包要求 `>=3.10,<3.13` |
| `TORCH_VER` | `2.7.1` | 官方在 2.7.1 + cu128 + H20 上实测 |
| `HF_BASE_URL` | `https://aifasthub.com` | 官方 HF 与 hf-mirror 在本机都不通，见 Notes |
| `DOWNLOAD_JOBS` | `8` | 分块并发；单连接实测 ~1.3 MB/s |
| `INFINISPLAT_REPO` | `$HOME/repos/InfiniSplat` | 官方仓 |
| `DA3_REPO` | `$HOME/repos/Depth-Anything-3` | 钉死 commit 41736238 |
| `WEIGHTS_DIR` | `$HOME/model/infinisplat_v2/da3` | `config.json` + `model.safetensors` |
| `RESULTS_DIR` | `$HOME/output/infinisplat_v2_results` | Linux fs，跑完用 08 搬 /mnt/d |
| `LONG_SIDE` | `518` | 112–2044，内部对齐 14 的倍数（DA3 patch=14） |
| `MAX_VIEWS` | `32` | CLI 上限 50；50 视角示例要显式给 `--max-views 50` |
| `EPS2D` | `0.01` | 论文值 1e-2；gsplat 默认 0.3，渲染时显式覆盖 |
| `MODE` | `holdout` | `03` 的评测口径：holdout（NVS）/ rerender（自洽） |
| `N_INPUT` / `N_TARGET` | `8` / `4` | holdout 划分；4 对齐论文"每例 4 个目标视角" |

## 目录布局

```
/media_code/infinisplat_v2/          # ← 本目录（编排脚本）
├── _env.sh                          # 共享环境（proxy / conda / GPU / 路径）
├── 00_setup_env.sh                  # 建 env + torch + infinisplat + DA3 + gsplat
├── 01_download_models.sh            # 权重 + 官方示例（镜像站分块下载）
├── 02_run_inference.sh              # 前馈推理 → scene.ply
├── 03_eval_render.sh                # 自接 gsplat 渲染 + 两种口径评测
├── 04_run_dataset.sh                # 批量跑自有数据集 + 逐物体记显存/耗时
├── 05_pose_colmap.sh                # COLMAP SfM 供位姿 → cameras.npz
├── 06_nvs_holdout.sh                # 少视角真实 NVS（N 输入 → 留出 M 视角）
├── 08_move_output.sh                # 结果从 Linux fs 搬到 Windows 盘
├── run_all.sh                       # 一键
├── hf_download.py                   # 分块并发下载器（断点续传 + SHA256）
├── make_holdout.py                  # 输入/目标视角划分
├── render_check.py                  # PLY → 渲染 → PSNR/SSIM + 拼图
├── bench_run.py                     # 包一条命令，线程采样 GPU 显存峰值 + 计时
├── bench_stages.py                  # 拆耗时段 + warm 重复（测常驻复用成本）
├── rebuild_bench.py                 # 从已有结果目录重建汇总表（不重跑 GPU）
├── diag_pose.py                     # 相机几何自检（两两夹角/基线 + 逐视角指标）
├── EXPERIMENTS.md                   # 实验记录：实测数值、踩坑、复现边界
└── README_wsl.md                    # 本文件

~/repos/InfiniSplat/InfiniSplat_v2/  # 官方仓（无训练代码）
~/repos/Depth-Anything-3/            # DA3 几何基座（钉死 commit）
~/model/infinisplat_v2/da3/          # 权重
~/output/infinisplat_v2_results/     # 输出（scene.ply / cameras.npz / result.json / eval/）
```

## Notes

**网络（本机 2026-10-30 实测）**
- `huggingface.co` 主站 API 能通、**大文件下载被 reset**；`hf-mirror.com` 解析正常
  （160.16.86.14）但 TCP 完全不通 —— 两者都不可用。
- 可用的是 `aifasthub.com`（HF 兼容 API + Range），与 HF 同 SHA（`10eec5ca…`）。
- 镜像站 WAF 会拦 `Python-urllib` 默认 UA（403），下载器必须带浏览器 UA。
- 镜像站 **HEAD 返回 403**，取文件大小要改用 `Range: bytes=0-0` 读 `Content-Range`。
- `github.com` 直连可用，DA3 的 `git+https` 安装没问题。

**推理契约（源码验证，别凭直觉改）**
- Gaussian 头是 **11 通道**：rot(4) + scale(3) + opacity(1) + SH0 残差(3)，
  **没有位置通道** —— 中心 100% 来自深度反投影（`model.py` 里 `depth_to_world_coords_points`）。
- `scale = exp().clamp_max(0.3)`（RDNorm 归一化空间的硬上限，0.3 不是像素单位）；
  颜色是以输入 RGB 为基底预测残差 `(rgb-0.5)/SH_C0 + sh_residual`。
- 合并契约写死在 `config.json` 并在 `pipeline.py:from_pretrained` 硬校验：
  `normalization=mean_predicted_point_radius`、`merge=uniform_mean`、
  `voxel_multiplier=1.4`、`opacity_threshold=0.01`、`eps2d=0.01`。改任何一个都会
  直接抛 `Unsupported InfiniSplat inference configuration`。
- 体素合并的边长是 **median_scale×1.4 的自适应值**（代码），论文写固定 0.004 ——
  两者关系仓库里无训练代码可证。

**输出坐标系（最坑的一处）**
- `scene.ply` 与 `cameras.npz` 都在 **RDNorm 空间**：第一相机为原点，全体反投影点
  的平均径向距离归一化为 1。所以 PLY 的 xyz 不是米，`scale_i` 存 `log(s)`、
  `opacity` 存 logit（标准 3DGS PLY 约定）。
- 在这个空间里自洽渲染没问题；但**外部 GT 相机（COLMAP/标定）活在原始尺度**，
  直接拿来渲染会出现 `alpha=0`、画面全白、PSNR 为负。必须先用输入视角做锚拟合
  Sim(3) 把目标相机搬过去（`render_check.py --ref-cameras`，实测残差 1e-7 / 0.024°）。
- 要和外部资产（如 COLMAP 稠密点云）对齐时同理，必须先统一尺度。

**渲染（发布包不带）**
- `INSTALL.md` 明说不依赖 gsplat；要出图必须自接。`03_eval_render.sh` 用 gsplat
  `rasterization`，`eps2d=0.01`（论文值）。
- **gsplat 1.5.2 的输出已含 `+0.5` 直流偏置**（实测 f_dc=0、alpha≈0.98 → 像素≈0.49），
  不要再加一次；`colors` 必须给 `[C,N,1,3]`。
- GT 图必须用与 `prepare_images` 相同的 bicubic 中心裁剪缩放，内参同步乘同一个矩阵，
  否则 GT 与渲染差一个裁剪偏移，指标被系统性拉低。
- 只给输入视角的相机 → 想渲染新视角必须自己提供目标相机（provided 模式），
  这正是接入自家 COLMAP 相机的入口。

**与 vggt_human 的对接点**
- 可作为稀疏视角 `scene_gs` 初始化的 A/B 方案：输入图像 + COLMAP 相机 →
  `scene.ply`，与现行 COLMAP 点云初始化对比。
- 局限：训练数据全是室内合成/扫描（Hypersim/Raw3D/ScanNet++），无人体专项；
  几何冻结意味着 **MHR/3DMM 这类"要改几何"的环节借不上力**，只适合背景/场景。
- 相机只收 PINHOLE / SIMPLE_PINHOLE 去畸变模型（`cameras.py` 校验），
  广角/鱼眼要先自行去畸变。

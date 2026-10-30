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
├── run_all.sh                       # 一键
├── hf_download.py                   # 分块并发下载器（断点续传 + SHA256）
├── make_holdout.py                  # 输入/目标视角划分
├── render_check.py                  # PLY → 渲染 → PSNR/SSIM + 拼图
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

**输出坐标系**
- `scene.ply` 与 `cameras.npz` 都在 **RDNorm 空间**：第一相机为原点，全体反投影点
  的平均径向距离归一化为 1。所以 PLY 的 xyz 不是米，`scale_i` 存 `log(s)`、
  `opacity` 存 logit（标准 3DGS PLY 约定）。
- 该空间对渲染自洽（相机与点同一套），但要和外部资产（如 COLMAP 稠密点云）对齐时
  必须先统一尺度。

**渲染（发布包不带）**
- `INSTALL.md` 明说不依赖 gsplat；要出图必须自接。`03_eval_render.sh` 用 gsplat
  `rasterization`，`eps2d=0.01`。sh_degree=0 的输出是 SH 直流项，需按
  `0.5 + SH_C0*f_dc` 转 RGB。
- 只给输入视角的相机 → 想渲染新视角必须自己提供目标相机（provided 模式），
  这正是接入自家 COLMAP 相机的入口。

**与 vggt_human 的对接点**
- 可作为稀疏视角 `scene_gs` 初始化的 A/B 方案：输入图像 + COLMAP 相机 →
  `scene.ply`，与现行 COLMAP 点云初始化对比。
- 局限：训练数据全是室内合成/扫描（Hypersim/Raw3D/ScanNet++），无人体专项；
  几何冻结意味着 **MHR/3DMM 这类"要改几何"的环节借不上力**，只适合背景/场景。
- 相机只收 PINHOLE / SIMPLE_PINHOLE 去畸变模型（`cameras.py` 校验），
  广角/鱼眼要先自行去畸变。

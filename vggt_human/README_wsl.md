# VGGT-Human (WSL Ubuntu 24.04) — 本地复现指南

本文件是 [`README.md`](README.md) 的 **WSL 本地复现版**。照着做即可从零跑完全流程。

> 两个 README 共存：
> - [`README.md`](README.md) — 服务器流程（`doll` env clone + 公司代理 + `/usr/local/cuda`）
> - [`README_wsl.md`](README_wsl.md) — 本文件（WSL 从零建 env + 无代理 + conda 装 CUDA toolkit）

## 与服务器版的核心差异

| | 服务器 (`00_setup_env.sh`) | WSL (`00a_setup_env.sh`) |
|---|---|---|
| conda env | cp -a clone `doll`（含 torch） | conda create -n vggt_human python=3.10 + pip install torch cu121 |
| CUDA toolkit (nvcc) | 系统 `/usr/local/cuda-12.x` | conda install cuda-nvcc（env 内，无需 sudo） |
| gcc | conda gxx_linux-64=12 | 同（conda gxx_linux-64=12） |
| 代理 / CA | proxy.env 填公司代理密码 + CA bundle | 不需要（直连互联网） |
| pip 源 | 默认 PyPI | 清华镜像（`-i https://pypi.tuna.tsinghua.edu.cn/simple`） |
| HF 源 | 直连 huggingface.co | hf-mirror.com 镜像（`HF_ENDPOINT`） |
| 仓库位置 | `/mnt/c/code/`（与 media_code 同级） | `~/repos/`（Linux 文件系统，编译快） |
| 权重 / 输出 | `/mnt/c/code/model/` | `~/model/`、`~/output/vggt_human_results/`（Linux fs，训练 I/O 快） |

> **为什么仓库 / 权重放 Linux 文件系统？** `/mnt/c` `/mnt/d` 是 Windows drvfs（9p 协议），文件 I/O 慢 5-10 倍。编译 CUDA 扩展 + 训练时大量写文件，放 Linux fs 避免超时和性能问题。路径通过 `proxy.env` 自动覆盖，脚本 01-08 不需改。

## Windows 路径 → WSL 路径

WSL 自动把 Windows 盘挂到 `/mnt/`，命令里用正斜杠：

| Windows | WSL |
|---|---|
| `C:\code\media_code` | `/mnt/c/code/media_code` |
| `D:\dataset\sample` | `/mnt/d/dataset/sample` |
| `D:\output` | `/mnt/d/output` |

**路径策略**（哪些放 Linux fs，哪些放 Windows 盘）：

| 内容 | 位置 | 原因 |
|---|---|---|
| conda env + 仓库 + CUDA 扩展 | `~/` (Linux fs) | 编译必须放 Linux fs，drvfs 上 symlink 会坏 |
| 权重 (VGGT-Omega 4.6GB) | `~/model/` (Linux fs) | 读一次加载到 GPU，Linux fs 读大文件比 drvfs 快 5-10x |
| 输入图像 | `/mnt/d/...` (Windows fs) | 只读一次，慢一点无所谓 |
| 训练输出 (RESULTS_DIR) | `~/output/` (Linux fs) | step 04 训练大量写文件，放 drvfs 慢到影响训练 |
| 最终结果备份 | `/mnt/d/output/` (Windows fs) | 训练完后用 step 08 剪切过去，释放 Linux fs 空间 |

## 前提条件

1. **WSL Ubuntu 24.04** 已安装并运行
   ```powershell
   wsl -d Ubuntu2404
   ```
2. **NVIDIA 驱动** 在 Windows 上装好（WSL 内 `nvidia-smi` 能看到 GPU）
3. **Miniconda** 已安装：
   ```bash
   # 如果还没装 miniconda：
   curl -fsSL https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -o /tmp/miniconda.sh
   bash /tmp/miniconda.sh -b -p ~/miniconda3
   ```
4. **磁盘空间** ≥ 30GB（torch + CUDA toolkit + 权重 + 仓库）

## 首次准备（一次性）

### 步骤 0：安装环境

```bash
# 进入 WSL
wsl -d Ubuntu2404

# 进入 media_code 目录
cd /mnt/c/code/media_code

# 一键安装（创建 env + 装 torch + CUDA toolkit + gcc + 依赖 + clone 仓 + 编 CUDA 扩展）
bash vggt_human/00a_setup_env.sh
```

> 如果 `pip install` 很慢（默认 PyPI 在国内慢），手动用清华镜像装剩余包：
> ```bash
> conda activate vggt_human
> pip install -i https://pypi.tuna.tsinghua.edu.cn/simple opencv-python huggingface_hub \
>     plyfile tqdm torchmetrics lpips scipy trimesh matplotlib mediapipe==0.10.14 \
>     diffusers==0.32.2 transformers==4.49.0 peft==0.14.0 omegaconf kornia accelerate
> ```

安装完成后：
```bash
# 让 conda 在你的 shell 里可用
source ~/.bashrc
```

### 步骤 0b：配置 proxy.env

00a 会自动生成 `proxy.env`（含 WSL 路径覆盖）。需要手动补充两项：

```bash
nano /mnt/c/code/media_code/proxy.env
```

补充以下内容：
```bash
# HF_TOKEN（VGGT-Omega 权重是 gated 仓库）
# 1. 申请访问：https://huggingface.co/facebook/VGGT-Omega（自动审核）
# 2. 创建 token：https://huggingface.co/settings/tokens
export HF_TOKEN="hf_xxx"               # ← 填你的 token

# HF 镜像（huggingface.co 直连超时；用 hf-mirror.com 镜像）
export HF_ENDPOINT="https://hf-mirror.com"
```

### 步骤 0c：下载 VGGT-Omega 权重

```bash
cd /mnt/c/code/media_code
GPU=0 VARIANT=1b_512 bash vggt-omega/01_download_models.sh
```

> 用了 `HF_ENDPOINT=https://hf-mirror.com` 镜像，4.6GB 约 3 分钟下完。
> 权重**实际**落在 `proxy.env` 里 `MODEL_DIR` 指向的位置，本机是
> `/mnt/d/wheel/vggt_human_ms/VGGT-Omega/vggt_omega_1b_512.pt`；
> **不是** `~/model/VGGT-Omega/`（那里只有 HF 的 `.cache/`，没有 `.pt`）。
> 原因：`_env.sh` 会 source `proxy.env`，`MODEL_DIR` 非空时下载脚本的默认路径被覆盖。
> step 02 硬性检查 `$MODEL_DIR/vggt_omega_1b_512.pt`，路径写错会直接报错退出 —— 全流程命令里要显式用 D 盘这个路径。

### HYPIR 人脸增强（step 01/06）— 遗留问题

00a 默认会 clone HYPIR + 下载 SD2 base model。如果遇到问题（hf-mirror 没收录 SD2 base repo / HYPIR 的 `beauty_ppr50k` LoRA 权重未从服务器拷来），可 **跳过 step 01/06**，直接从 step 02 开始：

```bash
# 安装时跳过 HYPIR
SKIP_HYPIR=1 bash vggt_human/00a_setup_env.sh
```

准备好后再开启 HYPIR：
1. 确保 huggingface.co 可达或用镜像 `HF_ENDPOINT=https://hf-mirror.com`
2. 从服务器拷 `beauty_ppr50k` LoRA 权重到 `~/repos/HYPIR/experiments/`
3. 下载 SD2 base model：`git clone https://hf-mirror.com/stabilityai/stable-diffusion-2-base ~/model/HYPIR/sd2_base`
4. 跑 step 01 对原始图做人脸增强

## 路径布局（WSL）

```
~/                                # Linux 文件系统（WSL vhdx 100GB）
├── repos/                        # 官方仓库（00a clone，Linux fs 编译快）
│   ├── vggt-omega/               # VGGT-Omega 官方代码
│   ├── gaussian-splatting/       # 原版 3DGS
│   │   ├── submodules/
│   │   │   ├── diff-gaussian-rasterization/  # CUDA 扩展（dr_aa 分支）
│   │   │   └── simple-knn/                    # CUDA 扩展
│   │   └── third_party/glm/
│   ├── HYPIR/                    # 人脸增强（SKIP_HYPIR=1 时不 clone）
│   ├── DiffBIR/                  # 去噪（INSTALL_DENOISER=1）
│   └── SwinIR/                   # 去噪（INSTALL_DENOISER=1）
├── model/                        # 权重（Linux fs，训练 I/O 快）
│   ├── VGGT-Omega/               # vggt_omega_1b_512.pt（gated HF 下载）
│   └── HYPIR/
│       └── sd2_base/             # SD2 base model（HYPIR 启用时需要）
├── output/
│   └── vggt_human_results/       # 训练输出（跑完用 step 08 搬到 D 盘）
└── miniconda3/                   # conda 安装
    └── envs/vggt_human/          # conda env（python=3.10 + torch cu121 + nvcc + gcc）

/mnt/c/code/media_code/           # 编排脚本（Windows fs，git 仓）
├── proxy.env                     # WSL 路径覆盖 + HF_TOKEN + HF_ENDPOINT（00a 自动生成）
└── vggt_human/
    ├── 00a_setup_env.sh          # WSL 安装脚本
    ├── 01b_heic_to_jpg.sh        # HEIC→JPG 转换（iPhone 照片）
    ├── 01_face_enhance.sh        # 以下脚本与服务器共用，通过 proxy.env 自动用 WSL 路径
    ├── 02_run_inference.sh
    ├── ...
    └── 08_move_output.sh          # 训练完把结果从 Linux fs 剪切到 D:\output

/mnt/d/                           # Windows D: 盘
├── dataset/                      # 输入数据（只读，drvfs 慢但无所谓）
└── output/                       # 最终结果备份（step 08 搬过来）
    └── vggt_human_results/
```

## 基线验证实测（RTX 3090 / 2026-09-02）

`02 → 03 → 04 → 05` 官方基线全流程跑通（无任何增强开关），输入 125 张 1440×1920。

### 核心结论：渲染不再是黑图

| 指标 | 旧 `train_pose.py`（已归档） | 官方 7000 iter | 官方 30000 iter |
|---|---|---|---|
| 渲染平均亮度 | 0.8（全黑 + 白点） | 127.2 | **127.6** |
| GT 平均亮度 | 127.9 | 127.9 | 127.9 |
| 亮度比值 | ~0.006 | 0.995 | **0.998** ✅ |
| PLY 大小 | 17,912 高斯（远小于正常） | 214 MB | **290 MB** |
| 场景尺度 (x-span) | 162（爆炸） | 3.35 | 3.35 |

### 各步耗时（RTX 3090）

| 步骤 | 耗时 | 关键产物 |
|---|---|---|
| 02 推理 | 35s | 2,000,000 点，`extent=[3.35, 1.62, 1.72]` |
| 03 npz→COLMAP | ~1 min | 125 图 + ~200k 初始点 |
| 04 训练 7000 iter | 7 min | `iteration_7000/point_cloud.ply` 214 MB |
| 04 训练 30000 iter | **42 min** | `iteration_30000/point_cloud.ply` 290 MB |
| 04 渲染评估 | 2 min | `train/ours_{7000,30000}/{renders,gt}/` |
| 05 新视角 | 14s | `source_aug/` 130 cameras（125 原图 + 5 新增） |

> Loss 0.34 → 0.040。训练速度随高斯增密递减（17.6 → 9 it/s），所以 30000 iter 不是 7000 的 4 倍耗时。
> **7000 iter 亮度比值已达 0.995，赶时间够用；出最终交付 PLY 用 30000（0.998）。**
> ⚠️ 重跑 04 会覆盖同名 `iteration_*` / `ours_*` 目录（想留旧结果先备份）。

### 全流程复现验证（2026-09-03，增强加回之后）

所有增强模块加回后，用全新 `RESULTS_DIR` 完整重跑 02→03→04(30000)→05
（全部开关默认关闭，确认默认链路 = 纯官方基线不受影响）。
脚本 `.tmp_diag/run_verify_full.sh`，总耗时 50 min，四步全 EXIT=0，
04 分支确认走官方 train.py、dynamic 预处理被正确跳过。

| 指标 | 原 30000 基线（/mnt/d） | 全流程复现（~/vggt_human_verify） |
|---|---|---|
| PSNR（125 帧同源对比） | 27.37 dB | **27.37 dB** |
| L1 | 0.0219 | 0.0220 |
| 渲染/GT 亮度比值 | 1.002 | 1.002 |
| 初始点云 | 7,163 | 7,163（02/03 输出逐位一致） |
| 05 新视角 | 5 张达覆盖阈值 | 6 张达覆盖阈值（`source_aug` 129 cams） |

**结论：模块加回与开关化改造对默认链路零影响，基线完全可复现。**
（04 训练内评估 PSNR：7000=25.96、30000=26.92，与原基线一致量级。）

### 增强 P1-1：depth-normal consistency（已加回 + 修复）

加回 `depth_normal_cons.py` + `train_depth_normal.py`（官方 train.py 副本 + 单一 hook）。实测 30000 iter：

| 指标 | 官方基线 | depth-normal |
|---|---|---|
| 渲染亮度 | 127.6 | 126.0 |
| GT 亮度 | 127.9 | 125.7 |
| 亮度比值 | 0.998 | 1.002 |
| PLY 大小 | 290 MB | 292.6 MB |
| 训练耗时 | 42 min | 45 min |

> DN Loss 在 densify 结束后（>15000 iter）才启用，densify 阶段跳过（否则点数爆炸导致 8.5s/iter）。
> DN Loss 稳定在 0.46，约束生效但未损害亮度指标。几何质量提升需看 PLY 细节。

**修复了原模块的 3 个 bug：**
1. **法线公式错误**（致命）：`(-dz/dx/fx, -dz/dy/fy, 1)` → `(-Zu*fx, -Zv*fy, Z)`。原版把 fx/fy 乘除弄反（差 fx² 倍），nz 用常数 1 而非深度 Z，导致法线退化成 (0,0,1)、DN Loss≈1.0 等同随机噪声、约束静默失效。
2. **缓存尺寸不匹配崩溃**：densify 后点数变化，缓存的法线索引对不上当前点云，`tensor a (7163) must match b (9679)`。
3. **densify 阶段速度崩塌**：点数爆炸导致投影+赋值开销线性增长，5450 iter 后从 16 it/s 崩到 8.5 s/iter。改为 densify 阶段跳过 DN Loss。

**用法：**
```bash
GPU=0 INPUT_DIR=... RESULTS_DIR=~/output/vggt_human_results_dn \
  USE_DEPTH_NORMAL=1 DEPTH_NORMAL_WEIGHT=0.05 \
  bash vggt_human/04_train_3dgs.sh
```

### 增强 P1-2：pose_refine（已加回，但当前后端不可用）

加回 `pose_refine.py` + `train_pose_refine.py`（可学位姿：四元数 + 平移）。修完 3 个 bug
后暴露出**根本性限制**：

1. `world_view_transform` 原用 `torch.zeros(4,4)` + in-place 赋值构造 → 断梯度，改 `torch.cat`
2. `PoseRefinedCamera` 缺 `alpha_mask` 等属性 → 加 `__getattr__` 委托 base Camera
3. `_rotmat_to_quat` 用了 `np` 但 numpy 在条件块内 import → 移到顶部

| 梯度检查 | 结果 |
|---|---|
| `world_view_transform.sum().backward()` | 位姿参数**有**梯度 ✅（说明 cat 构造确实修好了梯度链） |
| `render(...)` 后 L1 loss 再 backward | 位姿参数 `grad=None` ❌ |

**根因**：`diff_gaussian_rasterization` 的 CUDA rasterizer 把 viewmatrix 当**常量**用于投影，
只对高斯参数（means/scales/rot/opacity/SH）求梯度，**不支持可微相机位姿**。
所以位姿精炼在当前后端下拿不到梯度，无法工作。

**可行替代（未实施）**：① 换 gsplat 后端（原生支持可微位姿，但改动面大）；
② 在 03 之后加一轮 COLMAP bundle adjustment（不依赖 rasterizer 梯度）。

> `USE_POSE_REFINE=1` 时 `04_train_3dgs.sh` 会打印告警并退化为官方 train.py，不会静默跑坏。

### 数据集动态性诊断（决定要不要开"噪声抑制类"增强）

加 `noise_negating` / `dynamic_mask` / `dynamic_filter` 之前，先判断这个数据集到底有没有
动态内容。用基线 30000 的 `renders/` vs `gt/`（125 帧）做统计：

| 指标 | 实测 | 解读 |
|---|---|---|
| 训练 loss（基线 30000） | 0.34 → **0.040** | 静态场景的典型收敛值；若有动态物体，3DGS 拟合不了会卡在 0.1 以上 |
| 误差空间集中度（top-20% 块贡献的残差） | **46.8%** | 误差高度集中，不是随机欠拟合 |
| 高误差块跨视角重合 IoU | 0.175（随机基线 0.111） | 略高于随机。但相机绕人转、各视角图像坐标系不可比，**该指标参考性有限** |

**结论：这是静态场景**（多视角拍摄静止人物）。误差集中在难重建区域（图像边缘、头发、
反光），而不是跨视角不一致的动态物体。

- `noise_negating` / `dynamic_mask` / `dynamic_filter` 三件套**大概率有害**：
  MLP 会把"始终高误差的静态难区"当成动态区域永久屏蔽，反而丢细节。
- 后续优化应转向**提升静态重建质量**（位姿精度、点云密度、几何约束），
  而不是动态区域抑制。

### 增强 P0-3：noise negating（已加回 + 修复，但对**静态人物场景有害**，默认关闭）

加回 `noise_negating.py` + `train_noise_negate.py`：DINOv2 ViT-S/14 提特征 +
轻量 MLP（384→16→1）在线学习每帧动态掩码，loss 只在静态像素上计算。

实测 7000 iter 同源对比（125 帧全量 `renders/` vs `gt/`）：

| 指标 | 官方基线 7000 | noise negating 7000 | 差异 |
|---|---|---|---|
| PSNR | 25.97 dB | 24.35 dB | **−1.62 dB** ❌ |
| L1 | 0.0269 | 0.0314 | +16.7% ❌ |
| 亮度比值 | 1.001 | 0.996 | −0.005 |
| 训练耗时 | 7 min | 10 min（另加 3 min DINOv2 加载） | 慢 ~85% |

**为什么不适用**：MLP 把"始终高误差的静态难区"（图像边缘、头发、反光）当成动态区域屏蔽，
这些区域失去监督后高斯既不被优化、也因 densify 梯度不足而缺发育 → PSNR 掉 1.6 dB。
训练中 `Static%` 稳定在 90~99%（只屏蔽 7% 左右），说明**MLP 自己学到了"场景基本静态"**，
但残余的这点屏蔽就足以造成明显损失。

> 与上面「数据集动态性诊断」完全吻合：**静态人物场景不要开动态抑制类增强**。
> `USE_NOISE_NEGATE` 默认 0，代码与文档保留，将来有真动态数据（街景、含行人）时再用。

**修复了原模块的 7 个 bug**（单元测试全过，见 `.tmp_diag/test_nn.py`）：

1. **MLP 在全分辨率跑**（致命）：原实现把 384 通道特征插值到 `(384,H,W)` 再跑 MLP，
   1440×1920 下会产生 ~4GB 中间激活。改为在 DINO 特征图 `(F,F)` 上推理再上采样 mask
   —— 顺带让监督信号（cosine 不相似度）与 MLP 输出分辨率对齐。
2. **mask 未 detach**（致命）：3DGS 重建 loss 会顺着 mask 反传到 MLP，MLP 为最小化重建
   loss 会学会"屏蔽所有高误差区域"，形成对抗性塌缩。已 detach，单元测试验证 MLP 参数无梯度泄漏。
3. **masked L1 除以全像素数**：loss 被系统性缩小，与官方 `lambda_dssim` 配比、densify
   梯度阈值语义脱节。改为除以静态像素数（全 1 mask 时与官方 L1 逐位相同）。
4. **无动态比例兜底**：MLP 随机初始化输出饱和在 0.5，固定阈值 0.25 会让几乎全图判为
   动态、loss 归零崩塌。改为 `thr = max(固定阈值, 分位数(1−NN_MAX_DYNAMIC_RATIO))`，
   即使输出饱和也能保证 ≥50% 像素参与 loss（实测饱和时仍保持 66%）。
5. **残差边界项方向写反**：原 `relu(mask−upper)+relu(lower−mask)` 在"确定静态"区
   反而把 mask 推向 1（dynamic）。反证：旧实现给"正确标注"打 **1.0000** 分、
   给"标注反了"打 **0.0000** 分。已按 mask 语义（1=dynamic）转换边界。
6. **masked SSIM 先乘 mask**：屏蔽区两图同为 0 → SSIM≈1，虚低。反证：旧做法
   **0.8756** vs 左半真实 SSIM **0.7513**。改为 ssim map × mask 加权平均。
7. **接入点在 `no_grad` 块内**：官方 train.py 从 `with torch.no_grad():` 一直包到循环
   末尾，MLP 的 `loss.backward()` 直接报
   `element 0 of tensors does not require grad`。已用 `torch.enable_grad()` 包住。

**用法（不推荐用于静态人物场景）：**
```bash
GPU=0 INPUT_DIR=... RESULTS_DIR=~/output/vggt_human_results_nn \
  USE_NOISE_NEGATE=1 NN_MAX_DYNAMIC_RATIO=0.2 NN_WARMUP_EPOCHS=15 \
  bash vggt_human/04_train_3dgs.sh
```

### dynamic_mask + dynamic_filter 实测（2026-09-03）

机制：**训练前的点云预处理**（不是训练中的 loss）。`dynamic_mask.py` 用
GroundingDINO（文本→框）+ SAM2.1（框→掩码）给每帧训练图生成动态掩码；
`dynamic_filter.py` 把初始点云投到所有视角做多视角投票，落入掩码比例
> `DYNAMIC_THRESHOLD`(0.3) 的点删除——从源头减少动态物体污染的高斯。

**环境**：`sam2` 包（gh-proxy 镜像装，`--no-build-isolation` 复用环境 torch 2.5.1；
`configs/` 目录需从源码树补拷进 site-packages），GroundingDINO 走 hf-mirror 自动下载。
SAM2 config 文件名是缩写（`sam2.1_hiera_l.yaml`），代码已按 checkpoint 名自动映射。

**集成**：`04_train_3dgs.sh` 训练前新增第 0 步——
`ENABLE_DYNAMIC_MASK=1` 生成掩码（缓存于 `$RESULTS_DIR/dynamic_mask`，
`FORCE_DYNAMIC_MASK=1` 强制重生成）→ `ENABLE_DYNAMIC_FILTER=1` 过滤
`source/sparse/0/points3D.ply`（原始点云备份为 `points3D.ply.orig`，可随时还原）。
⚠️ prompt 默认 `TV screen monitor`，**刻意不含 person**：静态人物数据集里
person 是主体，过滤它等于删主体。真动态场景用
`DYNAMIC_PROMPTS="person TV screen"` 显式传。

**实测（7000 iter，prompt=person 刻意验证机制链路）**：

| 指标 | 官方基线 7000 | dynamic_mask+filter 7000 |
|---|---|---|
| 掩码生成 | — | 125 帧 ~23% dynamic，SAM2.1-large + GroundingDINO-tiny，~4 min |
| 初始点云 | 7,163 | 3,962（删 44.7%） |
| PSNR | 25.97 dB | 25.59 dB（−0.38 dB） |
| L1 | 0.0269 | 0.0281（+4.5%） |
| 训练速度 | ~17 it/s | ~13 it/s（点云减半后 densify 更快回血） |

结论：**机制端到端有效且损害温和**。即使删掉近半初始点云，训练中的 densify
也会重新长出被删区域（GT 监督仍含 person），只掉 0.38 dB。静态场景默认关闭
（`ENABLE_DYNAMIC_MASK/FILTER` 默认 0）；真动态场景（行人、屏幕闪烁）下
预期是正收益，因为那时删掉的点是被污染的。

**修复了原模块的 5 个 bug**（单元测试 `.tmp_diag/test_df.py` 全过）：

1. **GroundingDINO 后处理 API 变更**：transformers ≥4.46 把
   `post_process_grounded_object_detection` 的 `box_threshold=` 改名 `threshold=`，
   参数名不对抛 TypeError，且不在旧代码 `except AttributeError`范围内 → 直接崩。
   改为多组参数名依次尝试。
2. **SAM fallback 输入框格式错**：`SamProcessor` 期望 `[[[x1,y1,x2,y2],...]]`，
   原代码写成角点对 `[[[x1,y1],[x2,y2]]]`。
3. **有效视角分母只数有掩码的相机**（单测抓出）：原实现 `if name not in masks:
   continue` 跳过无掩码相机、分母不累加 → "125 帧只有 1 帧有掩码"时 ratio 恒为
   1/1，过度删除。改为分母数全部可见视角。
4. **过滤输出缺 nx/ny/nz 字段**：官方 `fetchPly` 读这三个字段，缺了直接
   `ValueError: no field of name nx`（外层静默吞成 `point_cloud=None` →
   `create_from_pcd` 崩 `'NoneType' object has no attribute 'points'`）。
   输出补齐字段（置 0）。
5. **`proxy.env` 无条件覆盖 `RESULTS_DIR`**：`_env.sh` 第 12 行先 source
   proxy.env，把外部传入的 RESULTS_DIR 冲掉 → 曾导致掩码/过滤写进原数据集目录
   （靠 `.orig` 备份恢复）。改为 `:-` 条件赋值（00a_setup_env.sh 同步修）。

**用法（真动态场景）：**
```bash
GPU=0 INPUT_DIR=... RESULTS_DIR=~/output/vggt_human_results_dmf \
  ENABLE_DYNAMIC_MASK=1 ENABLE_DYNAMIC_FILTER=1 \
  DYNAMIC_PROMPTS="person TV screen" \
  bash vggt_human/04_train_3dgs.sh
```

## 全流程命令

> 假设已完成「首次准备」，输入图像在 `D:\dataset\sample\image`。
> 每条命令在 WSL 内 `/mnt/c/code/media_code` 目录下执行。
> `proxy.env` 已写入 WSL 路径覆盖，脚本自动用 `~/repos/`、`~/model/` 等路径。

```bash
cd /mnt/c/code/media_code

# ── 0) 安装环境（首次，见上方「首次准备」）──

# ── 1b) HEIC → JPG 转换（iPhone 照片是 .heic，VGGT-Omega 不认）──
#    输入：D:\dataset\sample\image（.heic 文件夹）
#    输出：D:\dataset\sample\image_jpg（.jpg 文件夹，自动创建在同级目录）
GPU=0 \
INPUT_DIR=/mnt/d/dataset/sample/image \
bash vggt_human/01b_heic_to_jpg.sh

# ── 1) 前处理人脸增强（HYPIR 遗留问题，暂跳过）──
#    跳过此步，直接用原始图像（或 step 1b 的 JPG）喂 step 02
#    准备好 HYPIR 后再启用：
# GPU=0 INPUT_DIR=/mnt/d/dataset/sample/image_jpg \
#   RESULTS_DIR=~/output/vggt_human_results \
#   bash vggt_human/01_face_enhance.sh

# ── 2) VGGT-Omega 前馈推理（图像 → 位姿+深度 → predictions.npz + scene.ply）──
#    INPUT_DIR 指向 step 1b 的 JPG 输出（如果不是 .heic 则直接用原始图夹）
GPU=0 \
INPUT_DIR=/mnt/d/dataset/测试数据sample/3fe0604320a24d66a8bde164edf18c11/image_jpg \
MODEL_DIR=/mnt/d/wheel/vggt_human_ms/VGGT-Omega \
RESULTS_DIR=~/output/vggt_human_results \
MAX_POINTS=2000000 \
bash vggt_human/02_run_inference.sh

# 输出：~/output/vggt_human_results/vggt/<scene>/
#   predictions.npz   # 位姿+深度+点云+置信度
#   scene.ply          # 点云（供检查）
#   frames/            # 喂给模型的图

# ── 3) npz → COLMAP 转换（自适应置信度过滤 + 体素降采样 ~200k + 坐标系对齐）──
#    ALIGN=1：把场景居中到原点（官方 train.py 直接吃这个坐标系即可收敛）
GPU=0 \
TARGET_POINTS=200000 \
ALIGN=1 \
RESULTS_DIR=~/output/vggt_human_results \
bash vggt_human/03_npz_to_colmap.sh

# 输出：~/output/vggt_human_results/source/
#   images/*.png              # 训练图
#   sparse/0/cameras.txt      # 内参
#   sparse/0/images.txt       # 外参 w2c
#   sparse/0/points3D.txt     # ~200k 初始点

# ── 4) 原版 3DGS 训练 + 渲染 ──
#    默认恒走官方 gaussian-splatting 的 train.py（干净官方基线）。
#    增强（已加回、默认全关，见上方各实测小节）：训练脚本切换
#      USE_DEPTH_NORMAL=1 / USE_NOISE_NEGATE=1（pose_refine 不可用）；
#    训练前预处理开关：ENABLE_DYNAMIC_MASK=1 / ENABLE_DYNAMIC_FILTER=1
#      （prompt 用 DYNAMIC_PROMPTS 覆盖，默认 "TV screen monitor" 不含 person）。
#    未加回的归档脚本见 archive/README.md。
GPU=0 \
ITERATIONS=30000 \
WHITE_BG=0 \
RESULTS_DIR=~/output/vggt_human_results \
bash vggt_human/04_train_3dgs.sh

# 输出：~/output/vggt_human_results/model_3dgs/
#   point_cloud/iteration_30000/point_cloud.ply    # 最终高斯
#   train/ours_30000/renders/*.png                 # 重建渲染
#   train/ours_30000/gt/*.png                      # GT

# ── 5) 渲染新视角 → 去噪 → AdaIN → 增强 COLMAP（可选）──
#    DENOISER 可选: diffbir（扩散，质量高）| swinir（前馈，快）| none（跳过去噪）
#    首次用 DiffBIR/SwinIR 需先: INSTALL_DENOISER=1 bash vggt_human/00a_setup_env.sh
#    ⚠️ 本步是 ITERATION（单数），step 04/07 是 ITERATIONS（复数），别写混。这不是小bug吗，怎么不改？
#       值必须等于 step 04 实际跑的迭代数，否则报 "3DGS checkpoint not found"。
GPU=0 \
DENOISER=diffbir \
NUM_NOVEL_VIEWS=10 \
ITERATION=30000 \
RESULTS_DIR=~/output/vggt_human_results \
bash vggt_human/05_denoise_novel.sh

# 输出：~/output/vggt_human_results/
#   novel_renders/*.png     # 3DGS 渲染的新视角
#   novel_alpha/*.png       # 覆盖度图
#   source_aug/             # 增强COLMAP场景（原图+去噪图）

# ── 6) 后处理人脸增强（可选，需 HYPIR 已就绪）──
#    对 source_aug/images/ 做人脸增强（与 step 01 同一个 face_enhance.py）
GPU=0 \
RESULTS_DIR=~/output/vggt_human_results \
bash vggt_human/06_face_enhance.sh

# 输出：~/output/vggt_human_results/source_aug_face/
#   images/  # 原图+去噪图，人脸区域已增强+融合
#   sparse/  # COLMAP 原样复制

# ── 7) 用增强场景训练 3DGS（可选）──
#    原图 + 去噪虚拟相机 + 人脸增强 共同监督
#    同样恒走官方 train.py（增强已归档到 archive/，需加回见 archive/README.md）
GPU=0 \
ITERATIONS=30000 \
WHITE_BG=0 \
RESULTS_DIR=~/output/vggt_human_results \
bash vggt_human/07_train_denoise.sh

# 输出：~/output/vggt_human_results/model_3dgs_denoise/
#   point_cloud/iteration_30000/point_cloud.ply    # 增强训练后的高斯

# ── 8) 训练完后：把结果从 Linux fs 剪切到 D:\output（释放 WSL 空间）──
# 预览（不实际移动，默认 DST=/mnt/d/output/vggt_human_results）：
# DRY_RUN=1 bash vggt_human/08_move_output.sh

# 实际搬运（默认 ~/output/vggt_human_results → /mnt/d/output/vggt_human_results）：
bash vggt_human/08_move_output.sh

# 若想搬到别的目录，才需要显式传 DST，例如：
# SRC=~/output/vggt_human_results \
# DST=/mnt/d/output/vggt_human_results \
# bash vggt_human/08_move_output.sh
```

- 结果：VGGT-Omega 推理 → `vggt/<scene>/predictions.npz`；COLMAP 场景 → `source/`；3DGS 高斯 → `model_3dgs/point_cloud/iteration_30000/point_cloud.ply`。跑完 step 08 后全部搬到 `/mnt/d/output/vggt_human_results/`。
- PLY 可拖到 [supersplat](https://playcanvas.com/supersplat/editor) 在线查看。

## 可能遇到的问题（WSL 专属）（可以迁移到pipeline.html中）

**1. `conda: command not found`（运行脚本时）**
00a 末尾执行 `conda init bash`。如果还没 `source ~/.bashrc`：
```bash
source ~/.bashrc
# 或每次手动 source：
source ~/miniconda3/etc/profile.d/conda.sh
```
`_env.sh` 已加 fallback：conda 不在 PATH 时自动找 `~/miniconda3`。

**2. `pip install` 很慢 / 超时**
默认 PyPI 在国内慢。用清华镜像：
```bash
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple <package>
```

**3. `huggingface.co` 连不上 / 下载超时**
用 HF 镜像。在 `proxy.env` 里加：
```bash
export HF_ENDPOINT="https://hf-mirror.com"
```
`huggingface_hub` 库会自动用镜像。VGGT-Omega 权重 4.6GB 约 3 分钟下完。

**4. CUDA 扩展编译失败（`nvcc: command not found`）**
00a 通过 conda 装 cuda-nvcc。如果失败：
```bash
conda activate vggt_human
which nvcc           # 检查
# 手动装：
conda install -y -c nvidia/label/cuda-12.1.1 cuda-nvcc cuda-cudart-dev cuda-cccl
```

**5. `torch.cuda.OutOfMemoryError`（step 02）**
RTX 3090 有 24GB，但 VGGT-Omega 1B 对帧数敏感。降压：
```bash
RESOLUTION=256 MODE=max_size GPU=0 ... bash vggt_human/02_run_inference.sh
```

**6. 编译极慢（超过 30 分钟）**
确认仓库在 Linux 文件系统（`~/repos/`）而非 `/mnt/c/` 或 `/mnt/d/`。drvfs 上编译慢 5-10x 且 symlink 可能坏。

**7. 想切 cu124**
```bash
CUDA_TOOLKIT_LABEL=nvidia/label/cuda-12.4.0 \
TORCH_INDEX_URL=https://download.pytorch.org/whl/cu124 \
bash vggt_human/00a_setup_env.sh
```

**8. WSL vhdx 空间不足**
跑完 pipeline 后执行 step 08 把结果搬到 D 盘：
```bash
bash vggt_human/08_move_output.sh
```
如果 vhdx 本身太大（即使删了文件也不缩小），在 Windows PowerShell 里压缩：
```powershell
wsl --shutdown
diskpart
# select vdisk file="C:\WSL\Ubuntu2404\ext4.vhdx"
# compact vdisk
```

## 通用问题

其他问题（CUDA 扩展编译、GLM 缺失、simple-knn clone 失败、CRLF 行尾等）见 [`README.md` 的「可能遇到的问题」](README.md#可能遇到的问题)，排错方法通用。

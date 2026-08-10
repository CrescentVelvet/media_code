# Pi3 + 2D Gaussian Splatting — 视频 → 位姿估计 → 三维重建

从一组图像或视频（典型：`wan22_rotate` 生成的 360° 旋转视频）出发，先用 [π³ (Pi3)](https://github.com/yyfz/Pi3)（ICLR 2026，前馈式位姿+点云估计）一次性推出相机位姿 + 稠密点云，导出为 COLMAP 文本格式，再用 [2D Gaussian Splatting (2DGS)](https://github.com/hbb1/2d-gaussian-splatting)（SIGGRAPH 2024，几何精确的二维高斯泼溅）训练出可渲染/可导网格的高斯场景。

本目录只含编排脚本——Pi3 官方代码在 `../Pi3`、2DGS 官方代码在 `../2d-gaussian-splatting`（00 自动 clone），权重在各算法的目录下。两步共用同一个 conda env（CPython 3.10 + torch 2.6.0+cu124，复用 wan22_rotate 本地 cp310 wheel）。

## 常用命令

> 假设已进入容器（脚本自动激活 `pi3_3dgs` env）；`GPU=0` 按需换卡。首次跑前先做下方「首次准备」。

```bash
# ── 一键：Pi3 位姿估计 → COLMAP 导出 → 2DGS 训练 → 渲染+网格 ──
# INPUT: 视频(.mp4) 或图像文件夹。默认指向 wan22_rotate 的 rotate_360.mp4
GPU=0 INPUT=../wan22_rotate_results/rotate_360.mp4 \
  bash pi3_3dgs/run_all.sh

# ── 分步 ──
# 0) 环境准备：clone Pi3 + 2DGS 仓，装依赖，编两个 CUDA 子模块
INSTALL_DEPS=1 BUILD_CUDA=1 bash pi3_3dgs/00_setup_env.sh
# 1) Pi3 推理 + COLMAP 导出（视频抽帧 → 模型推理 → 写 cameras/images/points3D）
GPU=0 INPUT=../wan22_rotate_results/rotate_360.mp4 \
  bash pi3_3dgs/01_pi3_recon.sh
# 2) 2DGS 训练（用 01 导出的 COLMAP 场景）
GPU=0 SOURCE_DIR=../pi3_3dgs_results/source \
  bash pi3_3dgs/02_train_2dgs.sh
# 3) 渲染 + 提取网格（TSDF fusion）
GPU=0 MODEL_DIR=../pi3_3dgs_results/model SOURCE_DIR=../pi3_3dgs_results/source \
  bash pi3_3dgs/03_render_2dgs.sh

# ── 自定义 ──
# 用图像文件夹代替视频
GPU=0 INPUT=/path/to/image_folder bash pi3_3dgs/run_all.sh
# 加大抽帧密度（默认 10fps × 6s = 60 帧；Pi3 显存随帧数线性增长）
GPU=0 INPUT=... FRAME_FPS=5 FRAME_MAX=120 bash pi3_3dgs/run_all.sh
# 复现论文级训练（2DGS 默认 30000 步；快速 demo 用 7000）
GPU=0 INPUT=... ITERATIONS=7000 bash pi3_3dgs/run_all.sh
# 高分辨率网格 + 无界 TSDF（适合人像在白色背景中的轨道视频）
GPU=0 MESH_RES=2048 UNBOUNDED=1 bash pi3_3dgs/03_render_2dgs.sh
# 输入是白底分割图（wan22_rotate 的 rotate_360.mp4 即是）→ 训练用白底
GPU=0 INPUT=... WHITE_BG=1 bash pi3_3dgs/run_all.sh
# 覆盖默认相机内参（Pi3 是 affine-invariant，不输出内参；默认假设 fx=fy=max(W,H)）
GPU=0 INPUT=... FX=1248 FY=1248 CX=624 CY=352 bash pi3_3dgs/run_all.sh
```

- 结果：Pi3 预测 → `predictions.npz` + `dense_cloud.ply`；COLMAP 场景 → `source/{images,sparse/0}`；2DGS 训练产物 → `model/point_cloud/iteration_<N>/point_cloud.ply` + `cfg_args`；渲染 + 网格 → `model/test/ours_<N>/{renders,mesh.ply,...}`。

## 首次准备

本流程建一份独立的 `pi3_3dgs` env（**CPython 3.10**，复用 wan22_rotate 的本地 cp310 torch 2.6.0+cu124 wheel；不要用 3.11 或 clone doll——cp310 wheel 装不进 3.11），把 Pi3 + 2DGS 两套依赖装在一起，并编 2DGS 的两个 CUDA 子模块（`simple-knn` 在 gitlab.inria.fr，`diff-surfel-rasterization` 在 github）。

```bash
cd <your-code-dir>            # e.g. /data_3d/<uid>/code
git -c http.sslVerify=false clone https://github.com/CrescentVelvet/media_code.git
cd media_code && cp proxy.env.example proxy.env   # 填 http_proxy / https_proxy
# ⚠️ 确认 proxy.env 中 http_proxy / https_proxy 两行已取消注释并填好地址，
#    否则 pip 装依赖会报 "Network is unreachable"

# 1. 建 pi3_3dgs env（CPython 3.10，匹配本地 cp310 torch/triton wheel）
conda create -n pi3_3dgs python=3.10 -y && conda activate pi3_3dgs

# 2. clone Pi3 + 2DGS 仓，装两套依赖，编两个 CUDA 子模块，验证 imports
#    INSTALL_DEPS=1 用本地 cp310 wheel 装 torch 2.6.0+cu124 + nvidia 依赖，
#    BUILD_CUDA=1 编 simple-knn + diff-surfel-rasterization（需 CUDA 12.4 toolkit）
INSTALL_DEPS=1 BUILD_CUDA=1 bash pi3_3dgs/00_setup_env.sh

# 3. 下 Pi3 权重（公开，免 token；约 1GB）
#    手动放到 $MODEL_DIR/Pi3/model.safetensors
mkdir -p ../../model/Pi3
wget --no-check-certificate -O ../../model/Pi3/model.safetensors \
  https://huggingface.co/yyfz233/Pi3/resolve/main/model.safetensors
# 或在浏览器下载后上传：https://huggingface.co/yyfz233/Pi3/tree/main
```

权重需已在 `$MODEL_DIR`（默认 `../../model`）下：
```
$MODEL_DIR/
  Pi3/
    model.safetensors              # Pi3 推理权重 (yyfz233/Pi3, 公开 CC BY-NC 4.0)
  # 复用 wan22_rotate 的本地 cp310 wheel（同目录下，00 自动找）
  torch-2.6.0+cu124-cp310-cp310-linux_x86_64.whl
  torchvision-0.21.0+cu124-cp310-cp310-linux_x86_64.whl
  triton-3.2.0-cp310-cp310-...whl
  nvidia_*.whl  ...
```

> 2DGS 无预训练权重，每个场景从头训出高斯点云（即「模型」）。Pi3 的 checkpoint 是前馈网络权重（一个场景直接前向一次出位姿+点云），不是 2DGS 的初始化。

---

以下为详细参考（流程原理 / 各步骤参数 / 排错 / 目录布局）。

## Pipeline（流程详解）

```
INPUT (video.mp4 / image_folder)
    │
    ▼
[01] Pi3 (pi3_3dgs env)
    │  ├─ 抽帧（视频按 FRAME_FPS 抽 / 文件夹直接复制） → frames/
    │  ├─ load_images_as_tensor  (Pi3 自带, 自动 resize 到 ≤255k 像素的 14 倍数)
    │  ├─ Pi3 前向  (1 次推理 → 所有视角的位姿 + 稠密点云)
    │  │     ├─ camera_poses  (1, N, 4, 4)   c2w, OpenCV (z前 y下 x右)
    │  │     ├─ points         (1, N, H, W, 3) 全局点云 (per-pixel 世界坐标)
    │  │     ├─ local_points   (1, N, H, W, 3) 每视角局部点云
    │  │     └─ conf           (1, N, H, W, 1) 置信度 logits (sigmoid 后 [0,1])
    │  ├─ 保存 predictions.npz + dense_cloud.ply (置信度 + 边缘过滤)
    │  └─ 导出 COLMAP 文本格式
    │        ├─ cameras.txt   PINHOLE, 每图一个 (默认 fx=fy=max(W,H), cx=W/2, cy=H/2)
    │        ├─ images.txt    image_id, w2c 四元数+平移, camera_id, name
    │                       (c2w → w2c: R_w2c=R^T, t_w2c=-R^T·t)
    │        └─ points3D.txt  采样稠密点云 + RGB (体素下采样到 ≤ MAX_POINTS)
    ▼
$RESULTS_DIR/source/{images, sparse/0/}
    │
    ▼
[02] 2D Gaussian Splatting  (pi3_3dgs env)
    │  ├─ Scene 读 COLMAP  → 相机内外参 + 初始点云 (from points3D.txt)
    │  ├─ 高斯初始化        → simple-knn 算初始 3D 高斯
    │  ├─ 可微光栅化        → diff-surfel-rasterization 把 2D 高斯盘 splat 成图像
    │  ├─ 损失              → (1-λ)·L1 + λ·(1-SSIM)
    │  │                       + λ_normal · 法向一致性  (2DGS 创新)
    │  │                       + λ_distortion · 深度畸变  (2DGS 创新)
    │  └─ 致密化+剪枝        → 周期性 (前 15000 步) 按梯度分裂/克隆/剪枝
    ▼
$RESULTS_DIR/model/point_cloud/iteration_<N>/point_cloud.ply
    │
    ▼
[03] Render + Mesh
    │  ├─ render.py 渲染 test 相机 (可选 train)
    │  └─ TSDF fusion 提网格
    │        ├─ bounded   (默认, 适合 bounded 场景, 需调 depth_trunc)
    │        └─ unbounded (空间收缩 + 自适应截断, 适合无限延伸场景)
    ▼
$RESULTS_DIR/model/test/ours_<N>/{renders,gt,depth}/*.png + mesh.ply
```

### Step 01 — Pi3 位姿估计 + COLMAP 导出 (`01_pi3_recon.sh` → `pi3_recon.py`)

**Pi3 前馈几何**：Pi3（π³，ICLR 2026）是 permutation-equivariant 的前馈几何网络——输入 N 张图（无固定参考视角），一次前向出 N 个相机位姿 + N×H×W 的稠密点云 + 置信度。**无优化阶段**（与 DUSt3R/Mast3R 同属 feed-forward 流派，但去除了对参考视角的依赖，鲁棒性更好）。

- **输入**：视频（按 `FRAME_FPS` 抽帧，默认 10fps；6 秒视频 → 60 帧）或图像文件夹（全部使用）。Pi3 的 `load_images_as_tensor` 自动把每张图 resize 到 `≤255k 像素` 的 14 倍数（patch 对齐）。
- **输出**：
  - `predictions.npz` — 原始 Pi3 张量（`points`/`local_points`/`camera_poses`/`conf` + `images` + `frame_names`），供后续分析。
  - `dense_cloud.ply` — 置信度过滤 + 边缘过滤后的稠密点云（Pi3 的 `example.py` 同样做法），最多 100 万点。仅用于人工检查。
- **COLMAP 文本格式导出**：
  - `cameras.txt` — 每张图一个 PINHOLE 相机，`fx=fy=max(W,H)`（典型 50° FOV）、`cx=W/2, cy=H/2`。
  - `images.txt` — Pi3 的 `camera_poses` 是 c2w（OpenCV 约定：z 前 y 下 x 右，与 COLMAP 一致），转 w2c（`R_w2c = R_c2w.T`，`t_w2c = -R_c2c.T @ t_c2w`，对正交矩阵数值稳定），再转四元数（Shepperd 分支法）。
  - `points3D.txt` — 从 Pi3 的 `points`（全局稠密点云，每视角每像素一个世界坐标）按 conf+edge 过滤，体素下采样（Open3D，无则随机采样）到 `MAX_POINTS`，RGB 取对应像素。**track 留空**——2DGS 只用 3D 点位置做初始化，不用 track。

> **内参为何是「假设」的**：Pi3 是 affine-invariant（位姿）+ scale-invariant（局部点云）——模型本身不输出相机内参。但 `camera_poses + points` 来自同一个网络，**彼此一致**（互相投影自洽）。3DGS 训练把相机内外参固定、只优化高斯，所以即使假设内参稍有偏差，高斯会被光度损失驱动去自适应匹配图像（重建视觉上合理，高频细节可能有轻微 artifact）。若你的视频有已知内参（真实相机拍摄），用 `FX/FY/CX/CY` 覆盖默认值即可。

### Step 02 — 2D Gaussian Splatting 训练 (`02_train_2dgs.sh` → 2DGS `train.py`)

2DGS（SIGGRAPH 2024）用 **2D oriented disks**（surfel）代替 3D 椭球，做 perspective-correct 的可微光栅化，相比 vanilla 3DGS 在**几何精度**（表面、法向、深度）上更好。两个核心正则：

- `--lambda_normal`（默认 0.05）— 法向一致性：相邻 surfel 法向接近。
- `--lambda_distortion`（默认 100）— 深度畸变：同一射线被多个 surfel 命中时深度一致。

依赖：`diff-surfel-rasterization`（CUDA 光栅化，hbb1/diff-surfel-rasterization）、`simple-knn`（初始点云，gitlab.inria.fr/bkerbl/simple-knn）。两个都在 2DGS 仓的 `submodules/` 下，`00_setup_env.sh` 用 `BUILD_CUDA=1` 编译。

### Step 03 — 渲染 + 网格提取 (`03_render_2dgs.sh` → 2DGS `render.py`)

2DGS 的 `render.py` 有两种模式：

1. **渲染相机**：对 train/test 相机光栅化输出 `renders/*.png` + `gt/*.png` + `depth/*.png`，用于评测。
2. **TSDF fusion 提网格**（2DGS 招牌功能）：
   - **bounded**（默认）：在 bounded 体积内做 TSDF fusion，需调 `--depth_trunc` / `--voxel_size`。2DGS 自带自动估计（基于相机包围盒）。
   - **`--unbounded`**：空间收缩到球内 + 自适应 TSDF 截断，适合场景延伸到无限远处（典型：人像在白色虚空中环绕的 wan22_rotate 视频）。**推荐对 wan22_rotate 输入用 `UNBOUNDED=1`**。
   - `--mesh_res`：TSDF 体素分辨率（默认 1024，越高网格越细但更耗内存）。

## Config (env vars, all optional)

### Paths & envs
| var | default | note |
| --- | --- | --- |
| `INPUT` | `../wan22_rotate_results/rotate_360.mp4` | 输入视频 / 图像文件夹（run_all 默认接 wan22_rotate） |
| `CONDA_ENV` | `pi3_3dgs` | conda env（CPython 3.10，Pi3 + 2DGS 共用） |
| `GPU` | _(unset)_ | physical GPU id，e.g. `GPU=0` |
| `PI3_DIR` | `../Pi3` | Pi3 官方代码 |
| `PI3_MODEL_DIR` | `../../model/Pi3` | Pi3 权重根 |
| `PI3_CKPT` | `$PI3_MODEL_DIR/model.safetensors` | Pi3 checkpoint 文件 |
| `GS2D_DIR` | `../2d-gaussian-splatting` | 2DGS 官方代码 |
| `WAN_MODEL_DIR` | `../../model` | 共享模型根（00 找本地 torch wheel 用） |
| `CUDA_HOME` | `/usr/local/cuda` | CUDA toolkit 根（编 CUDA ext 用；须 cu124） |
| `RESULTS_DIR` | `../pi3_3dgs_results` | 输出根 |

### Step 01 params (Pi3 + COLMAP)
| var | default | note |
| --- | --- | --- |
| `FRAME_FPS` | `10` | 视频抽帧 fps（Pi3 显存随 N 线性增长：~6GB @ 1 帧，~13GB @ 100，~43GB @ 500） |
| `FRAME_MAX` | `60` | 最大保留帧数（防 OOM；6 秒 @ 10fps ≈ 60 帧，足够一圈 360°） |
| `CONF_THRES` | `0.1` | `sigmoid(conf)` 阈值，过滤低置信像素 |
| `MAX_POINTS` | `100000` | 写入 `points3D.txt` 的最大点数（3DGS 会致密化，无需太多） |
| `FX` / `FY` | `max(W,H)` | PINHOLE 焦距（覆盖默认假设） |
| `CX` / `CY` | `W/2` / `H/2` | PINHOLE 主点 |
| `DEVICE` | `cuda` | 或 `cpu`（很慢） |

### Step 02 params (2DGS train)
| var | default | note |
| --- | --- | --- |
| `SOURCE_DIR` | `$RESULTS_DIR/source` | 01 输出的 COLMAP 场景（`-s`） |
| `MODEL_DIR` | `$RESULTS_DIR/model` | 训练输出（`-m`） |
| `ITERATIONS` | `30000` | 2DGS 论文默认；7000 = 快速 demo |
| `WHITE_BG` | `0` | `1` = 训练用白底（wan22_rotate 输入推荐） |
| `PORT` | `0` | viser viewer 端口（0 = 关闭） |
| `EXTRA_ARGS` | _(unset)_ | 转发给 train.py，e.g. `--lambda_normal 0.05` |

### Step 03 params (2DGS render)
| var | default | note |
| --- | --- | --- |
| `MODEL_DIR` | `$RESULTS_DIR/model` | 训练好的场景（`-m`） |
| `SOURCE_DIR` | `$RESULTS_DIR/source` | COLMAP 场景（`-s`） |
| `ITERATION` | `-1` | `-1` = 最新；或指定 step（如 30000） |
| `MESH_RES` | `1024` | TSDF 体素分辨率（mesh 质量 vs 内存） |
| `UNBOUNDED` | `0` | `1` = 无界 TSDF（推荐 wan22_rotate 输入） |
| `DEPTH_RATIO` | `0` | `0` = 均值深度（多数场景），`1` = 中值 |
| `SKIP_TRAIN` | `1` | `1` = 跳过 train 渲染（默认只渲 test） |
| `SKIP_TEST` | `0` | `1` = 跳过 test（只出 mesh） |
| `EXTRA_ARGS` | _(unset)_ | 转发给 render.py |

## 可能遇到的问题

**1. `01` 报 `OOM` (Pi3 显存不够)**
Pi3 显存随帧数线性增长。降压：
- 减帧：`FRAME_FPS=5` 或 `FRAME_MAX=30`
- 用更小分辨率（脚本未暴露；改 `pi3_recon.py` 调用 `load_images_as_tensor` 前 resize）
- A100 40G 上 ~60 帧可行；80G 可到 120 帧。

**2. `01` 写出的 `points3D.txt` 为空 / 几乎空**
置信度过滤太严。降低阈值：`CONF_THRES=0.05`（默认 0.1）。若 dense_cloud.ply 也几乎空，说明 Pi3 对该视频预测失败——通常是因为帧间重叠太少（环绕视频应至少 30+ 帧覆盖一圈）。

**3. `02` 训练发散 / 渲染全黑**
- 检查相机位姿是否合理：用 `dense_cloud.ply` + 任何 viewer 看（MeshLab / SuperSplat）。点云应当大致是人形/场景形状。
- Pi3 的 c2w 旋转约定可能与 COLMAP 相反——目前假设都是 OpenCV（z 前 y 下）。若点云方向反了，在 `pi3_recon.py` 里翻转 c2w 的某个轴即可。
- 默认假设内参可能不准。若视频是真实相机拍摄且有已知内参，用 `FX/FY/CX/CY` 覆盖。
- 加大正则：`EXTRA_ARGS="--lambda_normal 0.1 --lambda_distortion 1000"`

**4. `02` 报 `No module named 'diff_surfel_rasterization'` / `'simple_knn'`**
两个 CUDA 子模块没编。`BUILD_CUDA=1 bash pi3_3dgs/00_setup_env.sh`（需 `CUDA_HOME` 指向 CUDA 12.4 toolkit；torch 是 cu124）。

**5. `00` 编 `diff-surfel-rasterization` 报 `nvcc not found` / `CUDA_HOME` 错**
rasterizer 的 `setup.py` 用 `torch.utils.cpp_extension`，需要 `nvcc`（CUDA toolkit）。且 toolkit 大版本要和 torch 的 CUDA 版本对齐：
- `torch==2.6.0+cu124` → 装 CUDA 12.4 toolkit，`export CUDA_HOME=/usr/local/cuda-12.4`
- 验证：`$CUDA_HOME/bin/nvcc --version`
- 版本不匹配会编不过或运行时段错。

**6. `00` 编译报 `error: no member named '...' in 'at::...'` / ABI 不匹配**
torch 版本和 rasterizer 代码不兼容。`diff-surfel-rasterization` 的最新 commit 支持 torch 2.x；若拉到旧 commit 会失败。在 `2d-gaussian-splatting` 目录跑 `git submodule update --remote` 拉最新版重编。

**7. `01` 报 `import cv2` / `import safetensors` 失败**
env 缺 Pi3 依赖。`INSTALL_DEPS=1 bash pi3_3dgs/00_setup_env.sh` 重装。

**8. `03` 出来的 mesh 是一团乱 / 中空**
- 试 `UNBOUNDED=1`（无界 TSDF），特别是 wan22_rotate 输入（人像在白色虚空中）。
- 调 `MESH_RES`：升到 2048（更细）或降到 512（更粗但稳）。
- 调 `DEPTH_RATIO=1`（中值深度，有时更稳）。
- 训练步数不够：`ITERATIONS=30000`（默认）或更高 50000。

**9. 训练 OOM（显存不足）**
2DGS 显存随分辨率 + 高斯数增长。降压：
- 减帧（间接降低复杂度）：`FRAME_MAX=30`
- 提前停：`ITERATIONS=7000`
- `export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`

**10. 跑 `.sh` 报 `syntax error near unexpected token ('`**
CRLF 行尾污染。`find pi3_3dgs -name '*.sh' -exec sed -i 's/\r$//' {} +` 或 `git checkout -- pi3_3dgs/*.sh`（`.gitattributes` 强制 LF）。

**11. NumPy 坏了 / `import numpy` 报 ABI 不兼容 / python 变成了 GraalPy**
（同 wan22_rotate 排错 #8）`conda install` 任何包都加 `--no-update-deps`，否则 conda solver 可能把 python 掉包成 GraalPy。本仓 `00_setup_env.sh` 不用 `conda install` 装其他包（只用 pip），且建 env 时显式校验 CPython。若已坏，重建：
```bash
python -c "import platform; print(platform.python_implementation())"  # 输出 GraalPy 即中招
conda env remove -n pi3_3dgs
conda create -n pi3_3dgs python=3.10 -y && conda activate pi3_3dgs
INSTALL_DEPS=1 BUILD_CUDA=1 bash pi3_3dgs/00_setup_env.sh
```

## 目录布局
```
<code-dir>/
├── media_code/                  # 本仓
│   ├── proxy.env                # 代理 + 覆盖项, gitignored
│   ├── wan22_rotate/            # 上游：选图+分割 → rotate_360.mp4
│   └── pi3_3dgs/                # ← 本目录（编排脚本）
│       ├── _env.sh
│       ├── 00_setup_env.sh
│       ├── 01_pi3_recon.sh
│       ├── pi3_recon.py          #   Pi3 推理 + COLMAP 导出
│       ├── 02_train_2dgs.sh
│       ├── 03_render_2dgs.sh
│       └── run_all.sh
├── Pi3/                          # Pi3 官方代码 (00 clone)
├── 2d-gaussian-splatting/        # 2DGS 官方代码 (00 clone, 含 submodules/)
│   └── submodules/
│       ├── simple-knn/                  # gitlab.inria.fr/bkerbl/simple-knn (CUDA)
│       └── diff-surfel-rasterization    # hbb1/diff-surfel-rasterization (CUDA)
└── pi3_3dgs_results/             # 本流程输出
    ├── frames/                   # 抽帧 / 复制的源图
    ├── predictions.npz           # 原始 Pi3 输出
    ├── dense_cloud.ply           # 调试用稠密点云
    ├── source/                   # 2DGS source 目录
    │   ├── images/                #   训练图
    │   └── sparse/0/              #   COLMAP 文本格式
    │       ├── cameras.txt
    │       ├── images.txt
    │       └── points3D.txt
    └── model/                    # 2DGS 训练产物
        ├── point_cloud/iteration_<N>/point_cloud.ply   # 高斯点云
        ├── cfg_args                                    # train.py 写入的参数
        ├── cameras.json                                # 相机序列化
        └── test/ours_<N>/                              # 渲染 + 网格
            ├── renders/*.png
            ├── gt/*.png
            └── mesh.ply
```

## Notes
- 官方代码 & 权重遵循各自 license（Pi3 = BSD-3-Clause 代码 + CC BY-NC 4.0 权重；2DGS = research-only）。本目录只编排；未拷贝官方代码。
- Pi3 的 checkpoint 是前馈权重，**严禁商用**（CC BY-NC 4.0）；2DGS 同样仅限研究用途。
- `.gitattributes`（仓根）强制 LF，让 Windows 推送的脚本在 Ubuntu 上干净运行。
- `proxy.env`（代理凭证 + 路径 / env 覆盖）在仓内 gitignored——从不入库。切勿把凭证写进脚本。
- 公司 TLS 中间人代理下：pip 用 `--trusted-host` / `PIP_CERT`；`git` 用 `GIT_SSL_CAINFO`；`curl` 用 `CURL_CA_BUNDLE`（`_env.sh` 优先用 `~/.ca-bundle.crt`，由 `hypir/setup_ca_bundle.sh` 或等价脚本构建）。
- 与 [`vggt-omega/`](../vggt-omega/) 的区别：VGGT-Omega 也是 feed-forward 出位姿+点云，但只出**点云**（无高斯属性），无法直接做新视角合成；本流程接 2DGS 后能渲染新视角 + 导网格。VGGT-Omega 的 03 是把点云 splat 成视频，本流程的 03 是真正的 2DGS 高斯渲染。
- 与 [`deformable_gaussians/`](../deformable_gaussians/) 的区别：Deformable-3D-GS 是**动态场景**方法（D-NeRF / NeRF-DS），需时序 fid 输入；2DGS 是**静态场景**方法，更适合 wan22_rotate 的「环绕静态人物」视频。若你的输入是真实动态视频（人物动作），改用 deformable_gaussians。

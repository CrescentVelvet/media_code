# VGGT-Omega — 多视图图像前馈三维重建（位姿 + 深度 → 彩色点云）

用 [VGGT-Omega (CVPR 2026 Oral)](https://github.com/facebookresearch/vggt-omega) 从一组**场景图像**（或视频）在**单次前向推理**中预测 per-view **相机位姿 + 深度图**（+ 置信度），反投影成彩色点云。它是 **feed-forward** 模型（一次模型推理 → 所有视图的位姿和深度），**不**优化场景表示——没有 per-scene fitting、没有 novel-view 高斯。`03_render_video.sh` 因此是把重建出的点云沿螺旋路径 splat 成 mp4，而非光栅化高斯。

> 本目录与 [`pdfgs_human/`](../pdfgs_human/) 并列但**范式不同**：pdfgs_human 是"真实拍摄序列 → PDF-GS 抗微动 3DGS 重建"（优化场景表示，出高斯 + 渲染）；本目录是"**多视图图像 → 前馈重建**"（一次推理出位姿 + 深度 → 点云，无优化）。两者不共用 env（本目录复用已有 `doll` env，要求 torch>=2.3；pdfgs_human 用 `pdfgs` env，torch 2.5.1+cu121）。

## 为什么用前馈重建（而不是优化式 3DGS）

VGGT-Omega 的核心是**一次前向推理**即可从任意数量视图恢复相机位姿 + 深度，这决定了它适用与不适用的场景：

1. **快速场景重建**：不需要特征匹配 / SfM / bundle adjustment / per-scene 训练——喂一组图，一次前向就出位姿 + 点云。几秒到几十秒出结果（视帧数），适合**快速预览 / 粗重建 / 后续 pipeline 的位姿初始化**。

2. **无 novel-view 高斯**：VGGT-Omega 不优化场景表示，输出的是反投影的**彩色点云**（XYZ+RGB，无高斯属性）。要 novel-view 合成 / 3DGS 质量渲染，用 pdfgs_human（优化式 3DGS）或 wan22_rotate（合成视频 → 3DGS）。`03_render_video.sh` 的螺旋视频只是把点云 splat 出来给人看，不是高斯光栅化。

**结论**：要快速位姿 + 粗点云 → 用 VGGT-Omega；要高质量 novel-view 重建 → 用 pdfgs_human。两者互补，不冲突。

## 常用命令

> 假设已进入容器（脚本自动激活 `doll` env）；`GPU=0` 按需换卡。首次跑前先做下方「首次准备」。
> **铁律：每条命令都必须显式写出模型路径、输入路径、输出路径，不能全靠脚本里的默认值。** 用具体路径，不要用 `...` 占位。

```bash
# ── 分步 ──
# 0) 激活 env + 校验 torch (INSTALL_DEPS=1 一次装齐已知依赖, torch 不重装)
GPU=0 INSTALL_DEPS=1 bash vggt-omega/00_setup_env.sh

# 1) 下权重 (gated HF 仓库, 需 HF_TOKEN + 访问已批准)
GPU=0 VARIANT=1b_512 \
  MODEL_DIR=../../model/VGGT-Omega \
  bash vggt-omega/01_download_models.sh

# 输出：<MODEL_DIR>/vggt_omega_1b_512.pt   (512px 变体)
#   或 VARIANT=1b_256_text -> vggt_omega_1b_256_text.pt   (256px text-aligned 变体)

# 2) 前馈重建 (图像/视频/场景文件夹 -> 位姿+深度 -> 点云)
GPU=0 INPUT_DIR=../Reconstruction/dataset/B003_Human_Data_w_pose/test_task_id_3a8b3cc746304f49b9e3275e36aa9374 \
  MODEL_DIR=../../model/VGGT-Omega \
  OUTPUT_DIR=../../output/vggt_omega_results \
  VARIANT=1b_512 RESOLUTION=512 \
  bash vggt-omega/02_run_inference.sh

# 输出：<OUTPUT_DIR>/<scene>/
#   scene.ply            # 置信度过滤后的彩色点云 (原始世界坐标)
#   predictions.npz      # 原始模型输出 (depth, depth_conf, extrinsic, intrinsic, ...)
#   scene.glb            # 官方可视化 (点云 + 相机锥), 需 trimesh/scipy
#   frames/              # 实际喂给模型的图 (复制/抽帧)

# 3) 点云渲染视频 (.ply -> 螺旋环绕 mp4, gsplat splatting)
GPU=0 PLY_INPUT=../../output/vggt_omega_results/image \
  VIDEOS_DIR=../../output/vggt_omega_results/videos \
  bash vggt-omega/03_render_video.sh

# 输出：<VIDEOS_DIR>/image/scene.mp4  (+ scene.png 首帧, 用于快速检查相机朝向)
```

- 结果：点云 → `OUTPUT_DIR/<scene>/scene.ply`；原始预测 → `predictions.npz`；可视化 → `scene.glb`；螺旋展示视频 → `VIDEOS_DIR/<scene>/scene.mp4`。

## 首次准备

本流程**复用已有的 `doll` conda env**（要求 torch>=2.3，CUDA 可用），不为 VGGT-Omega 单建 env——避免重下 torch（几 GB）。runtime 依赖（numpy/einops/safetensors/opencv/scipy/trimesh/gsplat 等）按需装，或 `INSTALL_DEPS=1` 一次装齐（torch 不重装）。`vggt_omega` 包通过 `sys.path` 导入（`run_batch.py` 加 `VGGT_DIR`），无需 `pip install -e .`。

> ⚠️ VGGT-Omega 权重是 **gated** 仓库（`facebook/VGGT-Omega`），需先申请访问 + 建 HF read token。这是与 pdfgs_human（DINOv3 可用本地 vitl16 免 token）的不同之处。

```bash
cd <your-code-dir>            # e.g. /data_3d/<uid>/code
git -c http.sslVerify=false clone https://github.com/CrescentVelvet/media_code.git
cd media_code && cp proxy.env.example proxy.env   # 填 http_proxy / https_proxy
# ⚠️ 确认 proxy.env 中 http_proxy / https_proxy 两行已取消注释并填好地址,
#    否则 pip / hf 下载会报 "Network is unreachable"

# 申请 gated 权重访问 (一次性):
#   1. 到 https://huggingface.co/facebook/VGGT-Omega 点 "Request access" (自动审核, 通常很快)
#   2. 到 https://huggingface.co/settings/tokens 建一个 read token
#   3. 写入 proxy.env:  echo 'export HF_TOKEN=hf_xxx' >> proxy.env

# 建好 doll env (若还没有): torch>=2.3 + CUDA
# conda create -n doll python=3.10 -y && conda activate doll
# pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
# python -c "import torch;print(torch.__version__, torch.cuda.is_available())"  # 需 >=2.3 + True

# 装依赖 + clone 官方代码 + 下权重 (一次性)
GPU=0 INSTALL_DEPS=1 bash vggt-omega/00_setup_env.sh
GPU=0 VARIANT=1b_512 MODEL_DIR=../../model/VGGT-Omega bash vggt-omega/01_download_models.sh
```

需系统有 NVIDIA GPU（A100 40G/80G 理想）。显存随帧数线性增长：1 帧 ~6GB，100 帧 ~13GB，500 帧 ~43GB（624×416）。用 `RESOLUTION=256` 或 `MODE=max_size` 降显存。

### 权重目录布局

```
$MODEL_DIR/                         # 默认 ../../model (code-dir 上一级, 各算法共享)
  VGGT-Omega/                        # VGGT-Omega checkpoint (HF gated 下载)
    vggt_omega_1b_512.pt             # 默认变体 (512px, 无 text alignment)
    vggt_omega_1b_256_text.pt        # 256px text-aligned 变体 (读 text_alignment_embedding)
```

外部 clone 的官方代码（`00` 自动 clone，sibling of media_code）：
```
<code-dir>/
  media_code/vggt-omega/             # 本目录 (编排脚本)
  vggt-omega/                        # VGGT-Omega 官方代码 (含 vggt_omega/ 包 + visual_util.py)
```

---

以下为详细参考（流程原理 / 各步骤参数 / 排错 / 目录布局）。

## Pipeline（流程详解）

```
INPUT_DIR/                           (一组场景图像 / 视频 / 场景文件夹)
    │
    ▼
[02] VGGT-Omega 前馈推理 (doll env)  — 一次前向 → 所有视图位姿 + 深度图
    │  ├─ discover_scenes: 图像文件夹 -> 1 场景; 视频文件夹 -> 逐视频; 场景文件夹 -> 批量
    │  ├─ load_and_preprocess_images (balanced/max_size, RESOLUTION)
    │  ├─ VGGTOmega(images) -> pose_enc + depth + depth_conf (一次推理, 所有视图)
    │  ├─ encoding_to_camera(pose_enc) -> 外参 (N,3,4) + 内参 (N,3,3)
    │  ├─ unproject_depth_map_to_point_map -> 世界坐标点云
    │  └─ filter_points: 置信度百分位 + 深度边缘 + 可选背景过滤 -> scene.ply
    ▼
$OUTPUT_DIR/<scene>/
    scene.ply            # 置信度过滤后的彩色点云 (原始世界坐标; MeshLab/SuperSplat 看)
    predictions.npz      # 原始输出 (depth, depth_conf, extrinsic, intrinsic,
                         #   world_points_from_depth, images, pose_enc, ...) — 同官方 demo
    scene.glb            # 官方可视化 (点云 + 相机锥), visual_util.predictions_to_glb [需 trimesh]
    frames/              # 实际喂给模型的图 (从输入复制, 或从视频抽帧)
    │
    ▼
[03] 点云渲染视频 (doll env) — .ply -> 沿螺旋路径 splat -> mp4 (gsplat)
    │  ├─ 加载 .ply -> XYZ + RGB (无高斯属性)
    │  ├─ 每点 splat 为小各向同性高斯 (POINT_SCALE × 场景范围, 不透明度 1)
    │  ├─ 螺旋相机轨迹 (TURNS 圈, ELEV 仰角) -> 逐帧 rasterization
    │  └─ imageio -> mp4 + 首帧 png (检查相机朝向)
    ▼
$VIDEOS_DIR/<scene>/scene.mp4   (螺旋环绕展示视频)
$VIDEOS_DIR/<scene>/scene.png   (首帧, 快速检查)
```

### Step 00 — clone 官方仓 + 激活 env + 校验 torch (`00_setup_env.sh`)

脚本先 clone VGGT-Omega 官方仓到 `$VGGT_DIR`（若不存在），再 source `_env.sh`（代理 + CA bundle + conda activate + GPU 选卡），然后内联 Python 校验 `torch.cuda.is_available()` + 版本>=2.3（VGGT-Omega 要求）。`INSTALL_DEPS=1` 时一次装齐已知 runtime 依赖：核心（numpy<2, Pillow, einops, safetensors, opencv-python）、点云导出（scipy, trimesh, matplotlib, tqdm）、渲染（plyfile, imageio, imageio-ffmpeg, gsplat）。`vggt_omega` 包通过 `sys.path` 导入（`run_batch.py` 加 `VGGT_DIR`），无需 `pip install -e .`。

### Step 01 — 下权重 (`01_download_models.sh`)

从 HuggingFace `facebook/VGGT-Omega`（**gated**）下载 checkpoint。`VARIANT` 选变体：`1b_512`（512px，默认，无 text alignment）或 `1b_256_text`（256px，text-aligned，读 `predictions["text_alignment_embedding"]`；`02` 自动开 `VGGTOmega(enable_alignment=True)`）。只下选中的那个 `.pt` 文件（省带宽）。`HF_TOKEN` 必须设在 `proxy.env`，否则脚本拒绝运行。下载先试 `hf download`（走 CA bundle），SSL 失败后自动回退到 `_hf_download.py`（`snapshot_download` 禁用 SSL 校验）；`HF_DISABLE_SSL=1` 跳过首次尝试直接走回退。

### Step 02 — 前馈重建 (`02_run_inference.sh` → `run_batch.py`)

模型**加载一次**，跨场景复用（`run_batch.py`）。`INPUT_DIR` 支持四种输入（`discover_scenes`）：

- **图像文件夹**（jpg/png/webp/…）→ 一个场景，以文件夹名命名
- **视频文件**（.mp4/.mov/…）→ 一个场景，按 `VIDEO_FPS` 自动抽帧
- **场景文件夹的文件夹**（每个子夹有图，直接放或 `<子夹>/images/` 下）→ 批量，每子夹一个重建
- **视频文件夹** → 批量，每视频一个重建

每个场景：图复制/抽帧到 `frames/` → `load_and_preprocess_images`（`MODE=balanced` 用 token 预算调大小，`MODE=max_size` 限最长边=RESOLUTION 省显存）→ `VGGTOmega(images)` 一次前向出 `pose_enc` + `depth` + `depth_conf` → `encoding_to_camera` 出外参+内参 → `unproject_depth_map_to_point_map` 反投影 → `filter_points`（置信度百分位过滤 + 深度边缘过滤 + 可选背景过滤 + `MAX_POINTS` 上限）→ `scene.ply` + `predictions.npz` + `scene.glb`（需 trimesh/scipy，缺则跳过）。

> `predictions.npz` 的 key 与官方 `demo_gradio.py` 保存的一致：`depth`, `depth_conf`, `extrinsic`, `intrinsic`, `world_points_from_depth`, `images`, `pose_enc`, `camera_and_register_tokens`。

### Step 03 — 点云渲染视频 (`03_render_video.sh` → `render_video.py`)

VGGT-Omega 输出的点云是纯 XYZ+RGB（无高斯属性），`render_video.py` 把每点 splat 为一个小各向同性高斯（`POINT_SCALE × 场景范围`，不透明度 1，单位旋转），沿螺旋相机轨迹（`TURNS` 圈，`ELEV` 仰角）用 gsplat `rasterization` 逐帧渲染 → mp4 + 首帧 png。输出路径镜像 02：`VIDEOS_DIR/<scene>/<stem>.mp4`。

> **相机朝向调参**：先跑一次看 `scene.png` 首帧——若歪了，脚本会打印 `image-up (world)` 向量，设 `UP_VEC` 为它（`ROLL=0`）得到不倾斜的环绕；`UP_AXIS`（x/y/z）是轴对齐快捷方式。默认 `UP_VEC="0 -1 0"`（OpenCV 相机约定：y-down → up 是 −Y），适合前向移动相机素材。全黑帧用 `VIEWMAT_C2W=1` 或 `BG=0.5` 调试。点云显稀疏（dotty）调大 `POINT_SCALE`；糊了（blobby）调小。

## Config (env vars, all optional)

### Paths & envs
| var | default | note |
| --- | --- | --- |
| `INPUT_DIR` | `../vggt-omega/examples` | 图像文件夹 / 视频文件 / 场景文件夹（见 Step 02） |
| `GPU` | _(unset)_ | physical GPU id, e.g. `GPU=0` |
| `CONDA_ENV` | `doll` | conda env（torch>=2.3 预装；复用不重下 torch） |
| `VGGT_DIR` | `../vggt-omega` | 官方代码路径（00 自动 clone） |
| `VGGT_REPO` | `https://github.com/facebookresearch/vggt-omega.git` | clone 源 |
| `MODEL_DIR` | `../../model/VGGT-Omega` | checkpoint 路径（gated HF 下载） |
| `HF_REPO_ID` | `facebook/VGGT-Omega` | gated 权重仓库（需 `HF_TOKEN`） |
| `OUTPUT_DIR` | `../vggt_omega_results` | 重建输出根；每场景 → `OUTPUT_DIR/<scene>/` |
| `VIDEOS_DIR` | `../vggt_omega_results/videos` | 视频输出根；mp4 → `VIDEOS_DIR/<scene>/` |
| `INSTALL_DEPS` | `0` | `1` = 在 00 一次装齐已知 runtime 依赖（torch 不重装） |
| `HF_HUB_DISABLE_XET` | `1` | 禁 HF Xet/CAS Rust 通道（不认代理 CA） |
| `HF_DISABLE_SSL` | `0` | `1` = 跳过 `hf download` 首次尝试, 直接走禁 SSL 校验的 `_hf_download.py` |

### Step 02 params
| var | default | note |
| --- | --- | --- |
| `VARIANT` | `1b_512` | checkpoint 变体: `1b_512`（512px）或 `1b_256_text`（256px, text-aligned; 02 自动开 `enable_alignment`） |
| `RESOLUTION` | `512` | 输入图像分辨率（`1b_256_text` 用 `256`） |
| `MODE` | `balanced` | resize 模式: `balanced`（token 预算）或 `max_size`（最长边=RESOLUTION, 省显存） |
| `CONF_THRES` | `20` | 深度置信度百分位保留（0–100; 高 = 更稀疏但更干净） |
| `MAX_POINTS` | `2000000` | `scene.ply` 点数上限（0 = 不限） |
| `MASK_SKY` / `MASK_BLACK_BG` / `MASK_WHITE_BG` | `0` | 可选点过滤（sky 需 onnxruntime skyseg） |
| `VIDEO_FPS` | `1` | `INPUT_DIR` 为视频时的抽帧 fps |
| `DEVICE` | `cuda` | 推理设备 |

### Step 03 params
| var | default | note |
| --- | --- | --- |
| `PLY_INPUT` | `../vggt_omega_results` | .ply 文件或文件夹（渲染 03） |
| `WIDTH` × `HEIGHT` | `1280` × `720` | 渲染分辨率 |
| `TURNS` / `ELEV` / `FRAMES` / `FPS` | `1` / `-15°` / `120` / `30` | 螺旋轨迹参数 |
| `START_ANGLE` | `0` | 起始方位角（度） |
| `FOV` | `55` | 相机视场角（度） |
| `RADIUS_SCALE` | `1.15` | 相机距离 = `半径 / tan(FoV/2) × RADIUS_SCALE` |
| `UP_AXIS` | `y` | 相机 up 轴（x/y/z） |
| `UP_VEC` | `0 -1 0` | 物体 up 向量 "x y z"（覆盖 `UP_AXIS`）；设为首帧 `image-up` 向量得到不倾斜环绕 |
| `ROLL` | `0` | 相机绕前向轴 roll（度）；场景歪了试 90/-90/180 |
| `POINT_SCALE` | `0.002` | splat 半径占场景范围比例（dotty 调大, blobby 调小） |
| `VIEWMAT_C2W` | `0` | `1` = view matrix 用 c2w 约定（全黑帧调试用） |
| `BG` | `0.0` | 背景（0=黑, 0.5=灰, 1=白；全黑帧调试用） |

## 可能遇到的问题

**0. clone 本仓报 SSL / 认证**
公开仓免认证，加 `-c http.sslVerify=false` 即可（见上文「首次准备」）。若克隆的是私有仓且提示不能用账号密码，是 GitHub 已停用密码认证——改用公开仓或只读 PAT。

**1. `01` 报 `HF_TOKEN not set` / `401 Unauthorized`**
VGGT-Omega 权重是 gated 仓库。先在 https://huggingface.co/facebook/VGGT-Omega 申请访问（自动审核），再在 https://huggingface.co/settings/tokens 建一个 read token，写入仓根 `proxy.env`：
```bash
echo 'export HF_TOKEN=hf_xxx' >> proxy.env
GPU=0 VARIANT=1b_512 MODEL_DIR=../../model/VGGT-Omega bash vggt-omega/01_download_models.sh
```
若已带 token 仍 `401`，多半是访问尚未批准，等几分钟再试。

**2. `hf download` 报 `CAS service error : ReqwestMiddleware`**
HF 的 Xet/Rust 通道不认代理。`_env.sh` 已设 `HF_HUB_DISABLE_XET=1`；仍报则彻底卸载：
```bash
pip uninstall -y hf_xet
GPU=0 VARIANT=1b_512 MODEL_DIR=../../model/VGGT-Omega bash vggt-omega/01_download_models.sh
```

**3. `hf download` 报 `SSLCertVerificationError`**
代理根 CA 不在系统证书包。先一次性建包，`_env.sh` 会自动用 `~/.ca-bundle.crt`：
```bash
bash vggt-omega/setup_ca_bundle.sh    # 抓代理证书链 -> ~/.ca-bundle.crt, 并自检
GPU=0 VARIANT=1b_512 MODEL_DIR=../../model/VGGT-Omega bash vggt-omega/01_download_models.sh
```
- 自检 `[OK]` → 直接重跑 `01`。
- 自检 `[FAIL]`（代理握手没带根 CA）→ 把公司根 CA 追加后再重跑：
  ```bash
  cat /path/to/corporate_root_ca.crt >> ~/.ca-bundle.crt
  ```
  公司根 CA 常见于 `/usr/local/share/ca-certificates/`（脚本已自动并入该目录）。

> 若建包后仍报 SSL（CDN 端点 `us.aws.cdn.hf.co` 等用了不同的 MITM 证书），`01` 会自动回退到禁用 SSL 校验的下载器（`_hf_download.py`）；或直接 `HF_DISABLE_SSL=1 bash vggt-omega/01_download_models.sh` 跳过首次尝试。代理已全程 MITM，此处关掉校验可接受。

**4. pip 装 torch 报 `SSL:CERTIFICATE_VERIFY_FAILED`**
脚本已内置 `--trusted-host`。仍失败时手动信任：
```bash
pip config set global.trusted-host "pypi.org pypi.python.org files.pythonhosted.org download.pytorch.org"
GPU=0 INSTALL_DEPS=1 bash vggt-omega/00_setup_env.sh
```

**5. pip 装 torch 报 `HTTPSConnectionPool`（连接超时/断开）**
不是 torch 版本问题，是代理对大文件超时。加超时重试，或退回 PyPI 默认 torch（自带 CUDA，A100 可用）：
```bash
pip install --timeout 600 --retries 10 --trusted-host download.pytorch.org \
  --index-url https://download.pytorch.org/whl/cu118 torch torchvision
python -c "import torch;print(torch.cuda.is_available(), torch.__version__)"  # 需 True + >=2.3
```

**6. `torch.cuda.OutOfMemoryError`**
显存随帧数线性增长。降压：`RESOLUTION=256`、`MODE=max_size`，或喂更少帧（视频则降 `VIDEO_FPS`）。`run_batch.py` 会捕获 OOM 并继续下一个场景。

**7. 缺包 `ModuleNotFoundError`**
按需补，或一次性装齐已知小依赖：
```bash
pip install <包名>
# 或:
GPU=0 INSTALL_DEPS=1 bash vggt-omega/00_setup_env.sh
```
`scene.glb` 导出需要 `trimesh scipy matplotlib`（缺则跳过，仅产出 `.ply`/`.npz`）。`03` 渲染需要 `gsplat plyfile imageio imageio-ffmpeg`。

**8. 跑 `.sh` 报 `syntax error near unexpected token '('`**
CRLF 行尾污染。`find vggt-omega -name '*.sh' -exec sed -i 's/\r$//' {} +` 或 `git checkout -- vggt-omega/*.sh`（`.gitattributes` 强制 LF）。

> 通用：`proxy.env`（代理凭证 + `HF_TOKEN`）在仓内 gitignored，`~/.ca-bundle.crt` 在家目录，都不入库；切勿把凭证写进脚本。

## 目录布局
```
<code-dir>/
├── media_code/                     # 本仓
│   ├── proxy.env                   # 代理 + HF_TOKEN + 覆盖项, gitignored
│   └── vggt-omega/                 # ← 本目录（编排脚本）
│       ├── _env.sh                 # 共享: 代理 + CA bundle + conda activate + GPU
│       ├── 00_setup_env.sh        # clone 官方仓 + 激活 env + 校验 torch (INSTALL_DEPS=1 装依赖)
│       ├── 01_download_models.sh   # gated HF 下载 checkpoint
│       ├── 02_run_inference.sh     # 前馈重建 (图像/视频 -> 点云)
│       ├── 03_render_video.sh      # .ply -> 螺旋 mp4 (gsplat)
│       ├── run_batch.py            # 批量重建 (加载模型一次, 循环场景)
│       ├── render_video.py         # 螺旋点云渲染器 (gsplat + imageio-ffmpeg)
│       ├── setup_ca_bundle.sh      # 一次性: 抓代理证书链 -> ~/.ca-bundle.crt
│       ├── _extract_ca.py          #   helper (setup_ca_bundle.sh 调用)
│       └── _hf_download.py         #   snapshot_download 禁 SSL 校验 (01 回退)
├── vggt-omega/                     # VGGT-Omega 官方代码 (00 自动 clone)
│   ├── vggt_omega/                 #   模型包 (models/, utils/)
│   ├── visual_util.py              #   官方 .glb 导出 (predictions_to_glb)
│   └── examples/                   #   自带示例视频
├── model/                          # 权重根 (code-dir 上一级, 共享)
│   └── VGGT-Omega/                 #   checkpoint (gated HF 下载)
│       ├── vggt_omega_1b_512.pt     #   512px 变体 (默认)
│       └── vggt_omega_1b_256_text.pt #  256px text-aligned 变体
└── output/vggt_omega_results/      # 输出 (repo 外)
    ├── <scene>/                    # step 02: 重建
    │   ├── scene.ply               #   彩色点云 (置信度过滤)
    │   ├── predictions.npz          #   原始模型输出
    │   ├── scene.glb               #   官方可视化 (点云 + 相机锥)
    │   └── frames/                 #   喂给模型的图 (复制/抽帧)
    └── videos/                     # step 03: 渲染视频
        └── <scene>/
            ├── scene.mp4           #   螺旋环绕展示视频
            └── scene.png           #   首帧 (检查相机朝向)
```

## Notes
- VGGT-Omega is **feed-forward** (one model pass → poses + depth for all input views). It does **not** optimize a scene representation, so there is no train/test split, no per-scene fitting, and no novel-view gaussians — the orbit video in `03` just splats the unprojected point cloud.
- Two checkpoints exist: `vggt_omega_1b_512.pt` (default, 512px, no text alignment) and `vggt_omega_1b_256_text.pt` (256px, text-aligned; reads `predictions["text_alignment_embedding"]`). `01`/`02` select via `VARIANT`; `02` auto-enables `VGGTOmega(enable_alignment=True)` for the text variant.
- Official code & weights follow their own license. This folder only orchestrates; no official code is copied.
- `.gitattributes` (repo root) forces LF so Windows-pushed scripts run cleanly on Ubuntu.
- `proxy.env` (proxy creds + `HF_TOKEN` / path / env overrides) is gitignored — never committed. Don't put credentials in scripts.
- SSL behind a TLS-intercepting corporate proxy: pip uses `--trusted-host`; `hf`/`git` use the CA bundle (`_env.sh` prefers `~/.ca-bundle.crt`, built by `setup_ca_bundle.sh`); `01` falls back to `_hf_download.py` (SSL disabled) if the CDN endpoint uses a different MITM cert.

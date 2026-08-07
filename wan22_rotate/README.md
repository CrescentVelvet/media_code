# Wan2.2 Rotate — 环绕人物 → 正面选图 → 分割 → 360° 旋转视频

从一组环绕人物拍摄的图像中，自动挑选人物面向相机的正面图像，用 SAM 3D Body 识别并分割人物（背景置白），再用 Wan2.2-TI2V-5B + 已训练的 LoRA 生成 360° 旋转视频。

本目录只含编排脚本——SAM 3D Body 官方代码在 `../sam-3d-body`、DiffSynth-Studio 在 `../DiffSynth-Studio`，权重在各算法的 `$MODEL_DIR` 下。两步共用同一个 conda env（从 `doll` 克隆，装两套依赖）。

## 常用命令

> 假设已进入容器（脚本自动激活 `wan22_rotate` env）；`GPU=0` 按需换卡。首次跑前先做下方「首次准备」。

```bash
# ── 一键：选图+分割 → 生成视频 ──
# INPUT_DIR: 含 image/ 子文件夹的人物数据
# WEIGHT_PATH: 训练好的 LoRA
# OUTPUT_DIR: 可选，默认 ../wan22_rotate_results
GPU=0 INPUT_DIR=../Reconstruction/dataset/B003_Human_Data_w_pose/test_task_id_2d96a34a9df848b2ba80b194df0ae99b \
  WEIGHT_PATH=../../model/Wan2.2-TI2V-5B_lora_add_data_reload/step-66900.safetensors \
  bash wan22_rotate/run_all.sh

# ── 分步 ──
# 1a) 选图+分割 完整版（SAM 3D Body: 3D 姿态估计选正面图 + SAM 分割）
GPU=0 INPUT_DIR=../Reconstruction/dataset/B003_Human_Data_w_pose/test_task_id_2d96a34a9df848b2ba80b194df0ae99b \
  OUTPUT_DIR=../../output/wan22_rotate_results \
  bash wan22_rotate/01_pick_and_segment.sh
# 1b) 选图+分割 简化版（只 ViTDet 检测 + SAM 分割, 按人物面积最大选图, 不加载 3D body 模型, 更快）
GPU=0 INPUT_DIR=../Reconstruction/dataset/B003_Human_Data_w_pose/test_task_id_2d96a34a9df848b2ba80b194df0ae99b \
  OUTPUT_DIR=../../output/wan22_rotate_results \
  bash wan22_rotate/01b_pick_and_segment.sh

# 2) 只做视频生成（用上一步的分割图）
GPU=0 SEGMENTED_IMAGE=../../output/wan22_rotate_results/segmented_image_centered.png \
  WEIGHT_PATH=../../model/Wan2.2-TI2V-5B_lora_add_data_reload/step-66900.safetensors \
  OUTPUT_DIR=../../output/wan22_rotate_results \
  bash wan22_rotate/02_generate_video.sh

# ── 自定义 ──
# 换 prompt / 分辨率 / 帧数（portrait 默认 1248×704；landscape 用 704×1248）
GPU=0 INPUT_DIR=... WEIGHT_PATH=... \
  PROMPT="人物360度旋转展示，高质量。" \
  HEIGHT=1248 WIDTH=706 NUM_FRAMES=121 \
  bash wan22_rotate/run_all.sh
# 跳过选图步骤，直接用已有图片生成视频
GPU=0 SKIP_SEGMENT=1 \
  SEGMENTED_IMAGE=/path/to/image.png \
  WEIGHT_PATH=... bash wan22_rotate/run_all.sh
# 选出的图是背面？翻转正面判定方向
GPU=0 FRONTAL_SIGN=-1 INPUT_DIR=... bash wan22_rotate/01_pick_and_segment.sh
# 用 SAM2 分割器（需提前放好 sam2 仓库 + checkpoint）
GPU=0 SEGMENTOR_PATH=/path/to/sam2_repo \
  INPUT_DIR=... bash wan22_rotate/01_pick_and_segment.sh
```

- 结果：分割图 → `../wan22_rotate_results/segmented_image.png`；视频 → `../wan22_rotate_results/rotate_360.mp4`；调试信息 → `frontal_scores.csv` + `debug_mask.png`。

## 首次准备

本流程建一份独立的 `wan22_rotate` env（**CPython 3.10**，匹配本地 cp310 torch/triton 轮子——不要用 3.11 或 clone doll，cp310 轮子装不进 3.11），把 sam_3d_body + diffsynth 两套依赖装在一起（detectron2 用 `--no-deps` 装，`networkx==3.2.1` 对 diffsynth 无影响；gcc12 用 `conda install --no-update-deps` 装防 conda 把 python 掉包成 GraalPy，否则 numpy 全坏）。

```bash
cd <your-code-dir>            # e.g. /data_3d/<uid>/code
git -c http.sslVerify=false clone https://github.com/CrescentVelvet/media_code.git
cd media_code && cp proxy.env.example proxy.env   # 填 http_proxy / https_proxy
# ⚠️ 确认 proxy.env 中 http_proxy / https_proxy 两行已取消注释并填好地址，
#    否则 pip 装依赖会报 "Network is unreachable"

# 1. 建 wan22_rotate env（⚠️ CPython 3.10，匹配本地 cp310 torch/triton 轮子；
#    不要用 3.11 或 clone doll——cp310 轮子装不进 3.11）
conda create -n wan22_rotate python=3.10 -y && conda activate wan22_rotate

# 2. 装两套依赖 + SAM2 + 验证
#    INSTALL_DEPS=1 会用本地 cp310 轮子装 torch 2.6.0+cu124 + nvidia 依赖，
#    装 gcc12（--no-update-deps 防 GraalPy 掉包）、diffsynth、sam_3d_body、detectron2，
#    并 clone SAM2 仓库 + pip install + 下载 sam2.1_hiera_large.pt
INSTALL_DEPS=1 bash wan22_rotate/00_setup_env.sh

# 3. 下权重（两边各自的下载脚本）
#    完整版 01 需要 SAM 3D Body 权重（GATED）；简化版 01b 不需要
HF_TOKEN=hf_xxx bash sam_3d_body/01_download_models.sh   # SAM 3D Body（GATED，需先 Request access）
bash wan22/01_verify_models.sh                           # Wan2.2（确认权重在位）
```

权重需已在 `$MODEL_DIR`（默认 `../../model`）下：
```
$MODEL_DIR/
  Wan2.2-TI2V-5B/                          # DiT + T5 + VAE (wan22 用, 01_verify_models.sh 自动建 Wan-AI 符号链接)
  Wan2.1-T2V-1.3B/                         # tokenizer
  Wan2.2-TI2V-5B_lora_add_data_reload/     # 训练好的 LoRA
    step-66900.safetensors
  sam-3d-body/
    sam-3d-body-dinov3/                    # SAM 3D Body ckpt + mhr_model
    moge-2-vitl-normal/                    # MoGe2 FOV estimator
```

LoRA 权重路径示例：`$MODEL_DIR/Wan2.2-TI2V-5B_lora_add_data_reload/step-66900.safetensors`。

---

以下为详细参考（流程原理 / 各步骤参数 / 排错 / 目录布局）。

## Pipeline（流程详解）

```
INPUT_DIR/image/               (环绕人物拍摄的多张图像)
    │
    ▼
[01] SAM 3D Body  (sam_3d_body env)
    │  ├─ ViTDet 人体检测          → bbox per image
    │  ├─ MoGe2 FOV 估计           → 相机内参
    │  ├─ DINOv3 编码 + MHR 解码   → 3D body (global_rot, pred_vertices, ...)
    │  ├─ 正面评分 (global_rot)     → 选最佳正面图
    │  └─ 人物分割                  → 背景置白
    │       ├─ 优先：HumanSegmentor (SAM2/SAM3, 需配 SEGMENTOR_PATH)
    │       ├─ 备选：3D mesh silhouette (pyrender 渲染 pred_vertices)
    │       └─ 兜底：bbox 矩形掩码
    ▼
$RESULTS_DIR/segmented_image.png   (人物保留, 背景白色)
    │
    ▼
[02] Wan2.2-TI2V-5B I2V + LoRA  (wan22 env)
    │  ├─ 加载 pipeline (DiT + T5 + VAE, 从本地 $MODEL_DIR)
    │  ├─ 加载 LoRA (pipe.load_lora, alpha=1)
    │  └─ I2V 生成 (分割图作首帧 → 360° 旋转视频)
    ▼
$RESULTS_DIR/rotate_360.mp4
```

### Step 01 — 选图 + 分割 (`01_pick_and_segment.sh` → `pick_and_segment.py`)

**正面图挑选**：对每张图跑 SAM 3D Body 的 `process_one_image`（3D 人体网格恢复），输出含 `global_rot`（全局旋转）。将其转为旋转矩阵 R，计算人物朝向 `forward = R @ [0,0,1]`（SMPL 静止姿态面 +Z）。正面朝相机时 `forward[2] < 0`（朝 -Z 即相机方向），评分 `score = -forward[2]`，取评分最高的图。

> 如果选出来的是背面而不是正面，说明旋转约定相反——设 `FRONTAL_SIGN=-1` 翻转。

`global_rot` 支持 3×3 旋转矩阵、轴角 (3,)、四元数 (4,) 三种格式，自动识别。

**人物分割**：按以下优先级尝试，任一成功即用：
1. **HumanSegmentor**（SAM2/SAM3）— 像素级精确掩码。需 `SEGMENTOR_PATH` 指向 sam2 仓库（含 `checkpoints/` + `configs/`）。脚本尝试 `__call__` 和 `predict` 两种接口。
2. **3D mesh silhouette** — 用 `Renderer.vertices_to_trimesh` 构建 trimesh，pyrender 离屏渲染为二值掩码。无需额外配置，但只覆盖身体模型形状（可能漏掉头发、宽松衣物）。
3. **bbox 矩形** — 最粗略的兜底掩码（带 `PADDING` 边距）。

掩码应用：人物区域保留原像素，背景区域设为白色 `(255,255,255)`（`WHITE_BG=0` 改为黑色）。

### Step 02 — 视频生成 (`02_generate_video.sh`)

直接调用已有的 `wan22/02_run_inference.sh`（它自己 source `wan22/_env.sh` 激活 `wan22` env），传入：
- `INPUT_IMAGE=$RESULTS_DIR/segmented_image.png`（I2V 模式）
- `WEIGHT_PATH=<lora>.safetensors`（`pipe.load_lora(dit, weight, alpha=1)`）
- `PROMPT`、`HEIGHT`/`WIDTH`/`NUM_FRAMES` 等

默认 portrait（1248×704），121 帧 @ 15fps ≈ 8 秒，足够一圈 360° 旋转。更多参数见 `wan22/README.md` 的 inference 部分。

## Config (env vars, all optional)

### Paths & envs
| var | default | note |
| --- | --- | --- |
| `INPUT_DIR` | _(required)_ | 人物文件夹（含 `image/` 子文件夹） |
| `WEIGHT_PATH` | _(required for 02)_ | 训练好的 LoRA `.safetensors` |
| `GPU` | _(unset)_ | physical GPU id，e.g. `GPU=0` |
| `CONDA_ENV` | `wan22_rotate` | conda env（从 doll 克隆，装两套依赖） |
| `SAM3D_DIR` | `../sam-3d-body` | SAM 3D Body 官方代码 |
| `SAM3D_MODEL_DIR` | `../../model/sam-3d-body` | SAM 3D Body 权重 |
| `DIFFSYNTH_DIR` | `../DiffSynth-Studio` | DiffSynth-Studio 代码 |
| `WAN_MODEL_DIR` | `../../model` | Wan2.2 权重根 |
| `RESULTS_DIR` | `../wan22_rotate_results` | 输出目录 |

### Step 01 params
| var | default | note |
| --- | --- | --- |
| `HF_REPO_ID` | `facebook/sam-3d-body-dinov3` | SAM 3D Body 骨干（alt `facebook/sam-3d-body-vith`） |
| `CHECKPOINT_PATH` | `$SAM3D_MODEL_DIR/<repo>/model.ckpt` | SAM 3D Body checkpoint |
| `MHR_PATH` | `$SAM3D_MODEL_DIR/<repo>/assets/mhr_model.pt` | MHR asset |
| `DEVICE` | `cuda` | falls back to `cpu` if CUDA unavailable |
| `DETECTOR_NAME` | `vitdet` | `vitdet` \| `sam3` \| `` (disable → full-image bbox) |
| `DETECTOR_PATH` | _(unset)_ | ViTDet auto-downloads; set for offline |
| `SEGMENTOR_NAME` | `sam2` | `sam2` (needs `SEGMENTOR_PATH`) \| `sam3` \| `` (disable) |
| `SEGMENTOR_PATH` | _(unset)_ | sam2 repo dir w/ `checkpoints/` + `configs/` |
| `FOV_NAME` | `moge2` | `moge2` \| `` (disable → default FOV) |
| `FOV_PATH` | `$SAM3D_MODEL_DIR/moge-2-vitl-normal` | MoGe2 local dir |
| `BBOX_THRESH` | `0.8` | detector score threshold |
| `INFERENCE_TYPE` | `body` | `body` (skip hand decoder, faster) \| `full` \| `hand` |
| `FRONTAL_SIGN` | `1` | `-1` = flip front-facing criterion (if picks back-facing) |
| `WHITE_BG` | `1` | `1` = white background; `0` = black |
| `PADDING` | `0.1` | bbox padding ratio (for bbox mask fallback) |

### Step 02 params
| var | default | note |
| --- | --- | --- |
| `PROMPT` | `人物360度旋转展示，高质量，细节清晰。` | 生成提示词 |
| `SEGMENTED_IMAGE` | `$RESULTS_DIR/segmented_image.png` | I2V 输入图 |
| `HEIGHT` / `WIDTH` | `1248` / `704` | portrait（landscape 用 `704`/`1248`） |
| `NUM_FRAMES` | `121` | 生成帧数（4k+1，Wan 约束） |
| `FPS` | `15` | 输出 mp4 帧率 |
| `OUTPUT_NAME` | `rotate_360` | 输出文件名（不含扩展名） |
| `SKIP_SEGMENT` | `0` | `1` = 跳过 step 01（run_all.sh 用） |
| `LOW_VRAM` | `0` | `1` = 磁盘 offload（慢但省显存，详见 wan22 README） |

## 可能遇到的问题

**1. 选出的图是背面而不是正面**
SAM 3D Body 的 `global_rot` 旋转约定可能与你的人物数据相反。设 `FRONTAL_SIGN=-1` 翻转评分方向，重跑 step 01。

**2. 分割用了 bbox 兜底（掩码是矩形）**
说明 HumanSegmentor 和 mesh silhouette 都失败了。改善方法：
- 配置 SAM2 分割器：设 `SEGMENTOR_PATH=/path/to/sam2_repo`（含 `checkpoints/sam2.1_hiera_large.pt` + `configs/`）。
- mesh silhouette 失败通常是 pyrender OpenGL 问题：确认 `PYOPENGL_PLATFORM=egl`（`_env.sh` 默认设了），或试 `PYOPENGL_PLATFORM=osmesa`（需 `apt install libosmesa6-dev` + 重装 PyOpenGL）。

**3. step 01 报 `import sam_3d_body` / `cv2` 失败**
`wan22_rotate` env 缺 sam_3d_body 依赖。`INSTALL_DEPS=1 bash wan22_rotate/00_setup_env.sh` 重装。

**4. step 02 报 `import diffsynth` 失败**
`wan22_rotate` env 缺 diffsynth。`INSTALL_DEPS=1 bash wan22_rotate/00_setup_env.sh` 重装。

**5. 视频生成 OOM**
- 降分辨率 / 帧数：`HEIGHT=480 WIDTH=832 NUM_FRAMES=49`。
- 开磁盘 offload：`LOW_VRAM=1`。
- 详见 `wan22/README.md` 排错 #3。

**6. 多人物图像**
脚本取 bbox 面积最大的那个人（离相机最近）。如需指定其他人，修改 `pick_and_segment.py` 的 `best = max(outputs, ...)` 逻辑。

**7. 跑 `.sh` 报 `syntax error near unexpected token ('`**
CRLF 行尾污染。`find wan22_rotate -name '*.sh' -exec sed -i 's/\r$//' {} +` 或 `git checkout -- wan22_rotate/*.sh`（`.gitattributes` 强制 LF）。

**8. NumPy 坏了 / `import numpy` 报 ABI 不兼容 / python 变成了 GraalPy**
根因：`conda install -c conda-forge gxx_linux-64`（装 detectron2 编译要的 gcc12）**不带 `--no-update-deps`** 时，conda 求解器会把 env 的 python 实现掉包成 **GraalPy**（满足 python 槽位的另一个 conda-forge 包），而 numpy/torch 是 CPython ABI 编译的——GraalPy 下全坏；本地 cp310 torch/triton 轮子更直接装不上。本仓 `00_setup_env.sh` 的 gxx 步骤**已加 `--no-update-deps`** 防掉包，env 也强制 CPython 3.10（建完即校验 `platform.python_implementation()=="CPython"`，gxx 装完再校验一次）。若 env 已被掉包成 GraalPy，**重建即可**：
```bash
python -c "import platform; print(platform.python_implementation())"   # 输出 GraalPy 即中招
conda env remove -n wan22_rotate
conda create -n wan22_rotate python=3.10 -y && conda activate wan22_rotate   # CPython 3.10
INSTALL_DEPS=1 bash wan22_rotate/00_setup_env.sh                          # gxx 带 --no-update-deps
python -c "import numpy, torch; print(numpy.__version__, torch.__version__)"  # 验证
```
> 铁律：往 `wan22_rotate` env 里 `conda install` 任何包都加 `--no-update-deps`，否则 GraalPy 会回来把 numpy 干掉。

## 目录布局
```
<code-dir>/
├── media_code/                  # 本仓
│   ├── proxy.env                # 代理 + 覆盖项, gitignored
│   ├── wan22/                   # Wan2.2 推理/训练脚本 (step 02 调用)
│   ├── sam_3d_body/             # SAM 3D Body 推理脚本 (step 01 调用)
│   └── wan22_rotate/            # ← 本目录（编排脚本）
├── sam-3d-body/                 # SAM 3D Body 官方代码
├── sam2/                        # SAM2 官方代码 + checkpoints (00 clone, 01b 分割用)
│   └── checkpoints/
│       └── sam2.1_hiera_large.pt
├── DiffSynth-Studio-Human/     # DiffSynth-Studio 官方代码 (本流程专用, 00 clone)
├── wan22_experiments/           # LoRA 训练产物 (epoch-N.safetensors)
└── wan22_rotate_results/        # 本流程输出
    ├── segmented_image.png      #   正面图 (人物保留, 背景白)
    ├── front_facing_original.jpg#   原始正面图
    ├── frontal_scores.csv       #   各图正面评分
    ├── debug_mask.png           #   分割掩码 (调试)
    └── rotate_360.mp4           #   360° 旋转视频
```

## Notes
- Official code & weights follow their own licenses (Wan2.2 = Apache 2.0; SAM 3D Body = SAM License). This folder only orchestrates; no official code is copied.
- `.gitattributes` (repo root) forces LF so Windows-pushed scripts run cleanly on Ubuntu.
- `proxy.env` (proxy creds / env overrides) is gitignored — never committed.

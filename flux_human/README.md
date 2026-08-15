# flux_human — 图像生成解决人体微动三维重建模糊

用 SAM 3D Body 抽 SMPL + Flux1-dev ControlNet(depth) 生成多视角静止人体图像，做无模糊三维重建（3DGS/NeuS 重建与评估为待办）。本目录只含编排脚本——Flux 权重从 HuggingFace 下，SAM 3D Body 复用 [`sam_3d_body/`](../sam_3d_body/) 子目录（02 跨 env 调用）。设计文档见 [技术方案.md](技术方案.md)。

> ⚠️ **FLUX.1-dev 是 GATED**：先在 [black-forest-labs/FLUX.1-dev](https://huggingface.co/black-forest-labs/FLUX.1-dev) accept license，再建 read token，用 `HF_TOKEN=hf_xxx` 传给 `01`。sam_3d_body 权重同理（gated，见其 README）。

## 常用命令

> 假设已进入容器；首次跑前先做下方「首次准备」。
> **铁律：每条命令都必须显式写出模型地址、输入路径、输出路径，不能全靠脚本里的默认值。** 用具体路径，不要用 `...` 占位。

```bash
# ── 一键（建 env + 下权重 + SMPL 提取 + 渲染深度 + Flux 生成）──
INSTALL_DEPS=1 HF_TOKEN=hf_xxx GPU=0 \
  VIDEO=../data/subject.mp4 \
  MODEL_DIR=../../model/flux_human \
  RESULTS_DIR=../flux_human_results \
  bash flux_human/run_all.sh

# ── 分步 ──
# 0) 建 env（克隆 doll + 装 Flux1/ControlNet/pyrender）
INSTALL_DEPS=1 bash flux_human/00_setup_env.sh
# 1) 下权重（FLUX.1-dev + ControlNet + IP-Adapter）
HF_TOKEN=hf_xxx bash flux_human/01_download_models.sh
# 2) SMPL 提取（抽帧 + 调 sam_3d_body + 选静止参考帧；⚠️ 需先建 sam_3d_body env + 下其权重）
GPU=0 \
  VIDEO=../data/subject.mp4 \
  RESULTS_DIR=../flux_human_results \
  bash flux_human/02_extract_smpl.sh
# 3) 渲染骨骼深度图（pyrender orbit 相机）
GPU=0 \
  RESULTS_DIR=../flux_human_results \
  NUM_VIEWS=24 \
  bash flux_human/03_render_depth.sh
# 4) Flux1 多视角生成（ControlNet depth 锁几何 + 同 seed 锁随机性）
GPU=0 \
  MODEL_DIR=../../model/flux_human \
  RESULTS_DIR=../flux_human_results \
  NUM_VIEWS=24 \
  bash flux_human/04_generate_views.sh
```

- 结果：`view_<i>.png`（24 视角静止人体图像）+ `cameras.npz`（每视角相机位姿，给 05 重建用）。

## 首次准备
```bash
# clone 本仓 + proxy.env
cd <code-dir>
git -c http.sslVerify=false clone https://github.com/CrescentVelvet/media_code.git
cd media_code && cp proxy.env.example proxy.env
# ⚠️ 确认 proxy.env 中 http_proxy / https_proxy 已取消注释

# 建 flux_human env（从 doll 克隆 + 装 Flux1/ControlNet/pyrender）
INSTALL_DEPS=1 bash flux_human/00_setup_env.sh

# 下 Flux1-dev + ControlNet + IP-Adapter（⚠️ FLUX.1-dev gated，需先 accept license）
HF_TOKEN=hf_xxx bash flux_human/01_download_models.sh

# ⚠️ 还需建 sam_3d_body env + 下其权重（02 跨 env 调用 sam_3d_body）
INSTALL_DEPS=1 bash sam_3d_body/00_setup_env.sh
HF_TOKEN=hf_xxx bash sam_3d_body/01_download_models.sh
```

权重目录布局：
```
$MODEL_DIR/flux_human/
├── FLUX.1-dev/        # black-forest-labs/FLUX.1-dev (GATED) — base pipeline
├── controlnet/         # InstantX/FLUX.1-dev-Controlnet-Union (depth+canny+..., diffusers 格式)
└── ip-adapter/         # xlabs/flux-ip-adapter (可选; 加载 API 待验证)

$MODEL_DIR/sam-3d-body/   # sam_3d_body 权重 (02 跨 env 调用)
├── sam-3d-body-dinov3/   # 主权重 + mhr_model.pt
└── moge-2-vitl-normal/   # FOV 估计
```

---

以下为详细参考。

## Pipeline（流程详解）
```
输入视频
  │
  ├─[02] 抽帧(ffmpeg) → sam_3d_body 抽 SMPL(每帧 npz) → 选 pose 变化最小的参考帧 T*
  │       → 导出 reference.npz (vertices/cam_t/faces/kp3d/kp2d/focal/ref_image)
  │       (跨 env: flux_human 编排, sam_3d_body 跑推理 + 取 faces)
  │
  ├─[03] 用 T* 的 SMPL mesh 渲染 N 视角骨骼深度图 (pyrender orbit 相机)
  │       → depth_*.png (ControlNet condition) + cameras.npz (known pose)
  │
  ├─[04] Flux1-dev + ControlNet(depth) + IP-Adapter(可选) → 生成 N 视角静止图像
  │       (同 seed 锁随机性; depth ControlNet 锁几何; IP-Adapter 锁外观)
  │       → view_*.png
  │
  ├─[05] TODO: 3DGS/NeuS 重建 (用 view_*.png + cameras.npz known pose)
  │
  └─[06] TODO: 评估 (Chamfer/F-score 对比 PIFuHD 单图重建 / GT mesh)
```

## 待办（未实现）
| 脚本 | 状态 | 说明 |
|---|---|---|
| `05_reconstruct.sh` + `reconstruct.py` | TODO | 3DGS / NeuS 重建。输入 `view_*.png` + `cameras.npz`（known pose，不用 colmap）；对生成图像噪声鲁棒用 NeuS/SDF 类。 |
| `06_evaluate.sh` + `evaluate.py` | TODO | Chamfer Distance / F-score@1cm / Normal consistency，对比 PIFuHD 单图重建（下限）+ 真实多视角 MVS（上限）。 |
| IP-Adapter 接入 | TODO | xlabs flux-ip-adapter 加载 API 待验证（`generate_views.py` 已留接口，`USE_IPADAPTER=1` 启用）。 |
| ControlNet Union mode | TODO | InstantX Union controlnet 的 depth mode 参数待 introspect 确认（`generate_views.py` 已打印 mode/union attrs）。 |

## Config (env vars)

### 通用
| var | default | note |
|---|---|---|
| `CONDA_ENV` | `flux_human` | conda env（从 doll 克隆） |
| `GPU` | _(unset)_ | 物理卡 id，e.g. `GPU=0` |
| `MODEL_DIR` | `../../model/flux_human` | flux_human 权重根 |
| `RESULTS_DIR` | `../flux_human_results` | 输出根 |
| `INSTALL_DEPS` | `0`（run_all: `1`） | `1` = 00 装依赖 |

### 02 SMPL 提取
| var | default | note |
|---|---|---|
| `VIDEO` | _(unset)_ | 输入视频（用 ffmpeg 抽帧；不设则用 `FRAMES_DIR`） |
| `FRAMES_DIR` | `$RESULTS_DIR/frames` | 帧目录 |
| `FPS` | `2` | 抽帧率 |
| `INFERENCE_TYPE` | `body` | sam_3d_body 类型（`body`=跳过手部解码器，更快） |
| `WINDOW` | `5` | 选参考帧的滑动窗口（pose 变化最小） |

### 03 深度图
| var | default | note |
|---|---|---|
| `NUM_VIEWS` | `24` | orbit 视角数 |
| `ELEVATION` | `-10` | 相机俯仰角（度，负=俯视） |
| `IMG_SIZE` | `768` | 渲染分辨率（方形） |
| `FOV_DEG` | `35` | 相机垂直 FOV |
| `CAMERA_DIST` | _(auto)_ | 相机距离（空=按 mesh 尺度自动算） |

### 04 生成
| var | default | note |
|---|---|---|
| `NUM_VIEWS` | `24` | 生成视角数 |
| `SEED` | `231` | 同 seed 锁随机性（跨视角一致性） |
| `HEIGHT`/`WIDTH` | `1024` | 生成分辨率 |
| `NUM_INFERENCE_STEPS` | `28` | Flux 推理步数 |
| `GUIDANCE_SCALE` | `3.5` | CFG |
| `CONTROLNET_SCALE` | `0.7` | depth ControlNet 强度 |
| `USE_IPADAPTER` | `0` | `1` = 启用 IP-Adapter（API 待验证） |
| `OFFLOAD` | `model` | `model` / `sequential` / `none` |

## 可能遇到的问题

1. **sam_3d_body env 未建**：02 跨 env 调用需 sam_3d_body env + 权重。先跑 `sam_3d_body/00` + `01`。
2. **FLUX.1-dev gated**：01 需 `HF_TOKEN` + accept license；无 token 返回 401。
3. **ControlNet Union mode**：生成视角和 depth 不符时，看 04 输出的 `controlnet mode/union attrs`，补 union depth mode 参数。
4. **pyrender EGL**：无显示器的 GPU 服务器需 `PYOPENGL_PLATFORM=egl`（`_env.sh` 已设）；EGL 不可用换 `osmesa`（`apt install -y libosmesa6-dev`）。
5. **FluxControlNetPipeline 不可 import**：diffusers >= 0.31 才有；跑 `INSTALL_DEPS=1 bash 00_setup_env.sh` 升级。
6. **SSL 下载失败**：代理 MITM 时 `bash flux_human/setup_ca_bundle.sh` 建 `~/.ca-bundle.crt`，或 `HF_DISABLE_SSL=1`。

## 目录布局
```
<code-dir>/
├── media_code/
│   ├── flux_human/          # 本目录（编排脚本）
│   └── sam_3d_body/         # SMPL 提取（02 跨 env 调用）
├── sam-3d-body/             # sam_3d_body 官方代码
├── flux_human_results/      # 输出
│   ├── frames/               # 抽帧
│   ├── sam3d/                # sam_3d_body 推理（result/mesh/npz）
│   ├── smpl/                 # reference.npz + faces.npy
│   ├── depth/                # depth_*.png + cameras.npz
│   └── views/                # view_*.png
└── ../model/
    ├── flux_human/           # FLUX.1-dev + controlnet + ip-adapter
    └── sam-3d-body/          # sam_3d_body 权重
```

## Notes
- flux_human 用专用 env（从 doll 克隆），diffusers/ControlNet 版本与其他算法隔离。
- 02 跨 env 调用 sam_3d_body（`conda run -n sam_3d_body`），不污染 flux_human env。
- `.gitattributes`（仓根）强制 LF，Windows push 的脚本在 Ubuntu 跑无问题。
- `proxy.env`（代理凭证）gitignored，不入库。
- 设计思路与 MVP 分阶段验证见 [技术方案.md](技术方案.md)。

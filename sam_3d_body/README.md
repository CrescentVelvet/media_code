# SAM 3D Body runner

在 Ubuntu + NVIDIA 服务器上跑 [SAM 3D Body](https://github.com/facebookresearch/sam-3d-body)（Meta，单图全身 3D 人体网格恢复 HMR）的**推理**。本目录只含编排脚本——官方代码自动 clone、权重从 HuggingFace 下载。

> ⚠️ **SAM 3D Body 权重是 GATED（受限访问）**：必须先在 [facebook/sam-3d-body-dinov3](https://huggingface.co/facebook/sam-3d-body-dinov3) 页面点「Request access」（通常几分钟内自动通过），再建一个 *read* token（https://huggingface.co/settings/tokens），用 `HF_TOKEN=hf_xxx` 传给 `01`。无 access + token 时 HF 返回 `401 / repository not found`。ViT-H 骨干同理（`facebook/sam-3d-body-vith`）。

## 常用命令

> 假设已进入容器并 `conda activate sam_3d_body`；路径取各脚本默认值（可改）；`GPU=0` 按需换卡。首次跑前先做下方「首次准备」。

```bash
# ── 推理(02) ── 默认 DINOv3-H+ 骨干 + ViTDet 检测器 + MoGe2 FOV，无分割器(掩码推理关)
# 1) 测试官方自带示例图(notebook/images/dancing.jpg)
GPU=0 bash sam_3d_body/02_run_inference.sh
# 2) 指定输入路径(递归遍历子目录)
GPU=0 INPUT_DIR=/path/to/images bash sam_3d_body/02_run_inference.sh
# 3) 换 ViT-H 骨干(631M，略小；先 01 下载该骨干权重)
HF_REPO_ID=facebook/sam-3d-body-vith GPU=0 INPUT_DIR=/path/to/images bash sam_3d_body/02_run_inference.sh
# 4) 开掩码推理(需 SAM2 分割器，先装 sam2 并给 SEGMENTOR_PATH)
GPU=0 INPUT_DIR=/path/to/images USE_MASK=1 SEGMENTOR_PATH=/path/to/sam2_checkpoint_dir bash sam_3d_body/02_run_inference.sh
# 5) 只跑 body 解码器(跳过手部细化，更快)
GPU=0 INPUT_DIR=/path/to/images INFERENCE_TYPE=body bash sam_3d_body/02_run_inference.sh
```

- 结果：推理 → `../sam-3d-body/results/<输入夹名>/result/<相对路径>.jpg`（渲染叠加图）+ `mesh/<相对路径>_mesh_<pid>.ply`（每人 3D 网格，可用 [3dviewer.net](https://3dviewer.net) / Blender 打开）+ `npz/<相对路径>.npz`（每人数值输出）。

## 首次准备
```bash
cd <your-code-dir>            # e.g. /data_3d/<uid>/code
git -c http.sslVerify=false clone https://github.com/CrescentVelvet/media_code.git
cd media_code && cp proxy.env.example proxy.env   # 填 http_proxy / https_proxy
conda create -n sam_3d_body python=3.11 -y && conda activate sam_3d_body
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
INSTALL_DEPS=1 bash sam_3d_body/00_setup_env.sh          # 装 INSTALL.md 全套依赖
# ⚠️ 先在 HF 页面 Request access 并建 read token，然后：
HF_TOKEN=hf_xxx bash sam_3d_body/01_download_models.sh   # 下 SAM 3D Body ckpt + MoGe2 FOV
```
⚠️ SAM 3D Body 的 `detectron2 / networkx==3.2.1` 版本 pin 与本仓其他算法冲突，务必用专用 env（`CONDA_ENV=sam_3d_body`），别装进共享的 `doll`。

或一键（clone + 装依赖 + 下权重 + 跑示例图）：
```bash
INSTALL_DEPS=1 HF_TOKEN=hf_xxx bash sam_3d_body/run_all.sh
```

---

以下为详细参考（流程原理 / 各脚本参数 / 排错 / 目录布局）。

## Inference (02 — 更多用法)
`02_run_inference.sh` 调 `run_inference.py`：把 `sam-3d-body` 代码目录加进 `sys.path`，加载一次 SAM 3D Body 模型 + 可选检测器/分割器/FOV 估计器，递归遍历 `INPUT_DIR` 全部图像，逐图打印检测到的人数与耗时，输出到 `OUTPUT_DIR/result/<rel>.jpg`（渲染叠加图）+ `mesh/<rel stem>_mesh_<pid>.ply`（每人 3D 网格）+ `npz/<rel>.npz`（每人数值输出：`pred_vertices`、`pred_cam_t`、`pred_keypoints_3d/2d`、`focal_length`、`bbox`、姿态/形状参数等）。测默认骨干见上文「常用命令」#1/#2。

```bash
# 换输出目录 / 换骨干 / 只存渲染图(不要 ply+npz)：
GPU=0 INPUT_DIR=/path/to/images OUTPUT_DIR=../sam-3d-body/results/runA bash sam_3d_body/02_run_inference.sh
HF_REPO_ID=facebook/sam-3d-body-vith GPU=0 INPUT_DIR=/path/to/images bash sam_3d_body/02_run_inference.sh
GPU=0 INPUT_DIR=/path/to/images SAVE_NPZ=0 bash sam_3d_body/02_run_inference.sh
```

## Pipeline（推理流程详解）
对应官方代码 `sam_3d_body/sam_3d_body_estimator.py::process_one_image` + `sam_3d_body/models/meta_arch/sam3d_body.py` + `tools/build_*.py`。一张图从输入到输出经过：

1. **人体检测** — `HumanDetector`（默认 `vitdet`，detectron2 Cascade Mask R-CNN ViTDet-H）。ViTDet 权重 `model_final_f05665.pkl` 从 `dl.fbaipublicfiles.com` **运行时自动下载**并缓存到 `~/.cache/torch/`（首次联网；之后可离线）。检测阈值 `BBOX_THRESH=0.8`。也可换 `sam3` 检测器（需 `SAM3=1` 装好 sam3）。
2. **（可选）掩码生成** — `HumanSegmentor`（默认 `sam2`，但**仅当给了 `SEGMENTOR_PATH` 才加载**，否则跳过、掩码推理关闭）。`USE_MASK=1` 时用 SAM2 从 bbox 生成 mask，做掩码条件预测（类 SAM 家族的 promptable 推理）。SAM2 需你自己放好 sam2 仓库 + `checkpoints/sam2.1_hiera_large.pt` + `configs/`。
3. **FOV 估计** — `FOVEstimator`（默认 `moge2`，HuggingFace `Ruicheng/moge-2-vitl-normal`，**公开非 gated**）。`01` 已预下到 `$MODEL_DIR/moge-2-vitl-normal`，`02` 把 `FOV_PATH` 指向该本地目录，避免运行时联网。MoGe2 估计相机内参，喂给模型做透视正确的 3D→2D 投影。`FOV_NAME=` 关闭则用默认 FOV。
4. **Top-down 仿射 + 编码** — `prepare_batch` + `GetBBoxCenterScale` + `TopdownAffine(IMAGE_SIZE)` 把每个 bbox 裁成 256×256 输入；`VisionTransformWrapper(ToTensor())`。`cam_int` 来自 FOV 估计或默认值。
5. **SAM 3D Body 前向**（encoder-decoder） — DINOv3-H+ backbone 编码图像 token → promptable decoder（可接 2D 关键点 / mask prompt）→ MHR head 解码出 **Momentum Human Rig**（MHR）参数：身体/足/手的姿态 + 形状。`inference_type=full` 还跑**手部解码器**（body+hand），`body` 跳过手部（更快），`hand` 只出手。
6. **输出** — `pose_output["mhr"]` 含 `pred_vertices`、`pred_cam_t`（相机平移）、`focal_length`、`pred_keypoints_3d/2d`、各 MHR 参数。`estimator.faces` 是 MHR 网格面索引（固定拓扑）。

> 一句话：**ViTDet 检人 →（可选 SAM2 出 mask）→ MoGe2 估 FOV → top-down 裁 256 → DINOv3 编码 + promptable decoder + MHR head 出全身网格参数 →（full 再跑手部解码器）→ 渲染叠加图 / 导出 PLY**。

### 模型骨干
| 骨干 (size) | HF repo | 3DPW(MPJPE) | 备注 |
| :-- | :-- | :-- | :-- |
| **DINOv3-H+ (840M)** | `facebook/sam-3d-body-dinov3` | 54.8 | 默认，推荐（`01`/`02` 默认指它） |
| ViT-H (631M) | `facebook/sam-3d-body-vith` | 54.8 | 略小；`HF_REPO_ID=facebook/sam-3d-body-vith` 切换 |

两者均为 **gated**，需各自 Request access + HF_TOKEN。每个 repo 含 `model.ckpt` + `model_config.yaml` + `assets/mhr_model.pt`。

## 可能遇到的问题

公司代理做 HTTPS 中间人解密，下面按流水线阶段列出常见报错与修法（命令在服务器上、`sam_3d_body` 环境已激活时执行）。

**1. clone 本仓 / 官方仓报 SSL / 认证**
公开仓免认证，加 `-c http.sslVerify=false`（`run_all.sh` 克隆官方仓失败时也会自动带 `-c http.sslVerify=false` 重试）。git 连不上（非 SSL）就设全局代理（密码特殊字符必须 URL 编码）：
```bash
git config --global https.proxy http://USER:PASS@proxyhk.huawei.com:8080
git config --global  http.proxy http://USER:PASS@proxyhk.huawei.com:8080
```

**2. `01` 下权重报 `401 Unauthorized` / `repository not found`**
SAM 3D Body 是 gated。两种情况：没在 HF 页面 Request access，或没传 token。修法：
- 打开 `https://huggingface.co/facebook/sam-3d-body-dinov3` 点 Request access（ViT-H 同理点 vith 页面）；
- 建 read token：`https://huggingface.co/settings/tokens` → New token → Read；
- `HF_TOKEN=hf_xxx bash sam_3d_body/01_download_models.sh`。

**3. `01` 报 `SSLCertVerificationError`**
代理根 CA 不在系统证书包。先一次性建包，`_env.sh` 会自动用 `~/.ca-bundle.crt`：
```bash
bash sam_3d_body/setup_ca_bundle.sh    # 抓代理证书链 -> ~/.ca-bundle.crt，并自检
HF_TOKEN=hf_xxx bash sam_3d_body/01_download_models.sh
```
- 自检 `[OK]` → 直接重跑 `01`。
- 自检 `[FAIL]` → 把公司根 CA 追加到 `~/.ca-bundle.crt` 后重跑（常见于 `/usr/local/share/ca-certificates/`，脚本已自动并入）。
- 仍报 SSL（CDN 端点用了不同 MITM 证书）→ `01` 自动回退到禁用 SSL 校验的下载器（`_hf_download.py`）；或直接 `HF_DISABLE_SSL=1 HF_TOKEN=hf_xxx bash sam_3d_body/01_download_models.sh`。

**4. `01` 报 `CAS service error : ReqwestMiddleware`**
HF 的 Xet/Rust 通道不认代理。`_env.sh` 已设 `HF_HUB_DISABLE_XET=1`；仍报则彻底卸载：
```bash
pip uninstall -y hf_xet
HF_TOKEN=hf_xxx bash sam_3d_body/01_download_models.sh
```

**5. `00` 装 detectron2 报 `git+https` SSL 失败**
`pip install git+https://...` 走 git clone，受 `GIT_SSL_CAINFO` 影响（`_env.sh` 已设）。仍失败则手动 clone 再装：
```bash
git -c http.sslVerify=false clone https://github.com/facebookresearch/detectron2.git
cd detectron2 && git checkout a1ce2f9
pip install --no-build-isolation --no-deps .
```
detectron2 编译可能需要 CUDA toolkit（`CUDA_HOME` 指向 `/usr/local/cuda`）。

**6. 推理报 `pyrender` / OpenGL 相关错（`Couldn't find EGL` 等）**
渲染器（`visualization/renderer.py`）用 pyrender，无显示器的 GPU 服务器需 GL 后端。`_env.sh` 默认 `PYOPENGL_PLATFORM=egl`（GPU 服务器最稳）。EGL 不可用时换 osmesa：
```bash
apt install -y libosmesa6-dev
pip install --upgrade --force-reinstall PyOpenGL
PYOPENGL_PLATFORM=osmesa GPU=0 bash sam_3d_body/02_run_inference.sh
```

**7. 推理首次卡在 ViTDet 下载（`dl.fbaipublicfiles.com`）**
ViTDet 权重 `model_final_f05665.pkl` 由 detectron2 **运行时自动下载**（不走 HF）。代理下若失败：先 `bash sam_3d_body/setup_ca_bundle.sh`，或手动下到 `~/.cache/torch/hub/checkpoints/`：
```bash
# 代理下手动下（注意该 URL 在 tools/build_detector.py 里写死，路径可能变，以源码为准）
curl -L -o ~/.cache/torch/hub/checkpoints/model_final_f05665.pkl \
  https://dl.fbaipublicfiles.com/detectron2/ViTDet/COCO/cascade_mask_rcnn_vitdet_h/f328730692/model_final_f05665.pkl
```
或关掉 ViTDet、用整图当 bbox（`DETECTOR_NAME= GPU=0 ... bash 02`，`DETECTOR_NAME=` 留空即不检测，退回 `process_one_image` 的整图 bbox）。

**8. 推理首次卡在 MoGe2 下载（HuggingFace）**
`01` 已预下 MoGe2 并把 `FOV_PATH` 指向本地目录，正常不联网。若仍联网（路径没对上），确认 `FOV_PATH=$MODEL_DIR/moge-2-vitl-normal` 存在且含权重文件；或 `FOV_NAME=` 关闭 FOV 估计（用默认 FOV）。

**9. 推理 OOM（显存不足）**
DINOv3-H+ 840M 骨干较大。关掉手部解码器省显存：`INFERENCE_TYPE=body`；或换 ViT-H 骨干（631M，`HF_REPO_ID=facebook/sam-3d-body-vith`）。仍紧张时 `export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`。CPU 可跑但很慢（`DEVICE=cpu`，`run_inference.py` 会自动回退）。

**10. 跑 `.sh` 报 `syntax error near unexpected token ('`（CRLF 行尾）**
脚本被 CRLF 行尾污染（Windows→服务器非 git 方式同步时带过去）。修法：
```bash
sed -i 's/\r$//' sam_3d_body/*.sh
git checkout -- sam_3d_body/*.sh   # git 同步的：.gitattributes 还原 LF
```
预防：用 `git pull` 同步（本仓 `.gitattributes` 强制 LF）。

> 通用：`proxy.env`（代理凭证）在仓内 gitignored，`~/.ca-bundle.crt` 在家目录，都不入库；切勿把凭证写进脚本。

## Config (env vars, all optional)
| var | default | note |
|---|---|---|
| `CONDA_ENV` | `sam_3d_body` | conda env to activate（专用 env，pin 与其他算法冲突） |
| `GPU` | _(unset)_ | physical GPU id to pin, e.g. `GPU=0`; remaps `CUDA_VISIBLE_DEVICES` |
| `SAM3D_DIR` | `../sam-3d-body` | official code path |
| `MODEL_DIR` | `../../model/sam-3d-body` | weights path（ckpt 目录 + MoGe2 目录） |
| `SAM3D_REPO` | official GitHub URL | clone source (run_all) |
| `INSTALL_DEPS` | `0` (run_all: `1`) | `1` = install INSTALL.md deps in `00` |
| `SKIP_CORE` | `0` | `1` = skip the big core pip list (already installed) |
| `DETECTRON2` | `1` | `00`: install detectron2 @a1ce2f9 (ViTDet detector needs it) |
| `MOGE` | `1` | `00`: install microsoft/MoGe (moge2 FOV needs it) |
| `SAM3` | `0` | `00`: install facebookresearch/sam3 (only for detector_name/segmentor_name=sam3) |
| `HF_TOKEN` | _(unset)_ | **REQUIRED** by `01` for the gated SAM 3D Body checkpoint |
| `HF_REPO_ID` | `facebook/sam-3d-body-dinov3` | main checkpoint repo (alt `facebook/sam-3d-body-vith`) |
| `HF_FOV_REPO` | `Ruicheng/moge-2-vitl-normal` | MoGe2 FOV repo (public) |
| `HF_DISABLE_SSL` | `0` | `1` = download weights with SSL verification disabled |
| `HF_HUB_DISABLE_XET` | `1` | disable HF Xet/CAS Rust path (proxy-unfriendly) |
| `PYOPENGL_PLATFORM` | `egl` | GL backend for pyrender headless rendering (`osmesa` alt) |

### Inference (02)
| var | default | note |
|---|---|---|
| `INPUT_DIR` | `../sam-3d-body/notebook/images` | folder of images (walked recursively) |
| `OUTPUT_DIR` | `../sam-3d-body/results/<input_folder_name>` | writes `result/` + `mesh/` + `npz/` under here |
| `CHECKPOINT_PATH` | `$MODEL_DIR/<repo>/model.ckpt` | SAM 3D Body checkpoint |
| `MHR_PATH` | `$MODEL_DIR/<repo>/assets/mhr_model.pt` | MHR asset |
| `FOV_PATH` | `$MODEL_DIR/moge-2-vitl-normal` | MoGe2 local dir (passed to `from_pretrained`) |
| `DEVICE` | `cuda` | falls back to `cpu` if CUDA unavailable |
| `DETECTOR_NAME` | `vitdet` | `vitdet` (default) \| `sam3` \| `` (disable → full-image bbox) |
| `DETECTOR_PATH` | _(unset)_ | ViTDet auto-downloads; set for offline / custom |
| `SEGMENTOR_NAME` | `sam2` | `sam2` (needs `SEGMENTOR_PATH`) \| `sam3` \| `` (disable) |
| `SEGMENTOR_PATH` | _(unset)_ | sam2 repo dir w/ `checkpoints/` + `configs/` (required to enable sam2) |
| `FOV_NAME` | `moge2` | `moge2` (default) \| `` (disable → default FOV) |
| `BBOX_THRESH` | `0.8` | detector score threshold |
| `USE_MASK` | `0` | `1` = mask-conditioned inference (needs a segmentor) |
| `INFERENCE_TYPE` | `full` | `full` (body+hand decoders) \| `body` \| `hand` |
| `SAVE_NPZ` | `1` | `1` = save per-image `<rel>.npz` of numeric outputs |

## Outputs
- **02 inference**: `OUTPUT_DIR/result/<rel>.jpg`（渲染叠加图，含 2D 骨架 + 3D 网格投影，等同官方 `demo.py` 输出）+ `OUTPUT_DIR/mesh/<rel stem>_mesh_<pid>.ply`（每人 3D 网格，可 3dviewer.net / Blender 打开）+ `OUTPUT_DIR/npz/<rel>.npz`（每人 `pred_vertices`、`pred_cam_t`、`pred_keypoints_3d/2d`、`focal_length`、`bbox`、姿态/形状参数；`SAVE_NPZ=0` 关闭）。

## 目录布局
```
<code-dir>/
├── media_code/                  # 本仓
│   ├── proxy.env                # 代理 + 覆盖项, gitignored
│   └── sam_3d_body/             # 编排脚本(本目录)
├── sam-3d-body/                 # 官方代码(自动 clone 到 ../sam-3d-body)
└── ../../model/sam-3d-body/     # 权重(在 <code-dir> 上一级, 各算法共享)
    ├── sam-3d-body-dinov3/      # facebook/sam-3d-body-dinov3 (GATED)
    │   ├── model.ckpt           # SAM 3D Body 主权重
    │   ├── model_config.yaml    # 模型配置
    │   └── assets/mhr_model.pt  # MHR 资产
    └── moge-2-vitl-normal/      # Ruicheng/moge-2-vitl-normal (公开, FOV 估计)
```
默认：官方代码 `../sam-3d-body`、权重 `../../model/sam-3d-body`（相对本目录）；用 `SAM3D_DIR` / `MODEL_DIR` 覆盖。复用现有 conda env（默认 `sam_3d_body`），但 SAM 3D Body 的 `detectron2 / networkx==3.2.1` pin 与其他算法冲突——建议专用 env。

## Notes
- Official code & weights follow their own license (SAM 3D Body = [SAM License](https://github.com/facebookresearch/sam-3d-body/blob/main/LICENSE)). This folder only orchestrates; no official code is copied.
- `.gitattributes` (repo root) forces LF so Windows-pushed scripts run cleanly on Ubuntu.
- `proxy.env` (proxy creds / path / env overrides) is gitignored — never committed. Don't put credentials in scripts.
- SSL behind a TLS-intercepting corporate proxy: pip uses `PIP_CERT`/`--trusted-host`; `hf`/`git` use the CA bundle (`_env.sh` prefers `~/.ca-bundle.crt`, built by `setup_ca_bundle.sh`).
- ViTDet detector weights auto-download from `dl.fbaipublicfiles.com` at runtime (not via HF); SAM2 segmentor is opt-in (needs `SEGMENTOR_PATH`); SAM3 detector/segmentor is opt-in (needs `SAM3=1`).

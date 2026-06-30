# VGGT-Omega runner

One-click orchestration to run [VGGT-Omega](https://github.com/facebookresearch/vggt-omega) (CVPR 2026 Oral) feed-forward 3D reconstruction on an Ubuntu + NVIDIA GPU server.
This folder holds **only orchestration scripts** — no official code, no weights. The official VGGT-Omega repo is cloned automatically; the checkpoint is downloaded from HuggingFace.

VGGT-Omega takes **a set of images of a scene** (or a video) and predicts, in a single forward pass, per-view **camera poses** + **depth maps** (+ confidence), which are unprojected into a colored point cloud. It is **not** a novel-view-synthesis / 3D-Gaussian model (unlike TripoSplat); `03_render_video.sh` therefore splats the reconstructed point cloud along an orbit rather than rasterizing gaussians.

## Design
- **Reuses an existing conda env** (default name `doll`) that already has a CUDA-enabled torch (>=2.3) — no torch download, no venv creation.
- Runtime deps (numpy/einops/safetensors/opencv/scipy/trimesh/gsplat/…) are installed **on demand**; or run `INSTALL_DEPS=1 bash vggt-omega/00_setup_env.sh` to install the known set in one shot (torch is NOT reinstalled).
- The `vggt_omega` package is imported via `sys.path` (no `pip install -e .` needed); `visual_util.py` (repo root) is reused for the official `.glb` export.
- Code and weights live outside this repo (see Layout).

## Layout (when this repo is cloned under your code dir)
```
<code-dir>/
├── media_code/              # this repo
│   ├── proxy.env            # proxy + HF_TOKEN + optional overrides, gitignored
│   └── vggt-omega/
│       ├── _env.sh                # shared: proxy + CA bundle + conda activate
│       ├── 00_setup_env.sh
│       ├── 01_download_models.sh
│       ├── 02_run_inference.sh
│       ├── run_all.sh
│       ├── setup_ca_bundle.sh     # one-time: extract proxy CA -> ~/.ca-bundle.crt
│       ├── _extract_ca.py         #   helper used by setup_ca_bundle.sh
│       ├── _hf_download.py        #   snapshot_download with SSL verify off (01 fallback)
│       ├── run_batch.py           # batch reconstruction (load model once, loop scenes)
│       ├── 03_render_video.sh     # render point-cloud .ply -> mp4 along a spiral (gsplat)
│       └── render_video.py        #   spiral point-cloud renderer (gsplat + imageio-ffmpeg)
├── vggt-omega/              # official code (auto-cloned to ../vggt-omega)
└── model/
    └── VGGT-Omega/          # checkpoint (hf download, gated)
```
Defaults: official code at `../vggt-omega`, weights at `../model/VGGT-Omega` (relative to this repo). Override with `VGGT_DIR` / `MODEL_DIR`.

## Prerequisites
- Ubuntu, NVIDIA driver (CUDA 12.x OK), `git`, `conda`
- A conda env with a CUDA-enabled **torch >= 2.3** already installed (default env name `doll`). Create one if needed:
  ```bash
  conda create -n doll python=3.10 -y && conda activate doll
  pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
  # VGGT-Omega needs torch>=2.3; cu118 gives 2.x. Verify:
  python -c "import torch;print(torch.__version__, torch.cuda.is_available())"
  ```
- NVIDIA GPU (A100 40G/80G ideal). Memory scales with #frames: ~6GB for 1 frame, ~13GB for 100, ~43GB for 500 at 624x416 (see the [repo's memory table](https://github.com/facebookresearch/vggt-omega#runtime-and-gpu-memory)). Use `RESOLUTION=256` or `MODE=max_size` to cut memory.

## ⚠️ Gated checkpoint (do this once, before running)
The HF repo `facebook/VGGT-Omega` is **gated** (auto-reviewed access request):
1. Request access at https://huggingface.co/facebook/VGGT-Omega (fill the form; usually approved quickly).
2. Create a **read** token at https://huggingface.co/settings/tokens.
3. Add it to `proxy.env` (repo root, gitignored):
   ```bash
   export HF_TOKEN="hf_your_token_here"
   ```
`01_download_models.sh` refuses to run without `HF_TOKEN`. (TripoSplat's weights were public; VGGT-Omega's are not.)

## Setup (on the server)
```bash
cd <your-code-dir>   # e.g. /data_3d/<uid>/code
git -c http.sslVerify=false clone https://github.com/CrescentVelvet/media_code.git
cd media_code
cp proxy.env.example proxy.env
# edit proxy.env: http_proxy / https_proxy / HF_TOKEN / CONDA_ENV
# bash vggt-omega/run_all.sh
# `run_all.sh`: activate conda env -> verify torch -> clone official repo -> download checkpoint -> reconstruct.
#   (Stops at 01 if HF_TOKEN missing or access not yet granted — add the token / wait for approval, then rerun 01.)

## Step-by-step
# bash vggt-omega/00_setup_env.sh        # activate env + verify torch (INSTALL_DEPS=1 to install deps)
sudo docker exec -it ff3dgs_v3 /bin/bash
conda activate doll
VARIANT=1b_512 bash vggt-omega/01_download_models.sh         # gated hf download (needs HF_TOKEN)
GPU=7 INPUT_DIR=/path/to/images bash vggt-omega/02_run_inference.sh   # folder of images -> scene.ply + .npz + .glb
GPU=7 PLY_INPUT=../vggt-omega/output/<scene> bash vggt-omega/03_render_video.sh  # .ply -> mp4 (spiral)
```
Missing a package? Just `pip install <pkg>` in the conda env and rerun the failed step.

## Inputs (what `INPUT_DIR` can be)
`02` reconstructs **one scene per item**. `INPUT_DIR` is one of:
- a **folder of images** (jpg/png/webp/…) → one scene, named after the folder
- a **video file** (`.mp4`/`.mov`/…) → one scene; frames auto-extracted at `VIDEO_FPS` (default 1 fps), like the official Gradio demo
- a **folder of scene folders** → one reconstruction per subfolder (batch; images directly in each subfolder, or under `<subfolder>/images/`)
- a **folder of videos** → one reconstruction per video (batch)

Out-of-the-box test: point `INPUT_DIR` at the repo's bundled example videos:
```bash
GPU=7 INPUT_DIR=../vggt-omega/examples VIDEO_FPS=1 bash vggt-omega/02_run_inference.sh
# -> reconstructs desert_road / forest_road / lake_speedboat / snow_lift (one scene per video)
```

## Render to video (.ply -> mp4)
Render a folder of point-cloud `.ply` (or a single `.ply`) along a spiral camera path (gsplat point splatting). Output path mirrors 02: `VIDEOS_DIR/<input_folder_name>/<stem>.mp4`.
```bash
git -c http.sslVerify=false pull
GPU=7 PLY_INPUT=../vggt-omega/output/desert_road bash vggt-omega/03_render_video.sh
# -> ../vggt-omega/videos/desert_road/scene.mp4  (+ scene.png first frame for a quick check)
```
Deps: `pip install gsplat plyfile imageio imageio-ffmpeg`. VGGT-Omega's point cloud has no gaussian attributes, so each point is splatted as a small isotropic gaussian (`POINT_SCALE` × scene extent; default `0.002`). If the cloud looks dotty, raise `POINT_SCALE`; if blobby, lower it. If the scene is sideways: run once, check `scene.png` — if `ROLL=90` makes it upright, read the printed `image-up (world)` vector and set `UP_VEC` to it (with `ROLL=0`) for a clean, non-tilting orbit; `UP_AXIS` (x/y/z) is the axis-aligned shortcut. Default `UP_VEC="0 -1 0"` (OpenCV camera convention: y-down → up is −Y) suits typical forward-moving camera footage. If the frame is all black: `VIEWMAT_C2W=1` or `BG=0.5` to debug. Tweak via `TURNS ELEV START_ANGLE FRAMES FPS FOV RADIUS_SCALE WIDTH HEIGHT UP_AXIS UP_VEC ROLL POINT_SCALE BG`.

## 可能遇到的问题

公司代理做 HTTPS 中间人解密，下面按流水线阶段列出常见报错与修法（命令在服务器上、`doll` 环境已激活时执行）。

**1. clone 本仓报 SSL / 认证**
公开仓免认证，加 `-c http.sslVerify=false` 即可（见上文 Setup）。若克隆的是私有仓且提示不能用账号密码，是 GitHub 已停用密码认证——改用公开仓或只读 PAT。

**2. `01` 报 `HF_TOKEN not set` / `401 Unauthorized`**
VGGT-Omega 权重是 gated 仓库。先在 https://huggingface.co/facebook/VGGT-Omega 申请访问（自动审核），再在 https://huggingface.co/settings/tokens 建一个 read token，写入仓根 `proxy.env`：
```bash
echo 'export HF_TOKEN=hf_xxx' >> proxy.env
VARIANT=1b_512 bash vggt-omega/01_download_models.sh
```
若已带 token 仍 `401`，多半是访问尚未批准，等几分钟再试。

**3. pip 装 torch 报 `SSL:CERTIFICATE_VERIFY_FAILED`**
脚本已内置 `--trusted-host`。仍失败时手动信任：
```bash
pip config set global.trusted-host "pypi.org pypi.python.org files.pythonhosted.org download.pytorch.org"
bash vggt-omega/00_setup_env.sh
```

**4. pip 装 torch 报 `HTTPSConnectionPool`（连接超时/断开）**
不是 torch 版本问题，是代理对大文件超时。加超时重试，或退回 PyPI 默认 torch（自带 CUDA，A100 可用）：
```bash
pip install --timeout 600 --retries 10 --trusted-host download.pytorch.org \
  --index-url https://download.pytorch.org/whl/cu118 torch torchvision
python -c "import torch;print(torch.cuda.is_available(), torch.__version__)"  # 需 True + >=2.3
```

**5. `hf download` 报 `CAS service error : ReqwestMiddleware`**
HF 的 Xet/Rust 通道不认代理。`_env.sh` 已设 `HF_HUB_DISABLE_XET=1`；仍报则彻底卸载：
```bash
pip uninstall -y hf_xet
VARIANT=1b_512 bash vggt-omega/01_download_models.sh
```

**6. `hf download` 报 `SSLCertVerificationError`**
代理根 CA 不在系统证书包。先一次性建包，`_env.sh` 会自动用 `~/.ca-bundle.crt`：
```bash
bash vggt-omega/setup_ca_bundle.sh    # 抓代理证书链 -> ~/.ca-bundle.crt，并自检
VARIANT=1b_512 bash vggt-omega/01_download_models.sh
```
- 自检 `[OK]` → 直接重跑 `01`。
- 自检 `[FAIL]`（代理握手没带根 CA）→ 把公司根 CA 追加后再重跑：
  ```bash
  cat /path/to/corporate_root_ca.crt >> ~/.ca-bundle.crt
  ```
  公司根 CA 常见于 `/usr/local/share/ca-certificates/`（脚本已自动并入该目录）。

> 若建包后仍报 SSL（CDN 端点 `us.aws.cdn.hf.co` 等用了不同的 MITM 证书），`01` 会自动回退到禁用 SSL 校验的下载器（`_hf_download.py`）；或直接 `HF_DISABLE_SSL=1 bash vggt-omega/01_download_models.sh` 跳过首次尝试。代理已全程 MITM，此处关掉校验可接受。

**7. `torch.cuda.OutOfMemoryError`**
显存随帧数线性增长。降压：`RESOLUTION=256`、`MODE=max_size`，或喂更少帧（视频则降 `VIDEO_FPS`）。`run_batch.py` 会捕获 OOM 并继续下一个场景。

**8. 缺包 `ModuleNotFoundError`**
按需补，或一次性装齐已知小依赖：
```bash
pip install <包名>
# 或：
INSTALL_DEPS=1 bash vggt-omega/00_setup_env.sh
```
`scene.glb` 导出需要 `trimesh scipy matplotlib`（缺则跳过，仅产出 `.ply`/`.npz`）。`03` 渲染需要 `gsplat plyfile imageio imageio-ffmpeg`。

> 通用：`proxy.env`（代理凭证 + `HF_TOKEN`）在仓内 gitignored，`~/.ca-bundle.crt` 在家目录，都不入库；切勿把凭证写进脚本。

## Config (env vars, all optional)
| var | default | note |
|---|---|---|
| `CONDA_ENV` | `doll` | conda env to activate (must already have torch>=2.3) |
| `GPU` | _(unset)_ | physical GPU id to pin, e.g. `GPU=3`; remaps `CUDA_VISIBLE_DEVICES` so in-process `cuda:0` == that card |
| `VGGT_DIR` | `../vggt-omega` | official code path |
| `MODEL_DIR` | `../model/VGGT-Omega` | checkpoint path |
| `VGGT_REPO` | official GitHub URL | clone source |
| `HF_REPO_ID` | `facebook/VGGT-Omega` | gated weights repo (needs `HF_TOKEN`) |
| `VARIANT` | `1b_512` | checkpoint variant: `1b_512` (512px) or `1b_256_text` (256px, text-aligned; auto sets `enable_alignment`) |
| `RESOLUTION` | `512` | input image resolution (use `256` with `1b_256_text`) |
| `MODE` | `balanced` | resize mode: `balanced` (token-budget) or `max_size` (longest side = RESOLUTION; less memory) |
| `INPUT_DIR` | `../vggt-omega/examples` | folder of images/videos, or a single video (see Inputs) |
| `OUTPUT_DIR` | `../vggt-omega/output` | reconstructions root; each scene -> `OUTPUT_DIR/<scene>/` |
| `CONF_THRES` | `20` | depth-confidence percentile kept (0–100; higher = sparser but cleaner) |
| `MAX_POINTS` | `2000000` | cap on points saved to `scene.ply` (0 = no cap) |
| `MASK_SKY`/`MASK_BLACK_BG`/`MASK_WHITE_BG` | `0` | optional point filters (sky needs onnxruntime skyseg) |
| `VIDEO_FPS` | `1` | frame sampling fps when `INPUT_DIR` is a video |
| `INSTALL_DEPS` | `0` | set `1` to install known runtime deps in 00 |
| `HF_HUB_DISABLE_XET` | `1` | disable HF Xet/CAS Rust path (proxy-unfriendly) |
| `HF_DISABLE_SSL` | `0` | set `1` to download the checkpoint with SSL verification disabled |
| `PLY_INPUT` | `../vggt-omega/output` | .ply file or folder to render (03) |
| `VIDEOS_DIR` | `../vggt-omega/videos` | base video dir; mp4s go to `VIDEOS_DIR/<scene>/` |
| `WIDTH`×`HEIGHT` | `1280`×`720` | render resolution (03) |
| `TURNS`/`ELEV`/`FRAMES`/`FPS` | `1`/`-15°`/`120`/`30` | spiral trajectory params (03) |
| `START_ANGLE` | `0` | starting azimuth in degrees (03) |
| `FOV` | `55` | camera field of view in degrees (03) |
| `UP_AXIS` | `y` | camera up axis (x/y/z) (03) |
| `UP_VEC` | `0 -1 0` | object's up as "x y z" (overrides `UP_AXIS`); set to frame0 `image-up` for a clean orbit (03) |
| `ROLL` | `0` | camera roll around forward axis (deg); try 90/-90/180 if the scene is sideways (03) |
| `POINT_SCALE` | `0.002` | splat radius as a fraction of scene extent (03; raise if dotty, lower if blobby) |

## Outputs
`02` reconstructs each scene into `OUTPUT_DIR/<scene>/`. For each scene:
- `scene.ply` — confidence-filtered colored point cloud (raw world coords; view in MeshLab/SuperSplat, or feed to `03`)
- `predictions.npz` — raw model outputs: `depth`, `depth_conf`, `extrinsic`, `intrinsic`, `world_points_from_depth`, `images`, `pose_enc`, `camera_and_register_tokens` (same keys the official `demo_gradio.py` saves)
- `scene.glb` — official visualization (point cloud + camera frustums), built via `visual_util.predictions_to_glb` (needs `trimesh/scipy`; skipped otherwise)
- `frames/` — the images actually fed to the model (copied from the input folder, or extracted from the input video)

`03` writes `VIDEOS_DIR/<scene>/<stem>.mp4` (+ `<stem>.png` first frame).

## Notes
- VGGT-Omega is **feed-forward** (one model pass → poses + depth for all input views). It does **not** optimize a scene representation, so there is no train/test split, no per-scene fitting, and no novel-view gaussians — the orbit video in `03` just splats the unprojected point cloud.
- Two checkpoints exist: `vggt_omega_1b_512.pt` (default, 512px, no text alignment) and `vggt_omega_1b_256_text.pt` (256px, text-aligned; reads `predictions["text_alignment_embedding"]`). `01`/`02` select via `VARIANT`; `02` auto-enables `VGGTOmega(enable_alignment=True)` for the text variant.
- Official code & weights follow their own license. This folder only orchestrates; no official code is copied.
- `.gitattributes` (repo root) forces LF so Windows-pushed scripts run cleanly on Ubuntu.
- `proxy.env` (proxy creds + `HF_TOKEN` / path / env overrides) is gitignored — never committed. Don't put credentials in scripts.
- SSL behind a TLS-intercepting corporate proxy: pip uses `--trusted-host`; `hf`/`git` use the CA bundle (`_env.sh` prefers `~/.ca-bundle.crt`, built by `setup_ca_bundle.sh`).

# SplatAnything（InfiniSplat v3）— 前馈多视角 3DGS，跨任意场景（WSL 复现）

从多张 RGB 图（可有/无位姿）一次前馈重建 3D Gaussian 场景。相对 v2 的核心改动是
**Gaussian Spacing Normalization（GSN）**：把每个场景的中位"每像素前向间距"
（`depth / sqrt(fx·fy)`）归一到固定目标值 0.0013227513，深度与相机平移同步缩放，
于是体素合并可以用固定 `voxel_size=0.002`，不再逐场景自适应。

- 论文：SplatAnything: Gaussian Spacing Normalization for Feed-Forward 3DGS across Arbitrary Scenes（ZJU3DV + 腾讯混元）
- 官方仓：<https://github.com/zju3dv/SplatAnything>（本仓实测 commit `b518996`）
- 权重：<https://huggingface.co/ritianyu/SplatAnything>（apache-2.0，非 gated）
- 原理与坑：见 [NOTES.md](NOTES.md)；实测数字见 EXPERIMENTS.md
- 前作参照：[../infinisplat_v2/](../infinisplat_v2/)（DA3 基座 + 自适应 voxel）

## ⚠️ 复现边界（先读这段）

官方**只开源推理包**（`rasterization.py` 注释写明 "no rasterization or training"）：
没有训练代码、没有 refiner、没有光栅化器、没有官方评测划分。

| 项 | 状态 | 说明 |
| --- | --- | --- |
| 前馈推理（rgb / pose 两模式） | ✅ 可复现 | 代码 + 权重齐全，`02_run_inference.sh` |
| GSN + 固定体素合并 + scale/opacity 过滤 | ✅ 可复现 | 参数全在 `configs/inference.yaml` + 命令行 |
| 导出标准 3DGS PLY | ✅ 可复现 | SH DC / log scale / logit opacity / wxyz |
| 渲染与 PSNR/SSIM | ⚠️ 需自接 | 发布包不带 gsplat，`03_eval_render.sh` 自接 |
| 训练（stage-1/2） | ❌ 不可得 | 无训练代码 |
| 论文表里的精确数值 | ⚠️ 不可直接对标 | 官方划分未公开；本仓只做自洽 + 自建 holdout |

## 常用命令

> 首次跑前先做「首次准备」。每条命令都显式写出权重、输入、输出路径。

```bash
# ── A) 官方示例（先检出 examples，仓库 418 MB 只拉需要的那一类）──
git -C ~/repos/SplatAnything sparse-checkout add examples/object

GPU=0 EXAMPLE=object MODE=rgb \
  bash splatanything/02_run_inference.sh
#   → ~/output/splatanything_results/rgb/object/{gaussians.ply,points.ply,cameras.json,depth/,rgb/}

GPU=0 EXAMPLE=object MODE=pose RESOLUTION="504 504" \
  bash splatanything/02_run_inference.sh

# ── B) 自有场景（目录里必须有 images/，pose 模式再加 prior_camera.json）──
GPU=0 INPUT=/home/velvet/output/splatanything_results/_scenes/jiajia_hold MODE=pose \
  bash splatanything/02_run_inference.sh

# ── C) 华为 Remy 采集包 → 场景目录 → 推理（一条链）──
GPU=0 CAPTURE="/mnt/d/dataset/手部微动样例/01e954de1b8a4cde9315c976f4dcb964-佳佳抱枕" \
  SCENE=jiajia_hold N_VIEWS=24 \
  bash splatanything/07_run_capture.sh

GPU=0 INPUT="$HOME/output/splatanything_results/_scenes/jiajia_hold" MODE=pose \
  bash splatanything/02_run_inference.sh

# ── D) 渲染评测（自洽口径；缺 gsplat 会自动装）──
GPU=0 RESULT="$HOME/output/splatanything_results/pose/object" \
  bash splatanything/03_eval_render.sh

# ── E) 不加载权重验链路（CPU，秒级；改完代码先跑这个）──
python splatanything/04_smoke_cpu.py
```

## 首次准备

```bash
# 1) clone 本仓 + proxy.env
cd ~ && git clone <本仓地址> && cd media_code && cp proxy.env.example proxy.env

# 2) 建 env（python 3.11 + torch 2.7.1）+ clone 官方仓（稀疏）+ 装依赖 + 自检
bash splatanything/00_setup_env.sh

# 3) 下权重（16.6 GB，走 aifasthub 镜像，分块并发 + SHA256）
bash splatanything/01_download_models.sh
```

权重目录布局（`$SPLATANYTHING_MODEL_DIR`，默认 `/mnt/d/wheel/splatanything_ms`）：

```
/mnt/d/wheel/splatanything_ms/
├── SplatAnything.ckpt          11.3 GB  完整高斯学生（stage-1）
├── worldmirror2.safetensors     5.05 GB  WorldMirror-2 几何（teacher）
└── skyseg.onnx                  176 MB  可选，--sky_seg 才用
```

`01` 会自动在官方仓 `checkpoints/` 建同名软链（`configs/inference.yaml` 的路径锚在 repo root）。

## Config（env vars）

| var | default | note |
| --- | --- | --- |
| `MODE` | `rgb` | `rgb` = 模型自估位姿；`pose` = 读 `INPUT/prior_camera.json` |
| `RESOLUTION` | `756 756` | 模型输入宽高，必须 14 的正整数倍；文档称支持 1K+ |
| `INPUT` / `EXAMPLE` | — | 场景目录（含 `images/`）；`EXAMPLE` 指官方 `examples/<名>` |
| `OUT_ROOT` | `$RESULTS_DIR/$MODE` | 输出根；inference 再套一层 `<scene>/` |
| `SKY_SEG` | `0` | `1` = 天空分割（需 `skyseg.onnx`），天空像素同时不参与 spacing 统计 |
| `OPACITY_THRESH` | YAML `0.05` | 导出层的最小不透明度 |
| `EXTRA_ARGS` | — | 透传，如 `--no_cam_align`（不锚到首个视角） |
| `PROFILE` | `1` | 采样 GPU 峰值显存 |
| `CAPTURE` / `N_VIEWS` | — / `24` | 07：采集包路径 / 等间隔抽帧数（2..50） |
| `EPS2D` | `0.01` | 03：屏幕空间膨胀 |
| `CONDA_ENV` | `splatanything` | |

`configs/inference.yaml` 只接受固定字段（多一个就报错），`spacing_target` 和
`decode_chunk_size` 是代码常量、**不能**从 YAML 改。

## 目录布局

```
media_code/splatanything/
├── _env.sh                  # 代理 / conda / 路径（含 CDN 直连的 no_proxy）
├── 00_setup_env.sh          # 稀疏 clone + conda env + torch + 依赖 + 自检
├── 01_download_models.sh    # 3 个权重 + SHA256 + checkpoints 软链
├── 02_run_inference.sh      # 推理（含显存采样与产物统计）
├── 03_eval_render.sh        # 自接 gsplat 渲染 + PSNR/SSIM（自洽口径）
├── 03_render_eval.py
├── 04_smoke_cpu.py          # CPU 全链路自检（不加载真权重，28×28 图，秒级）
├── 07_run_capture.sh        # Remy 采集包 → 场景目录
├── 07_capture_to_scene.py
├── hf_download.py           # 镜像分块并发下载（含 --noproxy）
├── NOTES.md                 # 原理（GSN / 体素合并 / 预处理）与排错
└── EXPERIMENTS.md           # 实测台账

官方仓（Linux fs，稀疏检出）: ~/repos/SplatAnything
权重（D 盘）              : /mnt/d/wheel/splatanything_ms
输出（Linux fs）          : ~/output/splatanything_results
```

## 输入格式

```
my_scene/
├── images/               # jpg/png/webp/bmp/tif；文件名 stem 必须唯一（用作 camera_id）
└── prior_camera.json     # pose 模式必需；rgb 模式忽略（即使存在）
```

`prior_camera.json` 要 **OpenCV c2w**（X 右 / Y 下 / Z 前）+ **原图像素**内参：

```json
{ "extrinsics": [{"camera_id": "000", "matrix": [[...4x4 c2w...]]}],
  "intrinsics": [{"camera_id": "000", "matrix": [[fx,0,cx],[0,fy,cy],[0,0,1]]}] }
```

- `camera_id` = 图片**文件名去扩展名**，必须每张图都有条目
- 内参要求零 skew、主点落在原图内（否则 loader 直接抛错）
- 已有 COLMAP 的用官方 `scripts/convert_cameras.py colmap` 转；自有矩阵用 `arrays` 模式
- 相机中心会被重新锚到第一个视角（`--no_cam_align` 可关），且**导出坐标不是米制**（GSN 归一化过）

## Notes

- 预处理以**主点为中心裁正方形**再缩到 `RESOLUTION`：1920×1080 只保留中间 1080×1080。
  `rgb/` 存的是裁后的图，评测必须用它当 GT。
- 官方包两个已知缺陷：`validation.py` 缺失（导致 `tests/test_inference.py` 无法运行）、
  `examples/object/prior_camera.json` 的 `num_cameras` 与实际条目数不一致（不影响运行）。
  详见 NOTES.md 第 6 节。
- 下载慢先查 `no_proxy`：镜像站 302 到 `us.aws.cdn.hf.co` / `cas-bridge.xethub.hf.co`，
  这两个域名被 v2rayN 代理接管后实测只有 57 KB/s（直连 1.7 MB/s）。

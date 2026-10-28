# Hunyuan3D-2.1 — 单图/多视图 → PBR 纹理 GLB

腾讯 Hunyuan3D-2.1 图生 3D 的一键编排：shape（`hunyuan3d-dit-v2-1` 单图 3.3B，或 `hunyuan3d-dit-v2-mv` 多视图）→ PBR 纹理（`hunyuan3d-paint-v2-1` + RealESRGAN 增强）→ GLB。本目录只有编排脚本，官方代码由 `run_all.sh` 自动 clone，权重走 HuggingFace。

> ⚠️ **协议**：代码文件头原文为 *TENCENT HUNYUAN **NON-COMMERCIAL** LICENSE AGREEMENT*（研究/非商用），与 MIT 的 TRELLIS.2 不同，商用前必须核对条款。
> ⚠️ **2.1 没有自己的多视图权重**：`MODE=mv` 用的是 2.0 代的 `tencent/Hunyuan3D-2mv`（2025-03 发布），由 2.1 代码库加载（pipeline 的 `image` 参数接受 `dict{front,left,back,right}`）。视图接受任意非空子集，缺位不会重编号。

显存（官方数据）：shape 10 GB / texture 21 GB / 同时驻留 29 GB。`run_inference.py` **两段顺序加载**（shape 出网格后释放再载 paint），峰值 ≈21 GB，单卡 24 GB（3090）可跑。

## 常用命令
> 假设已进入 Linux/WSL 且 `conda` 可用；首次跑前先做下方「首次准备」。
> **铁律：每条命令都必须显式写出模型地址、输入路径、输出路径，不能全靠脚本里的默认值。**

```bash
# 单图（2.1 shape + PBR 纹理，质量最高）
GPU=0 MODE=single \
  IMAGE=/path/to/object.png \
  SHAPE_DIR=~/model/Hunyuan3D-2.1 DINO_DIR=~/model/dinov2-giant \
  OUTPUT_DIR=~/output/hunyuan3d_2_1_results \
  bash hunyuan3d_2_1/02_run_inference.sh

# 三视图（2mv shape + PBR 纹理；front/back/left/right 任意子集）
GPU=0 MODE=mv \
  FRONT=/path/to/front.png LEFT=/path/to/left.png BACK=/path/to/back.png \
  SHAPE_DIR=~/model/Hunyuan3D-2.1 MV_DIR=~/model/Hunyuan3D-2mv DINO_DIR=~/model/dinov2-giant \
  OUTPUT_DIR=~/output/hunyuan3d_2_1_results OUTPUT_NAME=my_object \
  bash hunyuan3d_2_1/02_run_inference.sh

# 只出几何不要纹理（快速验证 shape）
GPU=0 MODE=single IMAGE=/path/to/object.png SKIP_TEXTURE=1 \
  OUTPUT_DIR=~/output/hunyuan3d_2_1_results \
  bash hunyuan3d_2_1/02_run_inference.sh

# ── 批量：视频抽帧的三视图（角度未知）──────────────
# 01b 预处理：rembg 抠图 → VGGT-Omega 估相机位姿 → 方位角吸附到
# front/left/back/right 槽位（误差>45° 或弧度过窄自动降级单图），产 views.json
GPU=0 INPUT_ROOT="/mnt/d/dataset/物体三图数据" \
  bash hunyuan3d_2_1/01b_preprocess_views.sh

# 02b 批量推理（shape 全跑完再统一 paint，模型只加载两次）
GPU=0 bash hunyuan3d_2_1/02b_run_batch.sh
# -> $RESULTS_DIR/models/<物体名>/<物体名>_textured.glb
```

输出（`OUTPUT_DIR/` 下）：
- `<stem>_untextured.glb` — shape 阶段的无纹理网格
- `<stem>_textured.obj` + `<stem>_textured.glb` — PBR 纹理成品（GLB 可直接进 Blender / 3dviewer.net）
- paint 阶段中间产物 `white_mesh_remesh.obj` 也在同目录

**为什么 01b 必须跑**（官方源码实证）：
- shape pipeline 对路径输入**不做抠图**（`hy3dshape/preprocessors.py` 的 `load_image` 只在有 alpha 通道时才生成 mask；rembg 在官方 demo 层 `minimal_demo.py` 而不在 pipeline 里）——杂乱背景照片必须预处理成 RGBA PNG
- mv 模型给每个视图学了一个**槽位 embedding**（`MVImageProcessorV2`：front/left/back/right = 0°/顺时针90°/180°/270°），乱贴标签 = 给错误角度条件 → 几何互相打架。所以 01b 用 VGGT-Omega 估真实方位角再吸附槽位

## 首次准备
```bash
# clone 本仓 + proxy.env
cd <code-dir>
git clone ... && cd media_code && cp proxy.env.example proxy.env
# ⚠️ 确认 proxy.env 中 http_proxy / https_proxy 已取消注释
# WSL 用户在 proxy.env 里覆盖路径：HY3D_DIR=~/repos/Hunyuan3D-2.1、MODEL_DIR=~/model

# 建 env + 装依赖 + 编译两个 CUDA 扩展（需要 CUDA Toolkit，推荐 12.4）
CREATE_ENV=1 bash hunyuan3d_2_1/00_setup_env.sh
# 或复用已有 CUDA env：CONDA_ENV=<name> bash hunyuan3d_2_1/00_setup_env.sh

# 下权重（2.1 shape+paint ≈ 15 GB，2mv ≈ 5 GB，dinov2-giant ≈ 1.3 GB）
bash hunyuan3d_2_1/01_download_models.sh          # SKIP_MV=1 可不下 2mv
```

权重目录布局：
```
$MODEL_DIR/
  Hunyuan3D-2.1/          # tencent/Hunyuan3D-2.1 快照
    hunyuan3d-dit-v2-1/   # shape（3.3B）
    hunyuan3d-paint-v2-1/ # PBR 纹理
  Hunyuan3D-2mv/          # tencent/Hunyuan3D-2mv 快照
    hunyuan3d-dit-v2-mv/  # 多视图 shape
  dinov2-giant/           # paint 阶段的图像编码器
```

---

以下为详细参考。

## Config (env vars)
| var | default | note |
|---|---|---|
| `CONDA_ENV` | `hunyuan3d` | conda env 名（须含 CUDA torch；`CREATE_ENV=1` 时自动建） |
| `CREATE_ENV` | `0` | `1` = 新建 env（python 3.10 + torch 2.5.1+cu124，官方实测组合） |
| `INSTALL_DEPS` | `1` | `0` = 跳过 00 的依赖安装与扩展编译 |
| `GPU` | _(unset)_ | 物理卡号，重映射 `CUDA_VISIBLE_DEVICES` |
| `HY3D_DIR` | `../Hunyuan3D-2.1` | 官方代码路径（WSL 建议 `~/repos/Hunyuan3D-2.1`） |
| `MODEL_DIR` | `../../model` | 权重根目录（WSL 建议 `~/model`） |
| `SHAPE_DIR` / `MV_DIR` / `DINO_DIR` | `$MODEL_DIR/{Hunyuan3D-2.1,Hunyuan3D-2mv,dinov2-giant}` | 各权重快照 |
| `RESULTS_DIR` / `OUTPUT_DIR` | `../hunyuan3d_2_1_results` | 输出目录 |
| `MODE` | `single` | `single`（IMAGE=）/ `mv`（FRONT/BACK/LEFT/RIGHT 任意子集） |
| `INPUT_ROOT` | — | 01b：物体文件夹的根目录（每个子文件夹一个物体的多视角图） |
| `PREP_DIR` | `$RESULTS_DIR/preprocessed` | 01b 抠图与 views.json 输出 |
| `VGGT_OUT` | `$RESULTS_DIR/vggt` | 01b 的 VGGT-Omega 位姿缓存（已存在则复用） |
| `VGGT_CONDA_ENV` | `vggt_human` | 01b 第 2 步调 vggt-omega 所用的 conda env |
| `MAX_SLOT_ERR` | `45` | assign_views：方位角吸附槽位的最大误差（度） |
| `FLIP_LR` | `0` | 02b：左右互换（VGGT 方位角顺/逆时针约定与模型槽位相反时用） |
| `FORCE` | `0` | 02b：重跑已有成品 GLB 的物体 |
| `SEED` | `0` | 同 seed + 同输入 => 同网格 |
| `NUM_INFERENCE_STEPS` | `50` | shape 采样步数 |
| `GUIDANCE_SCALE` | `5.0` | shape CFG |
| `OCTREE_RESOLUTION` | `384` | shape 提取网格的 octree 分辨率 |
| `NUM_CHUNKS` | `8000` | shape VAE 解码分块 |
| `TEX_MAX_NUM_VIEW` | `6` | paint 选用的视角数 |
| `TEX_RESOLUTION` | `512` | paint 多视图扩散分辨率 |
| `SKIP_TEXTURE` | `0` | `1` = 只出无纹理 GLB |
| `CUDA_HOME` | `/usr/local/cuda` | 编译扩展用的 CUDA Toolkit |
| `HF_HUB_DISABLE_XET` | `1` | 关掉 HF Xet/CAS 通道（代理不友好） |
| `HF_DISABLE_SSL` | `0` | `1` = 权重下载直接走免 SSL 校验的 fallback |

## 目录布局
```
<code-dir>/
├── media_code/hunyuan3d_2_1/   # 本目录（编排脚本）
├── Hunyuan3D-2.1/              # 官方代码（run_all 自动 clone）
└── ../model/                   # 权重（见上）
```

## Notes（坑位速查）
- **cwd 敏感**：paint 管线的配置/ckpt 默认路径是相对官方仓根的（`hy3dpaint/cfgs/...`、`ckpt/...`），`run_inference.py` 已 `chdir(HY3D_DIR)` 并把所有路径改成绝对路径覆盖，自己改代码时注意别破坏这一点。
- **mv 输入要求**：front/back/left/right 应为**去背景、主体居中**的方图（官方 demo 均先 rembg）。随手拍照片建议先过一遍抠图（如 rembg / SAM），否则多视图条件会带偏形状。
- **编译失败**：两个扩展（`custom_rasterizer`、`DifferentiableRenderer`）需要 nvcc 与 `CUDA_HOME` 指向 Toolkit（非驱动版本）；多版本并存时 `CUDA_HOME=/usr/local/cuda-12.4` 显式指定后重跑 00。
- **HF 下不动**（WSL 常见）：先 `curl -sI https://huggingface.co` 复测网络；不通就换 `HF_ENDPOINT=https://hf-mirror.com`，或手动下载后把目录指给 `SHAPE_DIR/MV_DIR/DINO_DIR`。
- **代理 SSL**：报 `SSLCertVerificationError` 时，`HF_DISABLE_SSL=1 bash hunyuan3d_2_1/01_download_models.sh` 走免校验下载（代理已全程 MITM，可接受）；其余代理/CA 处理与 `trellis-2/README.md` 的「可能遇到的问题」一节相同。
- **许可证**：代码与权重均为腾讯混元非商用协议；其依赖（RealESRGAN 等）各有自己的协议。

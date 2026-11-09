# Scene Splatter — 单图 + 视频扩散动量引导 → 3DGS（CVPR 2025，清华/腾讯）

复现 [Scene-Splatter](https://github.com/shengjun-zhang/Scene-Splatter)：单张图 + 自定义相机轨迹 →
Flash3D 前馈初始化 3DGS → ViewCrafter（DynamiCrafter 系 I2V，~2B）动量引导视频增强 →
迭代 3DGS 优化（每轮 5000 步），输出可渲染 3DGS。

> ⚠️ 官方原版**不落盘 PLY**（只有过程视频）；本仓 `patch_save_ply.py` 补了 `scene.ply` 导出。
> ⚠️ 输入相机轨迹需自备：用本仓 `02_gen_traj.py` 生成（官方只给示例不给生成器）。

## 常用命令
> 假设已在 WSL 且 conda env `scene_splatter` 可用；首次跑前先做下方「首次准备」。
> **铁律：每条命令显式写出模型地址、输入路径、输出路径。**

```bash
# 一键：单图 → 3DGS（PLY + 过程视频）
GPU=0 IMAGE=<code-dir>/data/scene.jpg \
  YAW_SPAN=40 NUM_FRAMES=100 \
  bash scene_splatter/run_all.sh

# 分步 1) 生成相机轨迹（pickle.gz，w2c OpenCV 约定 + 归一化 flat-4 内参）
GPU=0 IMAGE=<code-dir>/data/scene.jpg \
  OUT=~/output/scene_splatter_results/cameras/scene.pickle.gz \
  NUM_FRAMES=100 YAW_SPAN=40 RADIUS=1.0 FOV_Y=60 \
  python scene_splatter/02_gen_traj.py

# 分步 2) 推理
GPU=0 IMAGE=<code-dir>/data/scene.jpg \
  CAMERA_PATH=~/output/scene_splatter_results/cameras/scene.pickle.gz \
  RESULTS_DIR=~/output/scene_splatter_results \
  bash scene_splatter/02_run_inference.sh

# 分步 3)（备用）PLY 自定义轨迹重渲染
GPU=0 PLY=~/output/scene_splatter_results/02_infer/<ts>/scene.ply \
  OUT=~/output/scene_splatter_results/03_render/orbit.mp4 \
  bash scene_splatter/03_render_ply.sh
```

## 首次准备
```bash
# clone 本仓 + proxy.env（media_code 仓级，已有则跳过）
cd <code-dir>
git clone ... && cd media_code && cp proxy.env.example proxy.env

# 建 env + 编译光栅器（WSL：torch 2.1.2+cu118，nvcc 走 conda）
INSTALL_DEPS=1 bash scene_splatter/00a_setup_env.sh

# 下三份权重（flash3d v2 / ViewCrafter_25 / DUSt3R ViT-L）
bash scene_splatter/01_download_models.sh
```

权重目录布局：
```
$SS_DIR = ~/repos/Scene-Splatter（官方仓，gaussianSplatting/flash3d/viewcrafter 均 vendored）
├── flash3d/model_re10k_v2.pth                        ← 软链 → D:\wheel\scene_splatter\
├── viewcrafter/checkpoints/model.ckpt                ← 软链（ViewCrafter_25）
└── viewcrafter/checkpoints/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth  ← 软链
```

---

以下为详细参考。

## Config (env vars)
| var | default | note |
|---|---|---|
| `IMAGE` | — | 输入单图（1024×576 内参口径，pipeline 内部 resize） |
| `CAMERA_PATH` | run_all 自动生成 | pickle.gz：`{intrinsics: [flat-4], poses: [w2c 4x4]}` |
| `NUM_FRAMES` | 100 | 轨迹总帧数；迭代轮数 ≈ (N/interval)/(per_video_length-overlap) |
| `YAW_SPAN` | 40 | 环绕角度（度），对齐 FlashWorld 默认 orbit |
| `RADIUS` / `HEIGHT` | 1.0 / 0.0 | 轨道半径 / 高度偏移 |
| `FOV_Y` | 60 | 垂直视场角 |
| `PER_VIDEO_LENGTH` | 16 | 每段视频帧数（ViewCrafter ckpt 口径） |
| `NUM_OVERLAP` | 2 | 段间重叠帧 |
| `CAM_INTERVAL` | 2 | 轨迹抽稀间隔 |
| `RESULTS_DIR` | ~/output/scene_splatter_results | 产物根（08 搬到 /mnt/d） |

## 输出
```
$RESULTS_DIR/02_infer/<时间戳>/
├── scene.ply            # 最终 3DGS（media_code 补丁导出）
├── flash3d_video.mp4    # Flash3D 初始化渲染
├── render_video_{i}.mp4 # 每轮迭代后渲染（最后一段 = 最终质量）
└── confidence_maps/     # 像素级动量融合权重
```

## 目录布局
```
<code-dir>/
├── media_code/scene_splatter/     # 本目录（编排脚本 + 补丁）
└── Scene-Splatter/                # 官方仓（WSL: ~/repos/Scene-Splatter）
```

## License
官方 MIT（含 vendored 的 flash3d/3DGS 各 LICENSE，见官方仓）。

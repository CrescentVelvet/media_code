# deformable_human — 单目视频人体重建去模糊/重影（canonical + 形变场）

针对的问题：**单目视频拍人，人物有动作 → 静态重建把多时刻几何平均，出现重影/拖影**。
路线：canonical 高斯 + 形变 MLP 显式建模运动（[Deformable-3D-Gaussians](https://github.com/ingra14m/Deformable-3D-Gaussians)，
CVPR 2024），先跑 vanilla baseline 量化去重影效果；Phase 2 再叠 MHR 3DMM 锚定
（canonical 模板初始化 + 骨架/landmarks 正则，复用 vggt_human 的 MHR 与 657 点 landmarks）。

与同仓兄弟目录的关系：
- **`deformable_gaussians/`**：同一官方仓的服务器侧编排（py3.7/torch1.13 独立 env，D-NeRF 复现 + NeRF-DS 真实序列）。
  本目录**复用**它的 `03_colmap_pose.sh`（COLMAP 建场景），不重写。
- **`vggt_human/`**：Stage A 复用它的 `01a_video_to_frames.sh`（抽帧+模糊门）；Phase 2 复用 MHR/landmarks 能力。
  文件做接口，**不复制代码**。
- 本目录 env 从 **vggt_human clone**（torch 2.5.1+cu121，nvcc 随 clone 继承），与 deformable_gaussians 的
  py3.7/torch1.13 env 互不干扰。

> 本仓训练型方法：无预训练权重，每个场景从头训一套 canonical 高斯 + 形变 MLP。

## 常用命令
> **WSL 本机请直接用 [README_wsl.md](README_wsl.md)**（主战场）。以下为服务器版。
> 铁律：每条命令显式写出输入路径、输出路径，不靠默认值。

```bash
# ── 一键（先完成首次准备）──
GPU=0 VIDEO_PATH=/data_3d/<uid>/data/xxx.mp4 SCENE_NAME=human_seq \
  RESULTS_DIR=../deformable_human_results \
  bash deformable_human/run_all.sh

# ── 分步 ──
# 1) 单目视频 → 帧（带模糊门）→ COLMAP 场景
GPU=0 VIDEO_PATH=/data_3d/<uid>/data/xxx.mp4 SCENE_NAME=human_seq \
  VIDEO_FPS=6 BLUR_THRESHOLD=100 \
  RESULTS_DIR=../deformable_human_results \
  bash deformable_human/01_prepare_data.sh
# 1c)（可选）位姿规整：主体居中 + 重力对齐 + 尺度归一化
GPU=0 SCENE_NAME=human_seq RESULTS_DIR=../deformable_human_results \
  bash deformable_human/01c_pose_adjust.sh
# 2) vanilla Deformable-GS 训练（canonical + 形变 MLP，NeRF-DS 模式）
#    可选：USE_POSE_REFINE=1 联合精炼位姿（可学四元数+平移，内参不学）
GPU=0 SCENE_NAME=human_seq RESULTS_DIR=../deformable_human_results \
  ITERATIONS=20000 \
  bash deformable_human/02_train.sh
# 3) 渲染 + PSNR/SSIM/LPIPS（看运动区域是否还有拖影：MODE=original）
GPU=0 SCENE_NAME=human_seq RESULTS_DIR=../deformable_human_results \
  MODE=render RUN_METRICS=1 \
  bash deformable_human/03_render.sh
```

## 首次准备
```bash
# clone 本仓 + proxy.env（略，见仓级 README）

# 建 env（从 vggt_human clone）+ clone 官方仓 + 装依赖 + 编 CUDA 子模块
CLONE_FROM=vggt_human INSTALL_DEPS=1 BUILD_CUDA=1 \
  bash deformable_human/00_setup_env.sh
```

权重目录布局：无需权重（训练型）。官方仓在 `$REPO_DIR/../Deformable-3D-Gaussians`。

---

以下为详细参考。

## Config (env vars)
| var | default | note |
|---|---|---|
| `VIDEO_PATH` | （必填） | 单目视频文件，01 的输入 |
| `SCENE_NAME` | `human_seq` | 场景名，贯穿 01/02/03 默认路径 |
| `VIDEO_FPS` | `6` | 抽帧 fps（动态序列要比静态的 2 密） |
| `BLUR_THRESHOLD` | `100` | 拉普拉斯模糊门，0=关（透传 vggt_human/01a） |
| `USE_GPU` | `1` | COLMAP SIFT 用 GPU（conda-forge colmap 3.11.1 实测带 CUDA） |
| `ITERATIONS` | `20000` | NeRF-DS 真实序列标配 |
| `IS_6DOF` | `0` | 1=6DoF 形变变体（略准、更慢） |
| `USE_POSE_REFINE` | `0` | 1=训练中联合精炼位姿（可学四元数+平移，内参不学） |
| `POSE_REFINE_WEIGHT` | `0.01` | 位姿正则权重（拉回 COLMAP 初值） |
| `POSE_ADJUST` | — | 01c：位姿规整开关（居中+重力对齐+尺度归一化） |
| `WHITE_BG` | `0` | 1=白底训练（输入抠图后开） |
| `MODE` | `render` | render/time/all/view/pose/original |
| `DG_DIR` | `../Deformable-3D-Gaussians` | 官方仓位置（WSL 由 proxy.env 覆盖为 ~/repos/） |
| `CLONE_FROM` | `vggt_human` | 00/00a 建 env 的 donor |

## 目录布局
```
<code-dir>/
├── media_code/deformable_human/      # 本目录（编排脚本）
├── Deformable-3D-Gaussians/          # 官方仓（sibling；WSL 在 ~/repos/）
└── deformable_human_results/         # 输出（sibling；WSL 在 ~/output/）
    └── <scene>/                      # 每个输入数据一个文件夹
        ├── frames/                   # 01 抽帧（模糊帧已剔除）
        ├── colmap_scene/             # 01 COLMAP 场景（images/ + sparse/0/）
        ├── model/                    # 02 形变模型 + 03 渲染产物
        └── model_static/             # （可选）静态 3DGS 基线（A/B 对比用）
```

## Notes
- **先 vanilla 后锚定**：02 不加任何人体先验，先看 canonical+形变把重影消到几成，
  残差（单目欠定的背面/遮挡）再决定 Phase 2 的 MHR 锚定形态。锚定导出在 01 里是
  `ANCHOR_EXPORT=1` 占位，尚未实现。
- 图像文件名必须是纯数字（Deformable-GS `dataset_readers.py` 按时序算 fid）——
  01 委托的 COLMAP 脚本已自动重命名。
- Phase 2 设计讨论见 `media_paper` 调研笔记
  `VideoGen-20260929-视频生成驱动动态人体重建调研.html` §11。

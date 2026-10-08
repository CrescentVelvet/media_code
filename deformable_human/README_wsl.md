# deformable_human (WSL Ubuntu 24.04) — 本地复现指南

本文件是 [`README.md`](README.md) 的 WSL 本地复现版。照着做即可从零跑完全流程。

## 与服务器版的核心差异
| | 服务器 | WSL 本机 |
|---|---|---|
| conda env | clone donor（CLONE_FROM） | **clone vggt_human**（torch 2.5.1+cu121，nvcc 12.1 随 clone 继承） |
| 官方仓位置 | media_code 的 sibling | `~/repos/Deformable-3D-Gaussians`（Linux fs，编译快） |
| GitHub | 直连 | **走 ghfast.top 前缀**（直连不通） |
| pip 源 | 默认 PyPI + trusted-host | 阿里镜像 + `--find-links /mnt/d/wheel` |
| colmap | apt 装 | 无 sudo → `INSTALL_COLMAP=1` 装 conda-forge 版（CPU，01 记得 USE_GPU=0） |
| 输出 | sibling `deformable_human_results/` | `~/output/deformable_human_results`，跑完 08 搬到 D 盘 |

## Windows 路径 → WSL 路径
| Windows | WSL |
|---|---|
| D:\dataset\... | /mnt/d/dataset/... |
| D:\output\deformable_human_results | /mnt/d/output/deformable_human_results |
| D:\wheel | /mnt/d/wheel（pip wheel 缓存） |

## 前提条件
- WSL2 + NVIDIA 驱动（3090，driver CUDA 13.2）+ Miniconda（`~/miniconda3`）
- 已存在 `vggt_human` env（donor：torch 2.5.1+cu121 + cuda-nvcc 12.1 + diff_gaussian_rasterization）
- `proxy.env` 已配（本机主要是 `HF_ENDPOINT` / `PIP_INDEX_URL`）

## 首次准备
```bash
cd /mnt/c/code/media_code

# 建 env（clone vggt_human）+ clone 官方仓 + 装依赖 + 编 CUDA 子模块
# ⚠️ 关键点：官方仓的 depth-diff-gaussian-rasterization 是 fork，编译后会遮蔽
#    clone 带来的 editable 版——这是刻意的，只影响本 env，vggt_human env 不动。
INSTALL_DEPS=1 BUILD_CUDA=1 bash deformable_human/00a_setup_env.sh

# （可选）装 colmap：conda-forge CPU 版，01 步用 USE_GPU=0
INSTALL_COLMAP=1 bash deformable_human/00a_setup_env.sh

# 装完清 pip 缓存释放 vhdx 空间
conda activate deformable_human && pip cache purge
```

## 全流程命令

```bash
cd /mnt/c/code/media_code

# ── 0) 安装环境（首次，见上方「首次准备」）──

# ── 1) 单目视频 → 帧（模糊门剔帧）→ COLMAP 场景 ──
#    输入：/mnt/d/dataset/ 下的单目视频（人物有动作）
#    输出：~/output/deformable_human_results/datasets/<scene>/{frames, colmap_scene}
#    可选开关：VIDEO_FPS（默认 6）；BLUR_THRESHOLD（默认 100，0=关）；
#             USE_GPU=0（conda-forge colmap 是 CPU 版时必设）；FORCE_RECOLMAP=1（重跑 COLMAP）
GPU=0 \
VIDEO_PATH=/mnt/d/dataset/sample/human.mp4 \
SCENE_NAME=human_seq \
RESULTS_DIR=~/output/deformable_human_results \
  bash deformable_human/01_prepare_data.sh

# 输出：~/output/deformable_human_results/datasets/human_seq/
#   frames/image/          # 抽帧（模糊帧已剔除）
#   colmap_scene/images/   # 去畸变图像
#   colmap_scene/sparse/0/ # cameras.bin / images.bin / points3D.bin

# ── 2) vanilla Deformable-GS 训练（canonical + 形变 MLP）──
#    输入：上一步的 colmap_scene
#    输出：~/output/deformable_human_results/train/human_seq/
#    可选开关：ITERATIONS（默认 20000）；IS_6DOF=1（略准更慢）
GPU=0 \
SOURCE_PATH=~/output/deformable_human_results/datasets/human_seq/colmap_scene \
MODEL_PATH=~/output/deformable_human_results/train/human_seq \
ITERATIONS=20000 \
  bash deformable_human/02_train.sh

# 输出：~/output/deformable_human_results/train/human_seq/
#   point_cloud/iteration_20000/point_cloud.ply   # canonical 高斯
#   deform/                                       # 形变 MLP 权重
#   cfg_args / cameras.json                       # 渲染时自动恢复参数

# ── 3) 渲染 + 评测 ──
#    MODE=render 出测试视角 + PSNR/SSIM/LPIPS；
#    MODE=original 出时间×视角组合渲染——目视看重影用它
GPU=0 \
MODEL_PATH=~/output/deformable_human_results/train/human_seq \
MODE=render RUN_METRICS=1 \
  bash deformable_human/03_render.sh

# 输出：~/output/deformable_human_results/train/human_seq/
#   test/ours_20000/renders/   # 渲染图（目视检查重影）
#   test/results.json          # PSNR/SSIM/LPIPS

# ── 4) 搬运到 D 盘（训练完必做，vhdx 空间有限）──
bash deformable_human/08_move_output.sh

# ── 5)（可选，Phase 2）MHR 锚定导出 ──
#    ⚠️ 未实现。计划从 vggt_human 03e/03f 适配单目逐帧 MHR + 657 landmarks。
#    vanilla baseline 的残差分析出来之前不用碰。
# ANCHOR_EXPORT=1 GPU=0 VIDEO_PATH=/mnt/d/dataset/sample/human.mp4 \
#   bash deformable_human/01_prepare_data.sh
```

## 可能遇到的问题（WSL 专属）

- **`depth-diff-gaussian-rasterization` 编译失败**：2023 年的 fork 代码在 torch 2.5.1 上
  可能需要小改（如 `setup.py` 的 extra flags、C++17）。先看报错贴上来；兜底是退回
  py3.7/torch1.13 独立 env（方案见 `deformable_gaussians/_env.sh` 头部注释），
  代价是与 MHR 生态隔离，Phase 2 时会更麻烦——所以优先修编译。
- **import 到的光栅化器不是 fork**：`python -c "import diff_gaussian_rasterization as d; print(d.__file__)"`
  应指向 env 的 site-packages（fork 编译产物），而不是 `~/repos/gaussian-splatting`（editable）。
  不对就重跑 `BUILD_CUDA=1 bash deformable_human/00a_setup_env.sh`。
- **COLMAP 掉帧严重**：动态人体占画面大、背景匹配点少。对策：抽帧加密（VIDEO_FPS 调高）、
  模糊门收紧（BLUR_THRESHOLD 调高）、保证背景有足够静态纹理。
- **conda-forge colmap 是 CPU 版**：01 必须 `USE_GPU=0`，慢但能跑；要 GPU 版得自己编 colmap。
- **fid ValueError**：图像名不是纯数字——01 委托的 COLMAP 脚本会自动重命名，若自己手动
  放过图像进 colmap_scene，检查文件名。
- **vhdx 空间**：训练产物及时 `08_move_output.sh` 搬走；pip 装完 `pip cache purge`。

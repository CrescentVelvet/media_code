# da3_human — DA3 官方复现 + 人像定制（WSL）

Depth-Anything-3（ByteDance-Seed）官方高斯头复现与人像采集包定制实验。
官方仓：`~/repos/Depth-Anything-3`（AGENTS.md 约定，与 infinisplat_v2 共用）；
权重：`depth-anything/DA3NESTED-GIANT-LARGE-1.1`（⚠️ CC BY-NC 4.0 非商用）。

## 先读顺序

| 文件 | 内容 |
|---|---|
| `ERRATA.md` | ⛔ **跑推理/训练/评估前必读**：坐标、尺度、导出和评测事故 + 防再犯清单 |
| `EXPERIMENT_PLAN.md` | **唯一实验总计划**：当前前馈直出 D0→D5 路线、冻结口径、评测和晋级条件 |
| `EXPERIMENTS.md` | 实验结论台账（日期倒序，报数前先查这里） |

`PLAN_REVIEW_3VIEW.md` 是已归档的三视角方案评审记录；`28_da3_gt_feedforward_plan.md` 是长篇技术附录。它们不再与 `EXPERIMENT_PLAN.md` 并列指导实验。

## 常用命令

```bash
cd /mnt/c/code/media_code

# 0) 环境 + 权重（首次）
INSTALL_DEPS=1 bash da3_human/00a_setup_env.sh
bash da3_human/01_download_models.sh

# 1) DA3 官方推理（32 视角口径；等价于横评 90b 的转发内核）
RAW_DIR="/mnt/d/dataset/手部微动样例/<采集包名>" \
INPUT_DIR="$HOME/output/da3_human_results/_captures/<场景名>/images" \
SCENE_NAME="<场景名>" \
OUTPUT_DIR="$HOME/output/da3_human_results/<场景名>" \
N_VIEWS=32 bash da3_human/02_run_da3_inference.sh

# 2) 位姿可视化验收（跑完 02 先看这个，弱纹理/重复纹理场景可能塌缩）
DA3_INFER_DIR="$HOME/output/da3_human_results/<场景名>" \
COLMAP_REF_DIR="<COLMAP sparse/0 目录>" \
~/miniconda3/envs/da3_human/bin/python da3_human/02b_viz_trajectory.py

# 3) 官方 DA3-GS 基线 / 批量人像管线 / NVS 评估
bash da3_human/04_run_da3_gs_baseline.sh
bash da3_human/21_batch_portrait.sh
bash da3_human/11_run_nvs_eval.sh

# 4) 海思十视角批量原生高斯直出（17 案例，复用模型）
ROOT="/mnt/d/dataset/测试数据/海思端侧重建拍摄数据" \
  PROCESS_RES=504 DA3_MODEL_DIR="/mnt/d/wheel/da3_human_ms" \
  bash da3_human/34_run_hisilicon.sh
# 每个 CASE 输出 depthanything3.ply，以及 depthanything3/{cameras.npz,summary.json}
# 输入只取已验收的 splatanything/images RGB，不用其相机或几何；米制、不裁剪、全 SH。
# 在 vggt_human env 回放交付 PLY 的源视角，输出 depthanything3/render_check.jpg：
CONDA_ENV=vggt_human ROOT="/mnt/d/dataset/测试数据/海思端侧重建拍摄数据" \
  python da3_human/35_render_hisilicon.py

# 5) 海思十视角 + 各案例原始 stage1 相机（不覆盖第 4 步 RGB-only 结果）
ROOT="/mnt/d/dataset/测试数据/海思端侧重建拍摄数据" \
  PROCESS_RES=504 DA3_MODEL_DIR="/mnt/d/wheel/da3_human_ms" \
  bash da3_human/36_run_hisilicon_cameras.sh
# 每个 CASE 输出 depthanything3_cameras.ply，及 depthanything3_cameras/{cameras.npz,summary.json}
# JSON OpenGL c2w -> DA3 OpenCV w2c；按 sourceIndex 配对，模型输入外参/内参。
# 高斯经与 API 相同的 Sim(3) 对齐，再与相机同步绕X轴180度转正显示，含米制倍率和完整 SH，不裁剪。
# 位姿版直接输出上述文件名，不再额外生成 upright 副本；转正不改善模糊或几何质量。
# 在 vggt_human env 回放全部输入视角：
CONDA_ENV=vggt_human ROOT="/mnt/d/dataset/测试数据/海思端侧重建拍摄数据" \
  PLY_NAME=depthanything3_cameras.ply \
  CAMERAS_NPZ=depthanything3_cameras/cameras.npz \
  OUT_NAME=depthanything3_cameras/render_check.jpg \
  python da3_human/35_render_hisilicon.py

# 仅消除上下显示差异，不把RGB-only版与位姿版配准到同一原点、尺度或相机轨迹。

# 6) 海思最终 COLMAP SfM 相机版（不覆盖第 4/5 步结果）
ROOT="/mnt/d/dataset/测试数据/海思端侧重建拍摄数据" \
  PROCESS_RES=504 DA3_MODEL_DIR="/mnt/d/wheel/da3_human_ms" \
  bash da3_human/38_run_hisilicon_sfm.sh
# 每例输出 depthanything3_sfm.ply 和 depthanything3_sfm/{cameras.npz,summary.json}。
# 输入 colmap_sfm_results/sparse/0，按名字只取已注册图像；case7 为 6/10，其余为 10/10。
# OPENCV 图像先去畸变，外参不变；高斯用完整 Sim(3) 对齐至 SFM 原生坐标，不转正、不是米制。
CONDA_ENV=vggt_human ROOT="/mnt/d/dataset/测试数据/海思端侧重建拍摄数据" \
  PLY_NAME=depthanything3_sfm.ply \
  CAMERAS_NPZ=depthanything3_sfm/cameras.npz \
  SOURCE_IMAGES_DIR=depthanything3_sfm/images \
  OUT_NAME=depthanything3_sfm/render_check.jpg \
  python da3_human/35_render_hisilicon.py
# 数值/相机/回放完整性核验：
ROOT="/mnt/d/dataset/测试数据/海思端侧重建拍摄数据" \
  python da3_human/39_verify_hisilicon_sfm.py
```

各脚本的 env 契约以脚本头注释为准。

## ⛔ 基线冻结清单（官方对比口径）

以下脚本 + 其默认参数构成**官方基线，用于对比测试**：

- `02_run_da3_inference.sh`（官方推理导出）
- `04_run_da3_gs_baseline.sh` + `04b_run_da3_gs_colmapintr.sh`（官方 DA3-GS 基线）
- `08_bench_official_gs.sh` / `09_eval_bench_exports.sh`（官方 bench 导出与评估）
- `14_verify_export.py`（导出正确性验证）

规则：

1. **改前先复制副本**（如 `06_train_da3_2dgs.py` 之于官方训练器）或开分支，直接在原文件上改属于破线；
2. 改完在 `EXPERIMENTS.md` 当日节记一笔（改了什么、为什么、影响哪个口径）；
3. 基线快照 tag：`da3-baseline-20261109`（旧版正式口径 20.23dB；当前交付口径为不裁剪 + 全 SH，报数以 `EXPERIMENTS.md` 对应实验节为准）。
   随时可核对基线是否被污染：`git diff da3-baseline-20261109..HEAD -- da3_human/02_run_da3_inference.sh da3_human/04_run_da3_gs_baseline.sh`；
4. 报数前三查（详见 ERRATA）：视角索引对齐 / 尺度是否双重施加 / 相机与高斯是否同一口径。

其余脚本（`05`/`06`/`19`~`21` 等）是定制实验区，可自由修改。

## 与 compare_4methods 的关系（单向依赖）

横评的 DA3 基线是本目录链路的**转发封装**（2026-11-10 收编，原独立目录 `da3/` 已撤销）：

| 横评脚本 | 转发到 |
|---|---|
| `compare_4methods/90a_check_da3_deps.sh` | preflight：校验本目录 env + vggt_human env + 权重 |
| `compare_4methods/90b_run_da3_inference.sh` | `da3_human/02_run_da3_inference.sh`（da3_human env） |
| `compare_4methods/90c_train_da3_3dgs.sh` | `da3_human/build_da3_colmap_scene.py` + `vggt_human/04_train_3dgs.sh`（vggt_human env） |

- 改本目录的**推理/导出接口**（env 变量名、输出目录布局）后，必须冒烟验证
  `bash compare_4methods/90a_check_da3_deps.sh` 和 90b 单场景，否则横评会静默断链；
- 横评结果落 `~/output/da3_results/`（与本目录的 `~/output/da3_human_results/` 隔离，避免基线产物与定制实验产物混淆）。

## Config 速查

| var | default | note |
|---|---|---|
| `DA3_MODEL_REPO` | `depth-anything/DA3NESTED-GIANT-LARGE-1.1` | CC BY-NC 4.0 |
| `DA3_MODEL_DIR` | `~/model/da3_human`（WSL 由 proxy.env 覆盖） | 缺失时回退 HF 缓存 |
| `N_VIEWS` | `32` | DA3 是稀疏视角模型，更多视角收益递减且耗时超线性（ERRATA #1） |
| `PROCESS_RES` | `504` | 不同档位的相机与高斯严禁混用（ERRATA #6） |
| `CONDA_ENV` | `da3_human` | 3DGS 训练在 vggt_human env |

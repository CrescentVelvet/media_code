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

# 5b) 旧17例历史相机PLY有严重重影，不直接作为有效对照；先独立审计stage1：
ROOT="/mnt/d/dataset/测试数据/海思端侧重建拍摄数据" \
  RESULTS_DIR="/mnt/d/output/results/da3_human_results/stage1_camera_audit_repeat" \
  python da3_human/46_audit_stage1_cameras.py
# 8源+2留出、同704、同seed42的RGB/pose双分支，串行避免24GB显存争用。
# 批处理要求RESULTS_ROOT预先不存在；续跑未完成案例须给新的根目录，不覆盖历史结果。
ROOT="/mnt/d/dataset/测试数据/海思端侧重建拍摄数据" \
  DA3_MODEL_DIR="/mnt/d/wheel/da3_human_ms" PROCESS_RES=704 \
  RESULTS_ROOT="/mnt/d/output/results/da3_human_results/hisilicon_camera_ab_repeat/stage1" \
  ~/miniconda3/envs/da3_human/bin/python da3_human/48_batch_stage1_control.py
# 完整双分支后统一渲染原生PLY与共同物理source/target相机，核验RGB/JSON/PLY身份。
DATASET_ROOT="/mnt/d/dataset/测试数据/海思端侧重建拍摄数据" \
  RESULTS_ROOT="/mnt/d/output/results/da3_human_results/hisilicon_camera_ab_repeat/stage1" \
  GSPLAT_CACHED_SO="/home/velvet/.cache/torch_extensions/py311_cu126/gsplat_cuda/gsplat_cuda.so" \
  ~/miniconda3/envs/splatanything/bin/python da3_human/49_render_camera_ab.py
# 不套用新glTF的中心取负；case7几何风险单列。替换默认入口前必须确认完整实验摘要。
# 用户2026-11-26明确允许覆盖旧失败产物；每批明确1–10例，先备份再替换PLY/相机/摘要。
ROOT="/mnt/d/dataset/测试数据/海思端侧重建拍摄数据" \
  RESULTS_ROOT="/mnt/d/output/results/da3_human_results/hisilicon_camera_ab_repeat/stage1" \
  BACKUP_ROOT="/mnt/d/output/_backup/da3_stage1_pre_holdout_repeat_case01" \
  CASE_FILTER=case1_nopose_recentered \
  CONDA_ENV=da3_human bash -c 'source da3_human/_env.sh; python da3_human/48a_install_stage1_pose.py'
# 需source项目环境加载DA3的SH/四元数旋转依赖；不能用缺DA3包的渲染env直接运行。
# 默认入口保留10个相机槽位：8个源视角前向，2个留出视角仅回放。旧文件移入BACKUP_ROOT。
# 原始图像/相机JSON与实验输出目录不动；新PLY为8源pose分支对齐输入世界后同步转正，不宣称质量通过。

# 6) 新增 model_man / model_plant（glTF 相机关键帧）
# 已替换的全10视角默认入口与下面8源留出实验不是同一口径；实验native.ply保留在RESULTS_ROOT。
ROOT="/mnt/d/dataset/测试数据/海思端侧重建拍摄数据" \
  RESULTS_ROOT="/mnt/d/output/results/da3_human_results/gltf_pose_install_repeat" \
  BACKUP_ROOT="/mnt/d/output/_backup/gltf_pose_pre_install_repeat" \
  CASES=model_man,model_plant DA3_MODEL_DIR="/mnt/d/wheel/da3_human_ms" \
  CONDA_ENV=da3_human bash -c 'source da3_human/_env.sh; python da3_human/50_run_gltf_pose_install.py'
# 仅对已审计两例、两个新输出根目录执行；先产出并核验490万高斯再替换，旧PLY/相机/摘要成套备份。
# 原始图像/相机JSON不动；不做显示转正，不宣称质量通过，不推广特殊坐标解释。
# 这两例的8源+2留出附录用此前的有效native.ply，不能用全10视角安装版替代留出结果。

# 原始 images/frame_0001..0010.png 分别与 *_camera.json 的10个 c2w_keyframes 按序对应；
# 这两例按标准glTF读取与图像不一致！先用动画yfov + 经极线验证的特殊坐标解释。
# 细节与限制见 EXPERIMENTS.md 2026-11-26「图像几何独立审计」；不能泛用于其它glTF。
# 不读取目录中既有 model_*.ply 或 renders/ 作为 DA3 输入。
ROOT="/mnt/d/dataset/测试数据/海思端侧重建拍摄数据" \
  CASE_FILTER=model_man PROCESS_RES=504 DA3_MODEL_DIR="/mnt/d/wheel/da3_human_ms" \
  bash da3_human/34_run_hisilicon.sh
ROOT="/mnt/d/dataset/测试数据/海思端侧重建拍摄数据" \
  CASE_FILTER=model_man PROCESS_RES=704 DA3_MODEL_DIR="/mnt/d/wheel/da3_human_ms" \
  GLTF_INTRINSICS_MODE=camera_path_yfov GLTF_CAMERA_CONVENTION=image_verified_negative_centers \
  PLY_NAME=depthanything3_cameras_verified.ply OUT_SUBDIR=depthanything3_cameras_verified \
  bash da3_human/36_run_hisilicon_cameras.sh
# 同口径RGB/原读取/修正读取对照（结果目录必须尚不存在，原文件保留）：
CASE_DIR="/mnt/d/dataset/测试数据/海思端侧重建拍摄数据/model_man" \
  DA3_MODEL_DIR="/mnt/d/wheel/da3_human_ms" PROCESS_RES=704 \
  GLTF_INTRINSICS_MODE=camera_path_yfov GLTF_CAMERA_CONVENTION=gltf \
  RESULTS_DIR="/mnt/d/output/results/da3_human_results/camera_control_repeat" \
  python da3_human/run_gltf_camera_control.py
# model_plant 同样将 CASE_FILTER 改为 model_plant；单例摘要在各自 depthanything3*/ 下，
# 不覆盖根目录17例的总表。35号回放对原始 images/ 目录自动适配。

# 7) 海思最终 COLMAP SfM 相机版（不覆盖第 4/5 步结果）
ROOT="/mnt/d/dataset/测试数据/海思端侧重建拍摄数据" \
  PROCESS_RES=504 DA3_MODEL_DIR="/mnt/d/wheel/da3_human_ms" \
  bash da3_human/38_run_hisilicon_sfm.sh
# 每例输出 depthanything3_sfm.ply 和 depthanything3_sfm/{cameras.npz,summary.json}。
# 输入 colmap_sfm_results/sparse/0，按名字只取已注册图像；case7 为 6/10，其余为 10/10。
# OPENCV 图像先去畸变，外参不变；高斯用完整 Sim(3) 对齐至 SFM 原生坐标，不转正、不是米制。
# 尺度链（2026-11-26 修订）：用含 adapter a 的相机拟合一次 Sim(3)，不再额外乘 Nested sf 或除 a。
# 旧 PLY 会备份至各案例 depthanything3_sfm_backup_*；新版另存 native.ply/native_cameras.npz
# 以区分前向预测退化与输入世界对齐退化。COLMAP 是 OpenCV，不套 glTF 的中心取负规则。
# 独立 A/B 可用新的 OUT_ROOT（一次性实验根，已有输出拒绝覆盖）：
ROOT="/mnt/d/dataset/测试数据/海思端侧重建拍摄数据" \
  PROCESS_RES=504 DA3_MODEL_DIR="/mnt/d/wheel/da3_human_ms" \
  CASE_FILTER=case14_nopose_recentered \
  OUT_ROOT="/mnt/d/output/results/da3_human_results/sfm_scale_ab_new" \
  bash da3_human/38_run_hisilicon_sfm.sh
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

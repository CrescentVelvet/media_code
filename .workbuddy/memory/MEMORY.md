# media_code 项目长期约定

> 详细历史已按日期沉在 `.workbuddy/memory/YYYY-MM-DD.md`，本文件只留跨会话必须遵守的结论与索引。

## da3_human / splatanything 专属（只在该项目跑推理/训练时适用）

- **2026-11-19 当前路线（覆盖下方旧排期）**：用户判断 SHARP 几何不适合作初始化，停止 SH2/SH3。
  SA 场景/物体较好、人脸薄弱。网络训练前先按主计划 **P0～P3 做有预算的局部诊断**：完整 RGB 进 SA，
  人物 mask 保护、脸 mask 选高斯/损失；身体/头发/场景冻结，同 renderer 合成。
  **P1 仅颜色｜P2 属性(scale/quat/opacity)但固定中心｜P3 受限中心移动**；每臂独立从原生 SA 起步、
  ≤300 步、不增删点、固定相机。纪律：**mask 不是深度**；源图拟合与新视角几何分开验收；
  target 不参与优化或选参；高分辨率 ROI 监督 ≠ 提升骨干输入分辨率；32 视角 2K 失败不可外推 SA 三图。
  此方案当时未执行、非前馈成绩。v2 的 `14_finetune_gs.py` 锁死中心，**不能验证 P3**，需自实现。
- **2026-11-18 SHARP 复核**：SH1a 双重外参修正后三份 JSON 去重 16 对（同一人/6 图），S−identity 中位
  +3.62dB、S−planeH +2.11dB（14/16 正）；旧 +3.73 汇总与 v2 否证**均不沿用**。不能用跨视角 PSNR
  对比 SA 自渲来否证融合，单尺度扫描不是 Sim(3) 收益上限。SH1b 先修评测+同 target 配对。
- **2026-11-17 LoRA**：入口 `EXPERIMENT_PLAN.md` 顶部 L0～L4，冻结几何教师、学生 attention LoRA，
  人物数据与同预算通用数据配对；2DGS/纹理化暂缓；E-21 外部深度替换失败不证伪人物数据适配。
- **2026-11-12 三图路线**：三图 = 不同机位三张人像图（不是广角/主摄/长焦）。研发首选 SplatAnything，
  DA3 保留配对基线、暂停几何重训。SA 现有 FG 是预测 alpha mask，held-out 底色取自 target ——
  正式排名前需独立 mask/底色。用户报"人物鼻子变宽疑似重影"未证实，R2-N 须在域训练前区分
  相机/运动、中心深度、merge/footprint 与域/像素预算。分工：**Codex 做设计/证据审查，WorkBuddy 复现执行**，
  回填 `splatanything/ROUTE_VALIDATION_REPORT.md`。
- **DA3 复现错误 + 防再犯清单 + 训练事实**全在 `da3_human/ERRATA.md`，跑 DA3 前必读（其它项目不用看）。

## 工程纪律

- **commit**：中文 `<项目名>: <简短描述>`（英文冒号+空格），默认只写标题行，正文≤1-2 行。
  **改完当轮立即提交**，只 `git add <具体路径>`，**不 `git add -A`**。规范 AGENTS.md §8/§11。
- **AI 不 push**（含 --force）、不删远端分支/tag；push 由峰哥手动。
- **先读规范再动手**：git 前读 AGENTS.md；WSL 训练前读 README_wsl.md（输出写 `~/output/` Linux fs）。
- **多落点**：`C:\code\media_code`（主）｜`h100:00950754@69.48.159.15:/data/00950754/code/media_code`。
  同一改动只在一处提交，另一侧 `git pull --rebase`。**`.workbuddy/memory/` 只在本机写**。两边设
  `core.fileMode=false`。
- **删除红线**：删前判 `is_symlink()`（曾穿透软链删掉 82 张 HEIC）；临时文件走"备份+搬移"到
  `/mnt/d/output/_backup/`。
- **入库边界**：只入可复用脚本与规格文档；成品图与一次性出图脚本不进 GitHub（否则放
  `D:/ppt/figures/<项目>/_build/`）。`.gitignore` 已排除 `*.png *.jpg *.mp4`。
- commit 后按需更新 `EXPERIMENTS.md`（结论）与 README（命令）分离。

## D 盘落点规范（2026-11-18，动 D 盘前必读 `D:\ppt\README.md`；仓内 AGENTS.md §15）

- 硬规则：①根目录禁建 `*_figures`/`*_report_*`/`*_ppt_*`/`*_assets`；②`.slide` 内不写绝对路径，
  图一律 `assets/xxx.png` 相对引用；③PPT 不放 output、实验产物不放 ppt；④成品进 `ppt/figures/`、
  原始素材进 `output/results/<项目>_assets/`；⑤搬进 decks 后不改相对路径；⑥删除前判 `is_symlink()`。
- 结构：`D:\ppt\decks\<YYYYMMDD>-<标题>\`｜`D:\ppt\figures\<项目>\`｜`D:\output\results\<algo>_results\`｜
  `D:\output\_backup\`。**`/mnt/d/output/` 之后必须先跟 `results/` 或 `_backup/`**。
- 搬目录 Permission denied 多为 explorer 持句柄 → `mkdir` 目标 + 逐项 `mv`。

## 关键文档索引（含已定论结论）

- **`splatanything/EXPERIMENTS.md`**（日期倒序）+ `README_wsl.md`（命令）+ `NOTES.md`（原理排错）+
  `FIGURES.md`（图台账）。入口脚本 `01～11_*`，共享 `_env.sh` / `_gsplat_env.sh`。
- **`da3_human/EXPERIMENT_PLAN.md`**：唯一执行入口。**`compare_4methods/`**：IMPROVEMENTS + ERRATA +
  FINDINGS + P0/P1_EXPERIMENTS。已定论：
  - 正式口径 = 3DGS EWA 光栅器 + 官方等价导出（不裁剪+全 SH）：全图 22.11dB、人脸 20.00dB/锐度 347。
  - **2DGS 路线终止**：只换光栅器 −0.29dB；盘法向换 GT 真相（oracle，覆盖 65.1%）反而 −0.71dB、
    人脸 −1.16 → "法向没学对"不是根因，I-04 撤销。2DGS 仅保留 `median_depth` 用途。
  - **区域指标必须逐视角定框**（固定框曾落书架上，人脸锐度虚高 +180）。报数前目视核对裁切图。
  - DA3 瓶颈是跨视角几何不一致；**前馈高斯正确用法 = 稀疏点云 init + 3DGS 自己 densify**
    （整套灌入更差更慢，opacity 卡在 prune 阈值上方）。甜点：点云+5k 步 ≈2.6min/22.2dB（30k 24.3dB）。
- **`deformable_human/BACKLOG.md`**：流水线 A~J 案状态总览（做优化前先查）。截至 10-17：A 无收益｜
  B 双度证伪｜C 否决｜D 已落地(默认关)｜E 首轮负｜F/G/H 零实现｜I 未立案｜J 已实测。
- **`som_human/EXPERIMENTS.md`** 与 **`vggt_human/`** 三文档（README=命令｜NOTES=原理｜EXPERIMENTS=结论）+
  `DESIGN_face_pipeline.md`（人脸链路 v2 八阶段，讨论前先读）。MHR 身份 45 维 + 72 FACS + 657 点。
- **`figkit/SPEC.md`**：PPT 配图规格 + 排版库（1920×1080 满版、§5.6 单方法页、§10 入库边界）。
- **som_human 标准输入**＝华为「Remy」包：`transforms.json` + `image/*.heic` + `pcd.ply`；
  `w2c = inv(M @ diag(1,-1,-1,1))`（实测判定，勿凭直觉改）；入口 `som_human/01_convert_capture.sh`。

## som_human 手部：根因分层与已证伪路线（2026-10-17，勿重复踩）

- 三种"糊"分开：①输入运动模糊本数据不存在（GT 手部锐度 4.28~4.33 > 静止区 2.75~2.81）；
  ②重建过平滑（渲染高频只剩 GT 9~12%，**头也糊**；提容量已证伪：更多高斯更易摊平）；
  ③**两态几何冲突**（垂手 vs 抬手占同空间）→ 静态 3DGS 本质限制，唯一有效解=按姿态分段只用主段。
- 两条路线都证伪：SOM（平移 0.0020~0.0238，仍小 2 个数量级，且手部锐度 38%→9%）｜可微位姿精炼
  （好初值上 −8.29dB）。共同根因：**弱纹理小手区域亮度梯度不足以驱动运动参数**。
- 论断边界（E-07）：已实测=SoM+本仓 deform 场；HiMoR 未跑过【待核实】；"所有学习方法都不行"不成立。
  → 换的是**运动信息来源**：从"梯度学"改"先验给"（MANO/MHR 参数化、2D 关键点/光流对齐）。
- **删帧判据是"姿态段"不是"偏离大"**；删前查相机方位角覆盖。**评估口径坑**：mask 内 FG PSNR
  抓不到幽灵/透明（幽灵在 GT mask 外），要配 mask 外泄漏指标或 alpha 渲染。
- 可复用资产：方案 J per-person 拆分、01c COLMAP 精化（+7.29dB）、SAM3 手部 mask 管线。
- `02_train.sh` 默认全分辨率（历史口径靠 `EXTRA_TRAIN_ARGS="-r 2"`）。`pkill -f train_som.py` 会
  匹配包装 shell 自身，杀完要单独验证目录清理。

## tools_99/（重建产物工具集；路径全部相对 cwd=media_code 解析，与脚本目录无关）

- 打包链：99a 收集 ply｜99b PLY→UWA MP4（`mp4_work/` 是 99c/99e 复用前提）｜99c 收窄视角重封装｜
  99d 复制三帧｜99e MP4→PLY｜99f 收集 ply+相机+front_image。体检：99g 轨迹/视角可视化｜
  99h Python 3.10 语法自查｜99k 采集包预处理（**主体居中必需**，出 `gs_camera_params.json`）。
- **居中为何必需**：下游把 az/el 当"绕世界原点"用，而 Remy 包原点是 AR 会话起点（实测 10 包
  |target|=半径 86%~198%，不居中 Δyaw 最大 174°）。
- ⚠️ 99b/99f 的 `gs_camera_params_final.json` 是**外部产物同名约定**，别跟着改名。
- 写 JSON 一律 `open("w", encoding="utf-8", newline="\n")`；gs json/center_report 用 `indent=4`。
- 本机跑 99g 贴实拍首帧需 pillow：用 `~/.workbuddy/binaries/python/envs/default` 的 python。

## 本机 WSL 环境（2026-09 实测，网络会漂移，动手前先 curl 复测）

- 网络：不走代理。`github.com` ✅ 直连｜HF ❌ 走迅雷｜hf-mirror/ghfast ❌｜`raw.githubusercontent` ❌；
  pytorch.org / anaconda / tuna pypi 可用。
- 硬件：WSL2，54GiB RAM / 128GiB swap；`/` 1007G｜`/mnt/d` 9.1T；i9-12900K 8 核；RTX 3090 24GB。
- conda `~/miniconda3`：4danyone / colmap / flame_human / minimax_h3 / osediff / sam3 / sharp /
  vggt_human / splatanything / infinisplat_v2。新任务优先新建独立 env。
- `colmap` env：conda-forge colmap 4.2.1；必须补钉 `openimageio=3.1`；别用 tuna conda 通道；
  4.x mapper 要求 `--output_path` 已存在；直调二进制需 `LD_LIBRARY_PATH=~/miniconda3/envs/colmap/lib`。

## 服务器侧「3DGS 相机位姿调整工具」（2026-09-28 峰哥提供；本机无）

- `crop_camera_by_angle.py`（视线交汇最小二乘求目标点 + 方位角循环补偿）｜`visualize_cameras.py`｜
  `generate_camera_files.py`｜`main.py`（Tkinter GUI 闭环）+ MP4 解封装页。同族能力：≈99b/99e/99g/99c。

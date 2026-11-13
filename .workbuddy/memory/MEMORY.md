# media_code 项目长期约定

## da3_human 专属知识（只在该项目跑推理/训练时适用）
- **三图路线更新（2026-11-12 用户澄清 + Codex 审查）**：三图指不同机位三张人像图，
  不是近光心广角/主摄/长焦。研发首选 SplatAnything，DA3 保留配对基线、暂停几何重训。
  最新排期为 `da3_human/EXPERIMENT_PLAN.md` §0.0 R0～R4：修评测→三图横评/几何与 ROI
  审计，独立准备最多 100 step 成本探测；不等 DA3 C0～C4、不扩旧 72-run 矩阵。
  8/13v 仅参考。用户观感不等于定量达标；当前训练仅 T0，后续 A1～A4 需据成本排期。
  SA 现有 FG 是预测 alpha mask，held-out 底色取自 target；正式排名先修独立 mask/底色。
  E-11 pose 为参考参数+假设 look-at，基线/raw depth 尺度混用；其 GT/退化归因与 fig10/11
  的结论已在实验台账顶部修正，不可外推否定不同机位三图。
  用户另报人物鼻子变宽疑似重影、物体很好；最新 R2-N 在域训练前区分相机/运动、中心深度、
  merge/footprint 与域/像素预算；人物数据少和重影均未证实，不能直接定为外观问题。
- **SplatAnything 路线验证分工（2026-11-12 用户指定）**：Codex 做实验设计/证据审查，
  WorkBuddy 接续复现并执行。唯一任务入口 `da3_human/EXPERIMENT_PLAN.md` §0（最新 §0.0；WB-0～5 保留定义），
  回填 `splatanything/ROUTE_VALIDATION_REPORT.md`；三个裁决为新基座是否达标、DA3 是否需
  几何修正、SplatAnything 训练改造成本。首轮训练仅做最多 100 step 成本冒烟，
  不把它当 per-scene 产品链或泛化结论；后续域训练据实测报告安排。
- **DA3 复现错误存档 + 防再犯清单 + DA3 训练事实** 全部在 `da3_human/ERRATA.md`；
  **跑 DA3 推理/训练/评估前必读**，其它算法项目不用看。

## 权重复用池 D:\wheel（2026-11-05 峰哥约定，通用）
- **下载模型前先在 `D:\wheel` 里搜**（如 `D:\wheel\vggt_human_ms\VGGT-Omega\` 早有
  VGGT-Ω 权重，别重复申请/下载）；**新下载的模型落盘到 `D:\wheel\<方法名>/`**，
  消费侧（`~/model/`、各 repo `checkpoints/`）一律软链指向 /mnt/d/wheel，不复制。
- querysplat / vggt_omega 的 `01_download_models.sh` 已内置三段式（wheel → 消费侧 →
  下载到 wheel）+ SHA256 校验 + 「收编」逻辑（消费侧已有实体文件自动 mv 进 wheel）。

## 工程纪律
- **commit**：中文，`<项目名>: <简短描述>`（英文冒号+空格）；默认只写标题行，正文最多 1-2 行。
  规范原文 AGENTS.md §8；远端操作禁令见 §11（**AI 不 push**）。
- **⏱ 改完就提交，不等提醒**（2026-11-03 用户批评）：每完成一个逻辑改动（新建/重写文档、
  改脚本、加实验记录）**当轮立即 commit**，不要攒着等下一步或等用户问。只 add 本次改动的
  文件（`git add <具体路径>`），**不 `git add -A`** —— 仓里常有其它会话的半成品。
  同一逻辑改动的 memory log 跟主改动一起提；与主改动无关的他人改动留在工作区。
  根目录跨项目文档前缀用 `media_code:`（先例 `394dbfa`）。
- **先读规范再动手**：改 git 前读 AGENTS.md 目录/命名/commit 章；跑 WSL 训练前读
  README_wsl.md 路径策略（训练输出写 `~/output/` Linux fs，跑完用 08 搬到 `/mnt/d/`）。
  历史事故：09-05 未读目录规范 git rm+restore 触发 9p 竞态，vggt_human 整目录被识别为删除。
- **文件删除红线**：清场删除前判 `is_symlink()`（99k 曾穿透软链删掉采集包 82 张 HEIC）；
  临时文件走「备份+搬移」到 `/mnt/d/output/media_code_*_backup_<date>/`，判定无用要落到
  「结论已入库 / 已被正式脚本取代 / 无进程在跑」三条。
- commit 后按需更新 EXPERIMENTS.md（实验结论），与 README（运行命令）分离。
- **入库边界：只入可复用脚本与规格文档**（2026-11-13 峰哥定）：成品图与一次性出图脚本
  **都不进 GitHub**。判断标准「换个内容还能直接用的入库，绑定某一张图的不入库」。
  落点：可复用的放 `figkit/`（SPEC + figlib/chartlib）与 `*/FIGURES.md` 台账；一次性脚本
  放 `D:/<项目>_figures/_build/`，成品放同目录 `figures/`，旧版 `_legacy_*/`。
  规范原文 AGENTS.md §14。`.gitignore` 已排除 `*.png *.jpg *.mp4`。

## 关键文档索引
- **`compare_4methods/IMPROVEMENTS.md` + `ERRATA.md` + `FINDINGS.md`**（2026-11-08 建）：DA3-GS 的
  归位与演进台账。**已定论（2026-11-09）**：
  - **正式口径 = 3DGS EWA 光栅器 + 官方等价导出（不裁剪 + 全 SH）**，全图 **22.11 dB**、
    人脸 **20.00 dB / 锐度 347**（3 视角、逐视角人脸窗）。历史 PLY 口径（裁剪+DC）= 18.97。
  - **2DGS 路线终止**：只换光栅器 −0.29 dB；把盘法向换成 GT 真相（oracle，覆盖率 65.1%）
    反而 −0.71 dB、人脸 −1.16 —— **"法向没学对"不是 2DGS 落后的原因**，I-04（重训法向正则）撤销。
    三个替代解释同时被证伪：确实是薄盘（s_min/s_mid 中位 0.0000）、α 覆盖没变薄、
    `eps2d` 在 gsplat 1.5.3 的 2DGS 路径上**被忽略**（三档读数逐位相同）。
    2DGS 只剩一个用途：`median_depth`（掠射一致性 0.41% vs 3DGS 0.80%），做**几何**时单独用。
  - **区域指标必须逐视角定框**：固定人脸框在 view 8 落在书架上，把人脸锐度虚高 +180、
    把 I-03 的锐度差从 −5 夸大成 −69。报数前必须目视核对裁切图。
- **`compare_4methods/P0_EXPERIMENTS.md` + `P1_EXPERIMENTS.md`**（2026-11-06 建）：四方法对比的
  加容量/加速实验台账（先写预测与判据、再填实测）。已定论：**DA3 的瓶颈是跨视角几何不一致**
  （质量随"相机中心到参考视角三维距离"单调衰减），加分辨率/SH/视角数三项全判 B；
  **前馈高斯的正确用法是"稀疏点云 init + 让 3DGS 自己 densify"**——整套高斯灌入反而更差更慢
  （opacity 中位数 0.036 卡在 prune 阈值上方，scale 0.8 mm 又要靠 densify 长回去）。
  可用甜点：DA3 点云 + 5k 步 ≈ 2.6 min / 22.2 dB（30k 基线 24.3 dB / 21.5 min）。
- **`deformable_human/BACKLOG.md`**：Reconstruction 流水线 A~J 案评审台账（状态总览表 +
  逐案原理 + 本地批注）。**做任何优化前先查状态总览**。截至 2026-10-17：
  A 无收益（−0.31dB）｜B 已实现但双度证伪｜C 零加速否决｜D 已落地（默认关）｜
  E 首轮负结果｜F/G/H 零实现｜I 未立案｜J per-person 已实测。
- **`som_human/EXPERIMENTS.md`**（2026-10-17 建）：E-01~E-07 实验归档，含手部糊根因分层、
  SOM 退化解剖、**E-07 论断边界（实测 vs 外推的分级）**。
- **`vggt_human/DESIGN_face_pipeline.md`**：人脸链路 v2 设计（FLAME+DECA+468 点+重心绑定，
  八阶段）。讨论人脸链路前先读。vggt_human 三文档分工：README=命令｜NOTES=原理排错｜
  EXPERIMENTS=实验结论（日期倒序追加）。MHR 身份 45 维（20 body+20 head+**5 hands**）+
  72 FACS 表情 + ExpTracker 657 点。
- **som_human 标准输入**＝华为「Remy」采集包：`transforms.json`（逐帧内参 + c2w/OpenGL）
  + `image/*.heic` + `pcd.ply`。位姿转换 `w2c = inv(M @ diag(1,-1,-1,1))`（实测判定，勿凭直觉改）；
  入口 `som_human/01_convert_capture.sh`，规范见 `som_human/README.md`。

## tools_99/（重建产物工具集，2026-10-15 从 vggt_human/ 迁出）
- **路径全部相对 cwd（media_code）解析**，与脚本目录无关——这是能整体搬走的原因。
- 打包链：99a 收集 ply｜99b PLY→UWA MP4（中间产物 `mp4_work/`，KEEP_WORK 默认保留，
  是 99c/99e 复用前提）｜99c 收窄视角重封装（依赖 99b 的 mp4_work）｜99d 复制三帧｜
  99e MP4→PLY｜99f 收集 ply+相机参数+front_image 到 `<批次>/ply_viewlimit/<task>/`（缺一即 skip）。
- 体检/工具：99g 采集包轨迹图与视角约束可视化｜99h 全仓 Python 3.10 语法自查（默认扫仓根）｜
  99k 采集包预处理（**主体居中必需**，出 `gs_camera_params.json`；出图=子进程调 99g，不复制其代码）。
- **居中为何必需**：json 的 az/el/r 相对球心、平移不变；但下游把角度当「绕世界原点」用
  （99b:435 target=[0,0,0]、99b:445 由 positions[0] 反算 init、99b:432 相机放 `[0.9r,1.2r]` 球壳），
  而 Remy 包原点是 AR 会话起点。实测 10 包 |target|=半径的 86%~198%，不居中则 Δyaw 最大 174°。
- ⚠️ 99b/99f 里的 `gs_camera_params_final.json` 是**外部产物**的另一套同名约定，别跟着改名。
- 区间常量与 99g 同源（`PHI_PAD/THETA_PAD/RADIUS_PAD/FULL_PHI_SPAN` + 视角裁剪
  `THETA_TOP/BOTTOM/PHI_LEFT/RIGHT`，语义同 99c 的 crop_axis）；`THETA_MODE=peak` 的
  gap-split 函数在 99k/99g **逐字同源**（`median_of`→`compute_view_limit` 之间应完全相同）。
  `THETA_GAP_MIN` 默认 2.5°（实测单圈内部最大间隙 1.71°）。env 改名对照见 tools_99/README。
- 写 JSON 一律 `open("w", encoding="utf-8", newline="\n")`（Windows universal newlines 会翻 CRLF）；
  gs json / center_report 用 `indent=4`，transforms.json 保持紧凑单行。
- 本机跑 99g 贴实拍首帧需 pillow：用 `~/.workbuddy/binaries/python/envs/default` 的 python
  （自带 pillow+pillow_heif），否则静默退无左栏版面。

## 本机 WSL 环境（2026-09 实测）
- **网络**：不走代理。`github.com` ✅ 直连 clone｜HF ❌｜hf-mirror ❌｜ghfast ❌ 证书失效｜
  `raw.githubusercontent.com` ❌。策略：GitHub 直连；HF 大文件走迅雷（峰哥手动）；
  pytorch.org / anaconda / tuna pypi 一直可用。**网络会漂移，动手前先 curl 复测**。
- **硬件**：WSL2，54 GiB 内存 / 128 GiB swap；`/` 1007 G｜`/mnt/d` 9.1 T；i9-12900K 8 核；
  RTX 3090 24 GB（驱动 595.95）。`.wslconfig` 的 `vhdxSize` 是无效键（会告警）。
- **conda**：`~/miniconda3`，现有 env：4danyone / colmap / flame_human / minimax_h3 /
  osediff / sam3 / sharp / vggt_human。新任务优先新建独立 env，别污染现有。
- **colmap env**（01c 依赖）：conda-forge colmap 4.2.1 cuda_129；**必须补钉 `openimageio=3.1`**
  （配方漏声明且 ABI 锁 3.1）；别用 tuna conda 通道（repodata 卡死）；4.x mapper 要求
  `--output_path` 目录已存在；直调二进制需 `LD_LIBRARY_PATH=~/miniconda3/envs/colmap/lib`。

## 服务器侧「3DGS 相机位姿调整工具」（2026-09-28 峰哥提供；本机无）
- 四模块：`crop_camera_by_angle.py`（视线交汇最小二乘求目标点 + 方位角循环补偿）｜
  `visualize_cameras.py`｜`generate_camera_files.py`｜`main.py`（Tkinter GUI 调参闭环）+ MP4 解封装页。
- 同族能力对照：正向封装≈99b｜反向解封装≈99e｜轨迹/视角可视化≈99g｜区间收窄≈99c。
- 汇报页已于 2026-10-16 删除（提交 `f1ae570`）；配色口径留档在 `2026-09-28.md`
  （靛蓝=数据 · 青=核心 · 琥珀=附加说明）。

## som_human 手部重建：根因分层与已证伪路线（2026-10-17）
- **三种"糊"分开看**：
  ① 输入运动模糊——本数据不存在（GT 手部锐度 4.28~4.33 > 静止区 2.75~2.81）；
  ② 重建过平滑——渲染高频只剩 GT 的 9~12%，**头也糊**；提容量**已证伪**
     （-r1/30k/268k 点 锐度保留 14% < -r2/10k/116k 的 24%，更多高斯更易摊平）；
  ③ **两态几何冲突**（垂手 vs 抬手占同一空间）→ 手臂半透明/幽灵。静态 3DGS 本质限制，
     唯一有效解 = 按姿态分段只用主段训练（hand10 实删抬手段，视觉变实、FG PSNR −0.24 抓不到）。
- **两条不同性质的路线都已证伪**（判据：运动基幅度 + 系数场熵）：
  - **SOM**：hand10 平移 0.0087 / 旋转 0.26° / 熵 2.10；hand06 三档
    （per-person / W=3 / W=5+lr×5+smooth÷10）平移 0.0020→0.0025→0.0238，
    调参能让基"动"12× 但仍小 2 个数量级，且**手部锐度 38%→9%**（错误微动把手拉糊）。
  - **可微位姿精炼（B 案）**：COLMAP 好初值上 −8.29dB。
  - 共同根因：**弱纹理小手区域的亮度梯度不足以驱动运动参数**（B 案二次独立验证）。
- **论断边界（E-07，勿外推）**：已实测=SoM 与本仓 deform 场；HiMoR **未跑过任何实验**，
  仅按"同读光度梯度"机理预判会退化【待核实】；"所有学习方法都不行"不成立（近景特写/
  强纹理/大运动+全覆盖三类未测）。→ 换的是**运动信息来源**：从"梯度学"改"先验给"
  （MANO/MHR 参数化拟合、2D 关键点/光流对齐）。
- **删帧判据是"姿态段"不是"偏离大"**：用 mask 顶部上移或面积骤降自动分段；
  删前必须查该段相机方位角覆盖（hand10 抬手段仅 3.7° → 删了零代价）。
- **评估口径坑**：mask 内 FG PSNR **抓不到幽灵/透明**（幽灵在 GT mask 之外），
  要配 mask 外泄漏指标或 alpha 渲染。
- **可复用资产**（与运动模型解耦，勿随路线否掉）：方案 J per-person 拆分（01g/01h，
  点数/体积 1/6、速度 3→32 it/s）、01c COLMAP 精化（+7.29dB）、SAM3 手部 mask 管线
  （01i 合并 / 01j 运动基退化判官）。
- **02_train.sh 默认全分辨率**（历史口径靠 `EXTRA_TRAIN_ARGS="-r 2"`，单跑必须显式传，
  否则慢 4 倍）。`pkill -f train_som.py` 会匹配到包装 shell 自身，杀完要单独验证目录清理。

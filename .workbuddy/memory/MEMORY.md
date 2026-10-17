# media_code 项目长期约定

## 工程纪律
- **commit**：中文，`<项目名>: <简短描述>`（英文冒号+空格）；默认只写标题行，正文最多 1-2 行。
  规范原文 AGENTS.md §8；远端操作禁令见 §11（**AI 不 push**）。
- **先读规范再动手**：改 git 前读 AGENTS.md 目录/命名/commit 章；跑 WSL 训练前读
  README_wsl.md 路径策略（训练输出写 `~/output/` Linux fs，跑完用 08 搬到 `/mnt/d/`）。
  历史事故：09-05 未读目录规范 git rm+restore 触发 9p 竞态，vggt_human 整目录被识别为删除。
- **文件删除红线**：清场删除前判 `is_symlink()`（99k 曾穿透软链删掉采集包 82 张 HEIC）；
  临时文件走「备份+搬移」到 `/mnt/d/output/media_code_*_backup_<date>/`，判定无用要落到
  「结论已入库 / 已被正式脚本取代 / 无进程在跑」三条。
- commit 后按需更新 EXPERIMENTS.md（实验结论），与 README（运行命令）分离。

## 关键文档索引
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

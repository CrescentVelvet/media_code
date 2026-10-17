# media_code 项目长期约定

## commit 规范（用户明确要求，2026-09-01）
- 中文，格式 `<算法名>：简短描述`
- **默认只写标题行，不写正文**；正文最多 1-2 行（规范原文在 media_code/AGENTS.md 第 8 节）
- 历史教训：406153d 写了多段正文，被用户纠正

## 先读规范再动手（2026-09-05，项目工作纪律）
- **操作 git 前先读 AGENTS.md 的目录结构 / 命名 / commit 规范章节**，不要凭印象套用旧路径或旧约定
- **跑 WSL 训练前先读 README_wsl.md 的路径策略章节**（训练输出写 `~/output/` Linux fs，跑完用 08 搬到 `/mnt/d/`；不要直接写 drvfs/9p）
- 通用原则：动手前先读相关 README 章节，再下命令
- 历史教训（vggt_human）：
  - 09-05 误删整个 vggt_human 目录：没读 AGENTS.md 目录规范就 git rm + git restore，触发 WSL 9p 与 git 索引竞态，工作区整目录被识别为删除（git restore 救回）
  - 09-05 训练输出写错位置：没读 README_wsl.md 路径策略就把 RESULTS_DIR 指向 /mnt/d（drvfs），违反"训练写 Linux fs"规范（已搬运修正）

## 关键文档索引（media_code）
- **`deformable_human/BACKLOG.md`（2026-10-13 建立）**：Reconstruction 流水线八案
  （A~H）评审台账——状态总览表 + 逐案原理/移植/验证 + 本地实验状态批注。
  **动手做任何优化方案前先查它的状态总览**（C 已否决、D 已实现、B 方向优先级最高）。
- **som_human 标准输入格式（峰哥 2026-10-13 定：今后输入统一为此）**＝华为手机
  「Remy」采集包：`transforms.json`（逐帧内参 k1..k3/p1/p2 + **c2w/OpenGL** 4×4）+
  `image/*.heic` + `pcd.ply`。位姿转换 `w2c = inv(M @ diag(1,-1,-1,1))`（实测判定，
  勿凭直觉改）；转换入口 `som_human/01_convert_capture.sh`，规范全文在
  `som_human/README.md`「输入数据格式」节。
- `vggt_human/DESIGN_face_pipeline.md`（2026-09-08 建立）：人脸链路 v2 设计方案与决策记录
  （FLAME+DECA+468 点+重心绑定+表情驱动，八阶段全链路）。**讨论人脸链路前先读它**。
- vggt_human 三文档分工：README=运行命令；NOTES=原理/排错；EXPERIMENTS=实验结论（按日期倒序追加）。
- **`tools_99/`（2026-10-15 从 vggt_human/ 迁出，git mv 保留历史）= 重建产物工具集**，三族：
  ① **打包收尾链 99a~99f**（预设路径写在各自 main() 里，非 env 默认）——99a 收集 ply
  （压平成 `<批次>/ply/<task>.ply`）｜99b PLY→UWA MP4（中间产物 `mp4_work/`，KEEP_WORK
  默认保留，是 99c/99e 复用的前提）｜99c 收窄视角重封装（mp4_crop，依赖 99b 的 mp4_work）
  ｜99d 复制三帧｜99e MP4→PLY｜99f 收集 ply+相机参数+front_image.jpg 到
  `<批次>/ply_viewlimit/<task>/`（三件成套，缺一即 skip）；
  ② **数据集体检 99g**（Remy 采集包轨迹图 / 视角约束可视化）；
  ③ **通用工具 99h**（全仓 Python 3.10 语法自查，**默认扫仓根**，som_human 也引用）；
  ④ **采集包预处理 99k**（2026-10-15 新增，提交 58eb1bb）：Remy 采集包 → 主体居中
  （球心默认 `sight` 最小二乘视线汇聚，`TARGET_MODE=anchor` 可切）+ 视角区间 →
  `gs_camera_params.json`（**2026-10-16 由 `gs_camera_params_final.json` 改名**，99b COLMAP
  模式消费 `setting.pitch_min/max`、`yaw_min/max`、`pitch_init`、`scene.radius`）+
  居中位姿/点云（`transforms.json`/`pcd.ply`）——**居中是必需项、不是可选项**（2026-10-16 评审）：
  json 里的 az/el/r 是相对球心算的、平移不变，但下游把角度当「绕**世界原点**的球坐标」用
  ——99b:435 写死 `target=[0,0,0]`、99b:445 由 `positions[0]` 绕原点反算 init 位姿、
  99b:432 `[0.9r,1.2r]` 直接把相机摆在原点球壳上；而 Remy 包的原点是 **AR 会话起点**
  （99g:621 记「实测离场景 0.5–5.1 m」）。实测 10 包：|target| = 环绕半径的 **86%~198%**
  （1.19~2.73 m），不居中则 99b 反算的 init 偏 Δpitch 最大 38.7°、Δyaw 最大 **174°**。
  `OUT_CENTERED=0` 只作诊断逃生口，默认必须居中。
  ⚠️ **99b/99f 里的 `gs_camera_params_final.json` 是另一回事、别跟着改**：那两处读的是
  重建流程（`code/Reconstruction/output/…`）与 `3d_gaussian_photo/…/UserData` 的**外部产物**，
  与 99k 只是同名约定；跟着改会让它们找不到文件。
  **源图不复制不挂链**（2026-10-16 删掉 `OUT_IMAGES`/`link_images`）：99k 出图时透传
  `PHOTO_ROOT=<原批次根>`，99g 在本包找不到 `image/` 时按**同名 ID 目录**回源包读实拍首帧
  → 图面照旧、源图只读（99g 单跑要自己设 `PHOTO_ROOT`；不设则退无左栏版面，不报错）；
  som_human 01 要吃居中位姿得拿原包副本覆盖 `transforms.json`/`pcd.ply`。
- **本机跑 99g 贴实拍首帧需要 pillow/pillow-heif**（2026-10-16 实测）：WorkBuddy 自带的
  python 3.13.12 **没装 PIL** → quad 静默退无左栏版面（看着像"没这张图"）；
  `~/.workbuddy/binaries/python/envs/default` 这个 venv 装了 pillow 12.3.0 + pillow_heif 1.8.1，
  出图时 `PLOT_PYTHON=<该 venv>/Scripts/python.exe` 即可。
  区间口径与余量常量与 99g 同源（同名 env：`PHI_PAD/THETA_PAD/RADIUS_PAD/FULL_PHI_SPAN`），
  **改一处要同步另一处**；`RADIUS_MODE` 默认 `mid`（令 99b 硬编码的 `[0.9r,1.2r]` 罩住实测半径带）。
  **观看视角约束裁剪**（2026-10-16 新增）：`THETA_TOP/THETA_BOTTOM/PHI_LEFT/PHI_RIGHT`
  在「实测±余量」之后再往里收（默认全 0 = 与历史产物逐位一致），语义/方向与 99c 的
  `crop_axis` **完全一致**（99c 原文 `Phi→[min+LEFT,max−RIGHT]`、`Theta→[min+BOTTOM,max−TOP]`）。
  ⚠️ Theta=90−el、Phi=−az 是**反向**的：`THETA_TOP` 收 Theta 上界 = 抬 `pitch_min`（去相机
  最低/最平视那端）；`THETA_BOTTOM` 压 `pitch_max`；`PHI_LEFT`→`yaw_min`↑、`PHI_RIGHT`→`yaw_max`↓。
  代码内部用 el/az 计算且**只裁被设的那一侧**（另一侧逐位不动），`Theta/Phi` 只做显示换算
  （90−(90−x) 有 ~7e-15° 浮点往返误差，实测 10 包 80 个角度 20 个不逐位相等）；裁剪发生在
  init 夹取**之前**（先裁后夹）；整圈时 `PHI_LEFT/RIGHT` 跳过并告警；收没了降级成一条缝
  （取初始视角，同 99c）。`trim_axis` 在两脚本同源但不属于逐字比对块。
  **env 改名**（2026-10-16）：`VL_*`/`PITCH_*`/`EL_*` → Phi/Theta 术语（`env_get(新名, 默认, *旧名)`，
  新名优先、旧名一次性提示）——`VL_TARGET→TARGET_MODE`、`VL_AZ_PAD→PHI_PAD`、`VL_EL_PAD→THETA_PAD`、
  `VL_R_PAD→RADIUS_PAD`、`VL_FULL_AZ→FULL_PHI_SPAN`、`PITCH_MODE→THETA_MODE`、`EL_*→THETA_*`、
  `VL_TRIM→LIM_TRIM`、`VL_MARK_ALT→MARK_ALT`。四向对照表见 tools_99/README「env 命名与旧名对照」。
  **仰角空洞保护**（2026-10-15，提交 `499a7ee`）：双峰数据（俯视+平视两圈，中间没拍到）
  原先 `min/max ± pad` 必然横跨空洞 → `THETA_MODE=peak` 按 gap split 只保留一圈。
  判据 `thr=max(THETA_GAP_MIN, THETA_GAP_K×中位帧间距)`；**THETA_GAP_MIN 默认 2.5°**
  （2026-10-16 由 8.0 下调，提交 `3e8db9f`）：服务器双峰样例的洞仅 3.29° 被 8.0 卡死；
  实测 10 包单圈内部最大间隙 0.11~1.71°、相对判据最多 0.41° → 拦门的恒为绝对下限，
  2.5 对实测最坏值留 46% 余量（1.5 会误切佳佳蹲）。
  这一组 env（`THETA_MODE/THETA_GAP_MIN/THETA_GAP_K/THETA_MIN_FRAMES/THETA_PICK/THETA_PAD_GAP/THETA_TIE_WARN`）
  在 99k 与 99g **同名同义**，且 99k/99g 各有**逐字同源**的四份函数
  （`median_of/split_el_segments/pick_el_segment/el_span`）—— **改一处必须同步另一处**，
  比对法：取 `def median_of` 到 `def compute_view_limit` 之间文本应完全相同。
  单圈数据下 peak 与 full 结果逐位相同（安全开关，默认 full）；pad 只加数据外侧、
  洞侧用 `THETA_PAD_GAP`；首帧落在被丢圈里时 `pitch_init` 夹进保留区间（记 `init_clamped`）。
  **出图 = 子进程调 99g**（2026-10-15 定，提交 5b569d0）：99k 的居中包就是 99g 的原生输入
  （把 `OUT_ROOT` 当 99g 的 `SRC_ROOT`），`OUT_PLOTS`/`PLOT_SRC(centered|origin|both)`/
  `PLOT_STYLE`/`PLOT_ROOT`/`PLOT_IDS`/`PLOT_PYTHON` 控制；图面默认落 `<批次>/plots/<模式>/`。
  实测 99g 在居中包上重算的球心 = 原点（1e-16）、Phi/Theta 与 json 逐样本一致（1e-14°）。
  **不复制 99g 的绘图代码、也不 import**（它是定稿汇报图面，复制必然分叉）。
  **gs json 的完整 schema**（2026-10-15，提交 eaf6f71）＝
  `setting{init_fov, yaw_init, yaw_min, yaw_max, pitch_init, pitch_min, pitch_max, roll_init}`
  + `camera_type:"pinhole"` + `intrinsics{width,height,fx,fy,cx,cy}` + `scene{center,radius}`；
  **99b 只读其中 5 个键**（四个区间 + `scene.radius`，`pitch_init` 仅做 1° 告警），多写不影响它。
  取值：`init_fov = 2·atan(h/(2·fx))·INIT_FOV_SCALE`（**与 99b:466 同一式**，默认系数 0.8）；
  `intrinsics` 取时间序首帧（width/height 写 int）；`roll_init=0`；`scene.center` 居中后为 `[0,0,0]`。
  输出格式约定：**gs json 与 center_report 用 `indent=4` 缩进，transforms.json 保持紧凑单行**。
  ⚠️ **Windows 上 `Path.write_text` 会把 `\n` 翻成 `\r\n`**（写出侧 universal newlines），
  本仓规范是 LF → 写 JSON 一律用 `path.open("w", encoding="utf-8", newline="\n")`。
  **99 系列路径全部相对 cwd（media_code）解析，与脚本所在目录无关**——这是它能整体搬走的
  原因；搬走时只改了 99j 一处硬编码绝对路径（已改为基于 `__file__`）与 3 处文档引用。

## 本机 WSL 网络与下载策略（2026-09-21 实测，跨会话通用）
- **本机不走代理**。`proxy.env` 里没有 `http_proxy`/`https_proxy`，只有 `HF_ENDPOINT=https://hf-mirror.com`、
  `HF_TOKEN`、`PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple`。
  → `AGENTS.md` 的 `_env.sh` 代理模板在本机是空转（不报错，但也不起作用）。
- 可达性（⚠️ 2026-10-11 复测，与 09-21 **相反**）：**`github.com` ✅ 200 直连可 clone（WSL/Win 均通）**｜
  `huggingface.co` ❌ RST｜**`hf-mirror.com` ❌ 直连超时（原镜像已挂）**｜**`ghfast.top` ❌ SSL 自签名证书失效**｜
  `raw.githubusercontent.com` ❌ 挂起（github.com 页面正常，raw 域名单独被墙）。
- 09-21 旧结论（ghfast 是 GitHub 唯一通道、HF 走 hf-mirror）**已失效**，网络状态会漂移，动手前先 curl 复测。
- 现行策略：**GitHub 直连 clone**；**HF 大文件走迅雷**（峰哥手动）；HF 小文件/API 可用 WebFetch（工具侧网络独立）。
  pytorch.org / repo.anaconda.com / tuna pypi 一直 ✅。

## 本机 WSL 硬件与 conda（2026-09-21 实测）
- WSL2，54 GiB 内存 / 128 GiB swap（`.wslconfig`：memory=56GB, swap=128GB, processors=8）；
  `/` 1007 G（可用 866 G）｜`/mnt/d` 9.1 T（可用 8.8 T，可写）；CPU i9-12900K 8 核。
- **`.wslconfig` 里的 `vhdxSize=100GB` 是无效键**，WSL 启动会报未知键警告（实际根分区 1007 G）。
- GPU：RTX 3090 24 GB，驱动 595.95，driver CUDA 13.2。
- conda：`/home/velvet/miniconda3`，env 有 4danyone / flame_human / minimax_h3 / osediff / sam3 / sharp / vggt_human。
  哪个 env 装了什么（py/torch/transformers/diffusers）见当日 memory log，新任务优先新建独立 env，别污染现有 env。

## 服务器侧「3DGS 相机位姿调整工具」工具链（2026-09-28 峰哥提供；本机无此工具）
- 四模块 + 一个反向入口：
  `crop_camera_by_angle.py`（核心算法：**视线交汇最小二乘求目标点** + ±180° 方位角循环补偿 +
  角度/高度/半径多维调整）｜`visualize_cameras.py`（3D 空间分布 + 极坐标角度分布，裁剪前后自动
  对比，输出 300 dpi PNG）｜`generate_camera_files.py`（自动生成 cameras_init.json /
  cameras_out.json / view_params.json，含角度范围·包围盒·半径）｜`main.py`（Tkinter GUI +
  PyInstaller 单文件 exe，**调参闭环：看图验证 → 调整参数**）｜MP4 解封装标签页（反向提取
  3DGS 模型与相机数据）
- 数据流：UWA 高斯模型（camera.json / camera_crop.json）+ COLMAP·实拍 → crop → 调整后相机 →
  generate → 三件套 json → 编码 & 封装 → 3DGS 壁纸产出
- 与本机同族能力对照：正向封装 ≈ 99b、反向解封装 ≈ 99e、轨迹与视角约束可视化 ≈ 99g、
  视角区间收窄回灌 ≈ 99c。术语与算法口径一致，可用 99g/99c 的实现作论据。
- 汇报页（峰哥汇报用，部门技术评审口径）**已于 2026-10-16 删除**（峰哥指示，提交 `f1ae570`）：
  原 `tools_99/camera_pose_tool_report.html` + `report_figs/`（4 组同名 svg + 3× png，3540 px 宽）
  + 99i/99j 两个脚本一并删掉，README 引用已清理。**历史口径留档**：图面配色 / 线宽曾于
  2026-09-28 定版（靛蓝 = 数据 · 青 = 核心 · 琥珀 = 附加说明/反馈），速查表见
  `.workbuddy/memory/2026-09-28.md`；若日后重建汇报页可沿用该配色口径。

## COLMAP 环境（2026-10-13 建立，som_human 01c 依赖）
- WSL 无系统 colmap（apt 版 CPU-only 且需 sudo）。已建独立 env：
  `~/miniconda3/envs/colmap`（conda-forge colmap 4.2.1 cuda_129，GPU SIFT 可用）。
- **必须补钉 `openimageio=3.1`**（4.2.1 配方漏声明且 ABI 锁 3.1，装 3.2 会缺
  `libOpenImageIO.so.3.1`）。
- **别用 tuna 的 conda 通道**（repodata 404/落后导致 solver 卡死）；官方
  conda-forge 直连 ~850KB/s 够用，libmamba 解依赖 14s。
- COLMAP 4.x 的 mapper 要求 `--output_path` 目录**已存在**（3.x 自动建）。
- 调用时若直接用二进制路径（不经 conda activate），需
  `LD_LIBRARY_PATH=~/miniconda3/envs/colmap/lib`（01c 脚本已自动处理）。

## 文件操作红线（2026-10-15，99k 图链事故）
- **清场式删除前必须判 `is_symlink()`**：`for p in dst.iterdir(): p.unlink()` 在目标为
  软链接时会**穿透链接删掉源目录里的真文件**。99k 第一版曾因此把采集包里 82 张
  HEIC 送进回收站（已逐张恢复）。规则：链接只摘链接、目录非空跳过、只回滚自己创建的文件。
- 本机删除会被沙箱转入回收站（`os.unlink` 也可救回）：`/d/$RECYCLE.BIN/<SID>/` 下
  `$I`（元数据，含 UTF-16 原路径）+ `$R`（实体）成对；恢复用 copy 回原路径，回收站留底。
- **临时文件清理走「备份 + 搬移」而非物理删**（2026-10-16 二次清理定法）：`.workbuddy/memory/`
  里 `_*` 前缀（旧批为 `tmp_*`）是一次性脚本与调试图，`mv` 到 `/mnt/d/output/media_code_*_backup_<date>/`。
  清理前先 `git status` 分跟踪/未跟踪——dated `.md` 与 `MEMORY.md` 是记忆本体，永不清理；
  判定「无用」要落到三条：结论是否已入库、是否已被正式脚本取代、是否还有进程在跑。

## som_human 手部重建：糊/透明的根因分层与评估口径（2026-10-17，hand10 实测）
- **三种"糊"要分开**：
  ① 输入运动模糊——本数据不存在（GT 手部锐度 4.28~4.33 反而高于静止区）；
  ② 重建过平滑/容量不足——渲染高频只剩 GT 的 9~12%（Δ −0.45~−0.52 log10 Lap 方差），
     **头也一样糊**，说明是普遍保真度问题。~~提容量可改善~~ **已证伪（2026-10-17 对照）**：
     -r1 + 30k iter + 268k 点的 hi 模型锐度保留率 14%，比 -r2/10k/116k 的 24% **更差**
     （更多高斯 = 更容易摊平）。容量不是手部糊的解。
  ③ **两态几何冲突**（大姿态变化，如垂手 vs 抬手占用同一空间）→ 手臂半透明/幽灵。
     这是静态 3DGS 的本质限制，**唯一有效解是按姿态分段、只用主姿态段训练**。
  ④ **SOM 运动基退化（2026-10-17 判决完毕，路线证伪）**：hand10 model_p07 的
     10 条基平移仅 0.0087、旋转仅 0.26°，系数场塌缩 → SOM ≈ 恒等。档 0 三档
     判决实验（hand06 写字，per-person + 手部加权 W=3/5 + basis_lr×5 +
     smooth÷10）**全部救不活**：调参能让基幅度升 12×（0.0020→0.0238），但仍
     比真实手部运动小 2 个数量级，且**手部锐度从 38% 暴跌到 9%**（错误微动
     拉糊高斯）。结论：**光度梯度驱动运动参数（SoM/HiMoR 同类）在手部弱纹理
     小目标上不可用，与 B 案同根因二次验证，勿再投入**；转向 MANO 手模先验
     （HaMeR 拟合→运动靠手模给）或 2D 对齐。工具：`01i_merge_hand_masks.py`
     （合并 SAM3 hand 多实例 mask）、`01j_som_anatomy.py`（运动基退化判官）、
     train_som.py 新增 `SOM_HAND_MASK_DIR`/`SOM_HAND_WEIGHT`（保留，勿删——
     判决实验的可复现依赖）。
  ⑤ 两类手部运动要分开：**手臂刚体大位移**（招手/抬手 → 两态冲突 → 删姿态段有效）
     vs **手局部形变**（写字/弹琴，手臂静止手持续动 → 删帧无效，需运动分解）。
- **删帧判据不是"偏离大"而是"姿态段"**：自动判据用 mask 顶部上移（抬手 → top 骤降）
  或 mask 面积骤降。删前必须查该段的相机方位角覆盖——本例抬手段只占 3.7°（采集末尾
  定点拍），删了零代价；若抬手段覆盖了大角度，删了会缺面，不能删。
- **评估口径坑（重要）**：mask 内 FG PSNR **抓不到幽灵/透明**——幽灵色块落在 GT mask
  之外（手臂在"其他时刻"的位置）。验证：trim105 删掉抬手段后视觉明显更实，但 FG PSNR
  反而 −0.24dB。评"透明"要用 mask 外泄漏指标或 alpha 渲染，别只看 PSNR。
- 残余的轻微幽灵来自"招手微动"本身 → 这是 som_motion 动态建模要解决的，删帧治不了。

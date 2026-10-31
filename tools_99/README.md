# tools_99 — 重建产物的封装收尾、采集包预处理与数据集体检工具集

2026-10-15 从 `vggt_human/` 迁出（`git mv`，保留提交历史）。本目录原属于
vggt_human 的下游「99 系列」，但它们的服务对象已超出人脸重建本身：既有对
UWA 三件套的封装收尾，也有全仓通用的语法自查工具，还有 Remy 采集包的居中预处理。
脚本文件名一律未改（保留 99 前缀的连续性）。

## 五个族

| 族 | 脚本 | 用途 | 运行依赖 |
|---|---|---|---|
| **A. 打包/收尾链** | 99a 99b 99c 99d 99e 99f | 重建产物 ↔ 手机可播放 MP4 的双向封装 | UWA 工具链 + ffmpeg + conda env |
| **B. 数据集体检** | 99g | Remy 采集包轨迹图 / 视角约束可视化 | **纯标准库**，Python ≥ 3.8 |
| **C. 通用工具** | 99h | 全仓 Python 语法兼容自查 | **需 Python ≥ 3.12** |
| **D. 采集包预处理** | 99k | 主体居中 + 视角区间 → `gs_camera_params.json`，并调 99g 出图 | **纯标准库**，Python ≥ 3.10 |
| **E. 运镜渲染** | 99l 99m 99n | 重建相机 → 六种运镜三件套 → PLY 渲染 MP4；99n = 多场景批量编排（发现 → 99l → 99m） | 99l 只要 numpy；99m 需 GPU + gsplat 1.5.2 + torch 2.6；99n 纯标准库 |

族 A 内部是**运行时数据依赖**（99c/99e 复用 99b 留下的 `mp4_work/`），没有代码级
import。族 B（可视化）与族 D（出参数）都吃 Remy 采集包，视角区间的算法与
余量常量同源，改一处务必同步另一处；族 D 跑完会**子进程调族 B（99g）**给居中包出图
（`OUT_PLOTS=0` 关掉）——为什么是调用而不是复制代码，见 D 节的「与 99g 的分工」。

---

## A. 打包 / 收尾链（99a~99f）

### 链路总览

```
99a 收集 ply          <源批次>/<task>/point_cloud_final.ply → <批次>/ply/<task>.ply
99b 封装成 MP4        图片序列 + ply + 三件套 json → <批次>/mp4/<task>.mp4
  └─ 中间产物保留在    <批次>/mp4_work/<task>/（bin / output.mp4 / uwa_json/ / 3DGS.glb）
99c 收窄视角重封装    复用 99b 的 mp4_work → <批次>/mp4_crop/<task>.mp4
99d 抽取三帧          每个 task 首/中/尾帧 → <批次>/frames3/
99e 反向解封装        <批次>/mp4/<task>.mp4 → PLY + 三件套 json（99b 的严格镜像）
99f 收集 ply+相机     <源批次>/<task>/ → <批次>/ply_viewlimit/<task>/（三件成套，缺一即 skip）
```

统一结果根默认 `../../output/recon_human_results`，批次名与源目录同名。

### 常用命令

> **铁律：调批次时显式写出源目录与输出目录，不靠脚本默认值。**
> 99 系列的路径默认值写在各自 `main()` 里（相对 cwd，cwd 应为 `media_code/`）。

```bash
cd /mnt/c/code/media_code

# 99a 收集 ply（压平成 <批次>/ply/<task>.ply）
SRC_ROOT=/mnt/d/output/recon_human_results/B003_Human_Data_w_pose \
  bash -c 'python tools_99/99a_collect_ply.py'

# 99b 封装 MP4（自动判 UWA / COLMAP 模式；COLMAP 型现场生成三件套）
SRC_ROOT=/mnt/d/output/recon_human_results/B003_Human_Data_w_pose \
TOOL_DIR=/mnt/d/model/UWA_Sample_Tool_v3 \
  python tools_99/99b_pack_ply_to_mp4.py
# 输出：<批次>/mp4/<task>.mp4 + <批次>/mp4_work/<task>/（KEEP_WORK 默认 1 保留）

# 99b 先探路不执行
DRY_RUN=1 python tools_99/99b_pack_ply_to_mp4.py

# 99c 收窄视角重封装（依赖 99b 的 mp4_work/，不覆盖 mp4/ 成品）
FORCE=1 RADIUS_RANGE_SCALE=0.5 THETA_BUFFER=0 PHI_BUFFER=0 \
INIT_FOV_SCALE=1.25 PHI_MARGIN=40 THETA_MARGIN=40 \
  python tools_99/99c_repack_view_limits.py
# 输出：<批次>/mp4_crop/<task>.mp4
# DRY_RUN=1 只打印收窄前后范围

# 99e 反向解封装
MP4=/mnt/d/output/recon_human_results/<批次>/mp4/taskA.mp4 \
  python tools_99/99e_unpack_mp4_to_ply.py
```

### 关键 env（99b / 99c 通用）

| var | 说明 |
|---|---|
| `TOOL_DIR` | UWA 工具链根（含 `encode.py` / `muxer.py` / `build/gltf_packer`） |
| `SRC_ROOT` | 批次根目录，其下每个子目录一个 task |
| `RESULTS_ROOT` | 统一结果根，默认 `../../output/recon_human_results` |
| `OUT_DIR` | 显式覆盖输出目录（99b 默认 `<批次>/mp4`，99c 默认 `<批次>/mp4_crop`） |
| `WORK_ROOT` | 99c 专用：99b 中间产物根，默认 `<批次>/mp4_work` |
| `MODE` | `auto`（默认）/ `uwa` / `colmap`，强制指定模式 |
| `ONLY` / `FORCE` / `DRY_RUN` | task 白名单 / 覆盖已存在 / 只探路 |
| `KEEP_WORK` | 99b：默认 `1` 保留中间产物（99c/99e 复用它的前提），`0` 成功后清理 |
| `PHI_MARGIN` / `THETA_MARGIN` / `RADIUS_RANGE_SCALE` | 99c 收窄量（默认 10° / 10° / 1 不动） |
| `INIT_FOV_SCALE` / `INIT_RADIUS_SCALE` | 99c 初始取景补偿（默认 1 不动；主体占满屏看不全时试 1.25） |

### 模式判定（99b）

| 模式 | 判据 | 三件套来源 |
|---|---|---|
| **UWA** | `init_camera.json` + `camera.json` + `view_limits.json` + `<ply>` 齐全 | 现成，直接用 |
| **COLMAP** | `cameras.txt` + `images.txt`（`points3D.txt` 可选）+ `<ply>` | 现场生成（另需 `gs_camera_params_final.json`，或 `GS_FALLBACK=1` 用相机位姿推断） |

两者都存在时 UWA 优先。COLMAP 型缺渲染帧时返回 `pending`（三件套已生成，补帧后重跑即可）。
角度字段映射（已验证）：`Phi = yaw`、`Theta = 90 - pitch`、`minR = 0.9r`、`maxR = 1.2r`。

⚠️ `init_fov` 生成时乘了经验系数 **0.8**（等效初始放大 ~37%），这是 99c 需要
`INIT_FOV_SCALE=1.25` 补偿的由来。

⚠️ `thetaBuffer` / `phiBuffer` 需打过 cgltf 补丁的封装链才会写进 MP4（旧成品解出来没有这两项）；
区间收窄是否生效待复验，建议重打包后用 99e 解回比对。原理详见
[../vggt_human/NOTES.md](../vggt_human/NOTES.md) 第 9、10 条。

---

## B. 数据集体检：Remy 采集包轨迹图（99g）

对批次目录下每个采集 ID 画一张自包含 HTML 轨迹图（图内标题与文件名都用文件夹 ID），
用于快速判断这批数据「拍得怎么样」：环绕是否完整、相机到目标距离是否稳定、视线是否对得准。
每个风格同时落一份独立 `.svg`（矢量原图，可直接拖进 PPT / 报告排版）。

**注意**：99g 也是 99k 的出图后端 —— 99k 跑完会 `subprocess` 调它，把**居中包**当
`SRC_ROOT` 画一遍（`PLOT_SRC=origin` 可换成画原始包），见 D 节「与 99g 的分工」。
所以本脚本的 `SRC_ROOT` / `STYLE` 等价物在 99k 侧叫 `PLOT_*`。

```bash
# 默认：quad 风格（四联对照图 2×2）
# ⚠️ 脚本里的 SRC_ROOT 默认值是**服务器路径**，本机测试必须显式覆盖成本机的 D:/dataset/测试数据sample
SRC_ROOT=D:/dataset/测试数据sample python tools_99/99g_plot_capture_trajectory.py

# 一次出全部风格（挑图用）+ 额外生成 index.html 总览页
SRC_ROOT=D:/dataset/测试数据sample \
DST_ROOT=../../output/remy_traj_html \
STYLE=all python tools_99/99g_plot_capture_trajectory.py

# 只要原始双联图（无约束层）
SRC_ROOT=D:/dataset/测试数据sample \
IDS=13a8ecadfeb448e890db319ac828befe \
STYLE=combo python tools_99/99g_plot_capture_trajectory.py
```

风格可选 `quad`（**默认**：四联对照图 2×2 —— 左列是原始采集轨迹（俯视 XZ / 等轴测含 Y 高度），
右列是**同一取景下**的视角约束（俯视角度标注 / 等轴测 3D 球壳），见下节）/ `combo`
（上「俯视图 XZ」+ 下「等轴测含 Y 高度」上下排列，两块绘图区统一 800×450 预留框、图例都铺在
各自绘图区下方占满整幅宽度，俯视轨迹按帧序做蓝→紫→玫红渐变，统计卡片竖排在右栏）/
`minimal`（浅色极简）/ `darkspace`（深色网格 + 时间渐变）/ `fov`（视锥扇形 + 点云底图）
/ `iso`（等轴测单图），或 `all`。纯标准库实现（json/math/struct），不需要 conda env，
**兼容 Python ≥ 3.8**。
输入的 `transforms.json` 字段含义与四条易踩的坐标系坑见
[../vggt_human/NOTES.md](../vggt_human/NOTES.md)「Remy 采集包」。

### 四联对照图（quad · 默认风格）

一张图里 2×2 四块面板，绘图区统一 800×450、左右两列严格对齐：

| | 左列（原始采集轨迹） | 右列（视角约束） |
|---|---|---|
| 上行 | ① 俯视图（世界系 XZ，帧序渐变） | ② 俯视图 + 角度标注（`lim_angle`） |
| 下行 | ③ 等轴测（含世界 Y 高度） | ④ 等轴测 + 3D 球壳扇块（`lim_shell`） |

```bash
SRC_ROOT=D:/dataset/测试数据sample DST_ROOT=../../output/remy_quad_html \
STYLE=quad IDS=<id1>,<id2> python tools_99/99g_plot_capture_trajectory.py
```

四条约定（改这个风格前先看 [../vggt_human/NOTES.md](../vggt_human/NOTES.md)「四联对照图」）：

- **右列的球心改用「最小二乘视线汇聚中心」，不使用 `anchor_point`**（见下节）。左列是原始
  采集轨迹，参考点仍是 `anchor_point`——两列本来就不是同一件事，不需要统一。
- **四张图共用同一取景与比例尺**（= 相机活动范围 ∪ 锚点±均值半径 ∪ 约束范围 ∪ 对照球心）。
  这是这张图存在的意义：左右逐点对照「约束落在轨迹的哪一段」。代价是左列比 combo 单看时小一圈
  （约束范围通常是相机范围的 1.2–1.9 倍）。
- **右列的约束层用青绿**（`QUAD_INK`/`QUAD_FILL`），不用 vlimit 的紫 —— 左列轨迹是按帧序的
  蓝→紫→玫红渐变，紫约束压上去会和轨迹糊成一片。`vlimit` 系列风格仍用原紫色。
- **右列省掉两样**：均值半径环（与约束区间圈冗余）和 ρ 的图内数值文字（面板下方数值行已有，
  图内那个位置被 anchor_point / frame 0 / 世界原点 三个底图标签占着）。

### 视角约束范围（view limit）叠加

把「以球心为心、方位角/仰角/半径三区间围成的球壳扇块」画进轨迹图。区间由各帧实测
min/max ± 余量推出（不照抄外部那版 C++ 移植代码，四点问题见 NOTES.md）。

**球心取哪个点**由 `TARGET_MODE` 决定，默认 `sight`：

| 取值 | 含义 |
|---|---|
| `sight`（**默认**） | 最小二乘视线汇聚中心——最小化各帧视线到该点的正交距离平方和，**完全不使用 `anchor_point`** |
| `anchor` | 用 `transforms.json` 自带的 `anchor_point`（旧口径） |

```bash
# ① 选型：六种画法内联在同一张 HTML 里竖向排开，一页看完再定
SRC_ROOT=D:/dataset/测试数据sample DST_ROOT=../../output/remy_vlimit_html \
STYLE=vlimit IDS=<id1>,<id2> python tools_99/99g_plot_capture_trajectory.py

# ② 定稿后单独出一种（落 <ID>__lim_<mode>.html + 独立 .svg）
STYLE=lim_edges SRC_ROOT=... python tools_99/99g_plot_capture_trajectory.py

# ③ 直接把约束叠进 combo 的俯视图（叠 shell 则进等轴测面板）
VLIMIT_MODE=edges STYLE=combo SRC_ROOT=... python tools_99/99g_plot_capture_trajectory.py

# ④ 回到 anchor_point 口径 / 关掉对照点标记
TARGET_MODE=anchor python tools_99/99g_plot_capture_trajectory.py
MARK_ALT=0        python tools_99/99g_plot_capture_trajectory.py

# ⑤ 关掉右列的贴身裁剪（回到实测 min/max ± 余量的完整区间）
LIM_TRIM=0 python tools_99/99g_plot_capture_trajectory.py

# ⑥ 观看视角约束裁剪：在「实测±余量」之上再往图里收（方向与 99c 一致，见 D 节）
THETA_TOP=3 THETA_BOTTOM=5 PHI_LEFT=10 PHI_RIGHT=10 \
STYLE=lim_angle python tools_99/99g_plot_capture_trajectory.py
```

画法：`lim_band`（扇环填充）/ `lim_edges`（四条边界虚线，对轨迹影响最小）/ `lim_angle`
（边界 + 角度/半径数值标注）/ `lim_rings`（米制同心环 + 方位刻度）/ `lim_hull`
（对照组：相机水平位置凸包外扩，不假设环绕）/ `lim_shell`（等轴测 3D 球壳扇块）。
余量可用 `PHI_PAD` / `THETA_PAD` / `RADIUS_PAD` 覆盖，整圈判定阈值用 `FULL_PHI_SPAN`。

**四联图右列的贴身裁剪**（`LIM_TRIM`，默认 `0.2`；只有 `STYLE=quad` 的右列吃这个参数）：
右列画出来/标出来的 az/el 区间在「实测 ± 余量」之上两端各再收 20% 跨度，但两端都不越过
frame 0 的实测角（`min(lo+span·k, init)` / `max(hi−span·k, init)`），免得用户一开场就落在
约束外。**只作用于右列约束层、不改取景** —— 四块共用取景，取景若跟着变窄会把左下 ③
等轴测一起缩掉，左列就不是原图了。整圈（`FULL_PHI_SPAN`）时 az 不裁。`LIM_TRIM=0` 关闭。
副作用：el 跨度收到 8° 以下时球壳不填色（既有阈值），④ 会退化成细网格带。
**别和观看视角约束裁剪混**：`LIM_TRIM` 只是绘制层的收拢（只影响这张图怎么画）；而
`THETA_TOP` / `THETA_BOTTOM` / `PHI_LEFT` / `PHI_RIGHT` 是**改区间本身**的真实裁剪
（99k 里会进 gs json，这里只影响图面，见图注）。

**仰角空洞保护**（`THETA_MODE`，默认 `full`）：双峰数据（俯视 + 平视两圈）下仰角区间会
横跨没拍到的空档，设 `THETA_MODE=peak` 让本脚本与 99k 一样按 gap split 只保留一圈。
参数与语义见「D. 采集包主体居中 → 相机参数（99k）」一节的「仰角空洞保护」—— 注意
99k 出图时会自动把这组 env 透传下来，手工单独跑 99g 时才需要自己设。

**`PHOTO_ROOT` —— 实拍首帧贴图的源图回退根**（2026-10-16）：quad 左栏那张实拍首帧默认读
**被画的那个包**自己的 `image/`；如果被画的包没有 `image/`（典型就是 99k 出的居中包，
只有 `transforms.json` + `pcd.ply`），就按**同名 ID 目录**去 `<PHOTO_ROOT>/<ID>/image` 取。
所以"不复制、不挂链"和"图上照样有实拍图"可以同时成立 —— 图面照旧，源图只被读、不动。

```bash
# 画居中包（无 image/），实拍首帧回原批次根取
PHOTO_ROOT=D:/dataset/手部微动样例 SRC_ROOT=../../output/recon_human_results/<批次>/centered \
  python tools_99/99g_plot_capture_trajectory.py
```

回退成功会打一行 `🖼️ 本包无 image/，实拍首帧取自源包: …`；两边都没有可用图（或无
pillow-heif）时打印提示并回退无左栏版面，不报错。99k 出图时**自动透传**这个变量
（= 它扫的原始批次根），手工单独跑 99g 才需要自己设。

图上会同时画出**球心**（实心圆 + 十字）与**对照球心**（另一个候选点，空心叉 + 虚线 + Δ 距离），
标题行给出两者的拟合质量对照（垂距 RMS / 视线夹角 / 正规矩阵条件数）。实测 10 个样本
（见 NOTES.md「球心取哪个点」）：`sight` 垂距 RMS 0.081 m、夹角 3.46°，`anchor_point`
0.205 m、9.54° —— 10/10 个样本都是 `sight` 更贴合视线。

---

## C. 通用工具

### 99h — 提交前跑一次语法兼容检查

本机 WorkBuddy 的 Python 是 3.13，而 `vggt_human` env 是 **3.10.20**。3.12+ 才合法的写法
在本机一路通过、到服务器直接 SyntaxError（99g 第 434 行的 f-string 嵌套同类引号就这么挂过）。

```bash
python tools_99/99h_check_py_syntax.py            # 默认扫仓根 *.py，命中则退出码 1
python tools_99/99h_check_py_syntax.py a.py b/    # 也可指定文件/目录
```

需要 Python ≥ 3.12 运行该检查（tokenize 的 FSTRING_* token）；判定规则都是在 3.10.20 上
逐条实测过的，不是照文档猜的。检查分两道：先 `ast.parse` 拦普通语法错误，再用 tokenize
查「本机能过、3.10 过不了」的 f-string 写法。**不能只做后者** —— tokenize 是纯词法分析，
像 `f()    f()`（两条语句撞在一行）这种它一路通过（2026-09-28 实测漏报过一次）。

> 本工具已跨项目复用（`som_human/README_wsl.md` 也引用它），2026-10-15 起默认扫描目录
> 从「脚本所在目录」改为「**仓根**」——原先它恰好在 `vggt_human/` 下，默认值等于扫
> vggt_human，搬出本目录后那个巧合不再成立。

---

## D. 采集包主体居中 → 相机参数（99k）

```bash
python tools_99/99k_make_gs_params.py                    # 跑 main() 里预设的批次根
SRC=D:/dataset/手部微动样例/01e954…-佳佳抱枕 python tools_99/99k_make_gs_params.py
DRY_RUN=1 python tools_99/99k_make_gs_params.py          # 只算不落盘
ONLY=<task_a>,<task_b> python tools_99/99k_make_gs_params.py
OUT_PLOTS=0 python tools_99/99k_make_gs_params.py        # 只要参数、不出图
PLOT_STYLE=lim_edges PLOT_SRC=both python ...            # 换图面风格 / 顺便画原始包做对照
THETA_MODE=peak python tools_99/99k_make_gs_params.py    # 双峰数据（俯视+平视两圈）只保留一圈
THETA_MODE=peak THETA_PICK=init python ...               # 保留首帧所在的那一圈（开场视角不变）
# 观看视角约束裁剪：在「实测 min/max ± 余量」之上再往里收（默认全 0 = 不裁）
THETA_TOP=3 THETA_BOTTOM=5 PHI_LEFT=10 PHI_RIGHT=10 python ...
```

**为什么需要**：Remy 采集包的世界原点是手机开机点附近，主体（视线汇聚点）实测离它
1.19–2.73 m（10 包，`sight` 口径）；而 UWA 侧 `view_limits` 的 `target` 被 99b 写死成
`[0,0,0]`（`99b_pack_ply_to_mp4.py:435`）。不居中 → 手机端环绕中心整个偏离主体。
**「能不能不居中、居中到底是不是必需项」的完整论证与 10 包实测见下「为什么必须居中」。**

**四步**：① 求球心（默认 `sight` = 最小二乘视线汇聚中心，与 99g 同源；`TARGET_MODE=anchor`
切到采集包自带 `anchor_point`）→ ② 以球心为原点逐帧算 az / el / r 并取 min/max + 余量
（解缠 / 整圈 / 余量常量全部沿用 99g）→ ②b 观看视角约束裁剪（可选，见下）→ ③ 落盘 → ④
**调 99g 出图**（默认画居中后的包）。

**产物**（`<OUT_ROOT>/<task>/`，默认 `<RESULTS_ROOT>/<批次名>/centered`）：

| 文件 | 用途 |
|---|---|
| `gs_camera_params.json` | 相机参数：`setting.pitch_min/pitch_max`、`yaw_min/yaw_max`、`pitch_init` + `scene.radius`（其余字段见下）。字段口径与 99b 的 COLMAP 模式所需**一致**，但 99b 默认找的是重建流程的产物名 `gs_camera_params_final.json` —— 要拿它喂 99b 需改名（99b 侧暂未加文件名覆盖口） |
| `transforms.json` | 居中后位姿（同名 drop-in），世界原点 = 主体中心 |
| `pcd.ply` | 居中后点云（按属性偏移平移 xyz，其它属性原样） |
| `center_report.json` | 球心对照 / 拟合质量 / 区间明细 / 自检（诊断用） |

**源图一律不复制、不挂链**（2026-10-16 起删掉 `OUT_IMAGES` 与 `link_images`）：居中包只含
位姿与点云，需要图就直接用原始采集包（`01_convert_capture.sh` 也照样吃原包）。
**这不影响出图**：99k 出图时透传 `PHOTO_ROOT=<原批次根>`，99g 在本包找不到 `image/` 时按
同名 ID 回源包读实拍首帧，quad 左栏的贴图照旧（详见 99g 一节的 `PHOTO_ROOT`）。

`gs_camera_params.json` / `center_report.json` 落盘为 **4 空格缩进、LF 换行、无 BOM**
（便于人工核对与 diff）；`transforms.json` 保持**紧凑单行**，与上游采集包书写风格一致
（两者对 `json.load` 完全等价）。

**gs json 的字段**（4 个顶层块）：

```json
{
    "setting": {
        "init_fov": 56.364194283649056,
        "yaw_init": 68.03129209261009,
        "yaw_min": -49.12659718819826,
        "yaw_max": 170.07489874016608,
        "pitch_init": 22.982417952841683,
        "pitch_min": 18.730548539102887,
        "pitch_max": 28.08402128802671,
        "roll_init": 0.0
    },
    "camera_type": "pinhole",
    "intrinsics": {"width": 1440, "height": 1920,
                   "fx": 1359.494873046875, "fy": 1361.6729736328125,
                   "cx": 716.0700073242188, "cy": 952.9739990234375},
    "scene": {"center": [0.0, 0.0, 0.0], "radius": 1.4989950484938566}
}
```

取值来源：99b 只读 `setting` 的四个区间键 + `pitch_init`（做 1° 一致性告警）与
`scene.radius`，其余按键向上游 3DGS 侧相机文件的既有格式补齐，供人工核对与下游直接消费。

| 字段 | 来源 |
|---|---|
| `init_fov` | `2·atan(h/(2·fx))·INIT_FOV_SCALE`，**与 `99b_pack_ply_to_mp4.py:466` 同一式**；默认系数 0.8（99b 口径，99c 用 `INIT_FOV_SCALE=1.25` 反向补偿），想要原始竖直视场角设 1.0 |
| `intrinsics` | 时间序首帧的 `w/h/fl_x/fl_y/cx/cy`；`width`/`height` 写整数，其余浮点。缺字段时告警并留空，不编数 |
| `camera_type` | 恒 `"pinhole"`（采集包 `camera_model=OPENCV` 的畸变系数不进本文件，99b/99c 侧也按 PINHOLE 处理） |
| `roll_init` | 恒 `0.0`（本链路无滚转） |
| `scene.center` | 居中后恒 `[0,0,0]`；`OUT_CENTERED=0` 时写主体中心在原始世界系的位置并告警（与 99b 写死的 `target=[0,0,0]` 对不上）；详见下「为什么必须居中」 |

**图面**（`<PLOT_ROOT>/<模式>/`，默认 `<批次名>/plots/`）：`<task>.html`（自包含，
quad 四联 = 左列轨迹 / 右列视角约束球壳）+ `.png`（有光栅化后端时）+ `index.html` 总览页。
模式 `PLOT_SRC=centered`（默认）/ `origin`（原始采集包，做对照）/ `both`。
（quad 固定文案里的「原始采集轨迹」指**未加约束**，不是指未居中的包。）

要把居中后的位姿送进 `som_human/01_convert_capture.py`（它要求 `RAW_DIR` 里
`transforms.json` + `image/` + `pcd.ply` 三件齐全，**因此吃不了只有位姿和点云的居中包**）：
拿原始采集包复制一份，用居中包的 `transforms.json` / `pcd.ply` 覆盖同名文件即可（同名
drop-in）。训练出的 ply 与相机天然以主体为原点，99b 的 `target=[0,0,0]` 才对得上。

**关键 env**：`TARGET_MODE`（sight/anchor）、`PHI_PAD` / `THETA_PAD` / `RADIUS_PAD` /
`FULL_PHI_SPAN`（同 99g）、`RADIUS_MODE`（默认 `mid`：让 99b 硬编码的 `[0.9r,1.2r]`
恰好罩住实测半径区间）、`INIT_FOV_SCALE`（默认 `0.8`，同 99b）、`SORT_BY_TIME`、
`PHI_NORMALIZE`、`OUT_CENTERED` / `OUT_PLY`、`OUT_PLOTS` / `PLOT_SRC` / `PLOT_STYLE` / `PLOT_ROOT` / `PLOT_IDS` /
`PLOT_PYTHON`、`ONLY` / `FORCE` / `DRY_RUN`，以及仰角空洞保护的一组
`THETA_MODE` / `THETA_GAP_MIN` / `THETA_GAP_K` / `THETA_PICK` / `THETA_PAD_GAP` / `THETA_TIE_WARN` /
`THETA_MIN_FRAMES`（见「仰角空洞保护」）和观看约束裁剪的 `THETA_TOP` / `THETA_BOTTOM` /
`PHI_LEFT` / `PHI_RIGHT`（见「观看视角约束裁剪」）。
**旧名仍生效**（`VL_*` / `PITCH_*` / `EL_*`），用到时会在终端打一条一次性更名提示；
新名 ↔ 旧名 ↔ 99c ↔ json 字段的四向对照见文末「env 命名与旧名对照」。

### 为什么必须居中（2026-10-16 定论：不可省）

**疑点**：99k 的产物是 json，json 里只有角度与半径、没有平移项 —— 那还有必要居中吗？
（`az` / `el` / `r` 都是**相对球心**算的，整个场景平移时逐帧数值不变 —— 这一点没错。）

**结论：必须居中。** 数值确实平移不变，但**语义会错**：这份 json 的下游把 `az` / `el` / `r`
当成「绕**世界原点**」的球坐标来消费（下三处），而采集包的世界原点是 **AR 会话起点**、不是主体
（`99g` 实测原点离场景 0.5–5.1 m，10 包里只有 4 包落在取景内）。两者不重合时，json 描述的
环绕中心与下游实际使用的环绕中心就差一个平移量，而这个平移量在 json 里**看不见**。

下游的三处「看不见的假设」（都在 99b）：

| 位置 | 代码 | 含义 |
|---|---|---|
| `99b_pack_ply_to_mp4.py:435` | `"target": [0.0, 0.0, 0.0]` | UWA `view_limits` 的环绕中心**写死世界原点** |
| `99b_pack_ply_to_mp4.py:444-446` | `p0 = positions[0]` → `asin(p0[1]/‖p0‖)` / `atan2(p0[0], p0[2])` | 开场位姿由首帧相机**相对原点**反算；只有原点 = 主体中心时，它才等于 json 里的 `pitch_init` / `yaw_init` |
| `99b_pack_ply_to_mp4.py:432` | `minRadius/maxRadius = 0.9·radius / 1.2·radius` | 相机被直接摆在**原点球壳**上 |

**实测（10 包，`D:/dataset/手部微动样例/`）**：

| 指标 | 实测 |
|---|---|
| `‖target‖`（球心到世界原点） | **1.19 ~ 2.73 m** |
| `‖target‖ / r_mean` | **86.0% ~ 198.1%**（极值：佳佳蹲 0.860 / 男生二郎腿 1.981） |
| 按 99b 反算的开场位姿 vs 正解 | **Δpitch 最大 38.67°、Δyaw 最大 173.96°**（99b 的一致性告警阈值是 1°，必报） |

复算口径：`target` 取 `center_report.json` 的 `target`、`r_mean` 取 `measured.r_mean`；
「正解」= 首帧相机相对 `target` 的球坐标（已与 99k 写进 json 的 `init.pitch` / `init.yaw`
逐包比对，10/10 严格相等，确保口径同源）。

**为什么 `transforms.json` / `pcd.ply` 是控制模型原点的唯一入口**：
`som_human/01_convert_capture.py` 的位姿段（约 105–120 行）把采集包的 `transform_matrix`
按 `w2c = inv(M @ diag(1,-1,-1,1))` 原样转进 COLMAP —— **重建模型坐标系 = 采集包坐标系**，
链路上没有别的环节能改模型原点。所以居中只能在打包这一步做，且这两个文件不能删
（`01_convert_capture.py` 要求 `transforms.json` + `image/` + `pcd.ply` 三件齐全）。

`OUT_CENTERED=0` 只是**诊断逃生口**（保留原始世界系、把主体中心写进 `scene.center` 并告警），
**不是「可选的省事模式」** —— 默认必须居中。

⚠️ **一处待实测**：`01c` 的 COLMAP SfM 是否会重置坐标系，尚未在服务器验证 —— 退化路径
（固定 VIO 位姿跑 SfM）会保留原 gauge（居中生效），正路径不确定。*推理（待实测）*：若 SfM
自行确定 gauge，模型原点改由重建决定、不再取决于输入位姿的平移，届时居中的收益需重新评估；
但两种情况下居中都不会更差（不居中在「原 gauge 保留」时必定错）。

### 仰角「空洞」保护（`THETA_MODE=peak`）

**问题**：绕物体拍两圈（一圈俯视、一圈平视）时仰角 `el` 呈**双峰**，两峰之间是没拍到的
空档。原来的区间取 `min/max ± THETA_PAD` —— 只看两个极值点、对分布形状无感 —— 必然把这个
空档包进 `pitch_min/pitch_max`，用户转到中间视角就穿帮。

**做法**：`THETA_MODE=peak` 时对 `el` 做 **gap split**：按仰角排序后看相邻样本的间距，
在「大间隙」处切段，只保留一圈。

```
thr = max(THETA_GAP_MIN, THETA_GAP_K × median(相邻间距))   # 相对倍率自适应帧密度 + 绝对下限防误切
切段 → 丢掉帧数 < THETA_MIN_FRAMES 的小段 → THETA_PICK 选一圈 → 外侧 +THETA_PAD、洞侧 +THETA_PAD_GAP
```

**为什么不用「直方图投票 + 区间长度系数」**：投票要先定 bin 宽度，投完还要再拍一个长度系数，
而这个系数对「数据在哪」无感。实测两圈样例（平视 −5~8°、俯视 30~45°）：全局跨度 56°、
洞 22°，按跨度各端收 20% 得到的区间**仍然横跨空洞**，要收掉得单端 ~30% —— 收多少完全随
数据变，不是常数。切段法的区间边界直接由数据给出，不需要长度系数。

| env | 默认 | 含义 |
|---|---|---|
| `THETA_MODE` | `full` | `full` = 原行为（min/max ± pad）；`peak` = gap split 只保留一圈 |
| `THETA_GAP_MIN` | `2.5` | 切段绝对下限（度）：相邻仰角差 ≥ 它才算「洞」；单圈采集内部最大间隙实测 1.71°，2.5 是「不误切单圈」的安全下限 |
| `THETA_GAP_K` | `5.0` | 切段相对倍率：阈值还要 ≥ K × 同圈中位帧间距 |
| `THETA_PICK` | `biggest` | 保留哪一圈：`biggest`（帧数最多）/ `top`（仰角最高 = 俯视圈）/ `bottom`（仰角最低 = 平视圈）/ `init`（含首帧的圈） |
| `THETA_PAD_GAP` | `0.0` | 「洞侧」余量；不往洞里加 pad，否则等于把空洞又拉回来一截 |
| `THETA_TIE_WARN` | `0.8` | 次大段 / 主段 ≥ 此值就告警（两圈帧数接近时 `biggest` 会随帧数抖动） |
| `THETA_MIN_FRAMES` | `5` | 段的最小帧数：不足的段当离群 / 噪声丢掉 |

**默认下限 2.5° 的来历（2026-10-16）**：服务器双峰样例的两圈挨得很近 —— 洞只有 **3.29°**
（el 33.5° → 36.8°），远小于原默认 8.0°，于是判据被**绝对下限**卡死、两圈没切开（相对判据
那一项只有 0.41°，根本不是瓶颈）。实测 10 包真实单圈数据后把下限降到 2.5°：单圈内部最大间隙
仅 0.11~1.71°、中位帧间距 0.013~0.08°（所以相对判据永远只到 0.41°，实际拦门的恒为绝对下限），
2.5° 比实测最坏值（1.71°，仅 68 帧的稀疏采集）仍高 46%，10 包全部不被误切；再降到 1.5° 就有
1 包被误切成两段。
**排查「某包为什么没切开」看 `center_report.json` 的 `pitch_split`**：`threshold > gap_max`
就是门槛过高；再看 `median_gap` 决定调哪个 —— 相对项小（≪ 下限）就调 `THETA_GAP_MIN`，否则调
`THETA_GAP_K`。单圈数据跑 peak 是 no-op，图上自然看不出差别，以报告数字为准。

**安全性质（默认关，向后兼容）**：单圈数据不存在 ≥ thr 的间隙 → 只有一段 → 两侧都走
「外侧」分支 → 结果与 `full` **逐位相同**。实测 10 包真实数据（手部微动样例）peak 与 full
产物完全一致（`pitch_min/max`、`pitch_init`、`yaw`、`radius`、`init_fov` 全等）；只有合成
双峰包才收窄。附带收益：单帧离群（手抖拍到 −60°）会自成一段并被丢掉 —— 这是 min/max 做不到的。

**首帧（开场视角）**：若首帧落在被丢掉的那一圈里，`pitch_init` 会被**夹进保留区间**
（`center_report.json` 的 `pitch_split.init_clamped` 记 true），否则用户一开场就落在约束外。
想让开场视角留在首帧那一圈，设 `THETA_PICK=init`。

**诊断**：`center_report.json` 新增 `pitch_split`（段列表 / 选中段 / 阈值 / 最大间隙及其跨度 /
切几刀 / 丢了几段 / tie 告警 / clamp 详情），跑批时也直接打在终端上。

**只作用于 `el`**：两圈都绕整圈时 `az` 走 `FULL_PHI_SPAN` 本来就不设限，无需处理（而且 az 是
环形量，做间隙还得处理解缠跨 ±180，风险大收益小）。

**99g 必须同步**：99k 调 99g 出图时会把上面 7 个 env（`THETA_*`）与观看约束裁剪的 4 个
env（`THETA_TOP` 等）一起透传 —— 99g 是独立进程、自己从 `transforms.json` 重算视角区间，
不透传的话图会画出横跨空洞 / 未裁剪的旧区间、与 json 对不上。
`99g_plot_capture_trajectory.py` 里有**逐字同源**的 `median_of` / `split_el_segments` /
`pick_el_segment` / `el_span` 四份实现（用脚本可比对：取 `def median_of` 到
`def compute_view_limit` 之间的文本应完全相同），改一处必须同步另一处。

### 观看视角约束裁剪（`THETA_TOP` / `THETA_BOTTOM` / `PHI_LEFT` / `PHI_RIGHT`，默认全 0）

99k 生成的 `gs_camera_params.json` 描述的是**可观看范围**，所以除了「数据覆盖到哪」
（实测 min/max ± 余量），还需要一个「允许用户看到哪」的口子 —— 就是这 4 个裁剪参数，
与「余量」是两个正交的方向：

| | 余量（`PHI_PAD` / `THETA_PAD` / `RADIUS_PAD`） | 裁剪（`THETA_TOP` 等） |
|---|---|---|
| 方向 | 从实测 min/max **往外扩** | 在余量之后**再往里收** |
| 目的 | 贴着采集范围留一点游走空间，防边缘穿帮 | 挡住不想开放的视角（残缺 / 伪影一侧） |
| 负值 | ❌ 报错（余量只往外扩，要收窄用裁剪） | ❌ 报错（裁剪只往里收，要放宽用余量） |
| 收没了 | ——（余量不会反转区间） | 降级成一条缝 `[c, c]`：c 优先取初始视角（99b 的 1° 告警锚点），只打印告警不报错 |

**四个参数的语义、方向与 99c 的 `crop_axis` 完全一致**（99c 文档原文：
`Phi → [minPhi+LEFT, maxPhi−RIGHT]`、`Theta → [minTheta+BOTTOM, maxTheta−TOP]`）。
但注意：99c 的输入是 UWA `view_limits`（Theta / Phi 值域），99k/99g 内部存的是 el / az，
换算 `Theta = 90 − el`、`Phi = −az` 是**反向**的，所以参数名照抄 99c，落到 gs json 的效果是：

| env | 收的是 | 对 gs json 的效果 |
|---|---|---|
| `THETA_TOP` | Theta 上界（= 相机仰角最低、最接近平视的那一端） | `pitch_min` 增大 |
| `THETA_BOTTOM` | Theta 下界（= 相机仰角最高、最接近垂直俯视的那一端） | `pitch_max` 减小 |
| `PHI_LEFT` | Phi 下界 | `yaw_min` 增大 |
| `PHI_RIGHT` | Phi 上界 | `yaw_max` 减小 |

⚠️ 「`THETA_TOP` 抬的其实是 pitch 下限」容易反直觉（Theta 与 pitch 反向）。拿不准方向就
看产物：裁剪非 0 时，终端「最终区间」行会带 `THETA_TOP 3° / THETA_BOTTOM 5° …`，
99g 图注会多一行紫色「✂️ 已按观看约束裁剪（…）」—— 以它们为准。

**计算顺序**：实测 min/max ± 余量 →（仰角空洞保护，可选）→ **裁剪** → 最后才把首帧
`pitch_init` / `yaw_init` **夹进裁剪后的区间**。必须先裁后夹，否则 init 会被夹到裁剪前
的区间、又落到裁剪外。

**整圈不设限**：方位角跨度 ≥ `FULL_PHI_SPAN` 时按整圈处理，`PHI_LEFT` / `PHI_RIGHT`
被忽略（打印告警，不静默吞掉）。

**诊断**：`center_report.json` 的 `view_trim` 块逐维记录 `in`（裁剪前，Theta/Phi 值域）/
`cut` / `out`（裁剪后）/ `degenerate`。⚠️ 代码内部始终用 el / az 计算、**只裁被设了的那一侧**
（另一侧逐位不动），in/out 换算成 Theta/Phi 只为显示 —— 所以全 0 时与历史产物逐位一致，
个别裁一侧时另一侧也不会出现 1e-15° 的浮点漂移。

### env 命名与旧名对照

2026-10-16 起，99k / 99g 的视角相关 env 统一换成 **Phi / Theta 术语**（与 99c、与 UWA
`view_limits` 的字段同名，好对齐）。旧名（`VL_*` / `PITCH_*` / `EL_*`）仍生效，用到时
终端打一条一次性提示：`⚠️ 环境变量 VL_AZ_PAD 已更名为 PHI_PAD（旧名仍生效，建议改用新名）`。
**新名优先**：同名新旧两个都设了时只认新名。

| 新名（现在用） | 旧名（仍生效） | 99c 里的对应 | 影响的 gs json 字段 |
|---|---|---|---|
| `TARGET_MODE` | `VL_TARGET` | —（99c 不求球心） | `scene.center` 的来源 |
| `PHI_PAD` | `VL_AZ_PAD` | —（99c 不加余量） | `yaw_min` / `yaw_max` |
| `THETA_PAD` | `VL_EL_PAD` | — | `pitch_min` / `pitch_max` |
| `RADIUS_PAD` | `VL_R_PAD` | — | `scene.radius` |
| `FULL_PHI_SPAN` | `VL_FULL_AZ` | — | `yaw_min` / `yaw_max`（整圈时不设限） |
| `THETA_MODE` | `PITCH_MODE` | — | `pitch_min` / `pitch_max` |
| `THETA_GAP_MIN` | `EL_GAP_MIN` | — | 同上 |
| `THETA_GAP_K` | `EL_GAP_K` | — | 同上 |
| `THETA_MIN_FRAMES` | `EL_MIN_FRAMES` | — | 同上 |
| `THETA_PICK` | `EL_PICK` | — | 同上 |
| `THETA_PAD_GAP` | `EL_PAD_GAP` | — | 同上 |
| `THETA_TIE_WARN` | `EL_TIE_WARN` | — | 同上 |
| `THETA_TOP` / `THETA_BOTTOM` | —（新增） | `THETA_TOP` / `THETA_BOTTOM`（**同名同义**） | `pitch_min` / `pitch_max` |
| `PHI_LEFT` / `PHI_RIGHT` | —（新增） | `PHI_LEFT` / `PHI_RIGHT`（**同名同义**） | `yaw_min` / `yaw_max` |
| `LIM_TRIM`（仅 99g） | `VL_TRIM` | — | —（只影响图面） |

### 与 99g 的分工（为什么调脚本、不复制代码）

99k 的产物「居中包」= `transforms.json` + `pcd.ply`，是 99g 认得的输入格式（缺 `image/` 由
`PHOTO_ROOT` 兜住，见上），把它当 99g 的 `SRC_ROOT` 直接喂进去即可，零适配。所以 99k 用
`subprocess` 调同目录的 `99g_plot_capture_trajectory.py`，并把 `TARGET_MODE` / `PHI_PAD` /
`THETA_PAD` / `RADIUS_PAD` / `FULL_PHI_SPAN`、仰角空洞保护的 `THETA_MODE` / `THETA_GAP_*` /
`THETA_PICK` / `THETA_PAD_GAP` / `THETA_TIE_WARN` / `THETA_MIN_FRAMES`、观看约束裁剪的
`THETA_TOP` / `THETA_BOTTOM` / `PHI_LEFT` / `PHI_RIGHT`，以及 **`PHOTO_ROOT`（= 原始批次根）**
一起透传，保证图上的约束与 json 里的参数同口径、且左栏实拍首帧照常有图。

为什么不把 99g 的绘图代码搬进 99k（或 `import` 它）：99g 是**已定稿的汇报图面**
（13 种风格、1000+ 行绘图与图例口径），复制一份必然分叉、以后改图要改两处；
`import` 则会把 99k 钉死在 99g 的内部 API（`build_ctx` / `process_one`）上。而数值
口径本就一致，不需要把中间量喂给绘图端——

* 「居中包上 99g 重算的球心 = 原点」（平移等变性）：实测 |球心| ≈ 4.6e-16 m；
* 重算的 Phi/Theta 区间与 99k 写进 json 的**逐样本一致**：Δ ≈ 1e-14°（10/10 包）。

代价只有多一个进程：10 个包 quad 风格实测 **4.2 s**。绘图失败**不影响**参数产物，
只打印告警（缺 PNG 后端时 99g 会自己回退成 `.svg`）。

⚠️ 一处**不是错的口径差异**：99g 图上画的半径带是「实测 r ± `RADIUS_PAD`」（目标带），
而 gs json 里 99b 实际用的是 `[0.9×radius, 1.2×radius]`（`99b:432`）。看 R 环对不上，
先想到这一层。

⚠️ **历史坑（2026-10-15，相关代码已删）**：99k 第一版的 `link_images` 在「目标已是软链接」
时会穿透链接删掉**源目录里的真图**（实测把 01e954…佳佳抱枕 的 82 张 HEIC 送进回收站，
已逐张恢复）。该功能已于 2026-10-16 整体删除；教训仍然通用——**清场式删除前先判
`is_symlink()`，链接只摘链接、不按目录内容清理**（见 `.workbuddy/memory/MEMORY.md`）。

---

## E. 运镜轨迹生成与渲染（99l / 99m / 99n）

99l 读重建输出里的 COLMAP 相机（`refined_camera/`），取某台真实相机作起点，沿
orbit_left / orbit_right / orbit_up / dolly_out / hitchcock / grammy 六种电影感轨迹
生成新位姿（COLMAP 三件套，写到 `<场景根>/traj_camera/<效果名>/`）；99m 用 gsplat
离屏渲染成 MP4。轨迹文件可以用 99g 画图目检、用 COLMAP GUI 核对后再渲视频。
**轨迹位置/尺度来自训练相机本身，天然与场景对齐**——渲糊了先怀疑覆盖不足（见下）。

```bash
cd /mnt/c/code/media_code
ENV_PY=~/miniconda3/envs/infinisplat_v2/bin/python

# ① 生成轨迹（默认批量全部 6 种；EFFECT= 可只跑单条）
SCENE=/path/to/<场景根> N=120 $ENV_PY tools_99/99l_gen_colmap_traj.py

# 中心点三种：固定坐标（默认 0,0,0）/ 沿初始相机视线 / 俯仰看向 y 轴
SCENE=$SCENE CENTER=forward $ENV_PY tools_99/99l_gen_colmap_traj.py
SCENE=$SCENE CENTER=yaxis PITCH=10 EFFECT=orbit_left $ENV_PY tools_99/99l_gen_colmap_traj.py

# 初始相机选择：默认 01 组中间帧；INIT_INDEX 支持负数（倒数）
SCENE=$SCENE GROUP=02 INIT_INDEX=-1 $ENV_PY tools_99/99l_gen_colmap_traj.py

# ② 渲染（小样先验：MAX_FRAMES=16 + KEEP_PNG=1 目检首中尾帧）
PLY=/path/to/gaussians.ply TRAJ_DIR=$SCENE/traj_camera/orbit_left \
MAX_FRAMES=16 KEEP_PNG=1 $ENV_PY tools_99/99m_render_trajectory.py

# ③ 批量：一次渲多条轨迹（逗号分隔；PLY 只加载一次）。
#    输出固定为 OUT_DIR/<轨迹目录名>.mp4，批量时 OUT 被忽略
PLY=$PLY TRAJ_DIR=$SCENE/traj_camera/orbit_left,$SCENE/traj_camera/hitchcock \
OUT_DIR=$SCENE/traj_camera/videos $ENV_PY tools_99/99m_render_trajectory.py

# ④ 多场景批量（99n 编排）：遍历 BATCH_ROOT 自身 + 一层子目录，找出同时含
#    point_cloud/ 与 refined_camera/ 的场景，逐个 99l 生成六种轨迹、99m 渲染，
#    视频写 <场景>/traj_videos/<效果名>.mp4；PLY 取 point_cloud/iteration_* 中
#    最大号（无迭代目录则退回 point_cloud/point_cloud.ply）。
BATCH_ROOT=/path/to/batch_root $ENV_PY tools_99/99n_batch_traj_render.py

# 99n 常用 env：DRY_RUN=1 只列场景；MAX_SCENES=1 先试跑一个；NO_GEN=1 轨迹已有
# 只渲染；SKIP_EXISTING=1 视频齐了的场景跳过；STOP_ON_ERROR=1 遇错即停
# （默认记录后跳过、最后汇总退出码 1）。99l/99m 侧参数（N/CENTER/FPS/SSAA/
# SH_LAYOUT/...）在环境里设了就自动透传给子进程。
```

**轨迹格式约定**（99l 写、99m 读，改格式必须两边同步）：

- 帧序 = `IMAGE_ID` 升序；`images.txt` 每条记录**占两行**（位姿行 + POINTS2D 行，后者可为空
  但行必须存在）——漏第二行是 COLMAP 解析错位的第一大坑；99m 按「位姿行字段数 ≡ 1 (mod 3)、
  POINTS2D 行 ≡ 0 (mod 3)」双重判据防错位吞行。
- `qvec = (qw,qx,qy,qz)`，`tvec` 为世界系→相机系（`x_c = R·x_w + t`，`qvec2rotmat` 与
  `99b_pack_ply_to_mp4.py:313` 同式）；相机中心 = `-Rᵀt`。
- 只写/只认 PINHOLE / SIMPLE_PINHOLE 相机（OPENCV 带畸变的直接报错，防止静默按针孔渲偏）。
- 99l 的位姿约定 OpenCV 相机系（z 前、y 下）：`look_at_to_w2c` 的第二个基向量是
  **`-up`（down）**，用 `+up` 画面会上下颠倒、且仍能通过「相机中心回算」检查（2026-10-31
  实测踩过，已写进该函数 docstring）。

**PLY 加载约定**（99m）：标准 3DGS activation（scale=exp、opacity=sigmoid、color 走 SH，
gsplat 对 SH 输入内部做 sigmoid）；`rot` 默认官方 3DGS 约定存 `(w,x,y,z)`，gsplat 要 xyzw，
加载时 roll（`ROT_ORDER=xyzw` 可切）；`scale`/`rot` 兼容向量属性与 `scale_0..2` /
`rot_0..3` 逐分量两种写法。若渲染出来模型像被「拧散」，先怀疑 rot 顺序。

**f_rest 有两种磁盘排布**（2026-10-31 实测踩坑）：系数主序（官方 3DGS：每基函数的
RGB 连续 3 列）与通道主序（部分 fork 导出少做一次 transpose：R/G/B 各一块）。排布
解析错的典型症状是 **DC 颜色大体正常、一动相机就整屏红绿蓝彩虹光影**（SH 系数在
颜色通道间交叉污染）。**`SH_LAYOUT` 默认 `channel`**——本仓实际资产（HMSD 等
fork）实测全是通道主序；官方 3DGS 导出的 PLY（如 flashworld / dtu_smoke）才需要
`SH_LAYOUT=coef`；`auto` 按「每 15 列一块、各块尾/头能量比」双统计量（中位/均值
取或）自动区分，但真实训练 SH 是重尾分布且有未收敛异常点，**auto 在真实数据上
不可靠**（2026-10-31 两次误判：单均值失灵 → 改双统计量仍误判），仅作参考。
（诊断脚本 99n_diag_ply.py 已于 2026-10-31 删除，其判据保留在 99m 的
`detect_sh_layout()` 里。）

**性能参考**（2026-10-31 实测，RTX 3090，WSL，infinisplat_v2 env）：flashworld
`gaussians.ply`（**405 万**高斯、degree-0 颜色）120 帧 1920x1080@30fps 渲染 **4.6s**
（≈26 it/s 全分辨率）——性能完全不是瓶颈，瓶颈只在运镜设计本身。

**出片质量上限**：取决于训练数据的相机覆盖。渲染相机离训练位置越远（大幅推近/仰角变化），
画面越容易糊/出伪影——某段突然变糊先怀疑覆盖不足，不是渲染 bug。

**99l / 99m 的路径与参数预设**：与 99a~99f 同风格——`main()` 里有一组预设值
（SCENE 路径、EFFECTS、N、CENTER 等，注释「改场景/批次时只改这几行」），env 只作覆盖；
不设 env 直接 `python tools_99/99l_gen_colmap_traj.py` 即可跑预设批量。

**99l 其余 env**：`SCENE`（场景根或 refined_camera 目录本身）、`GROUP`（01|02，
图像名前两字符分组，名字必须匹配 `^(01|02).+\.(jpe?g)$`）、`INIT_INDEX`（初始相机
序号，默认组内中间帧 `len//2`，负数=倒数，越界报错）、`CENTER`（`"x,y,z"` 默认 0,0,0
| `forward`/`view`：沿初始相机原视线前移 ||C0|| | `yaxis`/`y_axis`/`y轴`：
`center=(0, C0y+h·tan(pitch), 0)`，`h=hypot(C0x,C0z)`）、`PITCH`（yaxis 俯仰角，度）、
`N` / `ANGLE`（orbit_* 角度，度；orbit_up 越限自动截断并 WARN）/ `PARAM`
（dolly_out/hitchcock 默认 1.5、grammy 默认 2.0；m≤0 报错、m<1 或 ≈1 WARN）、
`EFFECT`（单条快捷方式）/ `EFFECTS`（批量列表）、`OUT`（单条=效果目录，多条=父目录；
缺省 `<场景根>/traj_camera/`）、`PREFIX`（图像名前缀，默认 03→`03000001.jpg`）、
`NO_POINTS3D`（=1 不拷贝 points3D.txt/.ply）。输出浮点一律 `%.12g`；
hitchcock 每帧一台相机、`fscale` 只缩放 fx/fy（cx/cy 不变），dolly_out fscale 恒 1。
姿态一律重新 look-at 中心点（不沿用原拍摄朝向）；首帧位置 = 初始相机位置。

**99m 其余 env**：`TRAJ_DIR`（支持**逗号分隔多条轨迹**批量渲染，PLY 只加载一次；
输出固定为 `OUT_DIR/<轨迹目录名>.mp4`，批量时 `OUT` 被忽略；单条时 `OUT` 可自定义
文件名）/ `OUT_DIR` / `OUT` / `FPS` / `SSAA`（1=关，2=渲染 2x 再降采样，
抗 splats 缩放边缘闪烁）/ `EPS2D`（gsplat 抗锯齿，默认 0.3）/ `NEAR` / `FAR` / `BG`、
`SH_DEGREE`（默认按 PLY 系数个数自动）、`ROT_ORDER` / `SH_LAYOUT`（channel 默认，见上节排布坑）/ `MAX_FRAMES`（小样调试）/ `VIEW_CHUNK`
（一次并行渲染的视角数，默认 8）/ `KEEP_PNG` / `W` / `H`（0=跟随 cameras.txt）/ `DRY_RUN`。

> 本机无 ffmpeg，99m 用 `cv2.VideoWriter(mp4v)` 直接写 MP4（体积偏大、播放器兼容一般）；
> 要更小的文件可事后用 ffmpeg 转 H.264：`ffmpeg -i render.mp4 -c:v libx264 -crf 18 out.mp4`。

### 服务器运行（PLY 资产在服务器侧时）

代码**零改动**——所有路径都是 env 传入、相对 cwd 解析，与 99a~99c 同一约定。
唯一前置：服务器建一个独立的 gsplat 渲染 env（别污染现有训练 env）：

```bash
source ~/miniconda3/etc/profile.d/conda.sh
conda create -y -n gs_traj_render python=3.10
conda activate gs_traj_render
# 公司代理 TLS 拦截：照 proxy.env 约定 export PIP_CERT / SSL_CERT_FILE 后再装
pip install torch --index-url https://download.pytorch.org/whl/cu124
pip install gsplat plyfile opencv-python-headless numpy

# 验证（须为 True；False = 驱动/CUDA 不匹配）
python -c "import gsplat, torch; print(torch.cuda.is_available())"
```

- 99l / 99n 服务器上用新 env 跑即可（99l 需 numpy；99n 纯标准库只编排子进程）。
- 99m 用新 env 跑，命令与 WSL 完全一致：`PLY=... TRAJ_DIR=... python tools_99/99m_render_trajectory.py`。
- ⚠️ `pip install gsplat` 优先走预编译 wheel；若服务器 CUDA/驱动组合没有对应 wheel 会退回
  源码编译（需要 nvcc，耗时长）。【待核实】服务器驱动版本能否直接吃 torch 2.6+cu124 的 wheel
  （需驱动 ≥ 525.60），装前先在服务器跑 `nvidia-smi` 确认。

---

## 首次准备（族 A 需要）

```bash
conda activate xcodec      # 或任意带工具链依赖（pymp4 等）的 env
conda install -y -c conda-forge ffmpeg
chmod +x /path/to/UWA_Sample_Tool_v3/build/gltf_packer
```

`TOOL_DIR` 默认 `../../model/UWA_Sample_Tool_v3`；`muxer.py` 依赖 pymp4，脚本自动把
`thirdparty/pymp4-1.4.0/src` 加进 `PYTHONPATH`，也可用 `PYTHON_BIN` 指定解释器。

## 文档索引

- **99c 的 thetaBuffer/phiBuffer 字段落地问题、99e 解封装排错**：见
  [../vggt_human/NOTES.md](../vggt_human/NOTES.md) 第 9、10 条。
- **99g 的坐标系坑、四联对照图约定、球心取哪个点的实测数据**：见
  [../vggt_human/NOTES.md](../vggt_human/NOTES.md)「Remy 采集包」等节。
  **99k 的球心口径与余量常量直接沿用 99g**，两边必须一致；99k 出图也是调 99g，
  所以改 99g 的 `compute_view_limit` / 余量常量时，两处（图与 json）会同时生效。
- **与服务器侧「3DGS 相机位姿调整工具」的四模块对照**（同族能力互证）：
  见 [../vggt_human/AI开发经验分享.md](../vggt_human/AI开发经验分享.md) 第 4.4 节。

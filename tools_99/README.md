# tools_99 — 重建产物的封装收尾、采集包预处理、数据集体检与汇报图面工具集

2026-10-15 从 `vggt_human/` 迁出（`git mv`，保留提交历史）。本目录原属于
vggt_human 的下游「99 系列」，但它们的服务对象已超出人脸重建本身：既有对
UWA 三件套的封装收尾，也有全仓通用的语法自查工具，还有 3DGS 相机位姿工具的
汇报图面。脚本文件名一律未改（保留 99 前缀的连续性）。

## 四个族

| 族 | 脚本 | 用途 | 运行依赖 |
|---|---|---|---|
| **A. 打包/收尾链** | 99a 99b 99c 99d 99e 99f | 重建产物 ↔ 手机可播放 MP4 的双向封装 | UWA 工具链 + ffmpeg + conda env |
| **B. 数据集体检** | 99g | Remy 采集包轨迹图 / 视角约束可视化 | **纯标准库**，Python ≥ 3.8 |
| **C. 通用工具 + 汇报图面** | 99h 99i 99j | 语法兼容自查 / SVG→PNG / 汇报页内联同步 | 99h 需 Python ≥ 3.12；99i 需 node + Edge |
| **D. 采集包预处理** | 99k | 主体居中 + 视角区间 → `gs_camera_params_final.json`，并调 99g 出图 | **纯标准库**，Python ≥ 3.10 |

族 A 内部是**运行时数据依赖**（99c/99e 复用 99b 留下的 `mp4_work/`），没有代码级
import；族 C 的三个脚本共同服务本目录的 `camera_pose_tool_report.html` +
`report_figs/`。族 B（可视化）与族 D（出参数）都吃 Remy 采集包，视角区间的算法与
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

**球心取哪个点**由 `VL_TARGET` 决定，默认 `sight`：

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
VL_TARGET=anchor python tools_99/99g_plot_capture_trajectory.py
VL_MARK_ALT=0    python tools_99/99g_plot_capture_trajectory.py

# ⑤ 关掉右列的贴身裁剪（回到实测 min/max ± 余量的完整区间）
VL_TRIM=0 python tools_99/99g_plot_capture_trajectory.py
```

画法：`lim_band`（扇环填充）/ `lim_edges`（四条边界虚线，对轨迹影响最小）/ `lim_angle`
（边界 + 角度/半径数值标注）/ `lim_rings`（米制同心环 + 方位刻度）/ `lim_hull`
（对照组：相机水平位置凸包外扩，不假设环绕）/ `lim_shell`（等轴测 3D 球壳扇块）。
余量可用 `VL_AZ_PAD` / `VL_EL_PAD` / `VL_R_PAD` 覆盖，整圈判定阈值用 `VL_FULL_AZ`。

**四联图右列的贴身裁剪**（`VL_TRIM`，默认 `0.2`；只有 `STYLE=quad` 的右列吃这个参数）：
右列画出来/标出来的 az/el 区间在「实测 ± 余量」之上两端各再收 20% 跨度，但两端都不越过
frame 0 的实测角（`min(lo+span·k, init)` / `max(hi−span·k, init)`），免得用户一开场就落在
约束外。**只作用于右列约束层、不改取景** —— 四块共用取景，取景若跟着变窄会把左下 ③
等轴测一起缩掉，左列就不是原图了。整圈（`VL_FULL_AZ`）时 az 不裁。`VL_TRIM=0` 关闭。
副作用：el 跨度收到 8° 以下时球壳不填色（既有阈值），④ 会退化成细网格带。

图上会同时画出**球心**（实心圆 + 十字）与**对照球心**（另一个候选点，空心叉 + 虚线 + Δ 距离），
标题行给出两者的拟合质量对照（垂距 RMS / 视线夹角 / 正规矩阵条件数）。实测 10 个样本
（见 NOTES.md「球心取哪个点」）：`sight` 垂距 RMS 0.081 m、夹角 3.46°，`anchor_point`
0.205 m、9.54° —— 10/10 个样本都是 `sight` 更贴合视线。

---

## C. 通用工具与汇报图面

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

### 99i / 99j — 汇报页图面

配套本目录的 `camera_pose_tool_report.html`（部门技术评审口径的汇报页）与 `report_figs/`
（4 组同名 `.svg` + 3× `.png`，3540 px 宽）。

```bash
# 99i：把 report_figs/*.svg 渲染成 3x PNG（供汇报页 / PPT 用）
node tools_99/99i_render_report_figs.js                          # 全部 4 张
node tools_99/99i_render_report_figs.js fig2_solution_architecture 1180 620   # 只渲一张
# 输出：report_figs/<name>.png；包裹页临时生成在系统 temp 下，渲染完自动清理

# 99j：把 report_figs/*.svg 的最新内容同步进汇报页的内联 figure（就地改 HTML）
python tools_99/99j_sync_report_figs.py
```

⚠️ **99i 的坑点**：Edge 154 起 `--headless --screenshot=xxx.png` 静默失效 —— 进程 exit 0、
不报错、也不产出文件。必须走 CDP（启动 headless 实例 → WebSocket → `Page.captureScreenshot`）。

⚠️ **改图面前先读** `camera_pose_tool_report.html` 的 `:root` 与「图内样式」两个 CSS 块 ——
图面配色/线宽口径已于 2026-09-28 定版（靛蓝 = 数据 · 青 = 核心 · 琥珀 = 附加说明/反馈），
按既定口径取色，不要自创色值。

---

## D. 采集包主体居中 → 相机参数（99k）

```bash
python tools_99/99k_make_gs_params.py                    # 跑 main() 里预设的批次根
SRC=D:/dataset/手部微动样例/01e954…-佳佳抱枕 python tools_99/99k_make_gs_params.py
DRY_RUN=1 python tools_99/99k_make_gs_params.py          # 只算不落盘
ONLY=<task_a>,<task_b> python tools_99/99k_make_gs_params.py
OUT_PLOTS=0 python tools_99/99k_make_gs_params.py        # 只要参数、不出图
PLOT_STYLE=lim_edges PLOT_SRC=both python ...            # 换图面风格 / 顺便画原始包做对照
```

**为什么需要**：Remy 采集包的世界原点是手机开机点附近，主体（视线汇聚点）实测离它
1.3–2.9 m；而 UWA 侧 `view_limits` 的 `target` 被 99b 写死成 `[0,0,0]`
（`99b_pack_ply_to_mp4.py:435`）。不居中 → 手机端环绕中心整个偏离主体。

**四步**：① 求球心（默认 `sight` = 最小二乘视线汇聚中心，与 99g 同源；`VL_TARGET=anchor`
切到采集包自带 `anchor_point`）→ ② 以球心为原点逐帧算 az / el / r 并取 min/max + 余量
（解缠 / 整圈 / 余量常量全部沿用 99g）→ ③ 落盘 → ④ **调 99g 出图**（默认画居中后的包）。

**产物**（`<OUT_ROOT>/<task>/`，默认 `<RESULTS_ROOT>/<批次名>/centered`）：

| 文件 | 用途 |
|---|---|
| `gs_camera_params_final.json` | 99b 的 COLMAP 模式直接消费（`setting.pitch_min/max`、`yaw_min/max`、`pitch_init` + `scene.radius`） |
| `transforms.json` | 居中后位姿（同名 drop-in），世界原点 = 主体中心 |
| `pcd.ply` | 居中后点云（按属性偏移平移 xyz，其它属性原样） |
| `image/` | 源图硬链接（跨卷退软链接，都失败给手工命令） |
| `center_report.json` | 球心对照 / 拟合质量 / 区间明细 / 自检（诊断用） |

**图面**（`<PLOT_ROOT>/<模式>/`，默认 `<批次名>/plots/`）：`<task>.html`（自包含，
quad 四联 = 左列轨迹 / 右列视角约束球壳）+ `.png`（有光栅化后端时）+ `index.html` 总览页。
模式 `PLOT_SRC=centered`（默认）/ `origin`（原始采集包，做对照）/ `both`。
（quad 固定文案里的「原始采集轨迹」指**未加约束**，不是指未居中的包。）

居中包可直接喂 `som_human/01_convert_capture.py --raw_dir <该目录>`，训练出的 ply
与相机天然以主体为原点，99b 的 `target=[0,0,0]` 才对得上。

**关键 env**：`VL_TARGET`（sight/anchor）、`VL_AZ_PAD` / `VL_EL_PAD` / `VL_R_PAD` /
`VL_FULL_AZ`（同 99g）、`RADIUS_MODE`（默认 `mid`：让 99b 硬编码的 `[0.9r,1.2r]`
恰好罩住实测半径区间）、`SORT_BY_TIME`、`PHI_NORMALIZE`、`OUT_CENTERED` / `OUT_PLY` /
`OUT_IMAGES`、`OUT_PLOTS` / `PLOT_SRC` / `PLOT_STYLE` / `PLOT_ROOT` / `PLOT_IDS` /
`PLOT_PYTHON`、`ONLY` / `FORCE` / `DRY_RUN`。

### 与 99g 的分工（为什么调脚本、不复制代码）

99k 的产物「居中包」= `transforms.json` + `pcd.ply` + `image/`，**就是 99g 的原生输入
格式**，把它当 99g 的 `SRC_ROOT` 直接喂进去即可，零适配。所以 99k 用
`subprocess` 调同目录的 `99g_plot_capture_trajectory.py`，并把 `VL_TARGET` / `VL_*_PAD` /
`VL_FULL_AZ` 一起透传，保证图上的约束与 json 里的参数同口径。

为什么不把 99g 的绘图代码搬进 99k（或 `import` 它）：99g 是**已定稿的汇报图面**
（13 种风格、1000+ 行绘图与图例口径），复制一份必然分叉、以后改图要改两处；
`import` 则会把 99k 钉死在 99g 的内部 API（`build_ctx` / `process_one`）上。而数值
口径本就一致，不需要把中间量喂给绘图端——

* 「居中包上 99g 重算的球心 = 原点」（平移等变性）：实测 |球心| ≈ 4.6e-16 m；
* 重算的 Phi/Theta 区间与 99k 写进 json 的**逐样本一致**：Δ ≈ 1e-14°（10/10 包）。

代价只有多一个进程：10 个包 quad 风格实测 **4.2 s**。绘图失败**不影响**参数产物，
只打印告警（缺 PNG 后端时 99g 会自己回退成 `.svg`）。

⚠️ 一处**不是错的口径差异**：99g 图上画的半径带是「实测 r ± `VL_R_PAD`」（目标带），
而 gs json 里 99b 实际用的是 `[0.9×radius, 1.2×radius]`（`99b:432`）。看 R 环对不上，
先想到这一层。

⚠️ **踩过的坑（2026-10-15）**：图链逻辑第一版在「目标已是软链接」时会穿透链接删掉
**源目录里的真图**（实测把 01e954…佳佳抱枕 的 82 张 HEIC 送进回收站，已逐张恢复）。
现行规则：目标若是链接 → 只摘链接本身；若是目录 → 非空跳过、空则 rmdir；
硬链接阶段只回滚自己创建过的文件。**改这段代码前先读 `link_images` 的 docstring。**

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

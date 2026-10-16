# tools_99 — 重建产物的封装收尾、采集包预处理与数据集体检工具集

2026-10-15 从 `vggt_human/` 迁出（`git mv`，保留提交历史）。本目录原属于
vggt_human 的下游「99 系列」，但它们的服务对象已超出人脸重建本身：既有对
UWA 三件套的封装收尾，也有全仓通用的语法自查工具，还有 Remy 采集包的居中预处理。
脚本文件名一律未改（保留 99 前缀的连续性）。

## 四个族

| 族 | 脚本 | 用途 | 运行依赖 |
|---|---|---|---|
| **A. 打包/收尾链** | 99a 99b 99c 99d 99e 99f | 重建产物 ↔ 手机可播放 MP4 的双向封装 | UWA 工具链 + ffmpeg + conda env |
| **B. 数据集体检** | 99g | Remy 采集包轨迹图 / 视角约束可视化 | **纯标准库**，Python ≥ 3.8 |
| **C. 通用工具** | 99h | 全仓 Python 语法兼容自查 | **需 Python ≥ 3.12** |
| **D. 采集包预处理** | 99k | 主体居中 + 视角区间 → `gs_camera_params_final.json`，并调 99g 出图 | **纯标准库**，Python ≥ 3.10 |

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

**仰角空洞保护**（`PITCH_MODE`，默认 `full`）：双峰数据（俯视 + 平视两圈）下仰角区间会
横跨没拍到的空档，设 `PITCH_MODE=peak` 让本脚本与 99k 一样按 gap split 只保留一圈。
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
PITCH_MODE=peak python tools_99/99k_make_gs_params.py    # 双峰数据（俯视+平视两圈）只保留一圈
PITCH_MODE=peak EL_PICK=init python ...                  # 保留首帧所在的那一圈（开场视角不变）
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
| `gs_camera_params_final.json` | 99b 的 COLMAP 模式直接消费：`setting.pitch_min/pitch_max`、`yaw_min/yaw_max`、`pitch_init` + `scene.radius`（其余字段见下） |
| `transforms.json` | 居中后位姿（同名 drop-in），世界原点 = 主体中心 |
| `pcd.ply` | 居中后点云（按属性偏移平移 xyz，其它属性原样） |
| `center_report.json` | 球心对照 / 拟合质量 / 区间明细 / 自检（诊断用） |

**源图一律不复制、不挂链**（2026-10-16 起删掉 `OUT_IMAGES` 与 `link_images`）：居中包只含
位姿与点云，需要图就直接用原始采集包（`01_convert_capture.sh` 也照样吃原包）。
**这不影响出图**：99k 出图时透传 `PHOTO_ROOT=<原批次根>`，99g 在本包找不到 `image/` 时按
同名 ID 回源包读实拍首帧，quad 左栏的贴图照旧（详见 99g 一节的 `PHOTO_ROOT`）。

`gs_camera_params_final.json` / `center_report.json` 落盘为 **4 空格缩进、LF 换行、无 BOM**
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
| `scene.center` | 居中后恒 `[0,0,0]`；`OUT_CENTERED=0` 时写主体中心在原始世界系的位置并告警（与 99b 写死的 `target=[0,0,0]` 对不上） |

**图面**（`<PLOT_ROOT>/<模式>/`，默认 `<批次名>/plots/`）：`<task>.html`（自包含，
quad 四联 = 左列轨迹 / 右列视角约束球壳）+ `.png`（有光栅化后端时）+ `index.html` 总览页。
模式 `PLOT_SRC=centered`（默认）/ `origin`（原始采集包，做对照）/ `both`。
（quad 固定文案里的「原始采集轨迹」指**未加约束**，不是指未居中的包。）

要把居中后的位姿送进 `som_human/01_convert_capture.py`（它要求 `RAW_DIR` 里
`transforms.json` + `image/` + `pcd.ply` 三件齐全，**因此吃不了只有位姿和点云的居中包**）：
拿原始采集包复制一份，用居中包的 `transforms.json` / `pcd.ply` 覆盖同名文件即可（同名
drop-in）。训练出的 ply 与相机天然以主体为原点，99b 的 `target=[0,0,0]` 才对得上。

**关键 env**：`VL_TARGET`（sight/anchor）、`VL_AZ_PAD` / `VL_EL_PAD` / `VL_R_PAD` /
`VL_FULL_AZ`（同 99g）、`RADIUS_MODE`（默认 `mid`：让 99b 硬编码的 `[0.9r,1.2r]`
恰好罩住实测半径区间）、`INIT_FOV_SCALE`（默认 `0.8`，同 99b）、`SORT_BY_TIME`、
`PHI_NORMALIZE`、`OUT_CENTERED` / `OUT_PLY`、`OUT_PLOTS` / `PLOT_SRC` / `PLOT_STYLE` / `PLOT_ROOT` / `PLOT_IDS` /
`PLOT_PYTHON`、`ONLY` / `FORCE` / `DRY_RUN`，以及仰角空洞保护的一组
`PITCH_MODE` / `EL_GAP_MIN` / `EL_GAP_K` / `EL_PICK` / `EL_PAD_GAP` / `EL_TIE_WARN` /
`EL_MIN_FRAMES`（见下节）。

### 仰角「空洞」保护（`PITCH_MODE=peak`）

**问题**：绕物体拍两圈（一圈俯视、一圈平视）时仰角 `el` 呈**双峰**，两峰之间是没拍到的
空档。原来的区间取 `min/max ± VL_EL_PAD` —— 只看两个极值点、对分布形状无感 —— 必然把这个
空档包进 `pitch_min/pitch_max`，用户转到中间视角就穿帮。

**做法**：`PITCH_MODE=peak` 时对 `el` 做 **gap split**：按仰角排序后看相邻样本的间距，
在「大间隙」处切段，只保留一圈。

```
thr = max(EL_GAP_MIN, EL_GAP_K × median(相邻间距))   # 相对倍率自适应帧密度 + 绝对下限防误切
切段 → 丢掉帧数 < EL_MIN_FRAMES 的小段 → EL_PICK 选一圈 → 外侧 +VL_EL_PAD、洞侧 +EL_PAD_GAP
```

**为什么不用「直方图投票 + 区间长度系数」**：投票要先定 bin 宽度，投完还要再拍一个长度系数，
而这个系数对「数据在哪」无感。实测两圈样例（平视 −5~8°、俯视 30~45°）：全局跨度 56°、
洞 22°，按跨度各端收 20% 得到的区间**仍然横跨空洞**，要收掉得单端 ~30% —— 收多少完全随
数据变，不是常数。切段法的区间边界直接由数据给出，不需要长度系数。

| env | 默认 | 含义 |
|---|---|---|
| `PITCH_MODE` | `full` | `full` = 原行为（min/max ± pad）；`peak` = gap split 只保留一圈 |
| `EL_GAP_MIN` | `8.0` | 切段绝对下限（度）：相邻仰角差 ≥ 它才算「洞」 |
| `EL_GAP_K` | `5.0` | 切段相对倍率：阈值还要 ≥ K × 同圈中位帧间距 |
| `EL_PICK` | `biggest` | 保留哪一圈：`biggest`（帧数最多）/ `top`（仰角最高 = 俯视圈）/ `bottom`（仰角最低 = 平视圈）/ `init`（含首帧的圈） |
| `EL_PAD_GAP` | `0.0` | 「洞侧」余量；不往洞里加 pad，否则等于把空洞又拉回来一截 |
| `EL_TIE_WARN` | `0.8` | 次大段 / 主段 ≥ 此值就告警（两圈帧数接近时 `biggest` 会随帧数抖动） |
| `EL_MIN_FRAMES` | `5` | 段的最小帧数：不足的段当离群 / 噪声丢掉 |

**安全性质（默认关，向后兼容）**：单圈数据不存在 ≥ thr 的间隙 → 只有一段 → 两侧都走
「外侧」分支 → 结果与 `full` **逐位相同**。实测 10 包真实数据（手部微动样例）peak 与 full
产物完全一致（`pitch_min/max`、`pitch_init`、`yaw`、`radius`、`init_fov` 全等）；只有合成
双峰包才收窄。附带收益：单帧离群（手抖拍到 −60°）会自成一段并被丢掉 —— 这是 min/max 做不到的。

**首帧（开场视角）**：若首帧落在被丢掉的那一圈里，`pitch_init` 会被**夹进保留区间**
（`center_report.json` 的 `pitch_split.init_clamped` 记 true），否则用户一开场就落在约束外。
想让开场视角留在首帧那一圈，设 `EL_PICK=init`。

**诊断**：`center_report.json` 新增 `pitch_split`（段列表 / 选中段 / 阈值 / 最大间隙及其跨度 /
切几刀 / 丢了几段 / tie 告警 / clamp 详情），跑批时也直接打在终端上。

**只作用于 `el`**：两圈都绕整圈时 `az` 走 `VL_FULL_AZ` 本来就不设限，无需处理（而且 az 是
环形量，做间隙还得处理解缠跨 ±180，风险大收益小）。

**99g 必须同步**：99k 调 99g 出图时会把上面 7 个 env 一起透传 —— 99g 是独立进程、自己从
`transforms.json` 重算视角区间，不透传的话图会画出横跨空洞的旧区间、与 json 对不上。
`99g_plot_capture_trajectory.py` 里有**逐字同源**的 `median_of` / `split_el_segments` /
`pick_el_segment` / `el_span` 四份实现（用脚本可比对：取 `def median_of` 到
`def compute_view_limit` 之间的文本应完全相同），改一处必须同步另一处。

### 与 99g 的分工（为什么调脚本、不复制代码）

99k 的产物「居中包」= `transforms.json` + `pcd.ply`，是 99g 认得的输入格式（缺 `image/` 由
`PHOTO_ROOT` 兜住，见上），把它当 99g 的 `SRC_ROOT` 直接喂进去即可，零适配。所以 99k 用
`subprocess` 调同目录的 `99g_plot_capture_trajectory.py`，并把 `VL_TARGET` / `VL_*_PAD` /
`VL_FULL_AZ`、仰角空洞保护的 `PITCH_MODE` / `EL_*`，以及 **`PHOTO_ROOT`（= 原始批次根）**
一起透传，保证图上的约束与 json 里的参数同口径、且左栏实拍首帧照常有图。

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

⚠️ **历史坑（2026-10-15，相关代码已删）**：99k 第一版的 `link_images` 在「目标已是软链接」
时会穿透链接删掉**源目录里的真图**（实测把 01e954…佳佳抱枕 的 82 张 HEIC 送进回收站，
已逐张恢复）。该功能已于 2026-10-16 整体删除；教训仍然通用——**清场式删除前先判
`is_symlink()`，链接只摘链接、不按目录内容清理**（见 `.workbuddy/memory/MEMORY.md`）。

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

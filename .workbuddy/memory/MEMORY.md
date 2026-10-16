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
  （球心默认 `sight` 最小二乘视线汇聚，`VL_TARGET=anchor` 可切）+ 视角区间 →
  `gs_camera_params_final.json`（99b COLMAP 模式消费 `setting.pitch_min/max`、`yaw_min/max`、
  `pitch_init`、`scene.radius`）+ 居中采集包（`transforms.json`/`pcd.ply`/`image/`）。
  区间口径与余量常量与 99g 同源（同名 env：`VL_AZ_PAD/EL_PAD/R_PAD/FULL_AZ`），**改一处要同步另一处**；
  `RADIUS_MODE` 默认 `mid`（令 99b 硬编码的 `[0.9r,1.2r]` 罩住实测半径带）。
  **仰角空洞保护**（2026-10-15，提交 `499a7ee`）：双峰数据（俯视+平视两圈，中间没拍到）
  原先 `min/max ± pad` 必然横跨空洞 → `PITCH_MODE=peak` 按 gap split 只保留一圈。
  这一组 env（`PITCH_MODE/EL_GAP_MIN/EL_GAP_K/EL_MIN_FRAMES/EL_PICK/EL_PAD_GAP/EL_TIE_WARN`）
  在 99k 与 99g **同名同义**，且 99k/99g 各有**逐字同源**的四份函数
  （`median_of/split_el_segments/pick_el_segment/el_span`）—— **改一处必须同步另一处**，
  比对法：取 `def median_of` 到 `def compute_view_limit` 之间文本应完全相同。
  单圈数据下 peak 与 full 结果逐位相同（安全开关，默认 full）；pad 只加数据外侧、
  洞侧用 `EL_PAD_GAP`；首帧落在被丢圈里时 `pitch_init` 夹进保留区间（记 `init_clamped`）。
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

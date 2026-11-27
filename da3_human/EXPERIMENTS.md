# da3_human — 实验记录

> 按日期倒序追加。实验计划/决策依据见 `EXPERIMENT_PLAN.md`，机制解释见 `NOTES.md`（阶段E补）。
> 三视角实验的规划与评审见 `PLAN_REVIEW_3VIEW.md`（D1~D5 主表）。

## 2026-11-27 海思视频对比补充：逐例输入总览页

- 用户认可上一版PPT，要求在每个样例六方法对比之前增加一页该样例所有输入图像。保持原19例与全部108段真实高斯视频，改为**38页、每例两页**：奇数页输入总览，偶数页方法对比，不添加封面或目录。
- 输入严格来自 `D:/dataset/测试数据/海思端侧重建拍摄数据/CASE/images/*.png`，每例10张、共190张、704×704；stage1逐张匹配相机JSON的 `sourceIndex`，case18/19按 `frame_0001..0010`。不把SA深度、DA3/Zip回放或视频截帧当作输入，不修改原始RGB。
- 输入页为5列×2行，216×216等比显示，标签位于对应图上方并写View/source或View/frame。视频对比页沿用已验收的六路/三路位置与标签。构建工程位于 `D:/ppt/_src/20261127-海思端侧重建效果视频对比/`，`input_image_validation.json` 与 `input_image_audit/` 保存190张输入的来源、SHA和总览；改页前19页完整PPT备份于 `D:/output/_backup/hisilicon_comparison_before_input_pages_20261127/`。
- `build_deck.py` 的输入页不放视频，媒体验收逐例检查输入页10张图且偶数页视频数正确，并核验输入图像SHA、图像/视频矩形与页面边界、108段源/内嵌视频SHA、编码及完整解码。页面引擎改页可能清除媒体，仍按“全部页面完成→统一封装→只读媒体验收”执行，不将旧19页验收冒充新38页验收。

## 2026-11-27 海思 v2：DA3 成品默认直接写样例目录

- 用户明确原版 `CASE/depthanything3.ply`、采集位姿 `CASE/depthanything3_cameras.ply`，与SplatAnything同目录使用。旧实验为同时保存native/input_world再复制安装，并非DA3的必需步骤。
- 51号默认直接安装：内存中完成坐标对齐，在样例目录临时写入最终PLY，完整校验后原子改名；原生/对齐副本不再默认另存。已有目标须身份及SHA一致才可核验跳过，不完整或不一致拒绝覆盖；原始RGB/JSON不改。诊断归档显式 `SAVE_NATIVE=1` 或 `SAVE_INPUT_WORLD=1`，49号按对应坐标相机回放。
- 旧实验成品按用户指定命名安装到v2；case14无合法PLY，不安装。case17未完成临时文件经确认后备份至 `D:/output/_backup/da3_all10_case17_interrupted_20261127/` 再移走。安装与旧存档分离，未删除正式实验高斯。
- 直接导出以模拟前向和合法41属性PLY回归了目标命名、无双副本、原子文件、摘要及免模型续跑；旧双PLY存档case1回放兼容通过。模拟测试只验证代码路径，不替代真实模型或新画质结论。

## 2026-11-27 海思视频对比改稿：竞品相机与版式修正

- 用户复核指出旧版竞品完全看不到主体、视频位置与文字错位。旧版直接对随包矩阵做 OpenGL→OpenCV 翻转并未适配 recentered PLY；近距离背面/空白不是竞品质量结论，也不是可保留的“相机契约待确认”。
- 竞品相机修复使用**渲染诊断得到的展示映射**，尚未由原始导出器验证，不能写成已证明的数学契约：源中心加 `recenterOffset` 后映射至主体正面侧，按源方向朝向 `bounds.center`，径向距离由随包 `minDistance/maxDistance` 约束，再做 OpenGL→OpenCV 与局部显示横滚。不能把 `recenterOffset` 或源 `position` 当米制平移直接套用。保留完整高斯/SH，不裁点、不筛点；只重渲染17段竞品，其他91段保持SHA不变。旧竞品视频、摘要和PPT备份于 `D:/output/_backup/hisilicon_comparison_before_competitor_fix_20261127/`。
- 全量首帧及4秒中段抽检发现 case7 原源方向会俯冲至人物头顶；仅调整半径不能解决。最终该例保留各源机位水平方位，以水平半径中位数构造 upright look-at 展示环绕，**不冒充原始姿态的严格内插**。case14 的 view3 半径为1.5207，约为其余机位中位数的12倍；仅对该离群源机位使用随包 `initDistance=0.119161`，显示横滚135°。首帧人物完整、正向，中段主体持续可见但偏远且带倾斜，记录实际边界，不宣称与竞品官方查看器逐像素一致。水平环绕方案对case14产生更多地面近景，已否决。
- 页面引擎把1280×720源画布换算为960×540实际PPT，旧视频却按源坐标嵌入，导致偏右、偏下及越界。改版视频与对应海报共用实际PPT矩形，文字仅在视频上方显示“方法名 · 有记录耗时”，删除页面内长说明。case18/19按可用方法顺序与三格海报配对，并逐格校验实际矩形，不再用六路全局索引；页面引擎统一把持久化海报命名为 `image.png`，不能按源文件名匹配。全部19页完成后统一内嵌，再检查位置、媒体SHA与完整解码。
- 最终 `D:/ppt/20261127-海思端侧重建效果视频对比.pptx` 已完成19页、108段内嵌视频封装。`_src` 同名工程 `media_validation.json` 记录：108个唯一内容SHA与源视频一致，全部704×704、120帧、15fps、H.264/yuv420p且完整解码通过；重新打开后逐格矩形与页面边界通过。第1/18/19页再由页面结构接口复核标签在视频正上方、六路/三路位置一致。`build_deck.py` 以源画布数值直接换算实际EMU（不能用 `Pt` 作分母），支持完整页幂等续跑、半嵌入状态拒绝，最终ZIP完整性与Python 3.10语法检查通过。
- 最新七张4秒中段总览已重新生成并目检：case7竞品恢复完整人物；其余竞品中段均可辨主体，源半径离群筛查只有case7/14需特例。case14竞品仍偏远、倾斜、地面占比较大，不隐瞒这一构图边界；全部抽检不等于全轨迹或PowerPoint/WPS人工播放认证，更不构成几何精度或算法排名结论。最终PPT约321.28MiB；本改稿未执行push。

## 2026-11-27 海思端侧19例六路高斯渲染视频对比展示

- 任务以 `D:/dataset/测试数据/海思端侧重建拍摄数据` 的19个样例为范围，每例一页；case1–17展示竞品、DA3原版、DA3采集位姿、DA3 SfM位姿、SplatAnything、ZipSplat六路，case18/19只展示真实存在的DA3原版、DA3采集位姿、ZipSplat三路，不伪造缺失结果。正式输出 `D:/output/results/da3_human_results/hisilicon_video_comparison_v3/`、PPT `D:/ppt/20261127-海思端侧重建效果视频对比.pptx` 与 `_src` 同名工程。
- 51号使用 `gsplat.rasterization` 渲染原生完整高斯和SH，不裁点、不筛选画质；704×704、120帧、15fps、8秒、H.264/yuv420p。逐帧RGB经stdin送FFmpeg，临时视频通过编码与帧数检查后才替换正式文件；不恢复不明来源的旧PNG残留。轨迹为对应方法原生输入相机的闭环内插：旋转SLERP、位置Catmull–Rom、内参线性，不作无依据360°外推。
- **本节竞品视觉处理已被上方改稿取代，不可继续引用：** 随包相机直接回放导致近距离、局部模糊、倒置或近乎空白，不能据此判断竞品画质或保留为有效展示。SfM使用实际 `input_extrinsics_original` 与对应内参，不强制与stage1构成单一刚体。采集位姿case15–17兼容 `original_world_rotation`，已逐例与摘要 `world_rotation` 验证一致，不是新坐标协议。case18/19用各自原生预测轨迹，不宣称共同物理相机。
- 耗时只来自原算法日志/摘要，逐格写单例及口径；缺失写不详，不把本轮展示渲染耗时当推理耗时，也不跨口径作速度排名。
- 续跑曾因旧整例记录先折叠后又追加而形成重复方法；51号现从空列表按固定六路顺序登记，子集运行按旧摘要保留未处理方法，并拒绝重复旧键或指定方法无旧登记的续跑。case1连续两次全量续跑均为6条且视频SHA一致；另在独立测试目录连续执行 `ONLY_METHODS=da3`、`ONLY_METHODS=zip`，均保留6个唯一方法且6段视频SHA不变。全量整理后每例6个方法槽位，108段正式视频未因摘要或脚本修复而重渲染。
- 页面逐页写入并检查布局，再用 `python-pptx add_movie` 真实内嵌视频。`media_validation.json` 实测19页、ZIP内108个MP4、108个唯一内容SHA；case1–17每页6段，case18/19每页3段，按真实页面顺序与源视频SHA一致，全部为704×704、120帧、15fps、H.264/yuv420p且完整解码通过。
- 中段目检已覆盖全部108段的4秒处（约第61帧），七张总览位于 `_src` 同名工程的 `video_midpoint_audit/`；结合此前首帧抽检，除竞品契约异常外，多数方法可辨对应主体。case7的SA与Zip中段帧为后视人物，case14的DA3原版与ZipSplat中段帧主要显示背景、人物缺失或很弱，均如实保留，不作整体画质放行。抽检不代表全轨迹、PowerPoint/WPS人工播放或几何精度认证。

---

- 用户明确要求全部10张图输入DA3，不再使用8源2留出。正式根目录
  `D:/output/results/da3_human_results/hisilicon_camera_all10_v2_20261127/`；不改原始RGB、相机JSON或v1。
  34例逐例两支，同一权重、seed42、704→700、saddle_balanced、完整SH2、不裁点。
  最终 **33例完整RGB+pose双分支，case14两支均无法按原生完整口径导出PLY**。
- 每支保存原生高斯/相机和输入世界派生高斯/相机，无显示转正，不安装回数据集。
  后者只用一次adapter相机尺度的Sim(3)对齐，不把坐标对齐说成前向质量修复。
  成功分支均为4,900,000高斯、41 float属性、完整SH2，每份803,601,007 bytes。
- 49号兼容全10图源回放，所有RGB均前向；回放分数不是留出NVS或几何真值。
- 相机图像几何独立审计已覆盖34例：case3、14、24的Sampson中位误差分别
  3.8869、2.2926、2.1563px，其他多为0.0863–0.8294px；此阈值只提示输入几何风险，不用于删例。

### 文件验收与源回放

- `all10_file_validation.json` 按真实案例目录核验34例，33完整；132份完整PLY均通过
  SHA-256、顶点数、41属性、预期字节数和全有限值检查，无`.partial.ply`。33例×2支×10视角
  共660个源回放，回放PNG及对应GT共1,320张，均通过图像身份、顺序、可读性和504×504检查。
  660张磁盘PNG逐张按原始浮点MSE和8bit量化误差上界复核通过；原始浮点指标不改写。
  这只是文件/数值验收，不是视觉或几何质量验收。
- 33例各自10视角均分：RGB-only为14.5656957791dB，pose为15.5769609707dB，
  平均Δ+1.0112651916dB；以±0.2dB为描述性阈值，改善22/持平4/退化7。
  统计见 `source_replay_summary.json`；没有留出PSNR，也不能称新视角泛化提升。
- 33例首帧已复核，case12/23/30/31另看0–5视角。pose并非普遍更好：case12仍有主体不明显
  和错位；case23两支持续雾化；case30视角0–4恢复人物但视角5雾化更差；case31仍有幽灵人物
  和错误背景；case11/13/26等回放均分下降。审计图入口为根目录 `all10_overview_33cases.jpg`、
  `all10_visual_audit_selected.jpg`、`all10_visual_audit_risks.jpg`；未声称逐像素人工目检全部660图。

### case14 最终诊断与续跑边界

- RGB正式分支检测到810个非正线性尺度分量，独立重试超过17分钟未导出后停止，无成功PLY。
  pose独立前向成功，约12.375秒，但检测到4,413个非正线性尺度分量，也没有成功PLY。
  失败目录保留在正式根、独立RGB retry目录和独立pose目录，不能把它们合并成完整A/B。
- pose只读诊断（`hisilicon_all10_nonfinite_case14_pose_20261127.pt`）实测：4,413个非正分量
  涉及2,942个高斯，集中于视角0、6，线性尺度最小−0.0006913375691510737；
  原始高斯属性均有限。公开的 `prediction.depth` 为0.2559365–4.9216385、全非正数量为0。
  但该模型启用 `pred_offset_depth`，官方 `gs_adapter.py:83–87,121–130` 实际以
  `depth + raw_depth_offset` 构造 `gs_depths`，再乘正的sigmoid尺度和分辨率乘子；
  所以不能用 `prediction.depth>0` 推断最终高斯深度为正，更不能宣称尺度sigmoid输出负数。
  PLY log导出要求严格正尺度；本轮未使用abs/clamp/删点伪造完整结果。
- 51号按唯一摘要重建总表，修复重复索引；无待跑时只核验、不加载模型。`CASE_FILTER`子集摘要
  单独写 `batch_summary_subset.json`，不覆盖全量总表；一次旧子集核验曾将总表缩为case1，
  已备份并从34个唯一案例摘要恢复33完整+case14失败，未改动任何PLY或回放图。
  49号接受嵌套PLY摘要及精确原子改名，单分支诊断必须显式scope，失败目录不能冒充完整对照。

---
## 2026-11-26 海思 v2：32例有效 SFM 的 DA3 重建完成，画质不作整体放行

- 用户完成 v2 SFM 后要求沿用38号重建。以 `colmap_sfm_results_batch_summary.json` 为准：
  **32例成功、319个注册视角**，31例10/10，case14为9/10（view07未注册）。
  case5、case6仅有 database/logs/sparse_raw，没有最终 `sparse/0`、文本模型或 summary；
  不用数据库、stage1或猜位姿补齐，因此本批不是34例完成。
- 38号发现阶段只接收最终 summary + sparse/0/txt/images.txt，跳过案例写入日志及根总表
  `skipped_incomplete_sfm`；新增可选 `EXPECTED_CASES` 防静默少跑。39号按实际待验收摘要
  发现案例，移除旧17例硬编码。其他项目未提交的脚本改动不纳入本轮。
- 已生成32份正式 `CASE/depthanything3_sfm.ply`，全部 `adapter_scaled_camera_single_sim3_v2`、
  504p、完整SH2、不裁点。31例各2,540,160高斯，case14为2,286,144；正式PLY合计
  **13,289,133,280 bytes**。另保留每例 native PLY、原始/adapter相机及SFM相机，不混用坐标。
- 319个注册源视角均完成504² EWA回放。39号严格核验32/32通过：正式/native PLY均41属性、
  全有限，SFM输入K/w2c与导出相机配对、adapter缩放前后相机及尺度链均通过；
  `depthanything3_sfm_validation.json` 是最终含回放存在性检查的报告。
- **文件/数值验收不等于画质验收。** 首视角总览覆盖32例，另逐视角查看case1/7/14/16及
  8例input08侧视角：case1/34主体完整但软化，case7背景拖影明显；case12/14/16/24/27
  有严重人像或物体雾化、重影，case14椅子、case16手与椅子尤其明显。人像总体存在软化。
  审计图在 `D:/output/results/da3_human_results/hisilicon_v2_sfm_visual_audit/`；
  这些是源视角回放，不是独立新视角或几何真值证明，不能宣称所有重影已解决。

---

## 2026-11-26 SFM 条件分支：adapter 尺度修复及17例正式重跑

### 日志与根因：需要重跑，不套用 glTF 坐标规则

- 旧正式日志 `logs/da3_sfm_inference.log` 每例均有 `Using camera conditions provided by the user`。
  SFM 输入经 DA3 相机条件前向，故会进入 `gs_adapter(gt_extrinsics=...)` 的尺度分支。
- 官方 adapter 对高斯中心、尺度和反投影深度都乘 `a`；旧38号使用缩放前 raw camera 拟合，
  又用 `sf*pose/a` 修正，造成重复尺度。新版用 `adapter_camera_extrinsics` 将相机 w2c 平移乘 a，
  然后只对 adapter 原生高斯施加一次 `input <- adapter` Sim(3)，不再额外乘 Nested sf 或除 a。
  版本标记 `adapter_scaled_camera_single_sim3_v2`；`total_gaussian_scale == pose_scale`。
- COLMAP 本来就是 OpenCV w2c，本轮不改变相机旋转/中心、帧序、内参、深度或前向输入。
  新 glTF 两包经图像验证的 `R不变/C取负` **不能推广到 SFM**。真实官方 adapter 合成尺度2回归通过。
- 旧36号 stage1 默认公式保持不动，本轮只让38号显式走 adapter 对齐分支；
  不据此宣称旧 stage1 已重新验收。36号函数默认 False 仅用于历史正式结果复现。

### case14 单例 A/B 与剩余前向问题

- 独立目录 `D:/output/results/da3_human_results/sfm_scale_ab_20261126/`。
  同 SFM、同去畸变 RGB、同权重、504p、seed42、saddle_balanced、完整SH2、不裁点。
  case14 实测 adapter a=2.6260230541；`pose_scale=3.5079156450`。
  新旧对齐旋转/平移数值基本一致，实际差别在尺度链及正确的相机配对。
- case14 旧版源视角平均 PSNR=11.8379471313dB，修复正式版=16.3920040674dB，**+4.5540569362dB**。
  10个视角中9个改善，正面视角略降约0.317dB；平均改善不能当作每视角都改善。
  人物从大面积缺失/错位恢复为完整主体，重影显著减弱，仍有软化与局部背景模糊。
- 同一次前向的 adapter 原生 PLY/相机回放=9.8845188045dB，仍存在明显错位与缺失。
  因此 SFM 前向预测本身尚未通过质量验收，修复 Sim(3) 不是全部根因；
  应另查输入相机条件如何改变前向几何，不得只凭对齐后的源回放称“所有重影已解决”。
- case7 备份旧版=10.2585937781dB，修复正式版=16.3941871987dB，**+6.1355934206dB**；
  仍严格只用6个 COLMAP 注册视角。目检人物完整性明显改善，但侧面面部仍有局部串色/拖影。
  case16 背景雾化仍明显，case12 主体完整但软化和局部拖影仍在。

### 17例正式输出与验收

- 原路径 `CASE/depthanything3_sfm.ply` 仍为正式交付名，17/17 重跑无失败；
  16例各2,540,160高斯，case7为1,524,096，全部41属性、数值有限、完整SH2、不裁点。
  正式 PLY 合计 **6,915,348,703 bytes**；新版每例另存 `depthanything3_sfm/native.ply`
  和 `native_cameras.npz`，与 SFM 输入世界 PLY 严格分开。
- 每例旧 PLY、相机、摘要及旧回放备份在 `CASE/depthanything3_sfm_backup_20261126_204420` 起的
  时间戳目录（各例时间不同，准确路径见新版 summary.backup）；无旧文件被删除。
  原根总表与 validation 属可再生元数据，不逐版本备份。
- 全部166个已注册源视角完成新版正式 PLY 回放。39号强化核验：SFM 输入外参/K配对、
  新旧尺度版本、adapter缩放前后相机、原生 PLY 与原生相机均逐例通过。
  `depthanything3_sfm_validation.json` 为最终17/17通过记录。
- 38号支持 `OUT_ROOT` 的一次性新实验目录；shell 只创建外层日志目录，已有目标案例拒绝覆盖。
  默认正式重跑需 FORCE=1，旧尺度结果不允许被跳过逻辑冒充新版。
- 收尾发现单例 `CASE_FILTER` 曾覆盖根目录17例总表（正式 PLY、各例摘要和 validation 不受影响）。
  现改为单例摘要写到 `CASE/depthanything3_sfm/depthanything3_sfm_single_summary.json`；根总表
  仅由全量运行更新，并已用17份现存摘要恢复。恢复时补注 `summary_status: restored_after_single_case_overwrite`
  及恢复时间，保留原始 `status`，不重新推理、不改变高斯或相机数值。

---

## 2026-11-26 用户指定16例相机条件共同留出对照（已完成附录）

用户明确旧17例 `depthanything3_cameras.ply` 有严重重影，不能作为有效相机分支直接复用。
本附录只覆盖用户指定的 case1–14、`model_man`、`model_plant`；case15–17不纳入本批统计。
默认入口的替换有独立备份，原始RGB/相机JSON不动，有效对照均取输出目录内新前向 `native.ply`。
`model_man/model_plant` 的特殊glTF解释不套到旧stage1。

### 统一协议与输入审计

- 本批14例stage1与 `model_man/model_plant` 均固定 source=`0,1,3,4,5,6,8,9`、target=`2,7`。
  target RGB不前向；同一 `DA3NESTED-GIANT-LARGE-1.1`、seed42、process_res=704（实际700×700）、
  saddle_balanced、完整SH2、不裁点，504×504 EWA。各分支8源前向均为392万高斯；
  glTF另外安装的全10视角版为490万，不能混用两种输入规模。
  仅用source相机拟合各分支一个Sim(3)，将相同物理target映射到原生世界；源回放另作诊断。
- 46号用图像独立SIFT ratio=.7、USAC_MAGSAC基础矩阵筛选，旧stage1除case7外16例原flip约定
  Sampson中位0.227–0.542px；case7为2.156px、双正深度93.55%、7对/558观测，风险单列。
  该检验不证明绝对尺度、动态几何或独立真值。结果：`stage1_camera_audit_20261126_v4/`。
- 正式输出 `hisilicon_camera_ab_20261126/stage1/`，主批处理串行运行47/48号。
  49号强制检查case、RGB与相机JSON hash、PLY归属/hash、视角名称及source/heldout角色；
  旧渲染同时复核GT像素与PNG量化分数，不放宽严格身份校验，不改原始浮点metrics。
- `model_man` 有效留出为18.9056→17.6972dB（Δ−1.2084），`model_plant` 为
  14.0257→16.2780dB（Δ+2.2523），盆栽两个留出均改善但仍有软化/纹理损失。
  盆栽有效目录只有 `camera_holdout_plant_true_20261126/`；此前man/plant错配前向目录排除出统计。
- 本次16例完整指标 `hisilicon_camera_ab_20261126/comparison_metrics_case1_14.json`：
  留出Δ均值+0.2858dB；按描述性阈值Δ>+0.2改善、|Δ|≤0.2基本持平、Δ<−0.2退化，
  **改善8 / 持平1 / 退化7**。均值只作分布摘要，不能当质量放行线。
- 逐例留出Δ（dB）：case1 +2.1231、case2 −0.1191、case3 −0.2659、case4 +1.1976、
  case5 +1.1384、case6 −0.3895、case7 −0.4350、case8 +1.3901、case9 −1.7395、
  case10 −0.9803、case11 −1.2212、case12 +1.2532、case13 +0.7195、case14 +0.8575。
- case14虽平均PSNR提升，实际第一帧人物几乎消失、第二帧人物破碎/偏移，明确**画面质量不通过**；
  case7几何审计风险与面部破碎/拖影/雾化另列，不能由源回放或均分掩盖。
- case14输入来自 `hisilicon_camera_ab_20261126/stage1_remaining_14_17/case14_nopose_recentered/`，
  只补已有双分支的渲染，未重跑case15–17。成品PPT共29页，第13–28页为16例真实双分支留出对照。
  全10视角安装版是另一实验口径，不与8源留出混算。

---

## 2026-11-26 图像几何独立审计：坐标解释错误坐实，修正不等于普遍增益

本节修正下方同日中间记录。保留用户原始JSON、RGB、两份正式PLY以及同事PLY，未覆盖。
输出根目录 `D:/output/results/da3_human_results/`。

### 独立证据与根因

1. 使用SIFT ratio=0.7，图像自身估计F（USAC_MAGSAC，1px）过滤误匹配，**不使用待测相机过滤**。
   人像35个有效图像对、4401个匹配观测，源704像素下Sampson距离：

   | 相机解释 | 中位 / p90（px） | 三角化双相机正深度占比 |
   |---|---:|---:|
   | 原flip + 顶层静态K | 44.314 / 100.053 | 74.14% |
   | 原flip + 动画K | 38.886 / 85.023 | 96.00% |
   | 去掉flip + 动画K | 0.316 / 0.917 | 0.045%（不合法的修复） |
   | 保持R、中心C取负 + 动画K | 0.316 / 0.917 | 99.955% |

   JSON矩阵记为`[R,C;0,1]`；图像支持的诊断相机为`w2c=inv([R,-C;0,1])`，
   所有R仍是SO(3)。这识别的是**图片/JSON之间的坐标及手性解释不一致**，不是证明
   JSON采集轨迹本身坏掉，更不是通用glTF转换。只凭当前包不能定位上游视频渲染/导出中的哪一环。
   帧序仍为0001→关键帧0…0010→关键帧9；直接反序、把c2w当w2c、其它轴翻转均不通过。
   盆栽独立复核：原动画K方案24.414px，特殊解释0.352px，99.865%正深度；本轮未再推理盆栽。
2. 顶层95°与K**内部一致**，`352/tan(yfov/2)=322.54852773`，不是此前记载的327.3。
   但它与实际动画yfov不同。动画K为687.318974@704；即便改正外参，顶层K仍产生16.234px中位误差。
3. 官方41736238的adapter会将预测相机中心与深度同时乘`a`。36号旧版保存的相机来自
   **乘a之前**；原生高斯实际已含a。旧review的a=1.7395488，不能将两者直接配对，
   也不能用该相机直接拟合对齐后声称“无需a修正”。现将捕获的w2c平移乘a后再拟合单一Sim(3)；
   等价于对未缩放相机拟合的尺度再除a，且不混入Nested sf。真实官方adapter合成尺度2回归通过。
   本轮不修改官方仓。旧stage1/SfM后处理公式保持原样，未声称这些历史结果已重新验收。

### 受控重建（同本地官方权重，700×700，完整SH2，seed42）

`camera_control_20261126/`：同10张RGB、同DA3NESTED-GIANT-LARGE-1.1、process_res=704（实际700）、
同saddle_balanced、同完整导出，每支490万高斯、41属性、全部有限。
每支adapter相机都正确包含a；全10视角504×504 EWA渲染，白背景、eps2d=.01。

| 分支 | 源视角平均PSNR | 观察 |
|---|---:|---|
| rgb | 21.2379dB | 人物完整，软化、局部细节偏差 |
| declared_gltf（动画K、原flip） | 13.3572dB | 多头、大范围叠影；仅修正adapter尺度无效 |
| negative_centers（动画K、图像验证坐标） | 21.8719dB | 严重多头消失，仍有局部软化/眼镜细节问题；9/10源视角PSNR更高 |

该表使用各分支自己的生成相机，只能证明源回放改善，不能代替共同target的评测。
同事两PLY实读仍为4,776,138/4,776,089点、17属性（DC），用户原始本地两PLY为2,540,160点、41属性；
配置和导出口径不同，不能从文件名逆推出同事是否实际启用CameraEnc，用户无同事脚本。

### 8源+2留出：没有支持“位姿一定更好”

`camera_holdout_20261126/`：固定source索引`0,1,3,4,5,6,8,9`，target为`2,7`（frame_0003/0008）。
两支各392万完整高斯，target RGB不送入模型。仅用8个source相机拟合一个非RANSAC Sim(3)，
将同一组图像验证target相机映射至各自adapter世界；target K均为动画K，704→504同比缩放。
未按target得分改参数或重选帧；这是一次诊断留出，坐标约定此前看过整包图像匹配，不冒充盲测基准。

| target | RGB-only | 修正位姿 |
|---|---:|---:|
| frame_0003 | 19.4205dB | 19.0716dB |
| frame_0008 | 18.3907dB | 16.3229dB |
| 平均 | 18.9056dB | 17.6972dB |

目检两支均仍有细节损失；修正版第8帧背景/边缘退化更多。不能以源视角+0.634dB掩盖留出-1.208dB。
当前结论是**原严重退化的主要原因已定位到错误相机解释，且另有配套相机尺度bug；
但本例未证明加位姿对新视角整体更好**。后续应预设多案例/多留出和人物ROI、全图及几何指标，
再评价位姿输入收益；不得只选有利视角。

### 工具与复现

- `audit_gltf_camera.py`：纯CPU图像/相机审计；`CASE_DIR`、`RESULTS_DIR`必填，输出目录须不存在。
  可选`AUDIT_SOURCE`检查旧native.ply与其相机a尺度；输出不覆盖输入。
- `run_gltf_camera_control.py`：三分支对照，设置`SOURCE_INDICES`/`BRANCHES`可复现留出。
  `render_camera_control.py`接受明确PLY、相机NPZ、原图目录，`EVAL_ROLE=heldout`标记留出。
- 36号新增显式`GLTF_CAMERA_CONVENTION=image_verified_negative_centers`，默认仍是标准glTF，
  特殊模式仅对已审计数据使用；记录转换与K模式，旧摘要缺失或不匹配拒绝静默复用。
  使用新的`PLY_NAME/OUT_SUBDIR`；读取相机和完整对齐的修复不自动覆盖用户原文件。
- 实际命令：

```bash
cd /mnt/c/code/media_code
CASE_DIR=/mnt/d/dataset/测试数据/海思端侧重建拍摄数据/model_man \
RESULTS_DIR=/mnt/d/output/results/da3_human_results/camera_audit_repeat \
~/miniconda3/envs/da3_human/bin/python da3_human/audit_gltf_camera.py

CASE_DIR=/mnt/d/dataset/测试数据/海思端侧重建拍摄数据/model_man \
DA3_MODEL_DIR=/mnt/d/wheel/da3_human_ms \
RESULTS_DIR=/mnt/d/output/results/da3_human_results/camera_holdout_repeat \
GLTF_INTRINSICS_MODE=camera_path_yfov GLTF_CAMERA_CONVENTION=gltf \
SOURCE_INDICES=0,1,3,4,5,6,8,9 BRANCHES=rgb,negative_centers PROCESS_RES=704 \
~/miniconda3/envs/da3_human/bin/python da3_human/run_gltf_camera_control.py

PLY_FILE=/mnt/d/output/results/da3_human_results/camera_holdout_repeat/negative_centers/native.ply \
CAMERAS_FILE=/mnt/d/output/results/da3_human_results/camera_holdout_repeat/negative_centers/heldout_cameras.npz \
IMAGE_DIR=/mnt/d/dataset/测试数据/海思端侧重建拍摄数据/model_man/images \
RESULTS_DIR=/mnt/d/output/results/da3_human_results/camera_holdout_repeat/negative_centers/heldout_render \
GSPLAT_CACHED_SO=/home/velvet/.cache/torch_extensions/py311_cu126/gsplat_cuda/gsplat_cuda.so \
EVAL_ROLE=heldout LABEL=corrected_pose \
~/miniconda3/envs/splatanything/bin/python da3_human/render_camera_control.py
```

---

## 2026-11-26 带位姿重建最终复核：原生前向多头，尚不能质量验收

- 最终重新对 `model_man`、`model_plant` 跑704输入（实际700×700）、动画 `camera_path.yfov`、`saddle_balanced`、`DA3NESTED-GIANT-LARGE-1.1` 十视角相机条件前向，日志均明确 `Using camera conditions provided by the user`。两例输入世界 `input_world.ply` 均为4,900,000高斯、41属性、完整SH degree2、全部有限、803,601,007 bytes，配套 `cameras.npz`；同前向另存 `native.ply/native_cameras.npz`。
- 最终 `transform_gltf_gaussians` 使用 adapter 捕获的预测相机拟合 `align_poses_umeyama(input,pred)` 返回的 pred→input 单一 Sim(3)，不再叠加 Nested sf 或除 adapter a。man/plant 尺度分别1.41561975/1.46105687，相机中心最大残差0.03914112/0.03477079输入单位；这些残差不是独立位姿真值或画质通过证明。
- 每例完成原生预测世界与输入世界各10个源视角回放，渲染K均由700×700缩放至504×504，使用同一渲染器与缓存扩展；结果图和参数JSON在各自 `depthanything3_camera_audit/review_20261126/`。man 两路均有多个叠合人脸/头部，plant 两路叶片和花盆多层混叠。**原生相机条件前向已经退化，完整Sim(3)不会消除这类重影，仍不可标为质量通过。**
- 同版本、同704处理、同504回放配置下，`model_man` RGB-only 原生预测世界十视角只表现软化、局部细节不准，未出现这种多重人脸。该对照在 `model_man/depthanything3_camera_audit/rgb_saddle_v2/native_render.jpg`，排除了“只要原生DA3输出就多头”的解释。
- 顶层95°K 与动画约54°yfov 的冲突确实存在，但动画焦距并未修复多头。撤回此前“动画焦距使原生人物接近RGB-only、未见多个头部”的结论；相关旧图受相机/K口径混杂影响，不作最终证据。仅传 intrinsics 时官方 CameraEnc 不生成 cam_token，也不是有效内参独立消融。first 与 saddle 本例输出逐字节相同，不能构成独立参考帧消融。
- 当前边界：可排除“仅靠修正尺度/世界变换即可解决”，但尚不能区分 glTF 外参或帧映射错误与 DA3 CameraEnc/前馈多视角深度不一致；同事 PLY 不能证明其权重、参考帧或调用方式。正式 `depthanything3_cameras.ply` 未覆盖，两例最终结果仅供用户目检。旧 stage1/SfM 共用后处理保持原路径，未借本轮结论批量重跑。

---

## 2026-11-26 同事 PLY 对照：输入焦距、原生尺度与处理分辨率

> 本节是中间调查记录；其中“动画焦距接近RGB-only、未见多头”与 `sf×pose_scale/a` 尺度结论已被上方最终统一回放复核修正，保留仅用于追溯，不作最终结论。

- 对照文件为 `D:/dataset/测试数据/陈镇枫DA3重建结果/0000-DA3-704-{NoPose,pose}-man.ply`。二者分别为 4,776,138/4,776,089 高斯，17 个 float 属性、仅 SH DC、每点68 bytes；本地此前的 2,540,160 点来自 `PROCESS_RES=504`：`10×504×504`，而非输入原始照片尺寸。官方预处理将 `process_res=704` 圆整到 **700×700**，完整点是 4,900,000，十视角700前向实测峰值约15.2GiB；同事点数约为其97.47%，精确筛选逻辑尚需其导出代码，不能凭点数宣称“原始像素更多”。
- 同事无位姿与位姿 PLY 都接近 DA3 自估相机世界。借用本地 RGB-only 相机回放两版，人物均清楚、无本地相机版多头部；直接配原始 glTF 相机则倒置/偏小，故不是原生高斯可任意配 API 最终输入相机的证据。逐点检查显示部分视角仅浮点末位不同，另一些视角确有变化，不能简化为“完全复用 RGB-only”。这些图像尚需按最终明确 K 基准重新复核，不能反推同事实现。
- 原生 700 前向同时捕获 `gs_adapter` 的真实预测外参/内参。顶层 K 条件时预测焦距约306–313/333–341px，原生预测相机回放人物显著偏小、背景大量漂浮球；动画 `camera_path.yfov=0.9466165900230408` 条件时预测焦距约665–671/699–703px，接近 RGB-only 的678/706px，人物比例与清晰度接近 RGB-only，未见原相机版那种多个头部。这坐实新增两例**静态节点95°K 与实际图像/动画约54°视场不匹配**，而不是704 RGB或权重本身导致退化。
- 原生高斯即使采用动画焦距，也不能直接配 API 最终输入相机（会倒置/缩小）；仍需同尺度完整 Sim(3)。此前自定义尺度链遗漏撤销 adapter 在 Nested 米制缩放前施加的中间尺度 `a`，旧 `sf×pose_scale` 使物体相对输入相机额外放大。36号现捕获 adapter 实际使用的原张量并计算 `sf×pose_scale/a`，禁止拿 Nested 后的相机反推 `a`。用错误a=3得到的中间文件不作为修复结果。
- 新增两例的默认 PROCESS_RES 改为704（实际700），`GLTF_INTRINSICS_MODE=camera_path_yfov`；旧 `top_level_K` 仅保留诊断。完整对齐的诊断输出放在 `depthanything3_camera_audit/final_yfov_sim3/`，未覆盖正式 PLY，待同输入多视角回放验收后再替换。
- 撤回“36号完全绕开 adapter 中间尺度、17例均无需复核”的绝对表述。官方 adapter 不应用完整R/t仍是事实，但其 `a` 已实际进入旧高斯；旧17例须单独按新尺度链复核，不能用本轮新glTF内参结论替代。

---

## 2026-11-26 新增 model_man / model_plant 的原生与 glTF 相机 DA3

- 输入均为原始 `images/frame_0001..0010.png`，每例10张704×704 RGB。原生版不读相机、已有 `model_*.ply` 或 `renders/`；相机版按文件名序号一一对应 `*_camera.json` 的10个 `c2w_keyframes`，使用顶层显式 `intrinsics.K`（`fx=fy=322.548528`），glTF c2w 转 OpenCV w2c。没有借用旧17例的抽帧清单或 stage1 JSON。
- 权重仍为本机 `DA3NESTED-GIANT-LARGE-1.1`（MD5 `4e54bf75031acc19cf210d3423a3954a`），504p、`saddle_balanced`、`infer_gs=True`。两次相机版日志均明确 `Using camera conditions provided by the user`。输出 `depthanything3.ply` 和 `depthanything3_cameras.ply`；相机版使用捕获的预测相机 + 完整 Sim(3) 对齐，保留 glTF 原生世界，不施加旧17例的180°显示旋转。
- 四份 PLY 均为 2,540,160 高斯、41属性、完整SH degree2、416,587,247 bytes，全部有限；两例两版共40源视角回放完成。`model_man`/`model_plant` 的相机尺度分别3.02814484/6.31354576，总高斯尺度1.28684797/1.37502991，预测—输入相机最大中心残差0.04689972/0.07070256（输入世界单位），不能认定为零误差。
- ⚠️ 工程通过不等于质量通过：RGB-only 的人物/盆栽主体可辨但偏软；相机版两例均有大范围重影、雾化，人像几乎不可辨，保留原始输出不裁点/美化。`depthanything3_new_models_validation.json` 记录四份哈希与40视角回放。顶层 yfov 与 K 也不一致（704px 下约327.3 vs 322.55），关键帧另有动画 yfov；本次严格使用顶层 K，**没有在无依据时切换参数或把重影根因归给其中一方**。
- 工具兼容：34/36 支持新 `images/` + glTF 相机格式，35 自动适配原始 RGB 回放；单例摘要在各自输出目录，不覆盖旧17例根总表。

---

## 2026-11-26 海思使用最终 SfM 相机重跑 DA3

- 输入为 17 例各自 `colmap_sfm_results/sparse/0` 的最终模型；按图像名只取已注册视角。16 例为 10/10，case7 仅 6/10，缺失的 4 帧不伪造位姿。SFM 的 OPENCV 畸变先用 `cv2.undistort` 去除，K 和外参保留原模型口径；DA3 504p 条件前向，日志逐例出现 `Using camera conditions provided by the user`。
- 输出每例 `depthanything3_sfm.ply` 与 `depthanything3_sfm/{cameras.npz,summary.json,render_check.jpg}`；高斯从预测世界经完整 Umeyama Sim(3) 对齐到 SFM 原生坐标，不施加 stage1 版显示转正。SFM 是任意尺度，不能标成米制；不裁剪，完整 SH degree 2。
- 17/17 文件与 166 个已注册源视角回放完成。16 例每例 2,540,160 高斯，case7 为 1,524,096；全部 41 float 属性、有限值，总计 6,915,348,703 bytes（约 6.44 GiB）。验证报告在数据集根 `depthanything3_sfm_validation.json`，批量摘要为 `depthanything3_sfm_batch_summary.json`。
- 工程验收不代表画质改善：总览未见整例空白或整体错位，但 case7 侧面、case14 多个侧面/全身视角仍有明显人物缺失和拖影，多例人物/背景雾化；未用裁点或替换视角掩盖。
- 工具：`38_run_hisilicon_sfm.{py,sh}`、`39_verify_hisilicon_sfm.py`。36 号的 `transform_gaussians` 新增可选世界旋转参数，默认保持原 stage1 转正路径；35 号支持无畸变图目录和按实际视角数排版。

---

## 2026-11-26 官方 gs_adapter 相机对齐 bug 复核：本轮不重跑

- 同事指出的 `gs_adapter.py` 相机条件分支确有问题：只缩放 c2w 平移与深度，丢弃 Umeyama 的旋转/平移；且异常时只把尺度置1，不能完整回退。但完整 Sim(3) 的点/中心变换是 `X'=sRX+t`，c2w 为 `R'=R Rc`、`C'=sR Cc+t`，**旋转矩阵不乘尺度**。`isfinite(pose_scales > 0)` 也恒为 True，应检查 `isfinite(pose_scales) & (pose_scales > 0)`。
- 海思相机版36号已捕获原始预测相机，并独立变换 means/scales/rotations/SH 与配套相机；RGB-only 的 `gt_extrinsics=None`。因此已有17例未直接依赖该不完整 adapter 输出，本轮不重新推理、不修改 PLY。17例保存的 R/t/s 均为有限正尺度和 SO(3)，pose_scale 范围 0.23403496–5.24071031；不能套用官方 `[1/3,3]` clamp。
- 官方 adapter 若另行修复，异常应回退 B×3×3 单位旋转、B×3 零平移和 B 个1的尺度，并校验全部参数；本批后处理路径选择显式报错，不静默输出错位结果。详见 `ERRATA.md` #13。

## 2026-11-26 海思相机版原地转正（取消 upright 命名）

- 按用户要求，转正结果直接替换每例 `depthanything3_cameras.ply`，配套 `depthanything3_cameras/cameras.npz` 同步替换；RGB-only 的 `depthanything3.ply` 不动。数据目录不再保留 `depthanything3_cameras_upright.ply` 或同名目录，不重新推理、不裁点、不美化。
- 直接复用前一节已验收的转正 PLY，先备份17例原 PLY/相机/记录，再同盘改名并核对 SHA-256。原版本与临时 upright 文件均移至 `D:/output/_backup/da3_cameras_pre_rotation_20261126/`，不是删除原始结果。根目录验收表为 `depthanything3_cameras_{batch_summary,validation}.json`。
- 当前路径下17例/170源视角重新回放完成；case16 目检无倒置、空白或整体错位，原有模糊/破碎不变。原 PLY 的备份 SHA-256 逐例匹配；所有交付 PLY 保持 2,540,160 高斯、41 属性，共 7,081,983,199 bytes。
- `36_run_hisilicon_cameras.py` 后续直接在 Umeyama 对齐后施加 `diag(1,-1,-1)`，不再依赖额外 upright 步骤。F 与 R 虽均为正交旋转，**不能合写成 `F@R` 后再乘 Umeyama 平移**；平移需保持 `t` 后再由 F 旋转。用 case16 原始预测相机与保存的 Sim(3) 参数验证：参数最大误差 0，直接输出与两阶段高斯位置最大差 3.25e-6（float32）。

## 2026-11-26 海思输入位姿 PLY 补 180° 显示偏转（后续已原地替换）

- 按用户对比需求，每例新增 `depthanything3_cameras_upright.ply` 和 `depthanything3_cameras_upright/cameras.npz`；原 RGB-only 与输入相机版均不覆盖。显示坐标 `X'=diag(1,-1,-1)X`，同步旋转位置、四元数、完整 SH 与配套相机，尺度、透明度、点数和深度不更改。
- `37_flip_hisilicon_display.{py,sh}` 复用官方 `quat_to_mat/mat_to_quat` 与 `rotate_sh`，后者采用 `(N,3,9)` 及 `(N,3,3,3)`，不能用渲染加载器的 `(N,9,3)` 直接替代。case16 十视角回放与原几何/颜色一致；这只是显示坐标转正，**不是两版注册，也不改善模糊或破碎**。
- 全部17例转换完成：每例2,540,160高斯、41属性、完整SH、全部有限，总计7,081,983,199 bytes（约6.59 GiB）；逐点核验X不变、Y/Z反号，尺度与opacity完全不变，相机同步世界旋转最大绝对误差为0。17张共170源视角回放完成，抽查case1/7/14/16无新增颜色或整体扭曲；case7侧面后脑、case14侧面全身原有破碎仍保留。汇总为数据集根目录 `depthanything3_cameras_upright_{batch_summary,validation}.json`。

## 2026-11-25 海思两版 DA3 的 180° 与模糊复查

- **180° 是相机参考系转换，不是人物额外倒转**：逐例计算位姿版 `summary.json` 的 Umeyama 旋转与 `R_input[0] @ diag(1,-1,-1) @ R_raw[0].T` 的差，17 例均小于 0.94°；DA3 的第一帧预测旋转距单位阵也均小于 0.073°。这解释了自估 OpenCV 相机世界到 JSON 的 OpenGL 世界上 Y/-Z 的翻转。两版 PLY 到同一输入世界都需近似 180° 变换；RGB-only 不能直接套输入相机渲染。
- **同相机检查**：`case16/depthanything3_cameras/compare_same_input_cameras.jpg` 中旧 PLY 直接配输入相机会严重错位，位姿版能正常回放，所以不可用跨坐标渲染判断清晰度。`compare_each_native_cameras.jpg` 为两版各用配套相机；case16 源视角 0/2/3 的 PSNR 为 RGB-only 18.15/14.98/15.25 dB、位姿版 17.86/11.51/12.82 dB；case7 为 18.09/16.63/14.86 对 15.49/8.20/8.71 dB。case16 位姿版的灰度梯度方差更高但 PSNR 更低，说明锐边/破碎增加不等于几何更好；不是全场景位姿版一定更模糊。
- **原生模糊来源**：官方 `da3.py::_process_gs_head` 明确仍用预测相机/深度反投影，输入位姿只作 CameraEnc 条件和尺度参考；不是给定相机下多视图优化。每例 10×504×504=2,540,160 高斯直接导出，无后续优化/去重。以实际高斯投影近似估算，case16 两版的中位高斯轴尺度都约 10–17 px，opacity 中位数不到 1%；宽高斯与重叠平均会直接造成柔化。case7 的侧面/后脑双脸与碎裂不是单靠修正世界朝向可以解决。
- **仍有对齐残差，不能反向证明输入位姿本身错误**：预测相机经同一 Sim(3) 后与输入相机的最大方向差多数低于 2.3°，case12/14 为 3.16°/3.48°；最大位置差多数低于 3.1 cm，case14 为 9.19 cm。按 case14 的 f=491.5px 粗算 3.48° 可约产生 30px 偏移，足以加剧拖影；这些误差也可能来自原始 stage1 位姿，尚无独立真值不能划给某一方。不能直接拿预测相机替换输入相机来掩盖误差。
- 两版 PLY 未改，新图仅作诊断。下一步若提高质量，应固定输入相机做多视图融合/优化并单独审计源位姿；重跑同一前馈分支不会消除表示和融合瓶颈。

## 2026-11-25 海思输入相机 DA3 重建与坐标修正

- 输入：17 例各自 `*_stage1_cameras.json` 与同一批十视角 RGB。按 `sourceIndex` 配对，OpenGL c2w 转 OpenCV w2c；每例十张 704×704 图、图像/相机 JSON SHA-256、SO(3) 和内参均校验。
- 前向：仍是官方 `DA3NESTED-GIANT-LARGE-1.1`、504p、`saddle_balanced`、`infer_gs=True`，日志逐例出现 `Using camera conditions provided by the user`。DA3 高斯头继续使用预测相机反投影，故这是**输入相机条件前向 + 高斯到输入世界的 Sim(3) 对齐**，不是直接用输入相机反投影。
- 修复：API 会用输入外参覆盖 `prediction.extrinsics`，需在对齐前保留真正的预测外参；十视角时复用官方 RANSAC 配置（`random_state=42`）。当前 `align_poses_umeyama(pred,input)` 实际返回 input→pred，显式求逆。高斯 means/scales 乘 `metric_scale_factor×pose_scale` 一次，同时变换旋转与 SH；官方转换函数的输出按 `xyzw` 分量使用。case16 初次试验因此出现的整体放大、错位已消除。
- 交付：每例 `depthanything3_cameras.ply`，均 2,540,160 高斯、41 float 属性、完整 SH degree 2、不裁边/深度、全部有限；总计 7,081,983,199 bytes（约 6.59 GiB）。同目录 `depthanything3_cameras/{cameras.npz,summary.json,render_check.jpg}`，不覆盖此前 `depthanything3.ply`。根目录 `depthanything3_cameras_batch_summary.json` 与 `depthanything3_cameras_validation.json` 留档。
- 验收：完成全部 170 个输入相机下的源视角回放，排除了早期坐标系/尺度整体错误。case7 的侧面/后脑近景、case14/15 的室外侧面与全身视角仍有严重破碎和拖影；其余场景也有不同程度模糊、局部缺失。**17/17 文件与回放工程验收通过，不代表所有视角的几何/画质通过**。
- 工具：`36_run_hisilicon_cameras.{py,sh}`，`35_render_hisilicon.py` 新增 PLY/相机/输出名配置及批量显存释放。坐标与四元数坑见 `ERRATA.md` #11/#12。


## 2026-11-25 海思十视角数据的原生 DA3 批量重建

- 输入：`D:/dataset/测试数据/海思端侧重建拍摄数据/` 的 17 个 case。每例复用 `splatanything/36_extract_hisilicon.py` 已登记的 10 张静止视角 RGB，校验图片 SHA-256；不输入海思或 SA 的相机与几何，DA3 用 RGB 自估相机。
- 权重及前向：官方 `depth-anything/DA3NESTED-GIANT-LARGE-1.1`，`process_res=504`、`saddle_balanced`、`infer_gs=True`。用 `da3_frame.gaussians_to_metric` 将高斯中心/尺度和相机深度统一到米制，只折算一次。
- 交付：每例 `depthanything3.ply`；原生高斯、不裁边、不按深度裁剪、完整 SH degree 2（41 个 float 属性）。相机/深度/置信度保留于同目录的 `depthanything3/cameras.npz`。
- 工具：`34_run_hisilicon.py` / `34_run_hisilicon.sh`。单进程复用权重，完整 PLY 跳过；`FORCE=1` 会备份旧文件。case1 首轮官方导出与连续数组导出的所有属性逐点一致。17 例均交付，每例 2,540,160 个高斯、约 397.3 MiB，合计约 6.59 GiB；所有字段为有限值。用 `35_render_hisilicon.py` 完成 170 个源视角回放，主体均可辨认；case8 人物边缘、case12 腿部/远景存在明显软化和拖影，未做独立视角几何验收。
- 数值修正：case1 有 39 个高斯的 opacity 精确饱和到 1，官方 `log(x/(1-x))` 得 `+Inf`。批处理改用数学等价的 `log(x)-log1p(-x)`，饱和点用 float32 可计算最大 logit `-log(eps)`，不删点、不改几何，其余字段按官方导出。交付前逐文件检查非有限值和字段数；这是模型/导出数值统计，不冒充场景重建质量验收。


## 2026-11-24 三源深度倍率与逐源偏移

两静态身份各两个留出，完成点云和 SA 最终高斯回放。Rosemary 仅用脸外纹理估得倍率，最终高斯鼻尖深度 p90 从 10.94/10.68 降至 4.02/6.74 mm；Daven 缺少可靠匹配，回退原输出。保留默认 SA，下一轮换有独立尺度参照的新身份。见[逐帧实测](../splatanything/EXPERIMENTS.md)和[总计划](EXPERIMENT_PLAN.md)。

## 2026-11-24 真值深度替换与三张源脸的刚性配准

静态两人四个留出机位的真值点云鼻区误差 p90 为 0.54–1.17 mm。黑衣 49 即使源鼻部明显收拢，移动后的下颌仍比原版离实拍更远。两种刚性策略都未通过跨身份验收。保留 P0，下一步在固定相机下拆分共同尺度、各源深度偏移与剩余形变。参见[实验与点云](../splatanything/EXPERIMENTS.md)和[执行顺序](EXPERIMENT_PLAN.md)。

## 2026-11-24 SA 三图免训练脸部对齐

五身份 A/B/C 已执行。局部深度微调改善佳佳两帧，但黑衣 49 双脸仍存；两组合成的鼻区 p90 综合降幅分别为 +21.2%、−41.5%。默认保留 P0。逐帧图、深度和失败原因见 [SA 记录](../splatanything/EXPERIMENTS.md)，后续验证见 [总计划](EXPERIMENT_PLAN.md)。

## 2026-11-24 三图人像主线的 SA 双影追踪

五个身份、九个可用留出脸框已逐层重放。黑衣 49 的双侧脸在三源投影时出现，灰衣 32 的眼鼻斑驳在同源体素合并时出现，佳佳 P3 会进一步放大碎影。DA3 为历史对照。本次逐帧图表与复跑命令见 [SA 实验归档](../splatanything/EXPERIMENTS.md)，下一轮局部加权和局部深度对齐见 [当前计划](EXPERIMENT_PLAN.md)。

## 2026-11-12 峰哥改动后回归验证：v2 全量重跑 + v1/v2 A/B（判定：改动零效果差）

**背景**：峰哥对 da3_human 做了一轮改动（29l/30 系列评测与几何审计工具、08 日志格式等），
要求按 v1 同口径把 55 包人像数据全量重跑 + 新评测，验证改动效果。
产物：`~/output/da3_portrait_batch_v2`（v1 = `~/output/da3_portrait_batch`）。

**管线**（与 v1 完全同口径）：21 批量（32 视角/504p/first/不裁剪+全SH/快速导出 + orbit 视频）
→ 31 批量 30 号 FF 多视角自一致评测（v1 补跑同口径评测做 A/B）。

**结果**：

| 检查项 | v1 | v2 | 判定 |
|---|---|---|---|
| 包数 / 失败 | 55 ok | 55 ok | ✅ |
| 图片数（逐包） | — | 55/55 与 v1 一致 | ✅ |
| PLY 大小（逐包） | 954M×55 | 954M×55 全一致 | ✅ |
| 高斯数（PLY 头） | 6,096,384 | 55/55 = 6,096,384 | ✅ |
| 30 号评测（31 批量） | 55 报告 0 失败 | 55 报告 0 失败 | ✅ |
| PSNR mean | 12.158 | 12.158（Δmean −0.0001，max\|Δ\| 0.003） | ✅ 逐包等价 |
| FG PSNR / SSIM / LPIPS / sharp_ratio | 11.894 / 0.300 / 0.545 / 0.221 | 同左（55/55 包 Δ<0.05） | ✅ 逐包等价 |
| 抽帧目视（2 包同位置帧） | — | v1\|v2 观感无差异 | ✅ |

**结论**：本轮改动**未触碰推理→导出→渲染主链路**——55 包产物规格逐包一致，
定量指标在浮点抖动量级（max 0.003 dB）内等价，视觉无差异。改动可安全合入。

**v2 指标分布**（FF 自一致口径，n=55）：PSNR mean 12.16 / median 12.19 / min 10.48 / max 15.09；
FG PSNR 11.89；SSIM 0.300；LPIPS 0.545；sharp_ratio 0.221；hole_pct ≈ 0.05%。
最低三包：22e33fb8(10.48)、804efb01(10.51)、747b7c74(10.59)；最高：451729fd(15.09)。

**工具**：`32_compare_v1v2.py`（产物规格回归比对）、`33_ab_metrics.py`（评测指标 A/B）、
`31_batch_eval.sh`（批量 30 号评测，断点续跑）。顺手修 21 号 n_pts 提取：
08 新日志「📦 PLY N 点」致 v2 汇总 gs_points 列空，grep 改 `PLY[：: ]+\K[0-9,]+` 兼容新旧（5863290）。

---

## 2026-11-11 晚：视角数分桶 + 多包基线摸底（训练数据集就绪前的纯推理测试）

工具：`30_run_viewcount_bench.sh`（08 前馈导出 + 30 通用多视角评测，断点续跑）、
`30_ff_multiview_eval.py`（任意视角数：PSNR/SSIM/LPIPS/锐度/前景 PSNR/空洞率 + Phase 0
方位角登记）、`30b_make_bench_summary.py`（PPT 汇总图，图像不入 git）。
产物：`~/output/da3_results/_bench_viewcount/`（WSL）/ `D:\output\results\bench_viewcount_view\`（目检副本，含 ppt\）。

### A. 视角数分桶（同一 32v 采集包「佳佳抱枕」，REF_STRATEGY=first，N_VIEWS=3/8/13/18）

| 桶 | PSNR mean | PSNR min | FG PSNR | SSIM | LPIPS | 空洞率 | 高斯数 | 渲染 |
|---|---|---|---|---|---|---|---|---|
| 3v | 15.38 | 10.43 | 15.25 | 0.402 | 0.458 | 1.26% | 0.57M | 0.8s |
| 8v | 13.32 | **10.79** | 13.24 | 0.300 | 0.528 | 0.66% | 1.52M | 1.2s |
| 13v | 12.66 | 10.22 | 12.67 | 0.265 | 0.537 | 0.07% | 2.48M | 1.4s |
| 18v | 12.38 | 10.35 | 12.52 | 0.256 | 0.547 | 0.04% | 3.43M | 1.7s |

- **mean 下降**是参考视角占比被稀释 + 新增视角离参考更远，非直接质量对比口径；
- **min 停在 10.4~10.8 dB 平台不升**：跨视角几何不一致不随视角数收敛（结构 confirms E4）；
- LPIPS/SSIM 随桶增大同步变差（新增视角自身渲染更难）；
- 空洞率随视角数大幅下降（多视角层叠加覆盖更满）——覆盖不是问题，质量才是；
- 高斯数 ≈ 190,512/层 严格线性（3v 571,536 = 3×190,512 复验），点数随视角数线性膨胀，
  D3-F 融合去重的收益空间随桶增大。

### B. 多包基线摸底（7 个真实包，全 32 视角，同一正式口径）

| 包 | PSNR mean/min | FG PSNR | SSIM | LPIPS | 空洞率 | 相邻角距最大° | 方位角跨度° |
|---|---|---|---|---|---|---|---|
| 女生写字 | 10.63 / 8.05 | 9.31 | 0.316 | 0.550 | 0.00% | 155 | 204 |
| 三人聚餐 | 11.24 / 8.51 | 10.21 | 0.226 | 0.584 | 0.01% | 89 | 147 |
| 男生二郎腿 | 11.68 / 9.73 | 11.60 | 0.246 | 0.585 | 0.01% | 166 | 190 |
| 佳佳抱枕_gs | 12.06 / 10.15 | 12.23 | 0.238 | 0.549 | 0.00% | 148 | 290 |
| 男生手机 | 12.40 / 9.97 | 12.49 | 0.284 | 0.529 | 0.00% | 77 | 216 |
| 佳佳蹲 | 12.90 / 10.45 | 12.64 | 0.251 | 0.492 | 0.01% | 115 | 301 |
| 男生室内 | 13.74 / 11.54 | 13.66 | 0.443 | 0.529 | 0.00% | 154 | 235 |

- 7/7 包复现同一病因：每包最差 1~2 个视角 PSNR 仅 8~11.5 dB（崩坏视角实样见 ppt_vc_2，
  各包最差视角方位角各不相同 → 系统性问题，非固定坏帧）；
- 空洞率全部 ≤0.01%：多视角叠加 + 不裁剪口径下覆盖无虞；
- **相邻角距最大 77~166°**：采集轨迹不均匀（折返/大缺口），方位角跨度 147~301° —
  Phase 0 登记必须逐包记录方位角，不能假设均匀环绕；训练集按桶抽样时要把轨迹形状纳入分层；
- 评测协议补全：SSIM/LPIPS 已并入 30 号评测脚本（lpips alex，querysplat env 可用）。

### 结论

视角数 scaling 与多包摸底共同确认：**E4 的 center 跨视角错位是跨包、跨桶稳定复现的
结构性病因**，加视角/换包均不可自愈。训练目标分布按 EXPERIMENT_PLAN Phase 2 覆盖
3/8/13/18 桶；数据集就绪后直进 G0。

---

## 2026-11-11 E4：几何与遮挡审计——center 跨视角错位实锤，Phase 1 收口

> 触发：EXPERIMENT_PLAN.md E4 行——区分 center 错位 / scale / rotation / 遮挡融合错误。
> 工具：`29p_e4_geometry_audit.py`（WSL）+ `29q_e4_summary_images.py`（Windows 配图）。
> 产物：`~/output/da3_results/佳佳抱枕_3v/e4_geometry/`（e4_report.json + PNG），
> 目检副本 `D:/output/results/e4_geometry_view/`（含 ppt/ 四张汇报图）。

### 关键手法：PLY 按层切回逐视角

32v 高斯数 6,096,384 = 32×190,512、3v 571,536 = 3×190,512 → PLY 是**逐视角预测层等块拼接**。
先用 3×3 在框率矩阵验证归属（行=层、列=渲染视角，%）：

| 层＼视角 | v0 | v1 | v2 |
|---|---|---|---|
| L0（v0 预测） | **93.0** | 33.8 | 14.3 |
| L1（v1 预测） | 57.4 | **42.9** | 12.1 |
| L2（v2 预测） | 37.8 | 28.7 | **44.7** |

对角占优（60.2% vs 非对角 30.7%），切块成立。⚠️ 在框率未到 90+ 是因为各层只覆盖本视角
视锥内的场景，跨视角天然大量出框——这正是后面重投影审计的定量版本。

### 四个假说逐项判决

**① center 错位（跨视角几何不一致）——实锤。**

- 逐层渲染分解：对角线（L_k 渲 v_k）PSNR 19.53/10.08/13.04 且人物区 α 量 0.83~0.95；
  **非对角全部 ≤8.7 dB**——每个视角的人物由本层自己贡献，跨层内容互不匹配。
- v0 选出 314,638 个人物高斯（全层、投影进 v0 人物 mask 且与预测深度一致）：
  投到 **v1 在框仅 37.7%**（其余 62.3% 出框或在相机后方），**v2 在框仅 14.1%**；
  在框部分的质心与人物 mask 质心位移 83.0px（v1）/ 119.2px（v2）。
- 人物高斯 3D 散点（俯视/侧视）：红绿蓝三簇**明显分开**，每视角各自预测了一份"人物表面"。

**② 遮挡融合错误——排除。** 融合权重没有分错层（本层人物 α 量 0.83~0.95 占主导）；
错的是各层几何本身不重合，任何权重都救不回。

**③ scale 过大（独立病因）——E3 已排除。** σ 封顶 1.5px 即塌覆盖（人物空洞 63~81%）。

**④ rotation 不匹配——非主因。** 各向异性短轴 vs 预测深度法向 |cos| 中位
0.40~0.54、≥0.7 仅 23.5~33.2% —— 前馈高斯本就非表面定向，旋转不含可靠几何信息，
"调 rotation"没有可修的抓手。

### 判读边界

人物 mask 与法向均来自 DA3 自身预测深度（非独立 GT）："命中"只代表自洽；
但**出框率是与 mask 无关的几何事实**——表面点在三张图里都可见，62~86% 投出画面只能
解释为跨层 3D 位置不重合。

### Phase 1 收口

E1~E4 全部完成，停止条件四条满足 → **冻结 DA3-Baseline-V1**（官方权重 + 不裁剪全 SH 导出，
基线数字：571,536 高斯 / v0 23.08 · v1 10.32 · v2 12.83 dB）。下一步 Phase 2：
**G0 官方几何头训练口径复刻 → G1 跨视角几何一致性（重投影 + visibility-gated 监督）**。

---

## 2026-11-11 E3：scale 原始值 vs 受限值 一次性因果对照（负结果，判 B）

> 触发：EXPERIMENT_PLAN.md Phase 1 收口——只允许一次 scale 因果对照，验收"远视角锐度提升
> 且空洞/颗粒不恶化"才保留，否则归因回几何不一致。
> 工具：`29n_e3_scale_clamp_ablation.py`（WSL querysplat 双臂渲染统计）+
> `29o_e3_summary_images.py`（Windows 汇报配图）。
> 产物：`~/output/da3_results/佳佳抱枕_3v/e3_scale_clamp/`（e3_report.json + 逐视角 PNG），
> 目检副本 `D:/output/results/e3_scale_view/`（含 ppt/ 四张汇报图）。

### 对照设计（预注册，不看结果调参）

- 固定输入：`gs_max/0000.ply`（571,536 高斯，不裁剪+全 SH 正式口径）+ 同批 cameras.npz/mask/GT，378×504 评测。
- 唯一改动：三轴 scale 封顶 `s'_i = min(s_i, 1.5 px × z_min / fx)`；`z_min` = 该高斯在 3 个输入相机中的**最近深度**（保证任意输入视角投影 σ≤1.5 px，fx=366.7）。center/rotation/opacity/SH/相机/渲染器/底色全部不动。

### 实测（base 臂与 D1 数字逐位复现：PSNR 23.08/10.32/12.83，σ 2.68/3.06/4.34 ✓）

| 臂 | σ 中位/视角 | PSNR v0/v1/v2 | 锐度(全图) v0/v1/v2 | 人物空洞率 v0/v1/v2 | 人物α中位 v0/v1/v2 |
|---|---|---|---|---|---|
| 原始 | 2.68 / 3.06 / 4.34 px | 23.08 / 10.32 / 12.83 | 348 / 172 / 77 | 0.01 / 4.30 / 0.00% | 0.990 / 0.984 / 0.990 |
| σ≤1.5px | 0.70 / 0.72 / 1.37 px | 8.87 / 6.31 / 7.90 | **910 / 856 / 1205** | **62.9 / 71.0 / 81.5%** | 0.399 / 0.263 / 0.199 |

- 487,101 / 571,536（**85.2%**）高斯被钳制；最坏视角 σ 分位 p50/p90/p99 = 6.28/36.2/320.1 → 1.5/1.5/1.5 px。
- 钳制臂背景空洞（α<0.05）v1/v2 = 24.1%/28.3%（原始 0%）；渲染耗时 0.3s（原始 1.3s，点变碎反而更快）。
- **锐度 ×3~6 全部是颗粒假象**：Laplacian 对空洞边缘和噪点同样敏感；人物区 α 中位从 0.96~0.99 崩到 0.20~0.40，PSNR 掉 8~14 dB。

### 判决

**判 B（负结果）**：未通过 E3 验收（空洞/颗粒灾难性恶化），scale 受限**不保留**。
糊不是独立的"scale 过大"问题——大 σ 高斯数量上是少数（ worst-view σ>6 px 约占一半，
但 >36 px 的尾巴只有 10%）却**承担覆盖**；压 σ 想不塌覆盖，必须同步 densify/重定位，
即只能走 D3 几何头重训。导出/后处理侧对糊无解，Phase 1 不再扫 scale/SH/opacity/fusion。

### Phase 1 停止条件核对

| 条件 | 状态 |
|---|---|
| ① 白洞 = 导出裁剪 | ✓ D2 闭环（2×2 消融，空洞 34.8%→0.7%） |
| ② 灰 = 跨视角外观退化 | ✓ E2 闭环（人物 α 0.96~0.99，饱和度 v2 −71%，非覆盖问题） |
| ③ 糊收敛到具体假说 | ✓ D1 衰减曲线 + D5 位姿排除（GT 位姿重渲几乎不变）+ E3 scale 排除（钳制即塌覆盖）→ 锁定**跨视角几何不一致（center/遮挡层叠）** |
| ④ 导出参数收益到顶 | ✓ 四档 2×2 + E3 钳制均为边际为负 |

E3 已用掉唯一一次 scale 对照额度。**Phase 1 只剩 E4 几何输出审计，随后冻结 DA3-Baseline-V1、进入 Phase 2（G0 官方复刻）。**

---

## 2026-11-11 D1 诊断包：灰/糊/白洞归因（三视角，双口径）

> 触发：峰哥观察 3v 对比（five_way_arc 视频 DA3-GS 面板：偏灰、模糊、背景白洞）。
> 工具：`29_run_d1_diag.sh`（08 四档导出 + `29_d1_diag_3view.py` 双口径渲染统计）。
> 产物：`~/output/da3_results/佳佳抱枕_3v/d1_diag/`（PNG + diag_report.json），
> 目检副本 `D:/output/results/d1_diag_view/`。日志 `~/output/batch_4methods/d1_diag_佳佳抱枕_3v.log`。

### 链路事实（先纠正认知）

| 发现 | 含义 |
|---|---|
| 峰哥本次 3v 批量跑的是 da3_human 02→03 链（[0,40,81] 抽帧），`infer_gs=0` **没落前馈高斯** | 峰哥看到的"DA3-GS"不是这条链的 03 精化产物 |
| `04_model_3dgs` 30k 精化的 train renders：PSNR 55~59 dB、锐度比 93~95%，颜色与 GT 逐位一致 | 三图下 30k 步是**极度过拟合**，训练视角上完美——不是灰/糊来源 |
| 3v 对比评估的对象是 compare_4methods 前馈直出 PLY（[0,7,31] 输入，self3 11.48 dB / 锐度比 4%） | 灰/糊/白洞全部来自**前馈直出口径** |
| PLAN_REVIEW 里 `90a_run_3view_jiajia.sh` 路径**不存在**（笔误），3v 对比实际入口 = batch_4methods 评估 + orbit 视频 | 已订正 |

### 实测（输入 [0,40,81]，378×504 统一评测口径）

**耗时**：模型加载 87s + 纯前向 2.0s + 四档导出 0.3s；3v 高斯 571,536（57 万）；显存峰值 7.58 GB；gsplat 渲染每口径 3 视角 ≈1s。

**三口径对照**（α=渲染累计透明度）：

| 口径 | 点 | α 中位 | 背景空洞率 | v0(参考) PSNR | v1 PSNR | v2 PSNR | σ 折像素 |
|---|---|---|---|---|---|---|---|
| FF_max（不裁剪+全 SH，正式口径） | 571,536 | 0.101 | **0.72%** | 23.08 | 10.33 | 12.93 | 2.7~4.5 |
| FF_official（裁剪+DC，历史口径） | 451,583 | 0.104 | **34.79%** | 15.88 | 8.42 | 11.50 | 2.3~4.1 |
| FT_30k（30k 精化，官方渲染器） | 758,698 | — | — | 59.44 | 57.51 | 55.81 | — |

**饱和度（HSV S 均值，人物区）**：GT v0/v1/v2 = 39.2 / 26.7 / 24.9；FF_max 渲染 = 34.1 / 21.8 / **7.2**（v2 衰减 71%）；FF_official 更低（32.6 / 18.2 / 8.4）。亮度：FF_max v1 人物偏暗 ~25 灰阶，v2 偏亮 ~9。

### 归因结论（三条，按确定性排序）

1. **白洞 = 导出裁剪（已闭环）**。官方默认导出 `prune_by_depth_percent=0.9`（按逐视角深度剪掉最远 10%）+ 边框裁剪，把远背景整片剪掉；3 视角只有 3 层叠加（α 中位 0.10，32 视角版是 1÷32 压 opacity 靠 32 层叠满），被剪区域直接透出底色。**不裁剪（max 档）后背景空洞 34.8% → 0.7%**。机制与 32v 归位结论（I-01 不裁剪 +1.99 dB）同源，三视角下效应放大 48 倍。
2. **糊 = σ 大 + 跨视角几何不一致（与 32v 同曲线）**。参考视角 23.08 dB，v1 掉到 10.33（三视角基线 2.2 m、夹角中位 99°，几何不一致被拉满）；σ 折像素 2.7~4.5；锐度比 21%/14%/9%。三视角下"离参考视角越远越烂"的衰减比 32v 更陡——这不是导出能修的，是高斯头/几何问题。
3. **灰 = 半透明叠加 + 饱和度衰减（叠加效应）**。① 3 层 α 中位 0.10，画面靠底色和高斯半透明色叠加 → 发灰；② 饱和度随离参考视角距离衰减（v2 人物 -71%），机制与跨视角 SH/外观预测不一致同源；③ 裁剪档进一步掉饱和度（v0 背景 67.6→35.5）。

**对 D2~D4 的指向**：白洞已闭环（导出改参数即可，零成本）；糊和灰都指向**前馈头本身**（三视角下跨视角一致性与 opacity/饱和度参数化），支持 D3（改前馈高斯头 opacity/尺度参数化，重训）优先于 D4 立项。精化路线（03）在三图下是训练视角过拟合器，holdout 仍崩（FINDINGS §8），与峰哥"前馈直出"目标无关。

### D1-补：v1/v2 "人物偏出画面"排查 → 位姿排除，实锤前馈几何（2026-11-11 晚）

> 触发：峰哥目检 ppt_2 提出"DA3 前馈直出 v1/v2 人物偏到画面外，是不是位姿出错？"
> 工具：`29c~29k_pose_step*.py`（核查链）。全部产物在 `d1_diag/final_{pred,gt}_v*.png`。

**判决实验链**（每一步都排掉一个假说）：

| 步骤 | 假说 | 结果 | 结论 |
|---|---|---|---|
| 29c/d | 预测位姿 vs transforms.json 逐视角比 | 两两中心夹角 GT/预测 87.8:88.1°、122.1:122.1°、150.1:149.9° 几乎重合 | 相机位置/顺序没错 |
| 29i | cameras.json rotation 是 w2c 还是 c2w | c2w 约定渲染 25.3/26.8/27.8 dB vs w2c 5.3/8.1/27.8 dB | **cameras.json rotation=c2w**；D1 诊断里 FT_30k 本脚本 gsplat 重渲值（5~28 dB）是我解释错约定，官方 55~59 dB 无误 |
| 29h | 相对位姿对比（约定/镜像/居中全部消掉） | v0→v1 旋转差 0.8°、平移方向差 0.7°；v0→v2 1.0°/0.8° | **DA3 预测位姿 ≈ GT（<1.1°）** |
| 29j | PLY 各视角高斯在自身相机系（未做 c2w 融合） | 分块施加 c2w 后全面恶化（v0 23.1→18.0） | 排除；PLY 确实在 ref/world 系 |
| 29k | 终审渲染：FF_max PLY × GT 位姿（Umeyama 对齐，残差 2~4 mm，s=0.504，无镜像） | GT 位姿 v1/v2 = 10.36/12.88 dB，与预测位姿 10.33/12.93 **几乎相同**；v0 反而降（23.1→19.1） | **换成 GT 位姿也救不回人物出框 → 位姿不是原因** |

**最终归因**：人物高斯的 3D 位置沿参考视角射线方向错了——从 v0 投影仍正确（深度错不挪像素，v0 有 23 dB），从 v1/v2 看整体位移出框。这是 FINDINGS "质量随离参考视角三维距离单调衰减"的极端形态（3v 基线 88°/122°，比 32v 稀疏一个量级）。**糊/灰/人物出框同源于前馈几何跨视角不一致，全部指向 D3（重训前馈高斯头）**。

**排查中修掉的两个工具 bug**（坑过后人）：
- Umeyama 行/列约定混用（`Q=U@Vt` 是行映射，套列向量必须转置）——不修则对齐残差假大 0.8~1.6 m，镜像结论也是它的 artifacts（det 实际 +1.000，无镜像）；
- 该数据包 transforms.json 的 `transform_matrix` 约定 ≠ som_human 标准包（cameras.json 侧是 c2w），跨数据包勿直接套 `w2c=inv(M@flip)`。

### D1 补充（2026-11-11 晚，评审闭环）：导出 2×2 消融 + 灰归因修正

评审意见：白洞对照同时改了裁剪和 SH，独立贡献未拆开；α=0.10 是全图中位，不能证明人物发灰。补跑 29m（四档同相机同渲染器同底色，数据 `d1_diag/ablation2x2.json`）：

| 档 | 裁剪 | SH | 背景空洞率 | v0 PSNR | 人物 α 中位（v0/v1/v2） |
|---|---|---|---|---|---|
| official | ✓ | DC | 24.73% | 15.88 | 0.976 / 0.964 / 0.986 |
| full_sh | ✓ | 全SH | **24.73%**（逐位相同） | 16.38 | 同上 |
| noclip | ✗ | DC | **0.00%** | 20.81 | 0.990 / 0.984 / 0.990 |
| max | ✗ | 全SH | **0.00%** | 23.08 | 同上 |

**① 白洞完全闭环**：SH 对空洞率零贡献（两档点数相同、α 逐位相同），**裁剪是唯一因素**。评审建议的 2×2 判读（B≈D 成立）：不裁剪即恢复，SH 无关。

**② 灰归因修正（重要，推翻 D1 初版双因素之一）**：人物前景 α 中位四档全部 0.96~0.99——**人物区不存在 α 叠不满**。此前"α 中位 0.10"是全图口径（被背景稀释），不能用于人物。人物区发灰的唯一来源是**视角相关外观误差**：饱和度 render vs GT = v0 34.1/39.2、v1 21.8/26.7（轻度），**v2 7.3/24.9（−71%）**，且全 SH 也救不回 v2（7.0~9.5）。灰与糊同根：前馈外观分支对远离参考视角的颜色预测失效。D3 立项依据由"opacity 参数化"改为"**跨视角外观/几何一致性**"。

**③ 输入视角分布补记**：三帧（0/40/81）实际方位角 **−176.2° / +12.8° / +112.2°**，间距 189.0° / 99.4°（近对跖、极不均匀）。v1 距参考视角 v0 达 189°，比 32v 工况极端一个量级，"离参考越远越烂"衰减曲线拉满，v1 PSNR 垫底（8.4 dB）与此一致。

**④ 本脚本两个口径坑（已修，入 ERRATA #10）**：cameras.npz 的 extrinsics 是 w2c 直接用（勿求逆）；其 intrinsics/depth 已是 378×504 口径（勿再 ×0.5）。两处都静默出"看似合理"的错图。

---

## 2026-11-10 批量人像验收：55 包 Remy 数据全量 DA3 高斯 + orbit 视频

> 触发：峰哥指示把 `D:\dataset\人像验收数据汇总-含佳佳`（55 个 Remy 采集包）全量跑
> DA3 出高斯渲染视频。管线 `20_convert_heic.py` → `08_bench_official_gs.py`（脚本 21 串三级）。

### 结果汇总（`~/output/da3_portrait_batch/_summary.tsv`）

| 指标 | 数值 |
|---|---|
| 完成 / 总数 | **55 / 55，零失败** |
| 输入图片 | 54~218 张/包（heic → jpg） |
| 高斯点数 | 全部 6,096,384（32 视角 × 504p 固定） |
| 纯前向 | 中位 **7.5 s**/包（min 5.7 / max 252.0，max 为首批 GPU 预热） |
| 产物/包 | PLY 954 MB + orbit.mp4 2~6 MB（540×720，30fps，240 帧） |
| 汇总状态 | 全部 `ok` |

### 质量抽查（3 包 × orbit_frame_120 目视）

| 包 | 场景 | 判定 |
|---|---|---|
| 451729fd | 室外草地侧卧人像 | ✅ 全身完整、构图正常 |
| d86393e8_input | 室内站立人像（203 图） | ✅ 清晰、背景陈设完整 |
| f5cf3ba0 | 室外双人合影 | ✅ 双人完整无鬼影；⚠️ 面部偏软（504p 上限，已知） |

**结论**：手持开弧采集（实测方位空洞 170°+）下 ARC 模式轨道稳定，
无拖影/空洞/结构崩坏；短板仍是 504p 决定的面部软，属模型上限非管线 bug。

### 产物位置（WSL）

- 高斯+相机：`~/output/da3_portrait_batch/<包名>/{gs_max/0000.ply, cameras.npz}`
- 视频：`~/output/da3_portrait_batch/<包名>/videos/orbit.mp4`
- 逐包日志 `_logs/`、汇总 `_summary.tsv`

### 工程记录

- 全量运行 8h（含 heic 转换与首批模型冷加载），断点续跑三级缓存均命中设计预期。
- 汇总表 `gs_points`/`fwd_s` 两列初跑漏捕获，后从 `_logs/` 正则补全（55/55）。
- 复跑入口：`bash 21_batch_portrait.sh <数据集目录> <输出目录> [单包名]`

---

## 2026-11-09 第一阶段达成：官方 DA3-GS「秒级 + 20.23 dB」正式口径

> 峰哥定的规矩：**先把官方效果完整复现（速度秒级、PSNR 足够高），在 bug 上的改进都是浪费**。
> 本轮清掉 5 个 bug，其中 4 个一直压着之前的全部数字。

### 一、被修掉的 5 个 bug

| # | bug | 位置 | 代价（实测） |
|---|---|---|---|
| 1 | **把 125 视角全喂给 DA3** | `02_run_da3_inference.sh` / 采样逻辑 | 前向 133.9 s vs **6.6 s**（视角 3.9×，耗时 20×）；2381 万高斯 vs 610 万，质量还更低 |
| 2 | 用官方**默认导出**（裁边 8/256 + 剪最远 10% 深度 + 只存 DC） | `save_gaussian_ply` 入参 | 白丢 **2.51 dB** |
| 3 | `07_eval` 的 `load_gt` 假设 `sorted(images)[vi]` 是第 vi 视角 | `07_eval_da3_gs.py` | 抽帧后 GT 整体错位 → PSNR 掉到 10 dB |
| 4 | NVS 评估额外乘 `sf32/sf125` | `12_eval_nvs_first.py` | PLY 与 extrinsics **各自已是米制**，再乘等于把 2.18% 的估计误差当真实尺度差 → −2.29 dB（17.94 vs 20.23） |
| 5 | 官方 `export_ply` 用 `list(map(tuple, ...))` 构造 610 万行 × 41 列 | `gsply_helpers.py:105` | 953 MB 写 **28.3 s**；改 numpy 单次 `tofile` 后 **1.8 s**（15.7×） |

**第 5 条的验证方式**（同一次推理、同一份 gaussians，两条导出路径逐字段比对）：
41 列 `max|Δ| = 0`，属性名与顺序完全一致（`14_verify_export.py`）。
跨文件比对曾有 x/y/z 3 cm 差异，追查后确认是**两次独立推理的浮点抖动**
（三次推理 sf 分别 2.1204 / 2.1203 / 2.1193），与导出实现无关。

**第 4 条尤其值得记**：DA3 的 `_apply_depth_alignment()` 已经把 depth 和
`extrinsics[:,:,:3,3]` 折成米制，只有 `gaussians.means/scales` 漏乘（ERRATA E-01）。
所以「PLY 乘 sf」之后，PLY 与 npz 相机**已经同尺度**，任何额外缩放都是错的。

### 二、正式口径（32 视角 / 504p / ref=first / 不裁剪 + 全 SH）

| 阶段 | 耗时（RTX 3090） | 备注 |
|---|---|---|
| 模型加载（热） | 19.0 s | 冷启动 45~110 s，首次含权重换页 |
| **前向（含 GSDPT 头）** | **6.6 s** | 显存峰值 13.0 GB |
| **导出（610 万点 / 953 MB / 全 SH）** | **1.9 s** | 快速导出 |
| 端到端（热） | **27.5 s** | 纯计算 8.5 s |

对照峰哥引用的 `compare_4methods` 13.5 s（= 前向 6.2 + 导出 6.8，32 视角 484 万点 DC 档）：
**本轮 6.6 + 1.9 = 8.5 s，比它快，且质量高 2.5 dB**（DC 档 → 全 SH 是 +2.51）。

### 三、质量（三种口径）

| 口径 | 定义 | PSNR | SSIM |
|---|---|---|---|
| in-sample 504p | 渲染那 32 个输入视角，504p | 21.34 | 0.7635 |
| **in-sample 1440×1920** | 同上，交付分辨率 | **20.45** | 0.7533 |
| **NVS（32→125）** | 32 视角输入，渲染**全部 125 视角** | **20.23** | — |
| ├ seen(32) | 输入过的视角 | 20.27 | — |
| └ **unseen(93)** | **完全没见过的视角** | **20.21** | — |

**seen ≈ unseen（差 0.06 dB）** 是 DA3 最有价值的性质：它生成的是真三维高斯，
不是逐视角贴图。这也说明 20.23 这个数字没有靠"背输入"作弊。

### 四、SH 旋转问题：用 `REF_STRATEGY=first` 绕过

NVS 要把 32 视角的高斯渲染到 125 视角，两次推理必须共用一个世界系。
DA3 默认按 `saddle_balanced` 选参考视角，两次选中的是不同帧 →
世界系相差 **83.6°**，而 SH 与方向绑定，不转 SH 白丢 **1.2 dB**
（DC-only 档只掉 0.90，全 SH 档掉 2.08，差值即 SH 项）。

改用官方参数 `ref_view_strategy=first`（强制用第 0 张定义世界系）后实测：

```
s=0.9985 | R 与 I 夹角 0.100° | 相机中心残差 mean 2.53 mm / max 5.51 mm
```

⇒ 只剩均匀缩放，**方向不变，SH 原样可用**，不需要实现 SH 旋转。
效果：NVS 从 18.37（COLMAP 口径 + 未转 SH）→ **20.23**。

### 五、分辨率扫描：+0.83 dB，收益偏弱

⚠️ **跨分辨率不能混用相机**：DA3 在不同 `process_res` 下估计的 FOV 不同
（504p `fx/W=0.9780` → FOV 54.2°；672p `0.9566` → 55.3°，差 2.2%）。
拿 504p 的相机渲染 672p 的高斯，实测掉 **2.87 dB**（20.23 → 17.36），
这是视锥不匹配，不是质量下降。**必须用各自分辨率自己的相机做 in-sample**。

| process_res | 高斯数 | 前向 | 显存峰值 | PLY | in-sample 1440×1920 |
|---|---|---|---|---|---|
| 504 | 610 万 | 6.6 s | 13.0 GB | 953 MB | **20.45** |
| 672 | 1084 万 | 12.5 s | 18.0 GB | 1695 MB | **21.28**（+0.83） |

⇒ 分辨率是**弱杠杆**：付 1.9× 时间 / 1.4× 显存 / 1.8× 体积，只换 0.83 dB。
这也侧面说明瓶颈不在「每像素一个高斯」的密度，而在这套前馈输出本身的上限。
（1008p 需 ≈40 GB 显存，24 GB 卡跑不动，已排除。）

### 六、复现

```bash
# 32 视角推理 + 四档导出（FAST_EXPORT=1 默认；VERIFY=1 做导出等价性比对）
DA3_MODEL_DIR=/mnt/d/wheel/da3_human_ms \
IMAGES_DIR=/home/velvet/output/vggt_human_results/source/images \
OUT_DIR=/home/velvet/output/da3_human_results/bench_first_32v_fast \
PROCESS_RES=504 N_VIEWS=32 REF_STRATEGY=first EXPORT=max \
python da3_human/08_bench_official_gs.py

# 125 视角相机（只为 NVS 提供渲染视角，不用它的高斯）
#   → PROCESS_RES=504 REF_STRATEGY=first EXPORT=none （前向 133.9 s，一次性）

# NVS 评估
bash da3_human/13_run_nvs_first.sh /home/velvet/output/da3_human_results/bench_first_32v_fast

# 分辨率自洽对照
bash da3_human/16_eval_resolution.sh bench_first_32v_fast bench_first_32v_672
```

### 七、判定与下一步

| 项 | 判定 |
|---|---|
| 官方 DA3-GS 复现 | ✅ **达成**：8.5 s 纯计算 / 20.23 dB NVS，无 bug |
| 「125 视角输入」 | ❌ 废弃，改用 32 视角 |
| 「默认导出」 | ❌ 废弃（丢 2.51 dB） |
| 06 双分支 2DGS | ❌ 保持终止 |
| 分辨率到 672p | ⚠️ 可选（+0.83 dB，1.9× 代价）；1008p 显存不够 |

**关键认知**：20.23 dB 是**前馈单次前向**的上限，没有 test-time 优化。
04 的 25.505 / 03 的 27.573 是 30K 步优化后的结果，不是同一类方法，不可直接比。
要往上走只有两条路：① 前馈之后接短程优化（P1 甜点：5K 步 + densify 到 23.79，7.7 min）；
② 用人物数据微调 backbone（LoRA，见下条讨论）。

---

## 2026-11-09 C-06 终止判定：densify 才是分水岭（P1 甜点在本场景复现）

> 起因：峰哥质疑「PSNR 15.2、1 小时，DA3 高斯头本来是秒级」。
> 复核后确认 **06 是双输**（比零训练的官方直出还低 2.7 dB，却慢 4.7 倍），
> 根因不是范式（前馈 vs 优化），而是 06 丢掉了 3DGS 的 densify。

### 一、决定性对照：最终点数几乎相同，PSNR 差 8.58 dB

**实验**：`04_run_da3_gs_baseline.sh` + `MAX_INIT_POINTS=20000 ITERATIONS=5000`
（= `compare_4methods` P1 ⑤ 的「DA3 稀疏点云 + 短步数 + 开 densify」在本场景的严格复现）

| 配置 | init 点 | **最终点** | 步数 | 训练耗时 | PSNR ↑ | SSIM ↑ | LPIPS ↓ |
|---|---|---|---|---|---|---|---|
| 官方 DA3-GS 直出 | — | 1891 万 | 0 | 203 s 前向 | 17.93 | 0.7104 | — |
| 06 v3 anchor + 2DGS | 683,879 | **683,879**（不动） | 30K | 3,779 s | 15.209 | 0.6489 | — |
| **P1 DA3稀疏 + 3DGS** | **18,057** | **616,420**（densify 长大） | **5K** | **461 s** | **23.79** | **0.8200** | 0.2570 |
| 04 DA3点云 + 3DGS | 459,433 | ~500 万 | 30K | ~3,600 s | 25.505 | 0.8613 | 0.1870 |
| 03 COLMAP + 3DGS | 7,163 | — | 30K | 2,751 s | 27.573 | 0.8854 | 0.1642 |

**这张表的核心是第 2、3 行**：最终点数 68.4 万 vs 61.6 万（几乎相同），
PSNR 15.21 vs 23.79（差 **+8.58 dB**），训练耗时 63 min vs **7.7 min**（快 8.2 倍）。
点数是同一量级，唯一的结构差别是「点从哪来」：

- **densify**：点在梯度大（细节/轮廓）处自动 split/clone，在 opacity<0.005 处 prune；
  scale 与位置自由学习，贴合真实表面 → 61.6 万点全是有用的。
- **voxel 硬分**：7.2 mm 均匀网格，与几何无关；scale 初值锁在 voxel 尺寸；
  位置被 ±1px/±2% 的 offset 界焊死 → 68.4 万点里大量是均匀铺开、无处可动的"占位符"。

⇒ **densify 决定的是「点的质量分布」，不是「点的数量」**。这是本次最可复用的结论。

### 二、06 的四个可定位缺陷（全在 `06_train_da3_2dgs.py`）

| # | 缺陷 | 位置 | 后果 |
|---|---|---|---|
| 1 | **完全没有 densify** | `M = len(xyz)` (L106)，五个参数全是 `(M,·)` | 30K 步一个点都没长，容量锁死 68.4 万 |
| 2 | 有界 offset 把错误初值焊死 | `bound_cam` L119-121（±1px/±2%），`LR_GEO=1e-4` L164 | B-04b 已证 DA3 深度噪声是主因，此处明令禁止修正 |
| 3 | 每 iter 从磁盘读图 | `load_image(img_paths[vi])` L217 | 1440×1920 JPG 逐帧 `cv2.imread` 解码，无缓存 |
| 4 | 每 iter 两次全量光栅化 | L225（全分辨率）+ L236（1/4） | 单步成本翻倍 |

**但 06 的单步并不慢**：04 是 120 ms/it，06 是 126 ms/it，基本持平。
所以「慢」不是实现低效，而是 **30K 步全在做无用功**——63 分钟换来比零训练低 2.7 dB 的结果。

### 三、2DGS 不是元凶（避免错误归因）

`compare_4methods` I-03 实测：同一 PLY、同一组相机、**只换光栅器**，全图仅 −0.29 dB。
所以这 10.3 dB 的差距（04 3DGS 25.5 vs 06 2DGS 15.2）**不能记在 2DGS 账上**，
只能记在「关 densify + 锁几何」上。2DGS 仍保留一个独立用途：
`median_depth` 掠射一致性 0.41% vs 3DGS 0.80%，做**几何**时单独用，不做外观主链。

### 四、判定

| 项 | 判定 |
|---|---|
| C-06 双分支 2DGS 作为外观主链 | ❌ **终止**（负结果封存，不再投入） |
| 「DA3 几何 → 稀疏化 → 3DGS + densify」 | ✅ **确认为主链** |
| 05 anchor 点云（683,879） | ⚠️ 降级为「几何分析/初始化候选」，不再是训练终点 |
| 盘面法向诊断结论 | 保留（identity 代价 4.55 dB 成立；「朝源相机」修法不成立） |

### 五、复现

```bash
# 结果：PSNR 23.79 / SSIM 0.8200 / LPIPS 0.2570，端到端 13m26s（训练 7m41s）
ITERATIONS=5000 MAX_INIT_POINTS=20000 \
SCENE_DIR=$RESULTS_DIR/da3_colmap_scene_p1 \
GAUSSIAN_DIR=$RESULTS_DIR/p1_da3sparse_5k \
bash da3_human/04_run_da3_gs_baseline.sh
```

**下一步候选**（待定，按性价比）：
1. 扫 `MAX_INIT_POINTS` ∈ {20k, 50k, 200k} × `ITERATIONS` {5k, 10k}，找本场景的性价比拐点
   （04 的 46 万点 + 30K = 25.505，说明还有 1.7 dB 空间）；
2. 把 `07_eval_da3_gs.py` 的 sweep 能力接过来，做**留出视角** NVS 口径（当前 125 视角全 in-sample）。

---

## 2026-11-09 官方 DA3-GS 基线复现 + C-06 盘面法向诊断

> 依据峰哥「先复现官方效果，再以它为唯一参照」的规矩。**本场景的官方参照不是 22.11**：
> compare_4methods 的 22.11 出自 `佳佳抱枕_p1m`（32 视角），studio_human 是 125 视角开弧采集，
> 两者不可混用（ERRATA E6 同理）。

### 一、官方 DA3-GS 在本场景的实测基线 = 17.93 dB

| 项 | 值 |
|---|---|
| 推理 | `INFER_GS=1 EXPORT_FORMAT=mini_npz-gs_ply`，125 视角 504p，前向 203s（另有 87s 加载） |
| 内存高斯 | **23,814,000**（GSDPT 头一次前向直出） |
| 导出 PLY | 1.2GB / **18,911,929** 点（官方默认剪枝 `prune_by_depth_percent=0.9` + `prune_border_gs=True`） |
| PLY 字段 | 17 个，**含 nx/ny/nz 法向**（DA3 自带法向输出） |
| SH | 内存 `harmonics (1,N,3,9)` → **degree 2**；默认导出 `save_sh_dc_only=True` 只剩 DC |
| opacity | logit 中位 −4.39 → **α 中位 0.0122**（远比 compare_4methods 的 0.036 低） |
| **scale_factor** | **1.8666**（DA3 `is_metric=1`；gaussians 同样未乘，E-01 复现） |

**尺度口径验证（ERRATA E2 增益扫掠）**：手扫 sf ∈ {1.0,1.2,1.5,1.8,2.0,2.2,2.5}，
PSNR 为 11.72 / 11.79 / 13.25 / **16.88** / 15.27 / 14.00 / 13.20 → **峰位 1.8**，
与官方内部 `scale_factor=1.8666` 一致（粗档落在 1.8~2.0 之间）。不折算会白丢 ~5 dB。

**正式评估（全 SH + 米制，1891 万点，125 视角全量）**：

| 口径 | PSNR | SSIM | per-view p10 / p90 | min / max |
|---|---|---|---|---|
| 504p（官方口径） | **18.11** | 0.6696 | 17.0 / 19.6 | 11.6 / 20.4 |
| 1440x1920（交付口径） | **17.93** | 0.7104 | 16.9 / 19.4 | 11.5 / 20.1 |

**与既有基线对照（统一 1440×1920 / 125 视角 / in-sample）**：

| 配置 | 点数 | PSNR ↑ | SSIM ↑ | 备注 |
|---|---|---|---|---|
| 03 COLMAP + 3DGS 30K | 7,163 init | **27.573** | **0.8854** | 有 COLMAP BA |
| 04 DA3 点云 + 3DGS 30K | 459k init | 25.505 | 0.8613 | 有 30K 训练 |
| **官方 DA3-GS 直出** | **1891 万** | **17.93** | 0.7104 | **零训练** |
| 06 v3 anchor + 2DGS 30K | 684k | 15.209 | 0.6489 | 有 30K 训练 |

⇒ **两个必须记住的事实**：
1. **官方 DA3-GS 在本场景只有 17.93 dB，比 03 低 9.6 dB**。它"效果好"是 32 视角数据集的结论，
   在 125 视角开弧场景（闭合差 1.42 m、视角跨度大）不成立——这与「DA3 质量随相机到参考视角
   距离单调衰减」一致，125 视角把远端推得更远。
2. **06 v3 距官方直出仅 2.7 dB**（15.21 vs 17.93），而不是之前误以为的 6 dB。
   追赶目标应是 17.93（先不亏），再谈 27.57。

### 二、C-06 盘面法向诊断（`quats` 初始化是否锁死世界 +Z）

**背景**：06 把 `quats` 初始化为 identity，gsplat 2DGS 的盘法向 = 旋转矩阵第三列（ERRATA E10），
identity → 全部法向朝世界 +Z。DA3 视角绕人一圈，若法向锁在 +Z，多数视角是掠射。

**实测（v3 checkpoint，683,879 高斯，4 视角零训练对照渲染）**：

| 指标 / 配置 | 值 |
|---|---|
| identity quat 残留占比 | **0.0000**（训练后全动过） |
| cos(盘法线, 源相机方向) | mean −0.252 / **med −0.382** / p10 −0.898 / p90 +0.656 |
| \|cos\|>0.9（正对相机）占比 | **13.6%** |
| \|cos\|<0.3（掠射）占比 | **26.2%** |
| cos(盘法线, 世界+Z) | med **+0.868**，\|cos\|>0.9 占 **46.2%** |

| quats 配置（其余参数不变） | PSNR(4 视角) | per-view | alpha mean |
|---|---|---|---|
| as-is（训练后） | **16.21** | [20.5, 15.7, 12.9, 15.8] | 0.922 |
| identity（全朝世界+Z，= 初始状态） | **11.66** | [10.2, 14.8, 12.3, 9.4] | 0.672 |
| 盘面朝源相机 | 14.63 | [16.8, 14.9, 12.1, 14.7] | 0.946 |
| 仅掠射档(26.2%)改朝相机 | 15.41 | [18.0, 15.5, 12.7, 15.4] | 0.941 |

**结论（推测部分成立，修法不成立）**：
- ✅ **identity 初始化的代价是 4.55 dB**（11.66 → 训练后 16.21），这是真实损失，初始化确实有害；
- ❌ **但「盘面朝源相机」不是正解**（14.63 < 16.21，反而低 1.6 dB）。一个高斯被多视角共享，
  盘法向应对齐**多视角一致的表面法向**，而不是任何单个相机方向；
- ⚠️ v3 的 alpha mean 0.922 / \|α>0.5\| 0.959，**覆盖已满**，不是 v1 那种「半透明叠化」问题；
  真正的短板在几何与点数密度（684k vs 官方 1891 万）。

**下一步候选（按性价比排序）**：
1. **anchor 密度对齐官方**（68.4 万 → 数百万，体素 7.2mm → ~2mm）：官方 1891 万点，我们差 28 倍
   （表面积缩放 → 线性间距差 ~5.3 倍）。这是唯一"零算法风险"的追赶动作。
2. 用 **DA3 自带的 nx/ny/nz 法向**初始化 quats（PLY 里有，官方白送），替代 identity。
3. 有界 offset 的界值（当前 ±1px / ±2% 深度）在点数提上去后需重扫。

---

## 2026-11-05 阶段 C-06 v2/v3：梯度隔离消融——detach 是主凶，v3 去 detach 重跑

> ⛔ **路线已于 2026-11-09 终止**（见本篇开头「C-06 终止判定」）：去 detach 只解决梯度隔离，
> 但 densify 缺失才是量级问题。以下内容保留作负结果追溯。

### C-06 v2：PSNR 15.102 —— opacity 修复后仍被 detach 拖累

v1 修复（纹理分支 `opac.detach()` → 可导）后重跑 30K：**PSNR 15.102**（比 v1 仅 +0.23dB）。
训练速度反而恢复 3.5×（3,779s vs v1 12,372s），排除缓存膨胀因素，锁定**双分支梯度隔离设计本身**。
v2 checkpoint opacity 已恢复学习（mean 0.119→0.19、max 1.0），但 tex L1 轨迹与 v1 几乎一致——
系统卡在同一个 basin。

### 隔离消融（同 anchor 起点、8K iters、单视角循环，排除多视角随机性）

| 配置 | 关键设计 | PSNR | 说明 |
|---|---|---|---|
| **A** | 单分支无 detach + 正则（dist/norm） | **21.05** | 上限参考（30K 预计 +1~2dB） |
| B | detach 几何 + 10×lr + 正则 | 17.51 | detach 仍拖后腿 3.5dB |
| C | detach 几何 + 默认 lr + 正则 | 19.21 | lr 放大可救一部分 |
| D | **anchor + 有界 offset 参数化 + 正则，无 detach** | **20.65** | 与 A 相当，核心参数化成立 |
| **v2 原配置** | detach 几何 + detach 颜色 + 有界 offset | 15.10（30K） | detach 双重隔离是主因 |

**结论**：`detach`（梯度隔离）是主凶，与之前发现的 opacity 冻结同根源——纹理分支
detach 几何参数后，1/4 分辨率几何分支成了几何参数的唯一梯度来源，信号太弱/太噪。
D 配置证明：**anchor + 有界 offset 参数化 + 正则（无 detach）可接近 free-means 上限**。

### C-06 v3：去 detach + 双路共享梯度（运行中）

改动：纹理分支去掉 `means/quats/scales.detach()`，几何参数在纹理（全分辨率）和
几何（1/4 分辨率+正则）双路都有梯度。预期 30K PSNR ≥ 22（8K 消融 A/D 水平 + 收敛增益）。

- 产物：`da3_2dgs_v3/ours/`（v2 目录已归档 `da3_2dgs_v2_detach/`）
- 消融脚本：`/tmp/abl_06.py`（未入库，一次性诊断；口径与 06 评测完全一致）

---

## 2026-11-05 阶段 C-06：双分支 2DGS 首跑（负结果）与根因修复

> ⛔ **路线已于 2026-11-09 终止**。v1 的 opacity 冻结是真实 bug、v2/v3 的 detach 归因也成立，
> 但整条路线缺少 densify，最终点数锁死在 68.4 万 → 无法与官方 1891 万 / 03 基线 300~500 万竞争。

### C-06 v1：PSNR 14.877 —— 不透明度冻结 bug

| 项 | 值 |
|---|---|
| 配置 | 684k anchors（C-05）+ 有界 offset（±1px/±2%深度）双分支 2DGS，SH3，30K iters |
| 结果 | **PSNR 14.877 / SSIM 0.6347 / LPIPS 0.3962**（vs 03 基线 27.573） |
| 耗时 | 训练 12,372s（末段 ~412ms/it，前期 128ms/it，疑与缓存分配器膨胀+逐 iter 磁盘读图有关） |

**症状**：结构完全对齐（墙/桌/柜位置全对），但全图「叠化糊」——人脸帽檐水平拖影、
细纹理（藤编/发丝）全丢，像每像素叠了几十层半透明 splat。

**根因（诊断脚本实锤）**：`06_train_da3_2dgs.py` 纹理分支把 `opac.detach()` 传进
光栅化（原意只 detach 几何），几何分支也 detach —— **不透明度在两个分支都无梯度，
30K iters 后全体冻结在初始 σ(−2)=0.119**（checkpoint 实测 mean/median/p90/max 全 = 0.1192）。
低透明度 × 684k splats 重叠 → 逐像素多层 alpha 叠化 → 全局模糊。
对照实验：强制 opac=1 反而更差（11.90dB，背景层全挡死），证实是「叠化」而非「几何错」。

**修复**：纹理分支恢复 `opac` 可导（opacity 归纹理分支学），几何分支保持 detach；
训练结束新增 opacity 终态打印（std≈0 即告警）。v2 重跑进行中，结果随后补录。

---

## 2026-11-05 阶段 B-04b：内参假设隔离（负结果）

### 实验设计

04 落后 03 达 −2.07dB，三条假设排序后做隔离实验：DA3 点云/位姿不动，**仅把每相机内参按
图像名替换为 COLMAP 标定值**（cam0: DA3 1448.4/1387.2 → COLMAP 1389.6/1399.8），
区分「内参系统性偏差（假设1）」vs「深度噪声+densify 膨胀（假设2）」。

- 场景：`da3_colmap_scene_colmapintr/`（459,433 点，构建单独验证通过）
- 脚本：`build_da3_colmap_scene.py` 新增 `INTR_REF_DIR` 分支 + `04b_run_da3_gs_colmapintr.sh`

### 结果 ✅ —— **假设 1 排除，主因锁定深度噪声（假设 2）**

| 配置 | init | 内参 | PSNR ↑ | SSIM ↑ | LPIPS ↓ |
|---|---|---|---|---|---|
| 03 COLMAP 全套 | 7,163 | COLMAP | **27.573** | **0.8854** | **0.1642** |
| 04 DA3 全套 | 459,433 | DA3（fx≠fy 差4%） | 25.505 | 0.8613 | 0.1870 |
| **04b DA3几何+COLMAP内参** | 459,433 | COLMAP | **25.374**（−0.13dB） | 0.8564 | 0.1946 |

- 按预先声明的判定标准（回升 <0.5dB → 假设 2 成立）：换内参**无回升反而略降**，
  DA3 预测内参偏差**不是**性能差距主因，−2.07dB 主要来自**深度噪声经 densify 放大的
  过平滑/摊平**（与 som_human「更多高斯更易摊平」经验一致：04/04b 最终 ply 均 >03）。
- 推论：阶段 C 不必强依赖 COLMAP 内参（零成本论证不成立但也不是瓶颈）；**核心矛盾在
  点云质量**——anchor 一致性筛选 + 有界 offset（D1） 的权重上调。
- 副产物：`pycolmap.Reconstruction()` 要求 cameras/images/points3D 三文件齐（只拷
  cameras/images 会读失败）；参考模型是 `.txt` 格式（预检已兼容）。

### 阶段 B 小结（03/04/04b 三配置收束）

| 配置 | 点云来源 | 位姿 | 内参 | PSNR |
|---|---|---|---|---|
| 03 | COLMAP sparse | COLMAP | COLMAP | **27.573** |
| 04 | DA3 反投影 | DA3 | DA3 | 25.505 |
| 04b | DA3 反投影 | DA3 | COLMAP | 25.374 |

**结论：在 per-scene 优化框架内，DA3 几何直出替换 COLMAP 当前不成立（−2.1~−2.2dB），
差距由深度噪声主导而非内参。** 这正是阶段 C（anchor 一致性筛选 + 双分支 2DGS +
有界 offset）要解决的问题——若 C 阶段能追平/反超 03，即证明「筛选+参数化」路线有效；
若不能，DA3 侧需升级到带 BA 精化的几何（退出本计划范围，回 BACKLOG 讨论）。

## 2026-11-05 阶段 B-03/04 基线训练

### 阶段 B-03：原版 3DGS 基线（COLMAP init）✅

| 项 | 结果 |
|---|---|
| 初始化 | COLMAP sparse **7,163 点**（vggt_human source，即现有 scene_gs 路径） |
| 训练 | 30,000 iters，全分辨率 1440×1920，**45m51s**（RTX 3090） |
| 训练内评估 | [ITER 30000] PSNR **27.10**（官方 train.py 日志口径） |
| 渲染后评估 | train 视角 125/125：PSNR **27.573** / SSIM **0.8854** / LPIPS **0.1642** |
| 产物 | `~/output/da3_human_results/baseline_3dgs_colmap/`（point_cloud/iteration_30000、train/ours_30000/{renders,gt}） |

定性观察（floater/重影对照现象）待截图补入——需与 04（DA3 init）同视角对比后统一记录。

**metrics.py 全 nan 根因（"Metric evaluation progress: 0it"）**：
1. 本仓 `metrics.py`（vggt_human 定制版）只扫 `test/` 分割，而 render.py 渲染的是 train 视角
   → 建软链 `test/ours_<iter> -> ../train/ours_<iter>` 解决；
2. ⚠️ 坑中坑：**render.py 会预先给 test split 建空的 `renders/`、`gt/` 目录**，直接
   `ln -sfn` 会把链接塞进该目录内部（`test/ours_30000/ours_30000 -> ...`），metrics 仍读空目录。
   必须先 `rmdir` 空目录再软链。`_train_3dgs.sh` 已内置该逻辑（含 rmdir），04 起自动生效。

### 阶段 C-05：anchor 点云构建 ✅

| 项 | 结果 |
|---|---|
| 输入 | 02 的 DA3 导出（深度 125×504×378 @504p，conf/extrinsics/intrinsics） |
| 筛选链 | conf≥P40（阈值 1.49）→ 8 最近邻视角深度一致性（τ_rel=3% + τ_abs=5cm，≥3 视角一致）→ 跨视角体素去重（voxel=0.004×中值深度=**7.2mm**） |
| 数量 | 反投影 **14,334,402** → 一致性后 **13,841,023**（保留 96.6%）→ 去重后 **683,879** anchors |
| 产物 | `~/output/da3_human_results/anchor_pointcloud/studio_human/`（anchors.npz / anchors.ply / anchor_summary.json） |

**观察**：
- 与 02 colmap 导出点数（14.29M）吻合，反投影链一致 ✓。
- 一致性保留率 96.6% 偏高：DA3 深度与位姿同一网络产出、天然自洽，不一致集中在遮挡边界
  （view 124 边缘视角 60986→7033 即典型）。τ 目前是宽松档，D 阶段消融扫 τ 时再收紧验证。
- anchor 密度 68 万 surfel 规模适中（原版 3DGS 03 基线 30K iters 后约 300~500 万高斯），
  双分支 2DGS 训练量可控。

### 阶段 B-04：DA3-GS 基线（DA3 点云初始化）✅ —— **负结果：朴素 DA3 替换全面落后 COLMAP**

| 项 | 结果 |
|---|---|
| 初始化 | DA3 深度反投影体素降采样 **459,433 点**（conf≥P40）+ DA3 自估位姿/内参 |
| 训练 | 30,000 iters，全分辨率，训练段 ~60min（densification 膨胀：iter 7000 已 ~5M 高斯 / ply 339MB） |
| 训练内评估 | [ITER 30000] PSNR **25.26** |
| 渲染后评估 | train 125/125：PSNR **25.505** / SSIM **0.8613** / LPIPS **0.1870** |
| 最终 ply | 383MB（03 对照 298MB） |

**03 vs 04 对照总表**（口径一致：train 视角 in-sample，30K iters 全分辨率）：

| 基线 | init 点数 | PSNR ↑ | SSIM ↑ | LPIPS ↓ | ply |
|---|---|---|---|---|---|
| 03 COLMAP init | 7,163 | **27.573** | **0.8854** | **0.1642** | 298MB |
| 04 DA3 init | 459,433 | 25.505 (−2.07dB) | 0.8613 | 0.1870 | 383MB |

**负结果分析（三条假设，按嫌疑排序，均待验证）**：
1. **DA3 预测内参偏差**：04 内参 fx=1444.61 **≠** fy=1389.17（差 4%），而 COLMAP 标定 fy=1399.84
   （正方形像素相机）。内参系统性偏差 → 全视角 misprojection → 纹理对不齐（糊化）。
   → **最可疑**，且可零成本验证：04 场景换 COLMAP 内参重跑。
2. **深度噪声 + densification 膨胀**：459k 含噪 init → 7000 iter 即 5M+ 高斯，
   噪声点被 densify 成半透明羽化 blob（ply 反大 28%，PSNR 反降，符合"越多高斯越摊平"的历史经验）。
3. **位姿一致性误差**：02 验收 Sim(3) 对齐 mean 4.9mm 是**对齐后**残差；04 用 DA3 原始坐标系，
   残差直接进入训练（但量级小，嫌疑低于 1/2）。

**定性观察**（`baseline_compare/compare_*.png`，view 0/30/60/90 三联）：
- in-sample 视角两基线都**无明显 floater/重影**——同事观察的跨视角不一致问题在
  in-sample 上不显著，NVS（留出视角）才能充分暴露。⚠️ 当前 125 视角全用于训练，
  **无留出测试视角**，NVS 口径待补（D 阶段或重跑带留出）。
- 04 的差距表现为**整体锐度下降**（面部/头发/货架细节糊化），与 LPIPS +0.023 一致；
  非局部坏点，更像全局对齐失配——支持假设 1。

**对核心方案的意义**：朴素「DA3 几何直接替换 COLMAP」在本场景**不成立**（−2.07dB）。
阶段 C 的有界 offset + 双分支 + pose residual（D5）动机更充分——不能只信任 DA3 几何，
要给它信任域内的修正能力。

---

## 2026-11-05 阶段 A/B 启动

### 执行前调研发现（修正计划前提）

1. **⚠️ DA3 权重许可矩阵（README Model Zoo 实测）——修正计划 §6 前提**：

   | 能力组合 | 可选模型 | 权重许可 |
   |---|---|---|
   | 相对深度 + 位姿 | DA3-BASE(0.12B) / DA3-SMALL(0.08B) | Apache 2.0 |
   | 度量深度（单目，无位姿） | DA3METRIC-LARGE(0.35B) | Apache 2.0 |
   | **度量深度 + 位姿一体** | **DA3NESTED-GIANT-LARGE-1.1(1.40B) / DA3-GIANT-1.1(1.15B)** | **CC BY-NC 4.0** |

   - 计划 §7 说「depth-anything/Depth-Anything-3 (Apache 2.0)」——仅**代码**许可成立；
     计划 §6「商用 pose-free 管线（DA3 Apache 2.0 解许可）」在**权重层面不成立**：
     度量+位姿组合无 Apache 选项。商用化路径只剩：BASE/SMALL（相对深度+位姿，尺度靠 COLMAP/BA）或等官方出 Apache 度量模型。
   - **本期选择**：内部验证用 `DA3NESTED-GIANT-LARGE-1.1`（质量最高：README 称 DA3-LARGE 已可比 VGGT，NESTED 为 any-view giant + metric 嵌套），CC BY-NC 仅内部研究可接受。

2. **数据确认**（计划 §2 标注「待确认」已落实）：
   - vggt_human 棚拍 = **单场景，125 视角，1440×1920 JPG**，COLMAP sparse/0 在位（`~/output/vggt_human_results/source/`）；`/mnt/d/output/results/vggt_human_results` 为同场景 08 搬运产物。
   - 满足计划数据要求（≥4 视角、有内参、静态）；主体人物基本静态（vggt_human/04 注释），动态重影风险低。
   - 局限：只有 1 个场景，D 阶段敏感性实验无跨场景重复性检验（记为已知限制）。

3. **环境/网络实测（2026-11-05）**：
   - `huggingface.co` 经 v2rayN(10808) API 200 但 7s；`hf-mirror.com` 复活（2.4s）→ **下载走镜像**（10-28 记录镜像已挂，网络确会漂移，每次复测）。
   - `download.pytorch.org` 目录 403 属正常；torch 走 tuna（torch 2.14.1 拉 cu13 nvidia 依赖 ~2.5GB，~2.3MB/s）。
   - DA3 官方仓已在 `~/repos/Depth-Anything-3`（HEAD 4173623），跳过 clone。
   - 模型导出格式支持 `-` 组合（如 `mini_npz-colmap-depth_vis`）；`colmap` 导出把 cameras/images/points3D.bin 写在 export_dir 根（需整理成 `sparse/0` 给原版 3DGS）。

### 阶段 A：环境搭建 ✅

| 项 | 结果 |
|---|---|
| conda env | `da3_human`，python 3.10.22（`00a_setup_env.sh` 一键建） |
| torch | 2.14.1+cu130（tuna；拉 cu13 nvidia 依赖 ~2.5GB） |
| xformers | 0.0.35 ⚠️ 二进制是 against torch 2.10+cu128 编的，C++/CUDA 扩展加载失败 |
| DA3 本体 | editable 0.0.0（`~/repos/Depth-Anything-3` HEAD 4173623） |
| 其他关键 | numpy 1.26.4（requirements 钉 numpy<2）、pycolmap 4.2.1、open3d 0.20.0 |
| 耗时 | 装包 ~44 min（tuna ~1.5MB/s） |

**xformers 不配对的影响评估**：DA3 attention 走 `F.scaled_dot_product_attention`
（PyTorch 原生，model/utils/attention.py:57），xformers 只用于 SwiGLU 融合 MLP
（Triton 算子，运行时编译，不依赖预编译 C++ 扩展）。**预计不影响推理**，由 02 实测最终验证；
若 SwiGLU 报错，回退方案 = torch 降到 2.10.0 对齐 xformers。

装包坑：pip 对 opencv-python 回退解析多轮（moviepy 依赖冲突），最终 opencv 4.11.0.86 + moviepy 1.0.3；
numpy 被 requirements 的 `numpy<2` 从 2.2.6 降级到 1.26.4（torchvision 装的是 2.x，无冲突）。

### 阶段 B-01：权重下载 ✅

| 项 | 结果 |
|---|---|
| 模型 | `depth-anything/DA3NESTED-GIANT-LARGE-1.1` → `/mnt/d/wheel/da3_human_ms` |
| 产物 | model.safetensors **6.3GB**（fp32 1.4B）+ config.json |
| 通道 | hf-mirror.com（直连 7s vs 镜像 2.4s；镜像吞吐 ~1MB/s，全程 63 min） |
| 复跑 | `bash da3_human/01_download_models.sh`（幂等，本地已存在则秒过） |

### 阶段 B-02：DA3 推理（125 视角）✅

| 验收项 | 结果 | 判定 |
|---|---|---|
| 深度可视化合理 | median **1.797m**，范围 0.48~6.10m，NaN=0（度量输出，棚拍尺度合理） | ✅ |
| 位姿无漂移 | vs COLMAP Sim(3) 对齐后 mean **4.9mm** / p95 8.2mm / max 10.1mm，125/125 视角匹配，轨迹逐段重合（viz/trajectory_3d.png） | ✅ |
| 耗时 | 模型加载 18s + 前向 34~54s + 导出 92s ≈ **2.5 min**（504p，RTX 3090） | — |

产物（`~/output/da3_human_results/da3_inference/studio_human/`）：
`exports/{mini_npz,npz}/results.npz`（深度/置信度/extrinsics/intrinsics）、
colmap 导出（cameras/images/points3D/frames/rigs .bin，**14,288,402 点**、conf≥P40）、
`depth_vis/` 逐视角深度可视化、`da3_summary.json`、`viz/` 轨迹对比图。

**关键数字解读**：
- Sim(3) scale=0.5077：DA3 度量重建比 COLMAP 坐标系大约 1.97×（COLMAP 尺度本就任意，无问题）。
- 首末相机距 1.42m ≠ 漂移：轨迹是带起始小回环的开弧（viz 图证实），非闭环采集。

**踩坑记录**：
1. DA3 `requirements.txt` 漏声明 `addict`（da3.py:19）→ 00 脚本已补 `pip install addict`。
2. COLMAP `images.txt` 解析：points2D 空行（无 2D 点的注册图像）不能过滤，否则 header 配对
   错位约一半匹配不上（首跑 63/125 → 修后 125/125）。
3. xformers 0.0.35/torch 2.14.1 不配对：推理正常（attention 走 F.sdpa；日志出现
   "using MLP layer as FFN" 说明 SwiGLU 融合层自动回退 MLP），**无实际影响**，不回退 torch。

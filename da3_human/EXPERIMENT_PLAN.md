# da3_human 实验总计划

> **唯一执行入口。** 当前目标是 DA3 人像前馈高斯直出质量；不做逐场景优化，不把
> InfiniSplat 的短训或 3000 步精调当成产品链路。实验结果按日期倒序写入
> [`EXPERIMENTS.md`](EXPERIMENTS.md)，推理/坐标/导出事故写入 [`ERRATA.md`](ERRATA.md)。
>
> 旧文档的分工：
> - [`PLAN_REVIEW_3VIEW.md`](PLAN_REVIEW_3VIEW.md)：2026-11-11 三视角方案的决策审查记录，保留历史取舍，不作为第二份主计划。
> - [`28_da3_gt_feedforward_plan.md`](28_da3_gt_feedforward_plan.md)：DA3-GT 的长篇技术设计附录，接口和训练细节以本计划的当前决策为准。
> - `EXPERIMENTS.md` 中 2026-11-09 以前的 per-scene/2DGS 内容是已终止路线的实验史，不重新启动。

## 0. WorkBuddy 执行任务单（2026-11-12，当前优先入口）

**分工：Codex 负责实验设计、证据审查与路线建议；WorkBuddy 负责复现、编写必要工具、
运行实验、保存原始产物和回填报告。** 用户正在用 WorkBuddy 复现 SplatAnything，
先接续已有工作，不重复下载/建环境，不覆盖正在跑的输出，不同时启动争抢显存的任务。
本轮只回答三个问题：①SplatAnything 是否达到迁移门槛；②DA3 是否需要重训几何分支；
③SplatAnything 做属性头微调的实际工程/算力成本。不是启动完整训练工程。

本文是唯一执行入口；[路线评审附录](SPLATANYTHING_ROUTE_REVIEW.md)提供矩阵和门槛细节；
WorkBuddy 回填 [路线验证报告](../splatanything/ROUTE_VALIDATION_REPORT.md)，实验事实分别追加到
`splatanything/EXPERIMENTS.md` / `da3_human/EXPERIMENTS.md`。不要只在聊天里报结论。
任务编号 WB 与附录 S 编号不是两套实验：WB-0/1=S0，WB-2=S1，WB-3=S2，WB-4=S3 成本探测。

### 0.1 执行纪律与报告方式

1. 先读根 AGENTS、MEMORY、最近日志、SplatAnything README_wsl/NOTES/EXPERIMENTS；
   涉及 DA3 再读 ERRATA。重新核对当前代码、权重、输出和在跑进程，历史记录不代替现场状态。
2. WB-0→WB-1 先通过。随后按 WB-2 小样本→WB-3→WB-4→WB-2 扩展/确认执行；
   不必等 UE 才做 WB-3 的非 GT 对照及 WB-4。某阶段缺数据，只挂起依赖它的判定，继续独立工作。
3. 在本任务单范围内自行完成必要工具和验证，每阶段记录状态与下一步，不需逐条等待确认。
   WB-4 最多做下述 100 step 冒烟；2k/5k 域训练、全骨干解冻、encoder 蒸馏留待报告评审后排期。
4. 状态只能填 `TODO / RUNNING / PASS / FAIL / BLOCKED / INCONCLUSIVE`。程序退出 0
   不等于质量 PASS；缺 GT、缺对照或 CI 不确定填 INCONCLUSIVE，不以主观观感填“达标”。
5. 每阶段提交：实际命令（完整输入/模型/输出）、git commit/补丁、权重哈希、日志、JSON/CSV、
   固定视角图片位置和一句判定。下一会话据报告恢复，不反复从零调研。日志与脚本同 commit，
   只提交自己的文件，检查 staged diff，不 push；生成图片/视频/权重不入库。
6. 结果保存到 WSL `~/output/splatanything_route_validation/<run_id>/`，按
   `wb0_inventory/wb1_smoke/wb2_compare/wb3_centers/wb4_train_probe` 分目录，run_id 唯一。
   提交报告必须填真实绝对路径，不能只留下此模板。新脚本先查编号占用，勿覆盖其他会话脚本。

### WB-0：接续复现，冻结输入与环境（无模型质量结论）

- 盘点现有权重是否完整及 SHA256、实际加载路径、官方源码 commit/本地补丁、conda/torch/
  CUDA/gsplat 版本、GPU 空闲显存。不要把文件大小匹配当 SHA256 校验通过。
- 复用 `splatanything/07_run_capture.sh` 转换、`02_run_inference.sh` 推理和正在开发的 03 渲染。
  **实际读代码再用**：02 当前环境变量是 INPUT/MODE/RESOLUTION/OUT_ROOT，03 是 RESULT/OUT；
  03 目前是输入视角自一致评测，WB-2 的 held-out 入口尚需实现/验收，不能拿 MAX_VIEWS 当留出。
- 核对 02 是否真的将 `SPLATANYTHING_MODEL_DIR` 指向的权重用于加载（当前依赖官方 checkpoints
  软链或显式 CLI）；记录 symlink 最终目标，避免“日志打印 A、实际加载 B”。记录实际模式 flags，
  不只信 wrapper 末尾的摘要。nvidia-smi 整卡采样不能替代单进程 allocated/reserved 峰值。
- 从既有真实数据选 6 包，冻结来源、身份分组、静态/运动标签、3/8/13v 嵌套帧号、固定参考帧、
  留出帧和 mask/脸框来源。先选包后看结果；原始数据只读，转换输出写新目录。
- 产物：`inventory.json`、`manifest.json`、`commands.sh`。PASS=能解释实际模型/数据/模式，
  源与目标帧无交叉；无可靠留出帧时明确标记 self-only，仍可进入 WB-1。

### WB-1：真权重推理、导出、相机和渲染验收

- 先一个官方小例，再一个静态人像包 3v@504，成功后 8v/13v。先别上 24/32v@756。
  官方小例若抽子集需明确帧号，只算链路冒烟，不宣称复现论文数字。
- 检查裁剪后人物头脚覆盖、RGB/K 分辨率、c2w/w2c、四元数顺序、log scale/opacity 激活、
  SH/颜色空间，以及 spacing/pose alignment 因子；保存每个变换，验证正向/逆向投影闭环。
- 同一批合并后高斯在“内存张量”和“导出 PLY 再载入”两条路径渲染：像素值 [0,1] 的
  RGB MAE≤1e-4、alpha MAE≤1e-4 为初始容差；超限先查 dtype/量化/渲染设置，不能直接放宽。
  输入质量低不等同于导出失败，二者分表记录。
- 产物：`io_audit.json`、`render_roundtrip.json`、原图/预处理图/投影叠图、RGB/alpha、
  分段耗时及 GPU 峰值。PASS=口径闭环且真权重运行；它只授权进入质量比较，不代表方法达标。

### WB-2：SplatAnything 是否值得切换（先小试，后正式判定）

- 先选定的 2 包×3v，确认 DA3 与 SplatAnything 共用输入和评测工具；再扩展到 6 包×
  {3,8,13}×{RGB-only,pose}。同分辨率控制面板共 72 个基础重建，已满足 manifest 的产物可复用。
- 统一源帧、FOV/裁剪、评测相机、渲染器/eps2d/底色/目标尺寸；DA3 保留不裁剪+全 SH，
  SplatAnything 保留原生 DC/合并。504 控制面板与各自默认分辨率面板分开，不混报增益。
- **实现真正的 held-out**：source-only 输入目录；target RGB 不进模型，不用于尺度/位姿拟合；
  只用 source 相机求全局对齐，再渲染 target。GT 不从模型输入 rgb/ 目录误取。
  无 UE/静态真实留出数据时只交自一致诊断表，迁移判定保持 INCONCLUSIVE。
- 按场景/身份聚合，报人物和脸 PSNR/LPIPS、锐度/颜色、几何/轮廓、空洞/泄漏、速度/显存/点数；
  分开 3v 桶、最差四分位和失败包，不能删失败项。独立测试集留到配置锁定后使用。
- 判据直接使用附录 §6：人物 LPIPS 相对降≥10%、PSNR 增≥1dB，改善 CI 不跨 0；
  三视角/脸/几何/覆盖不突破退化容差；同预算延迟≤DA3 1.5×、目标配置峰值建议≤22GiB。
  只部分模式/视角通过则限定适用范围；质量通过、训练未验证时只能说“推理基线候选”。
- 产物：`per_view_metrics.csv`、`paired_scene_metrics.csv`、`gate_results.json`（每项阈值、
  实测、PASS/FAIL/N/A）、固定对照图。空洞/泄漏定义沿用附录，缺人物分离不得拿背景 alpha 代替。

### WB-3：DA3 是几何先验错，还是高斯头 offset 错

- 从同次 DA3 前向缓存 raw depth、预测/输入 K/T、raw GS 与正式 Gaussians，冻结其它条件；
  跑附录 C0 原生、C1 xy=0、C2 depth offset=0、C3 两者=0、C4 仅换 means 保留原属性。
  **不要改输出通道布局，不要把 Nested 已缩放 depth 和未缩放高斯混在一起。**
- 先原坏例，再至少另 2 个独立静态包；以可见对应区域评重投影/轮廓/held-out，
  不能用整层非对角 PSNR 或自然遮挡造成的出框率判几何必错。
- UE 到位后，比较原生 depth 反投影与 GT；加 GT-center oracle 只诊断，不能参与正式模型排名。
  GT radial depth 先转 axial Z。scale/rotation 与新中心可能不匹配，因此 oracle 无改善
  不能单独否定几何病因；同时看直接点/深度/重投影误差。
- 判读：C3/C4 改善 + 原生 depth 几何可靠 → 优先固定中心属性头，**不重训几何骨干**；
  原生 depth 在有效对应上也明显错误 + GT-center/直接几何证据支持 → 几何修正候选；
  几何修正先训 depth head/小适配器，只有容量验证失败才考虑解冻 backbone。
  两边都错则记混合病因；offset 清零没用不等于已证明需要重训 backbone。
- 初始因果信号门槛：相对 C0，C3/C4 人物重投影中位误差降≥20%，且 held-out PSNR
  增≥0.5dB 或 LPIPS 降≥5%，同时覆盖/脸不突破附录守门。这是“值得试固定中心训练”的门槛，
  不是一般几何真值判据；没有 GT 时结论必须标暂定。
- 产物：`center_ablation.csv`、逐像素 offset/center 位移统计、可见性 mask、原生/GT 深度误差、
  三类结论之一：`固定中心优先 / 几何修正候选 / 证据不足`，附反证和结论适用数据范围。

### WB-4：量出训练改造成本（先可导验证，再最多 100 step）

**无需等大规模数据集**；WB-1 通过后就可准备。优先利用可信 UE 小场景；暂缺 UE 可用静态
真实包验证反传/过拟合能力，但不能把这项收益当泛化，也不进入产品 per-scene 优化链。

1. 列出缺口：移除学生 no_grad/inference_mode、可导光栅化、权重加载、source/target 数据集、
   loss、优化器、断点恢复、日志。教师保持冻结；不复制官方整套源码进 media_code，采用
   可复现的适配层/补丁，记录基底 commit。
2. 在任何更新前，训练模式前向与冻结推理前向对照，复用 WB-1 容差；差异先解释再反传。
   先冻结 geometry/encoder/centers，训练 GSFeatHead+decoder。保留原生 GSNorm、合并和过滤。
3. 执行 1 个 optimizer step：检查参与最终渲染的 scale/rotation/SH/merge weight 分支
   梯度 finite、参与计算的可训练参数有梯度且确实更新；教师参数不变。未被损失使用的原
   opacity 通道可能无梯度，应解释为合并规则结果，不能用它误判整个训练链坏了。
   若仅在禁用 merge/过滤后反传成功，明确记“替代路径通过，官方等价路径未通过”。
4. 连续最多 100 step，batch=1、3v、504 长边、固定 source/target 与初始权重；记录 step 0/10/100
   的 loss/渲染/参数 delta、峰值显存和分段时间。初始冒烟要求固定批次渲染 MSE 较 step 0
   降≥10%、无 NaN/Inf，且不是通过关闭渲染/减少点数实现。未达到先诊断，不自动无限加步数。
5. 若冻结 encoder 仍 OOM，可另测“缓存冻结特征/教师几何”的等价路径：缓存生成时间/体积也计入
   成本，确认缓存参数、帧组、crop 不变。不允许把 3v 降成 1v 后称原配置通过。
6. 保存并恢复一次模型+优化器+随机状态，确认恢复后可继续一步且数值有限；记录 trainable
   参数量、优化器状态大小、checkpoint 大小、前向/反向/merge/step 时间和 100-step 总 GPU 时间。
   只有 3v 实测则只估算 3v，8/13v 不能按视角数线性外推。

成本表必须把“已付出”和“预计剩余”分开：实际改动文件/模块、实现及排错工时（无法计时则
填未记录，不能推算）、依赖/编译次数、冷启动与稳态耗时、显存、缓存/存储、训练吞吐、
未解决阻塞。2k/5k 时间仅用实测稳态 step 时间作区间估算，加预处理/验证/存储开销，标明未跑。

训练可行性分级：`T0 仅推理` → `T1 官方等价路径单步反传` → `T2 100步+恢复通过`。
T2 只说明小规模属性头微调可行，**不代表全模型训练可行或人像泛化有效**。
单个阻塞做两种有依据的修复仍失败，就记录最小复现与下一项证据需求，停止该分支的环境试错；
其它独立实验继续。本轮不执行附录 2k/5k 训练，待 Codex 根据实测成本安排下一轮。

### WB-5：向 Codex 交付审查包（本轮终点）

回填 `splatanything/ROUTE_VALIDATION_REPORT.md`，结论逐项附证据：

| 问题 | 允许的结论 | 必需证据 |
|---|---|---|
| SplatAnything 是否达标 | 通过 / 限定模式通过 / 不通过 / 待数据 | WB-2 分桶指标、阈值、CI、固定图和失败清单 |
| DA3 是否重训几何 | 固定中心优先 / 几何修正候选 / 证据不足 | WB-3 C0～C4、直接几何误差及 oracle（有 GT 时） |
| SplatAnything 改造成本 | T0/T1/T2 + 已付出/剩余估计 | WB-4 参数变化、梯度、计时显存、断点恢复及缺口表 |

报告末尾只给一个下一步建议和支持/反对证据。Codex 再审查是否准入后续域训练，
不以“新方法更新”“能打开 PLY”或“一个包好看”替代决策。无需为了凑 PASS 一直跑。

## 1. 目标、边界与当前判断

### 1.1 目标

在保持前馈推理速度的前提下，改善 3～13 张稀疏人物图像生成的 3D Gaussian：

1. 远离参考视角时几何不出框、不漂移；
2. 颜色和饱和度跨视角稳定，减少灰化；
3. 降低尺度过大造成的模糊，并保留人脸细节；
4. 控制高斯数量和渲染延迟。

### 1.2 已冻结的边界

| 项 | 当前口径 |
|---|---|
| 推理链路 | DA3 前馈推理 → 高斯导出 → 直接渲染 |
| 不纳入目标链路 | per-scene 2DGS/3DGS 优化、InfiniSplat 短训、测试时位姿优化 |
| 导出 | **不裁剪 + 全 SH**；裁剪会把三视角背景空洞从 0.7% 放大到 34.8% |
| 参考视角 | 固定 `REF_STRATEGY`；跨运行比较不得让参考帧自动变化 |
| 分辨率 | 训练、相机、渲染必须使用同一 `PROCESS_RES`；当前官方口径为 504 |
| 评测 | 三视角真实数据只做输入视角自一致性；主要泛化结论用 UE 合成数据的 held-out 视角 |
| 区域指标 | 人脸框逐视角确定；固定框和全图 PSNR不能单独作为结论 |

### 1.3 当前病因判断（D1 已闭环）

- **白洞**：导出裁剪造成，已由 D2 零成本修复。
- **人物出框、模糊**：前馈高斯中心/尺度在跨视角时不一致；替换 GT 位姿几乎不改变 v1/v2 结果，不能继续把问题归因给相机位姿。
- **人物发灰**：人物前景 alpha 中位数为 0.96～0.99，问题不是前景覆盖不足，而是远离参考视角时外观/颜色预测失效；v2 饱和度相对 GT 下降约 71%，全 SH 也救不回。

因此当前方向仍是 **D3 跨视角 Gaussian Geometry 一致性**，人脸专门分支后置为 D4。
**2026-11-12 路线评审建议**：先执行下方 S0～S2 的 SplatAnything 对照与中心来源诊断，
再决定用哪一个基座训练；暂缓直接启动大规模 DA3 Geometry Head 重训。此建议尚无本仓
同口径 SplatAnything 人像质量结果支持，不能写成“已决定迁移”。

## 2. 评测协议

每个实验都要同时报告下面四组结果，禁止只报一个全图 PSNR：

| 组别 | 指标 | 用途 |
|---|---|---|
| 图像质量 | PSNR、SSIM、LPIPS、Laplacian 锐度、分频能量比 | 颜色和清晰度 |
| 人物/人脸 | 逐视角 mask 内 PSNR、锐度比、饱和度差 | 人像质量；框必须随视角重定 |
| 几何/覆盖 | 深度误差、Chamfer/F-score（有 GT 时）、alpha 空洞率、自由空间泄漏率 | 出框、白洞、漂浮点 |
| 泛化/效率 | 按输入视角数和视角距离分桶的 held-out 指标、高斯数、显存、前向/渲染时间 | 是否值得进入主线 |

真实三视角数据的 PSNR 只表示“输入视角重建自洽”，不能写成 NVS。合成 UE 数据要固定输入帧、参考帧和 held-out 相机，并按场景汇总中位数及离散度。

每次重建至少保存：实际输入 RGB/深度、人物 mask、alpha、Gaussian centers 及投影、opacity 分布、scale/rotation 的像素当量、RGB-GT 并排图、相机和视角覆盖审计。坐标约定、尺度因子和视角索引先用 `ERRATA.md` 的清单验收。

### Phase 0 数据登记

真实 Remy 包的原始序列、抽帧和角度必须单独记录：原始帧数、实际输入帧号、每帧相机中心/姿态、方位角和相机夹角。当前三视角样例不是均匀环：输入帧为 `[0,40,81]`，实际方位角约为 `-176.2°/+12.8°/+112.2°`，间距约 `189°/99.4°`。任何报告都不得把这三帧写成“均匀三视角”。

模型处理分辨率和评测分辨率分开登记：DA3 当前 `process_res=504`；不同方法最终统一到同一评测尺寸，GT、预测图、mask 和相机内参使用同一 resize/crop 变换。指标表中还要保留 Normal Error、Reprojection Error、Silhouette IoU、Boundary Error 和性能项，不能只留 RGB 指标。

## 3. 当前执行路线

### D0：基线冻结与数据审计（已完成，持续作为前置检查）

固定官方 DA3 权重、`PROCESS_RES=504`、参考策略、导出参数和渲染器。每次运行记录输入帧号、实际方位角、参考帧、相机/高斯尺度和产物路径。官方正式导出口径为“不裁剪 + 全 SH”。

### D1：灰/糊/白洞诊断（已完成）

使用 `29_run_d1_diag.sh`、`29_d1_diag_3view.py` 和 `29m_export_2x2_ablation.py` 完成：

- 四档导出 2×2 消融，确认白洞由裁剪单独造成，SH 不影响 alpha 空洞；
- 人物前景 alpha、饱和度和逐视角 PSNR 分离，修正“灰=前景 alpha 不足”的初版判断；
- GT 位姿重渲和相对位姿核查，排除位姿作为人物出框主因；
- 记录三输入视角的实际方位角和不均匀间距。

产物和数字见 `EXPERIMENTS.md` 的 2026-11-11 D1 节。

### D2：导出归位（已完成）

后续所有正式前馈结果统一使用不裁剪 + 全 SH。旧的裁剪+DC 结果只能作为历史对照，不能作为新实验 baseline。

### Phase 1 收口：E1～E4 可执行任务与停止条件

Phase 1 的职责是修复实现、导出和渲染口径，并把当前 DA3 的上限测清楚。它不负责通过连续扫参数“优化”出一个难以复现的版本。每个任务都必须固定输入、修改点和输出；完成停止条件后冻结 `DA3-Baseline-V1`，进入 Phase 2。

| 任务 | 固定输入 | 修改/检查文件 | 必须落盘的输出 | 关键指标与验收 | 状态 |
|---|---|---|---|---|---|
| E1 导出与背景覆盖 | 同一次 3v 前向、同一相机 | `08_bench_official_gs.py`、`29m_export_2x2_ablation.py` | 4 档 PLY、RGB/alpha、`ablation2x2.json` | 裁剪/SH 独立贡献；固定集背景空洞率 ≤1% 且记录高斯数/渲染耗时 | **完成** |
| E2 颜色与 alpha 诊断 | E1 的同一批 PLY、GT、人物/背景 mask | `29_d1_diag_3view.py`、颜色统计工具 | 前景/背景 RGB、HSV 饱和度、alpha 分布、误差图 | 明确灰化来自 alpha 透底、颜色管线还是视角相关外观；人物前景 alpha 不能用全图中位数代替 | **完成诊断** |
| E3 尺度与模糊诊断 | E1 正式 PLY、相机、逐视角 mask | `29n_e3_scale_clamp_ablation.py`（一次性 scale 因果对照） | 每个视角的 σ 折像素、2D footprint、人物轮廓锐度 | scale 改动只有在远视角锐度提升且空洞/颗粒不恶化时才保留；否则归因回几何不一致 | **完成（负结果，判 B：钳制即塌覆盖，糊归因回几何不一致）** |
| E4 几何与遮挡诊断 | E1 PLY、预测/GT 相机、UE mask/对应点 | `29p_e4_geometry_audit.py`（逐视角层切分+重投影审计） | center 投影、重投影误差、轮廓/遮挡对照、可见性统计 | 区分 center 错位、scale/rotation 不匹配和遮挡融合错误；Depth 只能作为辅助证据 | **完成（center 跨视角错位实锤；遮挡融合/scale/rotation 三项排除，Phase 1 收口）** |

Phase 1 停止条件：①白洞已经由导出策略解决；②灰化已经定位到 alpha/颜色管线或外观表达；③模糊/重影已经缩小到几何 center、scale、rotation 或遮挡中的具体假说；④在固定数据和口径下，继续改导出参数的收益已接近上限。E1～E4 已完成，Phase 1 已收口。不得继续无限扫 SH、opacity 或 fusion 阈值。

### Phase 1.5：视角数分桶与多包摸底（2026-11-11，数据集就绪前的纯推理测试，已完成）

训练数据集未就绪期间完成的推理/评测侧测试，结果全文见 `EXPERIMENTS.md` 2026-11-11 晚节：

- **视角数 scaling**（同一 32v 包，3/8/13/18 桶）：最差视角 PSNR 停在 10.4~10.8 dB 平台不升，
  LPIPS/SSIM 随桶变差，高斯数按 190,512/层线性膨胀（18v 达 3.43M）→ 跨视角几何不一致
  不随视角数收敛，确认 G1 必须以几何一致性损失为主；
- **多包摸底**（7 真实包 32v）：7/7 复现同一病因（每包最差视角 8~11.5 dB）；空洞率全部 ≤0.01%；
  采集轨迹不均匀（相邻角距最大 77~166°），Phase 0 登记逐包方位角，训练集按桶+轨迹形状分层抽样；
- **评测协议补全**：SSIM/LPIPS 并入 `30_ff_multiview_eval.py`，后续所有实验默认四组指标齐报。

### Phase 2 / D3：Geometry-First 跨视角一致性前馈头

#### 2026-11-12 新增：先选基座，再启动训练

详细依据、单变量矩阵、数据划分与数值门槛见
[`SPLATANYTHING_ROUTE_REVIEW.md`](SPLATANYTHING_ROUTE_REVIEW.md)。它是本计划的专项设计附录，
本文件仍是唯一执行入口。以下为**拟执行**顺序，未产生新实验结果：

| 顺序 | 任务 | 决定什么 |
|---|---|---|
| S0 | 相机/裁剪/尺度/渲染审计；真权重小样本推理冒烟 | 两方法是否可以公平比较，3090 是否可承载 |
| S1 | DA3 vs SplatAnything，3/8/13v 主桶，RGB-only 与 pose-conditioned 分表 | SplatAnything 是否值得作为首选改进基座 |
| S2 | DA3 原生 depth 反投影 vs GS centers；同次前向消融 xy/depth offset；UE GT 几何诊断 | 错误来自几何先验还是高斯属性头的中心偏移 |
| S3 | 仅对胜出候选重建可训练路径：冻结中心、属性头微调，再加几何置信度/可见性融合 | 是否有可复现的人像域增益 |
| S4 | 通过几何验收后，沿用 T1～T4 外观/Face ROI 对照 | 人脸细节是否独立提升 |

S1/S2 通过附录门槛且 S3 梯度/资源冒烟成立，才把 SplatAnything 升为主要开发基座。
否则继续 DA3，但先试“可靠深度反投影固定中心”的最小基线；仅在该深度几何也不可靠时
训练深度/几何修正分支。S2 是新引入的中心来源因果诊断，不重开 E3 的 scale 扫参。

新增约束：不得只在推理时替换归一化常数并称为 GSNorm 训练消融；不得直接共享两套
不同权重的 encoder 并宣称无损加速；不得把修改 `requires_grad` 当作 SplatAnything 已可训练
（发布 forward 带 `no_grad`，还需梯度光栅化与训练循环）。

D3 的问题定义是：**前馈网络是否能直接生成可渲染的正确 Gaussian Geometry**。Depth 正确不等于 Gaussian Geometry 正确；验收对象必须同时包含 center、scale、rotation、可见性/置信度和 Novel View 渲染结果。先做可归因的最小基线，再逐项增加能力：

| 子实验 | 只改变一项 | 目的 |
|---|---|---|
| G0 / D3-0 | 复刻官方冻结 backbone + Gaussian head 训练口径 | 确认训练/导出/渲染链没有新偏差 |
| G1 | 先固定可靠 depth 反投影中心；S2 证明先验不足时，才用几何监督训练受约束 center/depth 修正；属性分支不自由移动中心 | 解决出框和跨视角漂移，避免光度损失驱动中心偏离表面 |
| G2 | 与现有前馈几何方法做同输入、同渲染器适配对照 | 判断应继续训练 DA3 还是采用外部几何路线 |
| D3-A | 外观一致性监督：可见性加权的跨视角颜色/渲染损失 | 解决远离参考视角的灰化和饱和度衰减 |
| D3-F | 几何融合/去重和置信度阈值 | 在质量不降的前提下把高斯数压到约 25 万以内 |
| D3-C | G1 + D3-A + D3-F | 只在单项消融成立后合并 |

Phase 2 不以 Depth Error 单独验收。每个几何实验至少保存并报告：

- **Center**：对应表面点的 3D 位置误差、跨视角重投影误差和人物轮廓边界误差；
- **Scale**：3D scale 与屏幕 2D footprint 的分布，尤其是脸、头发、脚和轮廓区域；
- **Rotation**：局部表面方向/法向与 Gaussian rotation 的一致性；
- **Visibility/Confidence**：遮挡区域是否被错误融合，自由空间泄漏和重复 Gaussian 是否增加；
- **可渲染结果**：输入视角、插值视角和大角度 held-out Novel View 的 RGB、alpha、锐度和 LPIPS。

Depth、Normal 和 Point 监督用于解释几何误差，最终模型选择以 Gaussian Geometry + Novel View 表现为准。只有 Depth 下降同时带来重投影、轮廓和 Novel View 改善，才算几何有效。

Phase 2 保留两条候选路线，必须在同一输入、相机、导出和渲染器下比较：

| 路线 | 做法 | 适用条件 |
|---|---|---|
| G1-A 外部几何适配 | 接入已有前馈几何方法，只替换 Geometry 输出，外观头和渲染口径保持一致 | 外部方法能直接降低 center/轮廓/Novel View 几何误差 |
| G1-B DA3 Geometry Head | 冻结 backbone，训练 DA3 Geometry Head 输出 center/scale/rotation/confidence；必要时再逐层解冻 | 外部方法兼容性或数据适应性不足 |

Geometry 训练损失可按 `L_geo = λ_depth L_depth + λ_point L_point + λ_normal L_normal + λ_sil L_silhouette + λ_mv L_multiview + λ_scale L_scale` 逐项加入。多视角损失必须带 visibility/occlusion gating；不可见区域和错误遮挡不能作为硬监督。若路线 A 作为 teacher，也必须用 GT 和 held-out Novel View 检查误差没有被蒸馏继承。

训练必须覆盖真实推理分布：视角数至少包含 3、8、13、18，并额外抽样 32 视角作为越界压力测试；官方训练上限为 18，不能直接把 32 当成已知分布。训练目标至少含训练视角光度损失、held-out 渲染损失、深度/轮廓监督和跨视角一致性项。训练初期先不加入 LPIPS 或复杂融合，避免把几何错误隐藏在感知指标里。

D3 的实验矩阵必须固定：同一场景、同一输入帧、同一参考策略、同一导出和渲染口径。每次只改一个子实验；同时保留官方前馈、per-scene 5K 甜点结果和 30K 结果作为上下界参考，但后两者不属于目标链路。Phase 2 未通过几何验收，不进入 Phase 3。

### Phase 3 / D4：冻结几何的纹理与人脸对照

Phase 3 必须保留冻结几何对照，否则最终效果无法归因。以下四组共享同一份 Phase 2 Geometry，输入、相机、导出和评测全部固定：

| 版本 | 几何 | 外观分支 | 能回答的问题 |
|---|---|---|---|
| T1 | 冻结 | 普通分辨率 Appearance | 纹理解耦的最低基线 |
| T2 | 冻结 | 4× Appearance | 高分辨率纹理头本身带来多少收益 |
| T3 | 冻结 | 4× Appearance + Face ROI | 人脸局部分支的独立收益 |
| T4 | 受约束联合微调 | 4× Appearance + Face ROI | 几何联合微调相对冻结几何的增益/代价 |

T1→T2→T3 每次只增加一项；T4 最后运行，并限制 Geometry Head 学习率、center 偏移范围和可修改区域。禁止用 T4 直接替代 T1～T3。

Phase 3 的外观指标包括人物/人脸 ROI PSNR、LPIPS、锐度、分频能量、饱和度差、跨视角颜色方差和 alpha 泄漏。只有当 T2/T3 在几何指标不恶化的前提下改善外观，才保留高分辨率或 Face ROI 分支；已有未重训的 504→1008 负收益不能直接外推为 T2 结论。

分支只处理人脸局部高分辨率外观特征，不把“全局输入放大”当作方案；监督使用 UE 人脸 mask/关键点，验收仍用逐视角人脸框、锐度和分频指标。

实现上，T2/T3 使用独立的高分辨率 Appearance Encoder（轻量 CNN/FPN 或局部特征），不要简单插值 DA3 低分辨率特征，也不要在 4× 像素数上复制完整全局 Transformer。每个 Gaussian center 投影到输入视角后，只从可见且未遮挡的位置采样特征，再做多视角融合；曝光/颜色校正和 Face ROI 特征必须受 visibility 约束，不能用无约束纹理融合掩盖几何错位。

### 总体对照矩阵

| 版本 | 改动 | 作用 |
|---|---|---|
| B0 | 官方 DA3、历史导出 | 原始参考，不作为最终交付口径 |
| B1 | E1/E2 导出和颜色管线修复 | 排除实现问题，形成 `DA3-Baseline-V1` |
| B2 | 一次性 E3/E4 几何审计 | 区分 scale footprint 与 center/遮挡问题 |
| G1 | 外部 Geometry 适配或 DA3 Geometry Head | 验证 Geometry-First 路线 |
| T1 | 冻结 Geometry + 普通 Appearance | 纹理解耦基线 |
| T2 | 冻结 Geometry + 4× Appearance | 测量高分辨率纹理独立收益 |
| T3 | 冻结 Geometry + 4× Appearance + Face ROI | 测量人脸分支独立收益 |
| T4 | 受约束联合 Geometry/Appearance 微调 | 测量联合微调增益与代价 |

B0/B1/B2/G1/T1～T4 必须共享输入帧、参考策略、评测相机、导出和渲染器。T4 不得跳过 T1～T3，否则无法回答“收益来自纹理还是几何”。

### D5：鲁棒性与多场景验收（D3/D4 后）

在至少多个 UE 场景和一批真实 Remy 包上复测，按视角数、视角弧缺口、人物尺度和背景复杂度分桶。D5 只回答“改进是否可复现”，不再通过单个困难包继续改模型。

## 4. 预注册的停止与晋级条件

- 导出参数改动若不能在同一次前向的 2×2 对照中独立归因，停止继续扩展。
- 几何改动若只提高参考视角、却使远视角或 held-out 下降，判为失败。
- 外观改动若提高 LPIPS 但降低锐度/饱和度或增加 alpha 泄漏，判为失败。
- D3 只有在非参考视角的 PSNR、锐度和饱和度衰减同时收窄，且高斯数/延迟没有失控时，才进入 D3-C。
- D4 只有在 D3 通过后才启动；若全局指标改善但人脸指标不改善，停止继续堆 SH 或全局分辨率。

## 5. 已终止路线与历史资料

旧的 per-scene DA3+2DGS 方案曾测试 anchor、2DGS、offset 和 detach。C-06 的结论是：opacity 冻结和 detach 是真实 bug，但关闭 densify、固定点数和缺少可增长容量才是无法追上 3DGS 的主要原因；该路线不再作为当前目标。相关数字保留在 `EXPERIMENTS.md`，避免与 D3 前馈重训混淆。

## 6. 参考文件

- 事故和坐标约定：[`ERRATA.md`](ERRATA.md)
- 实验数字和失败记录：[`EXPERIMENTS.md`](EXPERIMENTS.md)
- 三视角方案的历史评审：[`PLAN_REVIEW_3VIEW.md`](PLAN_REVIEW_3VIEW.md)
- DA3-GT 详细接口/损失草案：[`28_da3_gt_feedforward_plan.md`](28_da3_gt_feedforward_plan.md)
- 横评证据：`../compare_4methods/FINDINGS.md`、`P0_EXPERIMENTS.md`、`P1_EXPERIMENTS.md`

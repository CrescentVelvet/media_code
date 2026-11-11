# da3_human 实验总计划

> **唯一执行入口。** 当前目标是 DA3 人像前馈高斯直出质量；不做逐场景优化，不把
> InfiniSplat 的短训或 3000 步精调当成产品链路。实验结果按日期倒序写入
> [`EXPERIMENTS.md`](EXPERIMENTS.md)，推理/坐标/导出事故写入 [`ERRATA.md`](ERRATA.md)。
>
> 旧文档的分工：
> - [`PLAN_REVIEW_3VIEW.md`](PLAN_REVIEW_3VIEW.md)：2026-11-11 三视角方案的决策审查记录，保留历史取舍，不作为第二份主计划。
> - [`28_da3_gt_feedforward_plan.md`](28_da3_gt_feedforward_plan.md)：DA3-GT 的长篇技术设计附录，接口和训练细节以本计划的当前决策为准。
> - `EXPERIMENTS.md` 中 2026-11-09 以前的 per-scene/2DGS 内容是已终止路线的实验史，不重新启动。

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

因此当前优先级是 **D3 前馈高斯头的跨视角一致性重训**，人脸专门分支后置为 D4。

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
| E3 尺度与模糊诊断 | E1 正式 PLY、相机、逐视角 mask | `29_d1_diag_3view.py`；必要时一次 scale oracle/clamp 对照 | 每个视角的 σ 折像素、2D footprint、人物轮廓锐度 | scale 改动只有在远视角锐度提升且空洞/颗粒不恶化时才保留；否则归因回几何不一致 | **已测分布，待一次性因果对照** |
| E4 几何与遮挡诊断 | E1 PLY、预测/GT 相机、UE mask/对应点 | `29c`～`29k_pose_step*.py` 及投影工具 | center 投影、重投影误差、轮廓/遮挡对照、可见性统计 | 区分 center 错位、scale/rotation 不匹配和遮挡融合错误；Depth 只能作为辅助证据 | **位姿已排除，几何输出仍待模型级验证** |

Phase 1 停止条件：①白洞已经由导出策略解决；②灰化已经定位到 alpha/颜色管线或外观表达；③模糊/重影已经缩小到几何 center、scale、rotation 或遮挡中的具体假说；④在固定数据和口径下，继续改导出参数的收益已接近上限。当前已满足前两项，位姿分支也已排除；只允许完成 E3 的一次 scale 因果对照和 E4 的几何输出审计，然后停止 Phase 1。不得继续无限扫 SH、opacity 或 fusion 阈值。

### Phase 2 / D3：Geometry-First 跨视角一致性前馈头

D3 的问题定义是：**前馈网络是否能直接生成可渲染的正确 Gaussian Geometry**。Depth 正确不等于 Gaussian Geometry 正确；验收对象必须同时包含 center、scale、rotation、可见性/置信度和 Novel View 渲染结果。先做可归因的最小基线，再逐项增加能力：

| 子实验 | 只改变一项 | 目的 |
|---|---|---|
| G0 / D3-0 | 复刻官方冻结 backbone + Gaussian head 训练口径 | 确认训练/导出/渲染链没有新偏差 |
| G1 | Geometry Head 输出 center/scale/rotation/confidence；加入跨视角重投影、深度/轮廓监督 | 解决出框、尺度过大和跨视角漂移 |
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

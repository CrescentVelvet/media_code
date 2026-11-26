# ZipSplat 实验记录

## 2026-11-26 · 海思端侧 19 例十图重建与交付

- 输入 `D:/dataset/测试数据/海思端侧重建拍摄数据/` 下 19 个 `images/` 子目录：
  `case1..case17_nopose_recentered` 及 `model_man`、`model_plant`。每例恰好 10 张 704² PNG。
  使用官方 `zipsplat-da3g-504.tar`、commit `73ab63f048fcc15ee54ada4ef9cdd79b925611a8`，
  前馈后默认 TokenBA，不读取已有 stage1/SfM/glTF 相机或已有 PLY。
- 最终每例 `zipsplat.ply` 为 TokenBA 后高斯，207,360 点、26 属性、binary little-endian、
  21,566,072 字节；19/19 全部属性有限、SHA-256、十图输入清单与推理记录一致。
  中间结果在 `D:/output/results/zipsplat_results/hisilicon_10v/<case>/`，日志在 `logs/`，
  独立验收报告 `final_validation.json`。原始图像、相机和其它算法 PLY 未改动。
- 抽查 `model_man`、`case10`、`case1`、`case7`、`case14`、`model_plant` 输入自渲：
  主体与局部环境可辨，未见整帧空白；人物侧后方和部分纹理有软化。输入自渲不代表未观测视角
  或独立留出已达标，本批未做跨方法几何/PSNR 结论。
- 复用入口 `04_batch_hisilicon.sh`、`batch_hisilicon.py`；核验 `validate_hisilicon.py`。
  只处理恰好十图的目录；成品先写临时 PLY、完整长度校验后改名；再次运行核对已有成品、输入
  清单与推理记录，不覆盖。`_env.sh` 会预设通用 RESULTS_DIR，本入口须在 source 前设置
  `ZIPSPLAT_RESULTS_DIR`，不能沿用普通 RESULTS_DIR 覆盖习惯。
- `model_man` 首次 TokenBA 为 132.25s，后续 18 例每例 35.9–49.5s；不据此宣称稳定每例耗时。
  运行中 shell 曾因换行时序出现末尾解析异常，最终入口已在全部完成后回归通过，19/19 跳过。
  中间归档时误复制的旧佳佳结果副本已移入本批 `unrelated_legacy/`，旧原目录未改动。

## 2026-11-25 · ZipSplat × SplatAnything 表示机制讲解（无新实验）

交付 `D:/ppt/20261125-ZipSplat与SplatAnything算法思路对比.pptx`（15 页）及 D 盘同名 `_src` 工程；官方机制图、机制示意与本机同帧实测分开标注，PPT / 图片不入 git。本轮只读源码与归档，不启动训练或 GPU 实验。

### 分水岭不是“有没有 token”

- 两者骨干均把图像转为 token。SA 的“逐像素”指高斯中心依附源像素射线：深度与相机反投影定中心；`SplatAnything/src/models/models/rasterization.py:170-187,212-298` 的 12 通道为 quat4 / scale3 / opacity1 / residual SH3 / merge weight1，**不含自由 XYZ**，但深度和相机仍属预测几何。GSN 后固定 voxel 加权合并不保证消除跨源深度错层或重影，合并后也不是一像素一高斯。
- ZipSplat 的 scene token 是多视角场景代理，不是语义物体或大像素。`models/networks/zipsplat.py:229-380` 以 scene token 为 Q、geometry token 为 K、视觉/颜色 payload 为 V 融合；`models/decoders/gaussian_head.py:64-84,197-231` 每 token 解出 G 个高斯，23 参数含自由 position3 / scale3 / quat4 / opacity1 / SH1 12。一个 token 可分工生成多高斯，但表达空间不等于正确率。
- 本地 `configs/zipsplat_ft504.yaml:8-19` 为 G=16、`query_sample_ratio=[.025,1]`；三图 378×504 推理取 1.0，`K>=V*T` 时跳过 k-means，故 2,916 tokens → 46,656 Gaussians，**不能说本机先压缩 token 再重建**，跨视角注意力仍发生。assignments + `models/losses.py:221-263` ConeLoss 可复用源 patch 投影链，但不是 UV、真实表面或遮挡保证。

### 对分辨率、重影与 LGTM 的边界

- SA 的源像素追踪在合并前较直接；Zip 有源 patch 归属，却须另建稳定 surfel、局部 UV、选源/遮挡和纹理渲染。两者都不能“直接贴原图”获得可靠高分辨率纹理，LoRA 也不会创造这些接口。
- 提高输入分辨率仍增加两者 backbone patch/注意力成本；SA 还增加逐像素深度、属性和候选高斯规模。自由中心不等于 2K 免费，内部 token 增多不保证输出锐度。
- 两条链均可能重影：SA 可检查合并前后跨源点层，Zip 需检查 token 融合与自由几何解码。锐度不能替代几何，覆盖不能证明无重影。
- P12 复用佳佳同三输入、29 留出、512² 黑底、source-only 对齐：SA pose 13.4221 dB / .3805 / 锐度比 .3078 / 覆盖 82.62%；Zip FF 无位姿 17.6399 / .4558 / .1149 / 100%；TokenBA 17.2682 / .4729 / .2953 / 99.99%。输入条件不同、TokenBA 属 TTO、人物与抱枕微动，单场景不构成总体排名。

## 2026-11-24 · ZipSplat × LGTM 接入调研（候选设计，未训练）

### 问题边界

本地已复现的是官方 `zipsplat-da3g-504` 权重推理，而非从零训练。当前目标是判断能否借用
LGTM 的**低分辨率几何 + 原始高分辨率图片取纹理 + 每基元局部纹理 + 可导 2DGS 渲染**，
改善 ZipSplat 纯前馈偏软的问题；不假定三图人像、2K 或 4K 已可实用。主计划 N2 仍冻结，
本轮只读核对官方仓 commit `73ab63f048fcc15ee54ada4ef9cdd79b925611a8`，未修改源码或启动 GPU。

### 本地实测给出的动机，不是提升承诺

- 佳佳三图 29 留出：Zip FF `17.6399 dB / SSIM .4558 / 锐度比 .1149 / alpha>.5 覆盖 100%`；
  TokenBA `17.2682 / .4729 / .2953 / 99.99%`。TokenBA 提升锐度与 SSIM，PSNR 反降 `.3717 dB`，
  不能当全面提升，更不能把测试时优化算作纯前馈成绩。
- 两个近静态真人身份 RGB-only：Zip FF 灰衣/黑衣两 target 整图 PSNR 均值 `15.48/16.99 dB`、
  四帧均保住人物、覆盖 `100%`，但脸软且有重影。Rosemary 主口径紧脸 ED 均值 `27.29 mm`，
  SA known-pose 为 `7.34 mm`；不支持“Zip 几何已独立证明更好”。
- 因而 LGTM 最可能先改善**有正确几何支撑区域的高频外观与 LPIPS**；对鼻部错层、眼嘴错位、
  遮挡错误和无观测区域不作几何改善承诺。投影纹理在错误盘面上会更锐地贴错位置。

### 源码接口与 LGTM 的天然接点

- `models/decoders/gaussian_head.py:64-84,197-231`：每 scene token 展开 G 个自由高斯；
  每高斯 23 维 = position3 / scale3 / quat4 / opacity1 / SH1 12；无局部 UV、纹理或 ray 交点。
  504 发布配置 `configs/zipsplat_ft504.yaml:8-19` 为 G=16、SH dropout .5。
- `models/networks/zipsplat.py:229-292,302-380`：逐 patch geometry token → 可微簇均值 scene token
  → 多层视觉/颜色 CA → GaussianHead；`assignments` 保存每个源 patch 属于哪个簇。
  `models/losses.py:221-263` 的 ConeLoss 已把高斯投回全部成员 patch，说明“高斯—源 patch—相机”
  所需归属和投影链可复用，比无观测来源的 learned query 更适合接 LGTM。
- **但每个 scene token 有多个源 patch、多个高斯**。LGTM 的像素网格一对一局部纹理不能直接照搬：
  须给每个自由高斯建立稳定局部切平面/UV、尺度与朝向，并按成员源视角选主源或做多源置信融合；
  视线角、投影 footprint、前表面可见性/遮挡、跨源接缝须纳入。Cone 只约束中心横向落点且允许
  2 patch 自由区，不保证平面贴合、完整像素对应或遮挡正确。
- `zipsplat/gaussians.py:175-190,213-272` 使用上游 gsplat 3DGS rasterization 与普通 3DGS PLY。
  必须换成支持 ray–surfel UV 与 texture_colors（以及可选 texture_alphas）的 LGTM 补丁版 gsplat，
  并新增图元、纹理图集、源视角/UV 元数据导出；原 PLY 无法保存这些属性。
- TokenBA 冻结 GaussianHead 并优化 scene tokens + 相机；新增纹理后，相机或 token 更新会改变投影取纹理。
  可选择重新取纹理/联合优化纹理 latent，但必须保留微分链和源成员关系，不能把旧纹理当静态贴片。

### 必改模块

1. **几何表示**：保留 DA3 backbone、geometry/scene token、k-means、GeoCE 与相机头；
   把 GaussianHead 拆成几何 readout 与外观 readout。普通 3DGS 尺度/旋转需适配成 2D surfel，
   规定稳定法向、切向、UV 原点和局部跨度，首阶段禁止自由移动中心以隔离变量。
2. **高分辨率纹理支路**：保留低分辨率 backbone；另传未降采样 source RGB 与一致 K。
   由 assignments 得到候选成员源 patch；将每个 surfel 的 UV 网格投回源图，按可见性/视角/
   footprint 选源或融合，采样 RGB 作 base，轻量 texture head 预测 delta；SH 留低频视角色。
3. **可导渲染与导出**：接入纹理 2DGS 光栅化及反向；RGB 先沿用高斯 alpha，alpha 纹理后测；
   旧 PLY/viewer 不能无损承载，需新导出格式。控制总 texel、纹理分辨率和重复源存储。
4. **训练与评测接口**：图像 crop/resize 同步变换原图 K；target 仅提供渲染位姿与 RGB 监督，
   不进 source 编码或取纹理。保存成员索引、可见性和冻结参数/几何 hash，排除“冻了名字没冻输出”。

### 分阶段训练建议

| 阶段 | 参数与数据 | 回答的问题 / 通过条件 |
|---|---|---|
| Z0 免训练基线 | 原 504 权重；提高 TokenBA source 渲染/监督分辨率，FF 与 BA 分列 | 高分辨率 TTO 本身能恢复多少；不冒充 LGTM 或纯前馈提升 |
| Z1 普通 2DGS | 冻结 backbone/token/中心；先适配 surfel 尺度/朝向，仍 SH 颜色；高分辨率独立 target 监督 | 表示转换是否退化。若普通 2DGS 已明显损失，先修几何，不能归咎纹理不足 |
| Z2 RGB 局部纹理 | 固定中心/几何 hash；训练新投影与 delta 纹理层；原图 source + target RGB/LPIPS | 高频外观、LPIPS、总 texel、显存/耗时是否改善，且几何/覆盖不塌 |
| Z3 受控扩展 | 逐项 alpha 纹理、T=2/4/8、纹理 TokenBA；仅必要时小 lr/LoRA 适配几何 | 哪项有独立增益；若变清但错层/重影增加则停止，不用更大骨干掩盖问题 |

正式 LGTM 在静态场景先高分辨率训练普通 2DGS，再冻结非纹理参数训练纹理层；这支持上述
优先级，但不证明 ZipSplat 只训 LoRA 就足够。Zip 原训练为 252、450K 步/16×GH200、4 target，
504 微调 100K 步；这些不是本机小试预算。按主计划若未来获批，首轮只做 ≤100 步工程门，
再选一个基座做受控人物/通用数据对照，不并行两套大训练，不宣称单卡 3090 可复现官方全量训练。

### 数据、损失与验收

- 训练需同静态场景/身份的高分辨率 source RGB、同步准确 K/位姿、独立 target RGB/相机；
  保留 cone/GeoCE/pose 作为原几何辅助约束，新增高频 target RGB+LPIPS。GeoCE 训练锚点来自
  DA3 + GT 相机预计算；pose-free 推理不等于训练无先验。人物近静态微动须单列。
- 首轮 `T=2/4`、RGB-only 是工程起点，不是最优档承诺。固定 backbone 内部分辨率、基元数、
  target 集合、低通对照；逐臂报输入/网络/输出/纹理四分辨率、总 texel、训练及推理显存/耗时。
- 主看未见身份、留出视角紧脸 LPIPS、衣服文字/抱枕五官细节、alpha 覆盖、轮廓/重影；
  合成可信 GT 才报鼻部 ED/p90，真人无深度不报毫米。输入自洽变锐只说明纹理获取能力。
- 停止条件：无纹理 2DGS 已退化、纹理只复制错位源图、跨源接缝/遮挡恶化、总 texel/渲染成本
  抵消少高斯优势、或 FF 无收益而只有 TokenBA 变锐。此时回到几何/相机/融合诊断，不继续加纹理。

预期结论按条件表述：**几何基本正确且源图可见处，最可能提升高频外观/LPIPS，减少为细节继续堆
高斯的需求，并允许 backbone 保持低分辨率；几何错层、跨视角冲突和无观测区域不会自动修复。**
LGTM 多视 VGGT 论文也只到约 2K，未实现多视 4K，故本方案不承诺三图人像 4K。

## 2026-11-23 — PPT 二次扩版为 24 页：每视角一页整行四联

- 用户反馈第 7–9 页多行网格被 PPT 框上下裁切 → 改为**每视角一页**：输入 ×3
  （帧 0/40/81）、内插 ×3（v004 / v024 / **v027** 新增）、外插 ×3（v008 / v011 /
  **v013 新增**），每页一条 1160×345 四联（**287² 完整方形，不再裁切**）+ 两卡片。
- **v013 为补齐外插组单独渲染**（SAVE_VIEWS=13，同协议同口径；ZipSplat 双变体一跑、
  SA 单独一跑，写入主 renders 目录；结果 v013：SA 12.95 dB/SSIM 0.360/覆盖 73.8%、
  FF 16.74、TokenBA 16.34）。观测弧图高亮同步更新（内插 4/24/27、外插 8/11/13）。
- 旧 10–18 页顺延为 16–24；视频页现为第 **19** 页（embed_video.py PAGE=18、断言 24）。
- 本轮新坑：`03b_eval_3view_compare.sh` 第 22 行 `SAVE_DIR="$OUT_DIR/renders"` 是
  **硬赋值**，环境变量传 SAVE_DIR 无效（要写隔离目录得改脚本）；素材脚本里
  `key.endswith("FF")` 对 `ZipSplat_FF_self` 失效（→两个 ZipSplat 变体都显示成
  TokenBA 指标，已按去掉 `_self` 后缀匹配修正）。

## 2026-11-23 — PPT 扩为 18 页：输入 / 内插 / 外插视角分组对比

- 新增 5 页（7–11）：三输入精确自渲网格（原始帧 0/40/81 × 原始/SA/FF/TokenBA）；
  内插 v004/v024、外插 v008/v011 四联（GT + 三方法，逐视角 PSNR·SSIM·覆盖率）；
  SA 黑缺定位页（观测弧俯视示意 + 覆盖率随离弧距离下降：v004 83.6% → v024 75.0% →
  v008 81.2% → v011 65.7%）；ZipSplat 细节放大页（v021 同帧裁切，nearest 不增强）。
- 视角分组口径：输入方位 −163°/23°/98°，最大观测缺口约 186°；v004/v024 在输入弧上为内插，
  v008/v011 落在缺口内为外插（最近邻距离小 ≠ 处在弧内）。
- 素材由 `_build/prepare_view_group_assets.py` 生成，宽高与页面框严格一致
  （该 DSL 渲染器的 objectFit 实为 cover：框与素材宽高比不一致会被裁切，
  首版 1160×1198 网格曾被裁成单行，已重做）。
- 重建链路硬约束：**全部页面 upsert 完成后才运行 embed_video.py**——编辑器重存会丢包内
  MP4（本轮复现一次）。成品 18 页、第 13 页内嵌 orbit_360.mp4 与源逐字节一致。
- 重建中恢复了两页丢失源码（16/17 号位：双重降采样修正页、结论页），从 13 页版 pptx
  逐形状转储（位置/字号/颜色）高保真还原为 .slide。
- 备份：`D:\output\_backup\media_code_zipsplat_ppt_view_groups_20261123\`（13 页版）。

## 2026-11-23 · N1 已完成，Zip FF 保留为真人覆盖基线

正式记录见 [SA 实验归档](../splatanything/EXPERIMENTS.md) 的 Rosemary v27 和真人 v18。Rosemary 两个固定 target 的紧脸 ED MAE 均值为 Zip 27.29 mm、SA 7.34 mm（Zip RGB-only、SA known-pose）；仅 source 估尺度会显著改变 Zip 局部误差。两个近静态真人身份的 RGB-only 公平轨中，Zip 四帧均保住人物，灰衣/黑衣整图 PSNR 均值 15.48/16.99 dB，SA 为 6.66/10.15 dB。Zip 脸部仍软且有重影，真人缺少独立深度。当前证据支持覆盖更稳定，几何教师和 N2 基座仍待选。

后续直接复用上述基线，按[总计划](../da3_human/EXPERIMENT_PLAN.md)增加两个真实中间机位检验鼻部收益。此项属于 DensiTok 思路的免训练诊断，尚无补全模型结果。

## 2026-11-21 — 用户头部形状观察与下一步（方案，未新增实验）

用户观察ZipSplat更模糊但头部形状更接近输入；此信息支持将Zip FF作为人脸几何候选，与SA
P0进行同三图、独立正侧面target比较。已核对共同评测JSON，`regions_holdout=null`，
现有全图PSNR/覆盖优势不能替代脸部几何验证。头轮廓/鼻部深度/可信对应与纹理指标分开报。

若多个静态身份上结构优势成立，优先尝试固定可靠几何的人物外观适配；TokenBA单列TTO，
不计作纯前馈提升。若只是模糊掩盖重影或覆盖优势，不以此更换基座/做教师。暂不拼接SA与
Zip的PLY，两模型没有逐高斯对应；真实source才是外观监督，SA锐利伪影不能当细节真值。
统一执行入口：[三图总计划N0～N2](../da3_human/EXPERIMENT_PLAN.md)。本轮仅分析与设计。

## 2026-11-21 — 三图主线、单向 360° 环绕与新版汇报

### 三图横评结论（沿用 11-20 已复跑的共同协议）

- 同一原始帧 0/40/81、29 留出、512² 黑底、source-only Sim(3)：ZipSplat 前馈
  **17.6399 dB / SSIM 0.4558 / 锐度比 0.1149 / 覆盖 100%**；TokenBA
  **17.2682 / 0.4729 / 0.2953 / 99.99%**；SA v3（pose 模式）
  **13.4221 / 0.3805 / 0.3078 / 82.62%**。
- 本组要留出 PSNR 与完整覆盖选 ZipSplat 前馈；要 SSIM / 锐度折中选 TokenBA。
  SA 输入自洽 PSNR 21.8432 高于 ZipSplat FF 18.7353、TokenBA 19.6733，但未转化为留出优势。
  锐度必须与覆盖共同解释；SA 约 17% 未覆盖，不能据锐度比单独宣称整体更清晰。
- SA 为已知位姿，ZipSplat 为无位姿输入；单场景结果不构成普适排名。
  算法差别：SA 中心由深度 / 相机反投影，GSN 归一化间距后固定体素合并；
  ZipSplat 从 scene tokens 解码高斯、从独立 camera token 解码输入相机。TokenBA 冻结高斯解码器，
  优化 tokens 和输入 se(3)，不是重训网络。

### 单向 360° 视频与对齐修复

- 新增 `render_3view_orbit.py` / `03c_render_3view_orbit.sh`。固定半径 1.743m、
  仰角 23.845°，方位单调递增并闭合；144 帧 / 24fps / 6s，单格 368²，三联 1144×488。
  真实输出包括 SA、ZipSplat FF、ZipSplat+TokenBA 独立 MP4 与三联 MP4，另存首/中/末 PNG、
  相机、轨迹协议和逐帧 alpha 覆盖率。四个 MP4 均完整解码通过。
- 旧官方调用 `viz.turntable(..., sweep_deg=None)` 是正弦 wiggle；新视频不再称它为 360° 环绕。
  新轨迹不是独立留出 GT 相机集，首帧与 cam0 的 look-at/roll 差 20.93°；完整环含无观测背面，
  只能目检，不替代 29 留出评分。
- 对齐修复：视频入口预降采样后 `refine_sim3()` 又 resize，改变 CUDA 插值 / 梯度路径，
  SA 精化前仅 8.738dB。现保留原图方形裁切，只对齐函数内部一次降采样；同一 PLY / K /
  三输入 / 初值下，SA 恢复 12.9358 → 22.1382dB、旋转修正 3.0879°，三分支对齐断言均通过。
- 目检：近观测侧 SA 右侧大片缺失；背面 SA 黑区 / 抱枕破碎更明显，ZipSplat 覆盖更完整
  但人物、抱枕或椅子出现拉伸。覆盖完整不等于无观测几何正确。

### 汇报与归档

- 新版 PPT：`D:\ppt\20261121-ZipSplat复现与对比.pptx`（13 页，三图主线，含 Query Token 与 TokenBA 原理详解）；
  源码 `D:\ppt\_src\20261121-ZipSplat复现与对比\`。已逐页截图检查，布局 lint 无错误。
  第 3 页为 Query Token：scene token 在交叉注意力中作为 Q、完整几何 token 为 K、骨干 / 颜色
  payload 为 V；动态生成而非固定场景无关查询，区分 TokenBA 优化的 Z 与 camera token。
  第 4 页为 TokenBA 流程，第 5 页为原理详解：L1 输入光度目标、冻结 GaussianHead、patch–Gaussian / robust PnP
  → token 更新 → tokens + se(3) 联合优化，以及默认迭代 / 停止条件与输入过拟合边界。
  第 8 页内嵌三联 360° MP4（2,904,249 字节）；PPT 包内媒体与源 MP4 逐字节一致。
- 视频与协议：`D:\output\results\zipsplat_results\jiajia_3v_orbit_360\`；
  独立三联成品：`D:\ppt\figures\zipsplat\three_view_orbit_360.mp4`，与归档逐字节一致。
- 16 图仅保留 TokenBA 补充证据：输入 +2.507dB、8 留出 +0.238dB；不与三图协议混报。

## 2026-11-20 — 三输入同协议横评与复现汇报（补充）

### 共同协议与实现修正

- 复用 `compare_4methods/audit_31_eval_3view.py`，ZipSplat 通过 `EXTRA_JSON` 接入同一
  PLY 读取、相机对齐、渲染与指标链，不另造评分口径。两个分支都校验真实输入文件名为
  `000_5093309607791922.png`、`001_5093317707502922.png`、`002_5093325907213922.png`，
  对应原始帧 0/40/81；与历史三输入 PNG 逐张 SHA-256 一致。各 46,656 高斯。
  TokenBA 分支前馈 0.9841s、TokenBA 2.2145s。
- 原始 1439×1919 → 中心 1435×1435（x0=0,y0=235）→ 512×512；固定 GT 内参。
  三输入、32 GT、29 留出。三输入 Umeyama 初值后，仅用三输入做 Sim(3) 光度精化并冻结高斯；
  留出只用于评分。**这不是 16v 的 Wahba 协议，两组读数不可混报。**
- 发现旧 `sim3_refine.py` 光度精化固定白底、E14 最终评分黑底不一致。新增 `bg_value`
  （默认 1.0 保持旧调用），共同评测显式传 0.0，DA3-GS / SA / ZipSplat 全部重跑黑底。
  因此旧 E13 白底 10.0658/11.2513、E14 黑底 12.8009/13.4261 不用于本次排行榜。
- `03b_eval_3view_compare.sh` 以 `RESULT_NAME` 分方法保存，避免覆盖结果。
  ZipSplat / DA3-GS 为无位姿输入；历史 SplatAnything v3 为 **pose 模式**。

### 29 留出实测（统一黑底，单场景、非普适排名）

| 方法 | PSNR (dB) | SSIM | 锐度比 | alpha>0.5 覆盖率 |
|---|---:|---:|---:|---:|
| DA3-GS | 12.9516 | 0.3315 | 0.0676 | 0.7939 |
| SplatAnything v3（有位姿） | 13.4221 | 0.3805 | 0.3078 | 0.8262 |
| ZipSplat 纯前馈（无位姿） | 17.6399 | 0.4558 | 0.1149 | 1.0000 |
| ZipSplat + TokenBA（无位姿） | 17.2682 | 0.4729 | 0.2953 | 0.99994 |

- 本组中 ZipSplat 覆盖近完整，但前馈锐度保留偏低；TokenBA 提升 SSIM 与锐度比，
  PSNR 却较纯前馈 −0.3717dB，**不能称全面提升**。与 16v 的 +0.238dB 不矛盾：协议不同。
- 同帧目检可见主体结构与室内布局成立，但人物脸、抱枕五官、衣服文字、地面纹理仍明显偏软；
  覆盖率不是细节恢复证明。人物与手持抱枕微动仍是混杂因素。

### 汇报与归档

- PPT：`D:\ppt\20261120-ZipSplat复现与对比.pptx`（11 页）；源码在
  `D:\ppt\_src\20261120-ZipSplat复现与对比\`，页面均使用 `assets/...` 相对引用。
  第 6 页内嵌 16v TokenBA 往复摆动 MP4（504×504、96 帧、24fps、4s）；视频仅供目检。
- 新增产物：`jiajia_3v_compare/`、`jiajia_3v_compare_feedforward/`、
  `eval_3view_r512_black/`，均归档在 `D:\output\results\zipsplat_results\`。
  同帧证据在 `eval_3view_r512_black/renders/`；DA3-GS 读数在 `DA3-GS_metrics.json`，
  SA 读数在 `metrics.json`，两文件也各附 ZipSplat 两分支。

## 2026-11-20 — 佳佳抱枕：官方 504px 权重推理复现

### 复现范围

- 官方 `cvg/ZipSplat` commit `73ab63f048fcc15ee54ada4ef9cdd79b925611a8`；没有修改官方实现。
- 官方 `zipsplat-da3g-504.tar`（5.59 GiB；16 Gaussians/token），SHA-256：
  `85d28887dbaee3f2116cf464ab0a15053a40b2f88bff69aa09e7b0f6fda46fd4`。
- WSL Ubuntu 24.04，RTX 3090 24GB；独立 conda env，Python 3.11.17、
  torch 2.8.0+cu128、gsplat 1.5.3。验证的是发布权重推理，不是从零训练。
- 佳佳抱枕 Remy 包：82 张 1440×1920 HEIC → 去畸变、ROI 后 1439×1919 PNG。
  16 输入为 `0,5,11,16,22,27,32,38,43,49,54,59,65,70,76,81`；
  8 留出为 `3,14,19,36,47,62,74,79`，不参与预测或 TokenBA。
- 网络实际输入 378×504；原始竖图中心裁剪由官方 `prepare_inputs` 处理。

### 实测结果

| 分支 | 高斯数 | 前馈 / TokenBA (s) | 峰值 allocated (MiB) | 输入自洽 PSNR / SSIM | 8 留出 PSNR / SSIM | 留出 alpha>0.5 |
|---|---:|---:|---:|---:|---:|---:|
| 16v 纯前馈 | 248,832 | 2.51 / — | 9,832.2 | 20.827 / 0.5088 | 19.757 / 0.4772 | 1.0000 |
| 16v 前馈 + TokenBA | 248,832 | 2.46 / 10.56 | 9,832.2 | 23.334 / 0.6453 | 19.995 / 0.4817 | 0.9998 |

- 模型加载约 23s；16v TokenBA 执行官方默认 4 轮上限、96 joint steps。
- TokenBA 对输入自洽 +2.507dB、SSIM +0.1365；对留出仅 +0.238dB、+0.0045 SSIM。
  **输入侧收益远大于留出，不能据此宣称具有普遍 NVS 增益。**
- 三视角冒烟（前馈、无 BA）也已跑通：46,656 Gaussians，0.76s 前馈，
  峰值 8,619.3MiB；首次全分辨率渲染触发 120.4s CUDA JIT 编译。
- TokenBA 输入相机：Wahba 平均旋转误差 0.273°；米制 ATE 0.0099m，但
  中心轨迹 PCA `[1.104,0.672,0.033]`，标记 `planar_degenerate=true`，ATE 只作诊断。

### 评测口径与目检

- 输入自洽使用预测内参渲染原分辨率 GT；留出 GT 相机只用 16 输入相机对齐，
  不进入模型或优化。最终留出对齐为 Wahba 全局旋转 + 输入中心估计尺度/平移。
- PLY 的 SH 为 channel-first 存储，加载时恢复 `(K,3)`；黑底、gsplat 渲染。
- 目检正/侧面输入视角和留出视角：人物、抱枕、椅子与室内结构可辨，未见全黑、
  大面积空洞或整体崩溃；但人脸、抱枕五官、衣服文字和地面纹理显著模糊，局部
  有重影。该包为人物＋抱枕＋环境，手持抱枕有微动，不是静态纯抱枕重建。
- 环绕视频只供目检，不是独立新视角证据。未测 LPIPS，未复现论文训练/基准。

### 实现修正

- 官方 `load_image` 接收路径，不能传 PIL Image；前馈日志改用 `pred["model_images"]`。
- 严格纯前馈需要独立 `BA=0` 运行以保存前馈相机；将前馈高斯与 BA 后相机组合的
  `jiajia_16v/evaluation_feedforward/` 是历史误配，**不得作为前馈自洽结果引用**。
  正确对照在 `jiajia_16v_feedforward/evaluation/`。归档保留该目录供审计，但不用其报数。
- `proxy.env` 只做 proxy/no_proxy 白名单解析，不 source 整个含密钥共享配置；
  曾因错误赋值使结果目录为空，已用显式路径完成模型、评测和归档验证。
- DrvFs 上 `cp -a` 的属主/时间戳保留会失败；改普通 `cp -r` 并逐文件长度校验。

### 产物

- `D:\output\results\zipsplat_results\jiajia_16v\`：前馈/BA PLY、相机、渲染、
  环绕视频及 `evaluation/metrics.json`。
- `D:\output\results\zipsplat_results\jiajia_16v_feedforward\`：严格前馈对照。
- `D:\output\results\zipsplat_results\jiajia_smoke_3v\`：三视角冒烟。
- 权重：`D:\wheel\zipsplat\zipsplat-da3g-504.tar`；代码与权重仅限各自许可范围，
  其中发布权重为 CC BY-NC 4.0，非商用。

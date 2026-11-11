# 【技术附录】DA3-GT 面向人物前馈式 3D Gaussian Splatting 的设计草案

> **文档状态：归档技术草案，不是执行计划。** 当前实验顺序、冻结口径和晋级条件只看 [`EXPERIMENT_PLAN.md`](EXPERIMENT_PLAN.md)；实测数字只看 [`EXPERIMENTS.md`](EXPERIMENTS.md)。本文保留架构、损失和训练接口的长篇讨论，若与主计划冲突，以主计划和 `ERRATA.md` 为准。

> **来源与定位**：2026-11-10 峰哥用 GPT 生成，小华子审阅后收录为储备方案。
> 本方案属**前馈网络训练**（DA3-GT），对应 `EXPERIMENT_PLAN.md` §6「下一期」中的
> **前馈蒸馏**方向——与 per-scene 主线是**先后关系**（per-scene 验证收益 → 再前馈化），
> 不是替代关系。执行前**必读**下方审阅批注 + `ERRATA.md` + `EXPERIMENTS.md`（2026-11-09 各节）。
> 正文为 GPT 原文，除批注外未改写。

## 审阅批注（小华子，2026-11-10）

**总体判断**：框架合格，可立项储备。Ghosting/Blur 分治、"4× 外观分支不治重影"、
冻结几何优先、Stage 0 代码审计、不以单 PSNR 论成败，均与本仓已定论一致
（`compare_4methods/P1_EXPERIMENTS.md`、`EXPERIMENTS.md` 2026-11-09）。
但有 **2 个硬冲突 + 3 个必须补的本仓事实**，不处理不能执行。

### 硬冲突

1. **与 C-06 负结果的关系必须写清**。本仓 per-scene 双分支 2DGS（低分几何 + 全分纹理 +
   焊死 offset）已于 2026-11-09 三轮负结果**终止**（`EXPERIMENTS.md`「C-06 终止判定」：
   关 densify + 焊死 offset 是分水岭， densify 才是收益来源）。本方案 Stage 1
   （冻结几何 + 4× 外观分支 + center 不由外观分支移动）与 C-06 的解耦思路**同源**。
   立项依据不能只引 QuerySplat / InfiniSplat v2 的外部证据，必须写明：
   per-scene 已证伪了什么、前馈化改变了哪个前提（监督从光度梯度变为 held-out 渲染损失 +
   跨视角一致性损失）。
2. **训练侧算力完全缺失**。推理延迟预算写得很细，训练成本零评估。参照系：DA3 官方主模型
   200K 步 × 128×H100（`ERRATA.md` §DA3 训练事实）；本仓现实是**单卡 RTX 3090 (24GB) WSL**。
   "冻结 backbone + 新增 4× 分支"在 3090 上的可行性、训练时长、显存，是立项第一问，
   应先出 profile 再排期。

### 必须补的本仓事实

3. **DA3 训练事实**（`ERRATA.md` §DA3 训练事实，2026-11-09 查证 arxiv 2511.10647）：
   官方视角数均匀采样 **[2,18]**——我们推理用 **32** 已超训练上限，微调采样必须覆盖 [8,32]；
   官方 GS-DPT 高斯头 = **冻结整个 backbone 只训 DPT 头**（几 M 参数），损失 = 渲染
   photometric（MSE+LPIPS）+ 观测视角 scale-shift-invariant 深度损失。Stage 0/1 的
   第一个训练基线应直接复刻它，而不是另起炉灶。
4. **已有 baseline 与评估资产，Stage 0 不必从零复现**。正式口径已立：官方等价导出
   （不裁剪 + 全 SH）= 全图 **22.11 dB**、人脸 20.00 dB（3 视角、逐视角人脸窗）；
   32 视角前向 6.6s；NVS 32→125 评估（`10_eval_nvs_32to125.py` / `12_eval_nvs_first.py`）；
   55 包批量验收（`21_batch_portrait.sh`，2026-11-10 全量通过）。
5. **评估口径坑**（报数前必查，全部有实测代价）：
   - 区域指标必须**逐视角定框**，禁止固定 ROI——固定人脸框在 view 8 落在书架上，
     人脸锐度虚高 +180（`compare_4methods/IMPROVEMENTS.md`）；
   - mask 内 FG PSNR **抓不到鬼影**（幽灵渲染在 GT mask 之外），必须配 mask 外泄漏
     指标或 alpha 渲染（som_human E-07 同型教训）；
   - 渲染与 GT 的视角索引必须按 `view_indices` 对齐（`07_eval` 教训，错位即掉到 10 dB）；
   - **2DGS 路线已终止**（只换光栅器 −0.29 dB；oracle 法向 −0.71 dB），V1 renderer
     应明确为 **3DGS EWA**；2DGS 仅保留 `median_depth` 几何用途。

### 其他

- 正文公式是 pandoc 未渲染的 tex 残留（`` `\pi`{=tex} `` 等），引用公式时以本节文字口径为准。
- §9 消融表 E0~E6 缺 **per-scene 甜点基线**对照（DA3 点云 + 5k 步 ≈ 2.6 min / 22.2 dB，
  30k 基线 24.3 dB / 21.5 min，`P1_EXPERIMENTS.md`）——前馈方案的对比对象必须包含它，
  否则"前馈 vs per-scene"的收益说不清。
- §4 数据集要求与本仓 `24_ue_render_spec.md` V1.1 已核对一致 ✅；§5.2 metric 对齐后禁
  scale-shift 归一化 ✅；三摄特征不可直接按图像坐标拼接 ✅（与本仓三摄 6DoF 记录约定一致）。

---

以下为 GPT 原文。

---

# DA3 面向人物前馈式 3D Gaussian Splatting 的优化方案计划书

> **项目代号**：DA3-Human / DA3-GT\
> **目标**：在保持前馈推理速度的前提下，改善稀疏视角人物 3DGS
> 的几何一致性、细节清晰度和新视角渲染质量，重点降低 Ghosting 与 Blur。\
> **数据基础**：UE 合成静态人物数据集，采用稀疏主摄输入、三摄同位补充和
> 360° 评估环。\
> **方案性质**：这是面向现有 DA3 代码库的工程与研究计划。涉及 DA3
> 内部特征层、相机输出和 Gaussian renderer
> 的具体接口，必须在代码审计后确定；本文不假设仓库中存在某个固定字段或模块名称。

------------------------------------------------------------------------

## 1. 项目背景与问题定义

### 1.1 当前目标

构建一个以 DA3 几何推理为基础的前馈式人物 Gaussian 重建系统：

-   输入 3--13 张稀疏人物 RGB 视角；
-   可选输入 Wide/Main/Tele 三摄图像；
-   输出统一世界坐标系下的 3D Gaussian 表示；
-   无需 per-scene test-time optimization；
-   在 held-out 相机视角渲染人物；
-   在质量提升的同时控制推理延迟、显存和 Gaussian 数量。

### 1.2 Ghosting 与 Blur 必须分开处理

  ------------------------------------------------------------------------------------------------------------------------------------------
  现象                                 典型原因                                               主要解决路径
  ------------------------------------ ------------------------------------------------------ ----------------------------------------------
  Ghosting：轮廓双边、重复表面、重影   深度/位姿错误、遮挡关系错误、重复                      几何保护、可见性约束、跨视角一致性、Gaussian
                                       Gaussian、跨视角点云错位                               去重

  Blur：纹理软、边缘不清晰             Gaussian scale                                         高分辨率 Appearance
                                       过大、外观特征分辨率低、视角特征混合、图像预处理损失   Branch、尺度约束、局部纹理采样

  Flicker：视角变化时细节跳变          不同视角属性预测不一致、融合不稳定                     共享 3D 属性、置信度加权融合、跨视角一致性

  Floaters：漂浮 Gaussian              低置信度深度、无效背景点、遮挡误判                     depth validity、confidence gating、opacity
                                                                                              pruning
  ------------------------------------------------------------------------------------------------------------------------------------------

核心原则：

> **几何决定 Gaussian 在哪里；外观决定 Gaussian
> 长什么样；置信度与可见性决定哪些观测可以参与融合。**

------------------------------------------------------------------------

## 2. 总体技术路线

采用 Geometry-Texture Decoupling 设计，简称 **DA3-GT**：

1.  保留 DA3 预训练几何推理主干，尽量不改变其核心几何表征；
2.  使用低分辨率 Geometry Branch 预测 Depth / Ray / Camera /
    Confidence；
3.  增加 4× 线性分辨率 Appearance Branch，提取高分辨率纹理特征；
4.  通过几何投影将高分辨率外观特征采样到 Gaussian 对应位置；
5.  Gaussian 中心由几何分支确定，外观分支主要预测
    scale、rotation、opacity、SH/color 等属性；
6.  引入 visibility- and confidence-aware 的跨视角 Gaussian Fusion；
7.  使用 held-out target rendering、深度和多视角一致性损失训练；
8.  逐阶段解冻，不一开始联合大幅更新全部 DA3 参数。

### 2.1 结构概览

``` text
Sparse RGB Views (+ optional Wide/Main/Tele)
                    |
              Shared DA3 Backbone
                    |
          +---------+----------------+
          |                          |
    Geometry Branch             Appearance Branch
    low resolution              4x linear resolution
          |                          |
   Depth / Ray / Camera        High-res texture features
   Confidence / Geo feature          |
          |                          |
   Back-project to 3D                |
          +------ Project Gaussian centers
                         |
                  Feature Sampling
                         |
           Geometry + Appearance + Confidence
                         |
               Gaussian Attribute Head
                         |
       Center / Scale / Rotation / Opacity / SH
                         |
        Visibility-aware Gaussian Fusion
                         |
                  Gaussian Renderer
                         |
              Held-out Target Views
```

------------------------------------------------------------------------

## 3. 架构设计

### 3.1 Geometry Branch：保留 DA3 几何能力

**目标**：提供稳定的 3D 位置、相机关系和几何置信度。

输入：多视角 RGB，具体组织方式沿用当前 DA3 的多视角接口。

输出建议统一为：

-   `depth`；
-   `ray` 或可从相机内参计算的 camera ray；
-   `camera_pose` / `w2c`；
-   `geometry_feature`；
-   `geometry_confidence`；
-   `valid_mask`。

注意：

-   必须先审计 DA3 当前输出的 depth 定义、尺度、ray 方向和坐标系；
-   不要假设 DA3 的深度一定是 metric depth；
-   不要假设中间特征已经暴露为 `features`；
-   若需取 decoder 特征，应在实际 DPT/dense prediction head
    中增加受控的中间特征返回接口；
-   DA3 预测相机与 UE GT 相机训练对齐时，要统一坐标系、尺度和位姿约定。

**训练策略**：

-   Stage 1：冻结 geometry backbone/head；
-   Stage 2：只解冻少量 adapter 或最后若干层（若验证确有收益）；
-   Stage 3：可选局部高分辨率几何残差，不默认全量解冻。

### 3.2 4× Appearance Branch：高分辨率纹理恢复

这里的 4× 指 **宽和高分别扩大 4 倍**，像素数量约为 16
倍，而不是总像素数扩大 4 倍。

若 geometry grid 为 (H/4 `\times `{=tex}W/4)，Appearance Branch 可在 (H
`\times `{=tex}W) 特征网格上运行；如果 DA3 原始 geometry stride 不是
4，应按真实 stride 重新定义。

建议结构：

-   轻量 CNN / depthwise separable blocks；
-   多尺度局部卷积；
-   不增加第二套全局多视角 Transformer；
-   可选轻量 FPN 或局部窗口交互；
-   输出 32--128 通道的高分辨率 feature map，具体通道数通过 profile
    决定。

输入策略：

-   Main camera 作为主要外观参考；
-   Wide 提供大视野上下文；
-   Tele 作为局部高频细节补充；
-   三摄特征必须先经过相机投影和可见性判断，不可直接按图像坐标拼接。

### 3.3 Geometry-guided Feature Sampling

对每个 Gaussian 位置 (P_i)，投影到可见输入相机：

\[ p\_{iv}=`\pi`{=tex}(K_v,R_v,t_v,P_i) \]

在高分辨率 Appearance Feature Map 上采样：

\[ F\^{app}\_{iv}=`\operatorname{GridSample}`{=tex}(F\^{app}*v,p*{iv})
\]

采样必须处理：

-   图像边界；
-   深度有效性；
-   遮挡；
-   视角可见性；
-   投影点与目标深度的前后关系。

多视角融合特征：

\[ `\bar `{=tex}F_i = `\frac{\sum_v w_{iv}F^{app}_{iv}}`{=tex}
{`\sum`{=tex}*v w*{iv}+`\epsilon`{=tex}} \]

权重建议由以下因素组成：

\[
w\_{iv}=w^{vis}*{iv}`\cdot `{=tex}w\^{conf}*{iv}`\cdot `{=tex}w^{angle}\_{iv}
\]

其中：

-   (w\^{vis})：可见性；
-   (w\^{conf})：几何置信度；
-   (w\^{angle})：视角质量权重，可抑制极端掠射角观测。

### 3.4 Gaussian Center 与属性预测解耦

每个几何网格位置的 Gaussian center 由深度和 ray 反投影：

\[ P_i=t_v+d_i r_i \]

具体形式取决于 DA3 的 ray/depth 定义。若 ray 非单位向量或 depth 为
camera-Z，必须先做正确换算。

Gaussian 属性：

\[ G_i={P_i,`\alpha`{=tex}\_i,q_i,s_i,SH_i} \]

其中：

-   (P_i)：中心；
-   (`\alpha`{=tex}\_i)：opacity；
-   (q_i)：rotation quaternion；
-   (s_i)：3D scale；
-   (SH_i)：颜色/球谐系数。

Attribute Head 输入：

\[ h_i = `\operatorname{MLP}`{=tex}
(F_i^{geo},`\bar `{=tex}F_i^{app},c_i) \]

输出：

-   opacity；
-   rotation；
-   scale；
-   SH/color。

**V1 不允许 Appearance Branch 任意移动 Gaussian center。**
这样可以避免纹理边缘误导几何位置。若后续确实需要位置
refinement，必须通过独立的小幅 residual、置信度门控和几何监督实现。

### 3.5 Confidence-aware Gaussian Fusion

多视角重建可能在同一表面附近产生重复 Gaussian。建议设计 GPU 友好的融合：

1.  将 Gaussian center 映射到 voxel/hash cell；
2.  对候选邻居执行局部距离检查；
3.  加入深度一致性、法线相似度和投影一致性 gating；
4.  仅对同一表面的候选 Gaussian 进行加权融合；
5.  保留薄结构、边缘和前后表面，禁止仅凭 voxel 相同就合并。

融合权重：

\[ w_i = c_i `\cdot `{=tex}v_i `\cdot `{=tex}q_i \]

其中 (c_i) 为几何置信度，(v_i) 为可见性，(q_i) 为观测质量。

**重要风险**：薄结构（手指、头发、衣物边缘）可能被错误合并。必须使用
depth gap / normal / silhouette gating，并对细结构单独评估。

------------------------------------------------------------------------

## 4. 数据集对模型训练的要求

使用配套文档《UE
人物多视角训练数据集构建规范（V1.1）》作为数据生产标准。

### 4.1 V1 必需数据

-   16 张 input RGB（13 张稀疏主摄 + 3 张三摄）；
-   125 张 eval RGB；
-   每张图像的 `K`、`w2c`、`c2w`；
-   radial depth；
-   z-depth；
-   depth valid mask；
-   foreground / silhouette mask；
-   character / environment / pose / sequence ID；
-   固定渲染配置和版本。

### 4.2 强烈建议数据

-   world normal；
-   world position；
-   part ID；
-   instance ID；
-   albedo / material ID；
-   per-view visibility。

这些真值可以先只用于 loss、mask 或评估，不要求全都成为网络输入。

### 4.3 数据划分

禁止同一 sequence 的不同相机视角跨 train/val/test。

建议至少报告：

-   Seen Character / Unseen Environment；
-   Unseen Character / Seen Environment；
-   Unseen Character / Unseen Environment。

如果第一批数据规模不足，先做 scene-held-out
测试，不要把同一人物资产的相邻视角随机拆分后宣称模型具有身份泛化能力。

### 4.4 训练输入配置

分别评估：

-   3-view：s03/s06/s09；
-   5-view：s00/s03/s06/s09/s12；
-   7-view；
-   13-view；
-   Main-only；
-   Wide + Main；
-   Main + Tele；
-   Wide + Main + Tele。

三摄组合必须作为独立实验组。Tele
只覆盖脸部时，应使用局部可见性监督，不能要求它为全身提供纹理。

------------------------------------------------------------------------

## 5. 损失函数设计

第一版优先保证损失定义正确和 mask 可靠，不要一开始堆叠过多复杂项。

### 5.1 Held-out Rendering Loss

\[ L\_{render}= `\lambda`{=tex}*{rgb}L*{charb}
+`\lambda`{=tex}*{ssim}L*{ssim} +`\lambda`{=tex}*{lpips}L*{lpips} \]

建议：

-   RGB Charbonnier 或 L1；
-   SSIM；
-   LPIPS；
-   可选 DISTS 作为离线评估指标；
-   对 foreground 与 background 分开统计；
-   对高频边缘可增加 edge-aware image
    loss，但不能把纹理边缘当作深度边缘。

### 5.2 Depth Loss

有 GT depth 时：

\[ L\_{depth}= `\frac{1}{|\Omega|}`{=tex}
`\sum`{=tex}*{p`\in`{=tex}`\Omega`{=tex}}
`\rho`{=tex}(D*{pred}(p)-D\_{gt}(p)) \]

其中 (`\Omega`{=tex}) 必须包含 valid mask。若预测 depth 存在 scale/shift
ambiguity，应先对齐尺度或使用 scale-shift invariant loss；若相机和深度已
metric 对齐，则不应任意做 scale-shift 归一化而抹掉尺度误差。

可加入梯度项：

\[ L\_{grad}=`\sum`{=tex}\_k
\|`\nabla`{=tex}*kD*{pred}-`\nabla`{=tex}*kD*{gt}\|\_1 \]

### 5.3 Multi-view Reprojection Consistency

将视图 A 的 3D 点投影到视图 B：

\[ p_B=`\pi`{=tex}(K_B,T\_{B`\leftarrow `{=tex}A}P_A) \]

仅在 B 中有效、可见且无遮挡的像素计算：

\[ L\_{cons}= `\sum`{=tex}*p M*{vis}(p),
`\rho`{=tex}(z\^{proj}\_B(p)-D_B(p_B)) \]

mask 至少排除：

-   图像外投影；
-   无效深度；
-   遮挡；
-   轮廓附近的歧义区域（可用软权重，而不是全部删除）。

### 5.4 Gaussian Scale Regularization

不能简单要求所有 Gaussian 都小。scale 应考虑局部点间距、表面方向和投影
footprint。

投影协方差：

\[ `\Sigma`{=tex}*{2D}=J W`\Sigma`{=tex}*{3D}W\^T J\^T \]

可对异常大的投影 footprint 加惩罚：

\[ L\_{proj-scale}= `\frac`{=tex}1N`\sum`{=tex}*i
`\max`{=tex}(0,`\lambda`{=tex}*{max}(`\Sigma`{=tex}\_{2D,i})-`\tau`{=tex})\^2
\]

阈值 (`\tau`{=tex}) 应由分辨率、目标距离和基线实验确定。

### 5.5 Silhouette / Normal Loss

若 UE 提供 mask 和 normal，可加入：

\[ L\_{sil}=1-`\operatorname{IoU}`{=tex}(M\_{render},M\_{gt}) \]

或 silhouette BCE / Dice。

法线监督只在有效可见表面上使用，注意 world normal 与 camera normal
的坐标约定。

### 5.6 总损失与启用顺序

\[ L= `\lambda`{=tex}*rL*{render} +`\lambda`{=tex}*dL*{depth}
+`\lambda`{=tex}*gL*{grad} +`\lambda`{=tex}*cL*{cons}
+`\lambda`{=tex}*sL*{proj-scale} +`\lambda`{=tex}*mL*{sil}
+`\lambda`{=tex}*nL*{normal} \]

推荐启用顺序：

1.  RGB + SSIM + depth；
2.  加入 LPIPS；
3.  加入 multi-view consistency；
4.  加入 scale / silhouette；
5.  最后评估 normal loss。

所有 loss 权重都应通过验证集消融确定，不把预设权重当作已验证结论。

------------------------------------------------------------------------

## 6. 分阶段训练计划

### Stage 0：代码审计与基线复现

**目标**：确认 DA3、相机、Gaussian renderer 的接口和坐标定义。

工作项：

-   记录 DA3 backbone、decoder、depth/ray/camera 输出 shape；
-   确认 depth 是 metric、relative、radial 还是 camera-Z；
-   确认相机坐标和矩阵方向；
-   确认 Gaussian renderer 要求的属性格式；
-   复现原始 DA3-GS baseline；
-   记录 FPS、端到端 latency、峰值显存、Gaussian 数量。

交付：

-   baseline checkpoint；
-   一套固定评估输入；
-   相机/深度单元测试；
-   性能 profile。

**放行条件**：相机重投影和深度反投影通过测试，baseline 可重复运行。

### Stage 1：DA3-GT Basic

**目标**：验证高分辨率外观分支能否在不破坏几何的情况下改善细节。

模块：

-   冻结 DA3 geometry backbone/head；
-   新增轻量 4× Appearance Encoder；
-   新增 Feature Sampler；
-   新增 Gaussian Attribute Head；
-   Gaussian center 由 geometry branch 决定；
-   暂不加入复杂融合，先保持原有 Gaussian 生成方式。

训练：

-   输入：3/5/7/13-view；
-   目标：held-out target views；
-   loss：RGB + SSIM + LPIPS（LPIPS 可在训练稳定后加入）+ depth；
-   GT pose 用于训练对齐时，推理仍使用 DA3
    预测相机；两者尺度和坐标必须统一。

验收：

-   PSNR / SSIM 不下降；
-   LPIPS 改善；
-   人物轮廓与深度误差不恶化；
-   额外推理延迟可控；
-   Gaussian 数量不显著增加。

### Stage 2：Confidence & Multi-view Consistency

**目标**：减少重影、重复 Gaussian 和跨视角不一致。

新增：

-   geometry confidence head（若 DA3 已有可靠
    confidence，可先直接使用）；
-   per-view visibility mask；
-   reprojection consistency loss；
-   confidence-aware feature aggregation；
-   voxel/hash candidate grouping；
-   depth/normal/projection gating；
-   opacity pruning。

训练：

-   使用多基线 target views；
-   加入 near/mid/wide baseline 分组采样；
-   加入遮挡边界和细结构困难样本；
-   使用 held-out views 监督，而不是把输入视角本身当作唯一监督。

验收：

-   Ghosting Ratio 下降；
-   Reprojection Inconsistency 下降；
-   silhouette error 不恶化；
-   薄结构保留率不下降；
-   运行时间和 Gaussian 数量符合预算。

### Stage 3：可选局部高分辨率几何细化

**目标**：只在几何不确定或真实深度边界处细化，不做全图高分辨率几何预测。

形式：

\[ D_h=U(D_l)+M\_{detail}`\Delta `{=tex}D_h \]

\[ M\_{detail}= `\mathbf{1}`{=tex}
\[\|`\nabla `{=tex}D_l\|\>`\tau`{=tex}\_e
`\lor `{=tex}C_g\<`\tau`{=tex}*c `\lor `{=tex}E*{mv}\>`\tau`{=tex}\_m\]
\]

其中 `E_mv` 是多视角不一致性。**不要直接用 RGB edge
作为唯一几何边缘依据**，纹理边缘并不一定对应深度边缘。

只在 Stage 1/2 证明高分辨率 Appearance 仍不足以恢复轮廓/薄结构时启动。

### Stage 4：写实扰动与鲁棒性

基础模型稳定后再逐步加入：

-   逐视角 ±0.3 EV；
-   seeded camera pose jitter；
-   轻微曝光/白平衡差异；
-   景深；
-   动态人物和视频序列；
-   更复杂环境与材质。

每次只增加一种扰动并单独评估，避免无法定位性能下降原因。

------------------------------------------------------------------------

## 7. 推理效率与资源预算

### 7.1 设计约束

-   不在高分辨率分支中运行第二个全局 Transformer；
-   Geometry Branch 保持低分辨率；
-   Appearance Branch 使用轻量 CNN；
-   不按每个高分辨率像素生成 Gaussian；
-   以几何网格或稀疏候选点作为 Gaussian 数量基准；
-   只在确有收益的细结构区域稀疏增密；
-   GPU 上执行 voxel/hash grouping 与 segment reduction；
-   避免 Python O(N²) Gaussian 邻域计算；
-   不做 per-scene test-time optimization。

### 7.2 4× 分辨率的成本说明

4× 线性分辨率意味着像素数约 16×。即便 Appearance Branch
很轻，也可能成为显存和延迟瓶颈。因此：

-   先测实际 feature stride；
-   采用较少通道；
-   可在高分辨率分支使用 FP16/BF16；
-   只对 Gaussian 投影位置做稀疏 feature sampling；
-   若整幅高分辨率特征图过大，评估 tile-based / multi-scale feature；
-   不要默认 4× 分支必然只增加 10--20% 延迟，必须实测。

### 7.3 性能目标

建议将"额外延迟不超过 10--20%"作为**研发目标而非保证**。分项测量：

-   DA3 backbone；
-   geometry head；
-   appearance encoder；
-   projection / grid sampling；
-   attribute head；
-   Gaussian fusion；
-   rasterizer。

每次提交记录：

-   batch size；
-   输入分辨率；
-   输入视角数；
-   Gaussian 数；
-   latency（平均和 P95）；
-   FPS；
-   peak VRAM；
-   GPU 型号和软件环境。

------------------------------------------------------------------------

## 8. 评估体系

### 8.1 图像质量

-   PSNR；
-   SSIM；
-   LPIPS；
-   DISTS；
-   foreground-only PSNR/SSIM/LPIPS；
-   silhouette IoU / boundary F-score。

### 8.2 几何质量

-   Depth AbsRel；
-   Depth RMSE；
-   median / P95 depth error；
-   Chamfer Distance（若有 GT mesh/point cloud）；
-   normal angular error；
-   silhouette reprojection error。

### 8.3 Ghosting 专项指标

建议至少实现：

**A. Reprojection Inconsistency**

把一个视角的可见表面投影到另一视角，与 GT depth / silhouette
对比，报告误差和有效匹配比例。

**B. Ghosting Ratio**

定义为违反多视角可见空间约束、或在已观测表面附近形成重复表面的 Gaussian
比例。必须明确：

-   哪些 Gaussian 属于有效观测空间；
-   如何排除遮挡和未观测区域；
-   重复表面阈值如何设定；
-   指标是否按 Gaussian 数量或 opacity 加权。

**C. Silhouette Double-edge Score**

在渲染轮廓与 GT 轮廓之间检测双边/偏移边缘，可报告 boundary distance
和双边比例。

**D. Temporal/Angular Stability**

对相邻 novel views 的轮廓、局部纹理和可见表面进行连续性分析。虽然 V1
是静态多视角，也可以按方位角排序评估角度连续性。

### 8.4 Blur 专项指标

-   LPIPS / DISTS；
-   edge sharpness；
-   高频能量保留率；
-   Laplacian variance（作为辅助，不单独作为质量结论）；
-   头发、眼睛、衣物纹理等 ROI 的局部指标；
-   Gaussian projected footprint 分布。

### 8.5 必须保存的可视化

每个实验保存：

-   input RGB；
-   GT target RGB；
-   rendered RGB；
-   absolute RGB difference；
-   rendered depth；
-   GT depth；
-   depth error；
-   silhouette overlay；
-   Gaussian center/scale/opacity/confidence；
-   每个视角的可见性与融合权重；
-   Ghosting/Blur ROI crop。

------------------------------------------------------------------------

## 9. 消融实验设计

建议按下表逐项验证，不要一次性加入所有模块。

  实验   Geometry           Appearance   Confidence   Fusion   目的
  ------ ------------------ ------------ ------------ -------- ----------------------
  E0     原始               原始         无           原始     Baseline
  E1     冻结               原始         无           原始     检查 geometry freeze
  E2     冻结               4×           无           原始     验证高分辨率外观
  E3     冻结               4×           有           原始     验证置信度
  E4     冻结               4×           有           有       验证跨视角融合
  E5     局部微调           4×           有           有       验证有限几何适配
  E6     局部高分辨率几何   4×           有           有       验证局部几何残差

输入视角消融：

-   3 / 5 / 7 / 13 views；
-   Main-only vs 三摄；
-   near-baseline vs wide-baseline；
-   训练视角与评估视角的角度间隔。

数据消融：

-   仅 clean；
-   clean + EV；
-   clean + pose jitter；
-   clean + depth/visibility masks；
-   人物/环境未见测试。

------------------------------------------------------------------------

## 10. 主要风险与应对

  -------------------------------------------------------------------------------------
  风险                    可能后果                   应对
  ----------------------- -------------------------- ----------------------------------
  DA3 depth/ray 定义与 UE 几何整体偏移或尺度错误     Stage 0 做单元测试和尺度对齐
  GT 不一致                                          

  相机矩阵转换错误        所有多视角监督失效         用 3D landmark 投影闭环验证

  4× feature map 显存过高 OOM / latency 激增         降通道、tile、稀疏采样、混合精度

  appearance feature      纹理重影                   visibility + depth gating
  混入错误视角                                       

  voxel fusion            手指、头发、衣物边缘消失   normal/depth/silhouette gating
  误合并薄结构                                       

  Gaussian scale          空洞、噪点、渲染不连续     按局部间距和投影 footprint 正则
  被过度压小                                         

  UE 数据过于干净         真实输入域泛化不足         V1 稳定后逐项加入扰动

  人物身份泄漏            泛化指标虚高               character/environment split

  背面未观测              模型背面细节不可控         分开报告 observed/unobserved
                                                     region
  -------------------------------------------------------------------------------------

------------------------------------------------------------------------

## 11. 里程碑与交付物

### M0：数据与几何链路验证

交付：

-   UE 数据集 V1.1；
-   camera/depth verifier；
-   相机投影闭环测试；
-   DA3 baseline；
-   固定 benchmark。

验收：坐标、深度、内参和 mask 通过验收。

### M1：DA3-GT Basic

交付：

-   Geometry Branch wrapper；
-   Appearance Encoder；
-   Feature Sampler；
-   Gaussian Attribute Head；
-   Stage 1 训练脚本和 checkpoint。

验收：外观质量改善，几何指标不显著恶化，推理成本可控。

### M2：Multi-view Consistency

交付：

-   visibility/confidence aggregation；
-   reprojection consistency loss；
-   Gaussian fusion；
-   opacity pruning；
-   Ghosting 指标脚本。

验收：Ghosting 指标和可视化改善，薄结构保留。

### M3：局部几何细化（条件启动）

交付：

-   edge/confidence-guided geometry residual；
-   消融报告。

验收：几何边界改善，且不造成大面积纹理边缘误判。

### M4：鲁棒性扩展

交付：

-   EV / pose jitter / DoF 数据；
-   域扰动实验；
-   最终性能与泛化报告。

------------------------------------------------------------------------

## 12. 推荐代码组织

以下为建议组织方式，需根据现有 DA3
仓库结构适配，不应为了匹配目录而重写原始模型。

``` text
models/
  geometry/
    da3_geometry_adapter.py
    ray_utils.py
    camera_utils.py
  appearance/
    appearance_encoder.py
    feature_sampler.py
  gaussian/
    attribute_head.py
    confidence_head.py
    fusion.py
  da3_gt.py

losses/
  render_loss.py
  depth_loss.py
  reprojection_consistency.py
  scale_regularization.py
  silhouette_loss.py

datasets/
  ue_human_dataset.py
  view_sampler.py
  camera_loader.py
  depth_loader.py
  verifier.py

training/
  train_stage1.py
  train_stage2.py
  train_stage3.py
  evaluate.py

configs/
  dataset_ue_human.yaml
  da3_gt_stage1.yaml
  da3_gt_stage2.yaml
```

### 12.1 代码审计时必须确认的接口

-   DA3 forward 输入和多视角 batch 结构；
-   backbone/decoder 中可用的 feature tensor；
-   depth/ray/camera 输出语义；
-   相机内外参的矩阵方向；
-   Gaussian renderer 的属性 shape、坐标系、scale 参数化；
-   rasterizer 对 SH、opacity、quaternion 的约定；
-   训练时 GT camera 与推理预测 camera 的切换机制。

------------------------------------------------------------------------

## 13. 第一版建议冻结的决策

为了避免范围失控，建议 V1 固定：

1.  只做静态人物；
2.  只做干净渲染；
3.  先不加景深、手持抖动和 EV；
4.  保留 DA3 几何主干；
5.  先做 4× Appearance Branch；
6.  Gaussian center 不由 appearance branch 自由预测；
7.  第一版先做简单可见性过滤，再做完整 Gaussian fusion；
8.  先验证 3/5/7/13-view；
9.  先完成 Stage 0 和 Stage 1，再决定是否进入 Stage 2；
10. 不以单一 PSNR 作为成功标准。

------------------------------------------------------------------------

## 14. 成功标准

项目成功必须同时满足三类目标：

### 几何

-   深度与重投影误差不恶化；
-   人物轮廓和薄结构保持；
-   跨视角可见表面一致；
-   不出现明显重复表面和漂浮 Gaussian。

### 外观

-   held-out view 的 PSNR/SSIM/LPIPS 综合改善；
-   头发、眼睛、服装等 ROI 更清晰；
-   Gaussian projected footprint 不再系统性过大；
-   角度变化时纹理不明显跳变。

### 效率

-   不引入 per-scene optimization；
-   Gaussian 数量不因高分辨率分支按像素数膨胀；
-   记录端到端 latency、P95 latency、FPS 和 peak VRAM；
-   Appearance Branch 的成本通过 profile
    量化，并符合项目预先设定的预算。

------------------------------------------------------------------------

## 15. 执行优先级

建议严格按以下顺序执行：

1.  **先修数据**：相机内参、坐标转换、radial/z-depth、mask、数据划分；
2.  **再复现 baseline**：固定输入、固定评估环、profile；
3.  **实现 DA3-GT Basic**：冻结几何 + 4× Appearance + geometry-guided
    sampling；
4.  **分析误差类型**：确认残余问题究竟是 Ghosting 还是 Blur；
5.  **实现 confidence / visibility / fusion**；
6.  **必要时再做局部高分辨率几何细化**；
7.  **最后增加写实扰动和动态数据**。

**最终建议**：不要把"4×
外观分支"当作解决重影的主要手段。它主要针对纹理模糊；重影需要依靠可靠相机几何、遮挡感知、跨视角一致性和谨慎的
Gaussian 融合。两条路径应分别实现、分别评估，再组合成完整系统。

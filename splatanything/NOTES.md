# SplatAnything（InfiniSplat v3）原理与排错

本文件记录源码机制、排错及明确标注的候选架构；服务器运行见 [README.md](README.md)，本机 WSL 运行见 [README_wsl.md](README_wsl.md)，
实测数字见 EXPERIMENTS.md。候选设计不是已实现功能或实验结论。

官方仓：<https://github.com/zju3dv/SplatAnything>（commit `b518996`，2026-11-12 clone）
权重：<https://huggingface.co/ritianyu/SplatAnything>（apache-2.0，非 gated）

---

## 深度倍率和场景整体缩放的区别（2026-11-24）

三台相机中心固定，每张源图的前景点沿自己的相机射线按同一个倍率调整深度。因为三条射线从不同位置出发，这一步会改变各源表面的相对位置，并能改变新视角下的鼻部层距。若将相机间距和整个场景一起缩放，投影关系会保留，无法做这个诊断。为避免用验收答案调倍率，取已知相机基线和源图的跨视图纹理对应作为距离参照；对应不足时回退。完整高斯与点云的可见性不同，应分别验收。[两静态身份实测](EXPERIMENTS.md)

## 单张点云精细，如何判断三张点云放得是否正确（2026-11-24）

单张点云可以保留鼻翼和眼窝的局部起伏；整张脸沿相机光轴偏远时，源图上依然可能轮廓清楚。从另一个机位看，相机光轴方向不同，几张精细表面就会错开。把各源脸配到同一张脸只约束彼此的位置，参考脸本身也可能偏离真实位置。用独立目标的鼻部、下颌和深度检查绝对位置，再用同一射线上多源表面的间距检查相对重合。将自动配准施加于真值点云，可量出配准本身引入的误差。见[静态真值与黑衣实测](EXPERIMENTS.md)。

## 鼻尖收拢与脸部归位如何分别验收（2026-11-24）

把来自不同源图的鼻尖投到留出相机，两点间距衡量预测之间的一致性。即便两点重合，它们仍可能一起偏离实拍脸。调深度还会改变轮廓遮挡和体素成员，需在最终导出的高斯上检查留出实拍、脸框外的第二张脸、有效覆盖和深度真值。32 号黑衣 49 的两源间距减少了约 11 px，最终脸框误差却上升。降低透明度也会改变沿视线的期望深度，即使没有移动任何中心，Daven 的 B 分支已有这一现象。见 [A/B/C 实测](EXPERIMENTS.md)。

## 三张来源如何形成一张留出图（2026-11-24）

教师为每张源图估计深度。解码器把每个像素沿自己的相机射线放到这个深度，作为高斯中心，同时预测大小、方向、颜色修正、不透明度和融合权重。把三个来源打上颜色再投到留出机位，就能看到哪张照片贡献了哪一块脸。即使每个源图单看都只有一个鼻子，投到同一机位时，中心位置不一致也会留下几条轮廓。

检查融合时应分别控制不透明度和体素分组。直接渲染解码 alpha 与正式合并输出之间同时改变了这两项。先把每个点的 alpha 换成融合权重，再用同一合并函数分源分组，最后允许跨源分组，才能分别观察各步贡献。相隔较远的脸不会落进同一个体素；同一张照片内的点也可能在一个体素被平均。按源单独渲染、来源着色、移除一源三者结合，能区分重复轮廓、露出背景和覆盖下降。实图见 [11-24 实验记录](EXPERIMENTS.md)。

## DensiTok 的接入位置与诊断顺序（2026-11-23，候选方案）

对于鼻部双像，先分清错误发生在哪一步。SA 若在同一静态脸上预测出两层表面，增添虚拟视角可能让两层都更密；若三张观测本来就对齐，只在两机位之间缺少表面支持，插值补全才更有希望。先跑[总计划的真实三图/五图配对](../da3_human/EXPERIMENT_PLAN.md)。记录同三锚点的位姿误差、教师深度、GSN 前后尺度、融合前后鼻部误差与覆盖。教师已错就优先检查相机与深度；教师稳定而融合后出现第二层，就优先查相邻视角的深度冲突和 voxel 分组。

| 接口 | 通过真实五图门后如何小试 |
|---|---|
| SA 的 GeometryWorldMirror 教师 | 论文的 WorldMirror 与本仓的 WorldMirror-2 权重、层号、patch 格式及相机规范需逐项对照。第一步仅让补全的上下文影响原三视角的相机/深度输出，仍用原三张 RGB、原属性学生和 GSN 出高斯。这样可以单独检验几何收益。记录原三视角输出是否随新增上下文变化；若对应 head 逐视角独立且观测 token 固定，这条接法不产生有效作用，跳过。 |
| SA 的 GaussianWorldMirror 学生 | 学生有独立编码器，DC 以真实 RGB 为底色。要从虚拟位置新增高斯，需同时提供与学生兼容的特征、相机、深度及颜色。先冻结模型，用真实五图中的两帧依次替换为投影颜色做接口对照。颜色从原三图按正深度、画内、前后深度一致及多源一致性取样，遮挡或冲突处保留无效标记。分别统计有效率、鼻缘双层和肤色偏移；通过后才尝试用预测 token 替代真图特征。 |
| ZipSplat | 解码器从场景 token 预测高斯。论文补出的逐视角 token 与场景 token 无直接对应。保留原生 FF 作配对基线；只有 Zip 自身在五图对照也取得稳定几何收益，才评估在输入特征聚合前接入的工程量。 |
| DA3 | 论文有 DA3 的同骨干实验，是公开代码/权重可用后最省接口改造的机制参照。核对权重版本、预处理与深度定义，再做三图原生/无虚拟位置/有虚拟位置配对。其结果只归属该 DA3 版本。 |

论文在两个真实输入视角的 AnySplat 消融中发现，仅编解码观测 token 的提升有限，增加虚拟位置才取得主要增益。这支持保留“零虚拟位置”对照。训练成本评估先报每样本 token 缓存体积、接口改动、显存和单步耗时，再估 GPU 时；论文使用四张 RTX 6000 Ada，训练墙钟未给出。论文结果限于静态场景轨迹内插，实际鼻部效果按总计划独立验收。[论文表 3 与附录](https://arxiv.org/html/2610.07958v1)

## 真实包评测：官方预处理 K 与原生回放（2026-11-22）

- 官方 `scene_io.preprocess_view()` 在第 43 行计算裁剪目标 K，却在第 45 行把旧 `K`
  交给返回变量 `K2`；于是返回内参的焦距已缩放、主点却少减整数 crop offset。
  本真实包的纵向偏移达 123.5～126px。这里是预处理链的实现问题，不是可用旧
  `official_preprocess_geometry()["K_reference"]` 反证全部原生相机错误的理由。
- **共同 target GT** 的 RGB 与实际投影 K 必须对应同一像素映射。RGB-only 的虚拟 K 只确定
  主轨官方居中 crop；target 投影使用真实采集 K，并施加完全相同的 crop/resize/integer-crop。
  SA RGB 与 Zip 共用 `group[0]=SA_RGB` 的主轨 RGB、投影 K 和主轨 ROI。known-pose 的采集 K
  同时决定它自己的 crop 和投影，因此辅助轨另有 RGB 画布与 ROI，不能与主轨混成同一逐像素表。
- `evaluation_geometry()` 的投影 K 在**图像 resize 之前**调用 `rescale_image_depthmap`，第三参为
  投影 K、返回值赋给 K；v16 曾误读已经 resize 的图，倍率被算成 1，真实 fx=1380.817 未缩放，
  而 known-pose 正确输出 fx=726.948。v11/v16 的 target 数值全部作废，不因脚本退出 0 而采用。
  旧 `official_preprocess_geometry()` 在 target K 已减 crop offset 后又减 bbox，属于第二次偏移。
  **SA 原生 PLY** 必须用 `cameras.json` 的实际导出 K；RGB-only 是几何网络逐视角预测
  fx/fy/cx=cy=378，不是虚拟预处理 K；known-pose 导出 K 则等于官方前馈实际接收的
  预处理返回 K。原生回放与共同 GT 评测不得共享一套错误的“统一原生 K”。
- Zip 原生长边 504、宽 378；`metrics.json` 的 `model_input_size=[378,504]` 为 `[H,W]`。
  其原图相机 `[W,H,fx,fy,cx,cy]` 按长边 504 等比，得到原生 `K=[fx*.26275,cx*.26275]`；
  与审计保存的 model camera 一致。主轨共同评测仍统一 756²/eps2d=.01，原生核验另用 eps=.3。
- 原生 PLY 回放门只比较同一原生相机/K 下的两个 PLY 读取器及 alpha；输入 RGB 对拍只是
  前馈 source 重建质量诊断，**不是**官方原生渲染参考，不能拿 3/255 判读取器失败。
- 此项影响历史 `21_geometry_audit_v2.py` 的 K 对账解释。Rosemary 单包正式数值本轮不回改；
  若复审历史几何数字，应先重跑受影响的预处理 K 契约，不沿用“K_reference 必为官方正确 K”。

## N1 对齐与评测契约（2026-11-22 已实测）

- `22_n1_head_compare.py` 正式输出为 `rosemary_v27`。原生 PLY 保持不变，将 GT 相机逆回模型系；
  `X_gt=s R X_model+T` 的逆相机使用 `1/s,R.T,-(R.T@T)/s`，ED 乘 `s` 回米制。SA 复用 M03/M13/N0
  `model_camera`，通用 PLY 读取器只作交叉诊断，避免 exp/sigmoid 重舍入污染正式结果。
- Zip degree-1 SH 的系数旋转为 `D=A R A.T`，CUDA 等变最大差 1.19e-7；原生 PLY、正向 Sim(3) 与共同 GT
  模型质量是三个独立门。SA 正向少量 RGB 极值仍是未通过诊断，不得改写成全部通过。
- `zipsplat/evaluate_jiajia.py` 旧 Wahba 函数 `Vt.T@D@U.T` 在其行向量 M 定义下方向不符；N1 使用
  `U@D@Vt`，最终 model→GT 为 `Rw.T`。精确随机相机直接对拍 `R_pred=A R_gt→R_model_to_gt=A.T`，
  往返误差 4.44e-16。旧历史 JSON 不重写；后续复跑旧脚本前必须迁移此修复。
- 本包 Zip 焦距 fx/fy 相对 GT 仅差 0.38%～1.70%；此前 36.5% 属错误比较，已撤回。原生 source RGB
  可达 20.27～24.77dB，共同相机只有 15.61～17.52dB，说明 pose/尺度和场景几何存在耦合，不能拿 source 自洽
  分数证明 target 结构。source 相机中心严格共面，Wahba 仅作预声明敏感性，不按 target 分数换对齐。
- source 紧脸 `median(GT/ED)` 的尺度诊断独立于相机中心；冻结到 target 后 Zip 的局部均值明显下降，
  但尾部和眼嘴错位仍在。尺度补偿不是默认主口径，更不是 target oracle。`eps2d=.3` 不是差距主因。
- ED 只在 `valid GT & ED>1e-5` 的规定 ROI 内评分；无深度背景不得进 full ED。整图 GT 覆盖与每 ROI
  alpha 覆盖分列。结论与命令见 EXPERIMENTS/README_wsl；单包不足以选择 N2 基座。

## N0 坐标/测量契约已实测（2026-11-22）

- 官方 `GaussianSplatRenderer.forward()` 浅拷贝 geometry：`depth/*.npy` 是 GSN 前教师；PLY、未融合
  点云和导出相机是 GSN 后量。本包教师深度总倍率 1.4036118984=pose-align 0.9428013563×spacing
  1.4887673855。原始 lifting 米制=`teacher*total/spacing`；PLY/ED 米制=`model_depth/s`。
- 行向量 model→pkg 是 `(points-T)@R/s`，不是 `@R.T`；后者曾让旧 selected IDs 的紧脸表面样本全为 0。
  主脚本有相机中心逆变换硬门。点云经过过滤/merge，不能按像素网格顺序切片。
- 本包 prior=C@R_manifest@C；五帧 recorded look-at 目标投影距主点≤0.0004055px，raw manifest 不通过。
  不要把“自洽法线中位数接近”误写成相机未定，也不要把 look-at 目标当鼻尖真值。
- GT Depth 为轴向 Z；Depth 反投影得到的点云/法线不是独立扫描网格。最近点距离不证明高斯在该视角
  可见，也不证明材质归属。原 FBX 路径缺失时必须显式保留这项限制。
- FaceMesh 紧脸可包含头发；小 ROI 必须目检。source38 排除误检鼻翼/眼眉，target34 排除左侧眼眉。
  遮挡筛选紧脸是法线朝向+深度连续性代理，不是皮肤分割。source38 鼻尖融合后尾部仍存在，N0 只证明
  测量契约可用，不裁决几何/属性分支。结果见 EXPERIMENTS 顶部及 `five_view_rigid_v28/`。

## 最新机制勘误：深度文件与相机尺度（2026-11-21）

20号F3的毫米精度/“相机错误”结论暂停使用，当前路线见[主计划N0～N2](../da3_human/EXPERIMENT_PLAN.md)。

- human_fbx_gt相机id从0开始、文件从1开始；必须查`cameras.json`的`file_rgb/file_depth`，
  不能按`id:06d`直接构造文件路径。现有20号错配了前一帧；17号的RGB按manifest选取是正确的。
- 当`X_m=s R X_p+t`，刚体相机为`R_m=R R_p,C_m=s R C_p+t`。完整矩阵`S@c2w_p`
  的旋转块是`s R R_p`，不是刚体旋转；两者投影可能相同、深度却相差s。使用标准刚体相机，
  明确`z_metric=z_model/s`，不要把单位转换的改善解释为cameras.json对错。
- 本包CPU对账证明cameras.json与正确刚体链一致；这不代替RGB/alpha/深度的渲染回归。
  还需校验原始K、resize半像素约定、target内参、near/far及背景；不只看相机中心残差。
- [Blender手册](https://docs.blender.org/manual/en/4.2/render/layers/passes.html)将Z pass描述为
  到可见表面的距离。仓内生成器把Depth输出直接写EXR，需已知几何测试其轴向/射线定义，
  再决定是否除以射线范数；先转换再有效区域重采样，不把注释当类型证明。
- 教师每视角GT尺度拟合属oracle诊断；正式几何评测只允许source估一个尺度，再冻结评target。
  期望深度是alpha混合结果，低MAE不证明所有中心/表面正确；须补局部鼻部与深度多层尾部。
- F1的触顶率不足以证明刚性平移；F0回退也不自动隔离参数因果。FLAME鼻尖深度顺序需按
  实际视角/GT几何定义，轮廓应比较同语义可见表面，不能以顶点splat对landmark bbox代替。

ZipSplat“模糊但头更像”先作几何对照假设。结构验收通过才谈固定几何改善外观；不要直接拼
两模型高斯、把锐度当结构，或以模糊隐藏的重影当成几何正确。当前不追加全新表示改造。

## 历史机制审计：为什么P3源视角拟合上升，脸仍然碎裂（2026-11-19）

状态：P0～P3已执行、target未过。最新执行入口为[主计划F0～F3](../da3_human/EXPERIMENT_PLAN.md)，
下方2DGS/纹理网络仍是历史候选。观察支持当前自由局部拟合泛化不足，但不唯一证明数据缺乏。

- `13_face_select.py`以投影中心、单侧期望深度上界和近似footprint计支持数，未计算遮挡后
  每高斯对脸像素的贡献；浮在脸前、被挡住或重复层的点可能仍被选中，未选中的错误层也会留在
  固定块。应以实际贡献/深度层审计验证，不能仅凭二维红点都在脸内声称3D归属正确。
- `14_face_local_opt.py`同时开放DC/opacity/scale/quat/means时，光度拟合允许几组参数
  互相补偿。P2已碎裂说明问题不限于中心自由度；P3触顶率20.8%说明边界活跃，但还不是因果证明。
- 位移及kNN位移差损失直接用模型坐标平方，非尺度不变；应按可信头宽或局部间距归一化，
  同时记录加权损失与梯度范数。简单“lambda调大十倍”无法保证跨包一致。相同颜色的邻居可能
  分属鼻翼两侧/重复脸层；颜色kNN不能替代皮肤语义、拓扑或法线门控。
- 当前scale只做绝对log范围裁剪，opacity均值低于0.05才停；局部点大幅增删透明度仍能通过。
  后续先冻结opacity/DC、限制相对初值变化、检查最差区域；再分阶段放开外观，避免同时自由拟合。
- source loss读取756²重采样图；face指标是扩展方框，person指标排除脸带。原图ROI监督、紧脸
  和鼻部指标须单独实现与标注；不能拿本轮结果宣称这些手段无效。最终RGB clamp也可能掩盖
  参数越界，需同时报告逐高斯DC范围及最终像素/局部肤色变化。
- 表面先验应减少歧义：共享FLAME形状+稳定面片对应+小残差，先保证鼻/颊为同一合理表面，
  再恢复纹理。直接在旧双层脸上叠新脸会留下双重贡献；若需表面替换，作为独立有回退的表示实验。
  [GaussianAvatars](https://arxiv.org/abs/2312.02069)展示面片绑定与局部位移思路，但不是三照片
  重建保证。[SparseGS](https://vgl.ict.usc.edu/Research/SparseGS/)讨论稀疏视角光度歧义和期望
  深度的局限，不能据此未经消融就照搬其删点/SDS模块到人脸。

## 0. 2026-11-16 候选：冻结几何先验 + 低分辨率 2DGS + 高分辨率局部纹理

> 2026-11-19：网络微调前先做[主计划P0～P3](../da3_human/EXPERIMENT_PLAN.md)的SA脸部
> 局部高斯诊断；人物数据准备可继续。mask只限定优化区域，不提供深度；场景参数优化是
> 测试时优化，不是前馈直出。SHARP初始化支线停止投入，以下网络改造继续后置。

> 2026-11-17 优先级更新：先按[主计划 L0～L4](../da3_human/EXPERIMENT_PLAN.md)做原生
> SplatAnything 高质量人物数据 S-LoRA。以下表示/网络改造暂缓；学生LoRA固定教师原深度，
> 不等于几何教师适配，是否进入几何LoRA由直接误差证据决定。

**结论：技术上可行，值得小规模验证；不是直接拼接两个现成模型，也未证明能消除鼻部重影。**
用户提出不同位置表达不同 RGB/alpha，本节将它具体化为前馈预测的 textured surfels。
状态仅为架构评审，无新增训练/渲染实测；不自动替换现有主线或停止服务器训练。
唯一执行入口仍为 [实验总计划](../da3_human/EXPERIMENT_PLAN.md)，鼻部归因 R2-N 保留。

### 0.1 借鉴边界与当前代码事实

- [QuerySplat 论文](https://arxiv.org/html/2608.01186v1)的可借鉴部分是冻结 VGM 与几何/外观
  分支解耦。它用非像素绑定查询产生普通 3D Gaussians，不是逐高斯 RGBA 纹理图，也不是 2DGS。
  本机源码 `~/repos/QuerySplat` commit `9703d6f`：`scripts/models/querysplat.py` 的
  `post_output_dim=color_dim+1`，`scripts/rendering/gs.py` 调普通 3D rasterization。
- [Textured Gaussians 项目](https://textured-gaussians.github.io/)更直接对应局部 RGB/alpha
  纹理表示。[作者公开重实现](https://github.com/bchao1/textured_gaussians)提供 ray/UV 与
  可导纹理渲染，使用精确 2DGS，且说明它与论文内部原实现有差异；其逐场景优化结果不能当作
  我们三图前馈泛化证据。可借用算子思路，需单独核对 CUDA/torch/gsplat 接口与梯度。
- SA 本机 commit `b518996` 的 `src/models/models/rasterization.py`：12 通道为
  quat4/scale3/opacity1/DC residual3/weight1，中心直接由教师深度反投影；合并后 opacity
  为 sum(w*w)/sum(w)。增加纹理不能只在该输出后补通道，渲染与合并接口都需修改。
- **训练状态更新**：本仓已有 `canada_h100/12_sa_train.py` 的可导前向、functional merge、
  gsplat 渲染、参数范围选择和分布式训练。当前源码 `slice_views` 取同批 source/target，
  损失为输入视角重渲染 MSE。本次只读代码及本地运行记录，未检查服务器实时作业或重验梯度；
  不应再说“本仓只有推理”，也不能把它称为本文 textured 2DGS 或跨身份 held-out 训练已通过。
- 日常相机模式遵从 E-18：默认 RGB 自估；参考 fx/look-at 不作 GT。可信标定/UE GT 可另做
  受控几何实验，不能以不可信 pose 的效果差证明某种表示更好。

### 0.2 建议的最小架构（设计，未实现）

1. **固定几何先验**：WorldMirror-2 encoder + camera/depth heads 冻结、eval，输入仅三张
   source；输出相机、深度、置信度及可用特征。冻结保留的是先验，不保证鼻部表面正确。
   SA 的几何教师与高斯学生是两个独立权重 encoder，不能以“冻结骨干”含混地只冻结其中一个。
   首轮两者均冻结，学生特征接轻量可训练头；后续单独评估是否删掉学生 encoder。
2. **低分辨率几何头**：低分辨率指生成 surfel 的格点密度，不必同时降低教师输入分辨率。
   先保留教师原分辨率，减少输出格点作为独立消融。输出中心、完整切平面朝向、两个切向尺度、
   基础 opacity/置信度。初始中心用深度锚点；法线可从有效局部深度估计，深度断层/低置信度
   处另作 mask，不跨轮廓平均。后期才放开有界深度/中心残差，并保留锚定与直接几何监督。
3. **高分辨率外观头**：同三张原始 RGB → 轻量 CNN/FPN → 按 surfel 的局部 UV 位置投影到
   各 source → 可见性/遮挡加权聚合 → 每 surfel 一张小纹理。输入 K、crop、resize、像素中心
   约定必须闭环。先采样真实高分辨率 RGB/特征，不将低分辨率特征插值称为恢复高频。
4. **可导纹理光栅器**：目标射线与 surfel 平面求交，得到局部坐标后采样纹理并 alpha 合成。
   单张纹理绑定三维 primitive，与目标相机共享；不按 target 图生成独立贴片。
   首版不加 view-dependent 网络，低阶 SH/方向残差作为后续单项对照。

建议先借鉴解耦思想，保留 depth-anchored 像素格点和轻量头，**不直接移植 QuerySplat
整套大查询 decoder**。若以后改为全局 scene queries，应单列“像素绑定 vs 场景查询”对照，
不能将改变摆点机制带来的收益归给纹理。

### 0.3 “四块纹理”必须明确的参数化

2DGS primitive 是处于三维空间的椭圆盘面，不是有唯一表面 UV 的体积球。
用中心 μ、切向单位向量 t1/t2 和正尺度 s1/s2 表示局部面：

```text
x(u,v) = μ + s1*u*t1 + s2*v*t2
C(u,v) = sample(T_rgb, UV(u,v))
alpha_eff(u,v) = alpha_base * exp(-(u²+v²)/2) * sample(T_alpha, UV(u,v))
```

其中纹理值有界，局部高斯坐标截断支持域映到 [0,1]²；用双线性采样，缩小时要评估低通/mip
抗锯齿。变法线、尺度或切向旋转也会改变 UV，因此需固定切向基准/纹理朝向规范，尤其不能
在合并时任意翻转/平均 quaternion 后直接逐元素平均纹理。

- 若“四块”指 **2×2 texels**：每 surfel 4 个采样值，是最小验证版；双线性表示能力有限，
  不是足够精细的人脸纹理保证。建议比较 2×2 与 4×4，之后才考虑脸部自适应 8×8。
- 若指四张 P×P 小图：明确它们组成一张 (2P)×(2P) atlas，处理边界滤波；首版不需要四套 UV。
- 若指四个子 Gaussian：这是几何细分，点数/排序成本增加，不能算同一个高斯内部的纹理。
  “2×边长 = 4×像素”不等于“4×边长 = 16×像素”，所有预算用 texel 数明确写出。

**先 RGB、后 alpha**：先固定中心/朝向/尺度/基础 opacity 与纹理 alpha=1，只学 RGB。
随后单独开放纹理 alpha 并记录总覆盖、空洞和鼻部几何；alpha 会改变可见支持，不能当作
纯外观变量。若它只是把错误鼻部层变透明，self 可能提高，但不等于真正修复鼻部几何。
用前景 silhouette/coverage、适度平滑和多视角监督约束，避免把正确轮廓也抹掉。

### 0.4 主要集成难点与收益边界

| 问题 | 首版处理 | 不可承诺的事 |
|---|---|---|
| 鼻部层间错位 | 延续 R2-N，检查 raw depth/中心、可信相机与人物动作 | 贴图不自动让不同位置的表面重合 |
| 低分辨率几何丢失曲率 | 脸/鼻/轮廓保留足够格点，几何密度单列消融 | 高分辨率贴图不能补回缺失的鼻翼几何、背面或新视角视差 |
| 3D→2D 表示迁移 | 重训/适配尺度与朝向头，深度先验初始化；保留普通2DGS控制 | 把3D scale某一维压零或换渲染器不等于训练好的2DGS |
| SA 合并破坏纹理局部坐标 | 优先先确定最终 surfel/朝向，再预测其纹理；绑定稳定 ID | 不能复用现有 DC/quat 加权平均直接合并纹理 |
| 重复表面与拓扑 | 基于位置、法线、深度可见性的去重作为独立变量；初期固定/缓存分组并记录局限 | hard voxel分组不连续，不能宣称整个合并拓扑已可导 |
| 渲染/导出 | 新增 UV/RGBA 可导路径；几何+atlas/纹理张量+元数据一起导出 | 标准 PLY/SuperSplat 不会自动显示纹理；普通PLY只能另导近似版 |
| 显存/时延 | 统计纹理预测、采样、排序/合成与反向峰值，分块处理并记录重算开销 | 冻结 backbone 只节省训练梯度开销，不自动减少前向耗时 |

以 50 万 surfels、FP16 RGBA 估算，纯纹理输出 2×2 为约 15.3 MiB，4×4 为61.0 MiB，
8×8 为244.1 MiB（N×K²×4×2字节）；这不含网络特征、梯度、临时排序缓冲或优化器状态。
纹理是网络预测输出，不默认给每场景每个 texel 建持久 Adam 参数。实际训练显存/时延须测，
不从输出大小直接推断 3090 可运行或性能提高四倍。

以前 DA3 的“仅换2DGS渲染器/GT法线无收益”否定的是那个具体方案，不直接否定本候选；
反过来也不能以本候选为由删除历史负结果。区别是表示、训练头、纹理及渲染算子一起变化，
因此更需要分项消融。表示层面的 RGB/alpha 纹理已有先例；潜在研究价值在三图前馈预测、
低几何/高外观预算分配及跨视角一致性，尚不能宣称这些组合已有新颖性或性能优势。

### 0.5 最小验证矩阵（只设计，未启动）

先对可导渲染器做平面色块/斜视射线验证：恒定 RGB + alpha纹理=1 必须退化为相同参数的
普通2DGS；纹理采样梯度作小规模有限差分检查，验证 UV 方向、遮挡及近掠射处理。
默认三图 RGB 模式；单独用 UE GT 几何/相机作控制，真实数据不喂假定 pose。

| 版本 | 唯一主要变化 | 可归因比较 |
|---|---|---|
| Q0 | 官方 SA 3DGS 前馈 | 最终基线；与2D迁移的差异不算纯纹理增益 |
| Q1 | 同冻结先验 + 适配/训练的普通2DGS头，无纹理 | 先确认表示迁移没有明显丢失几何/覆盖 |
| Q2 | 固定Q1所有几何/opacity/分组，仅预测2×2 RGB纹理 | Q2−Q1：空间颜色变化价值；Q1颜色头也给予同训练预算 |
| Q3 | Q2增加局部alpha纹理 | Q3−Q2：局部透明度收益与覆盖风险 |
| Q4 | Q2或Q3中已验证的版本，将纹理2×2改4×4 | 纹理分辨率收益与资源代价 |
| Q5 | 最后才开放受约束几何残差，仍冻结骨干 | 联合修正是否必要，及鼻部层间误差是否真下降 |

训练身份与验证/测试身份隔离；source严格只有三图，target只作独立监督/评分。先从固定
权重/种子/相机/渲染尺寸/点预算比较，再测同显存或时延预算的 Pareto；禁止一边同时加点数、
换底色、增输入又称纹理收益。保留 pose/RGB 与 oracle 控制的独立标识。
训练损失先 RGB L1/SSIM + 几何/轮廓守门，LPIPS后加；真实伪深度按置信度使用，UE GT 单列。

首轮延续主计划的至多100步工程探测范围，记录可训练参数、梯度、保存恢复、耗时显存；
它只验训练链。候选完整域训练须另按实测成本排期，不因已有服务器长训而默认本架构已获验证。
最终看人物/脸/鼻 held-out LPIPS与PSNR、鼻宽/对应误差、轮廓/空洞/重复表面、原生分辨率细节
及资源；沿用主计划守门。Q1几何退化先修几何；Q2无稳定收益不扩大atlas；Q3靠覆盖塌缩提分
则回退RGB-only。训练集变清晰而新身份/目标视角不改善，不晋级。

---

## 1. 推理链路：两个模型串行，学生不吃教师梯度

```
images (+ 可选 prior_camera.json)
        │
        ├─► GeometryWorldMirror   (worldmirror2.safetensors)   "teacher"
        │      ViT-L/14 (embed 1024 × 24 层)
        │      ├ CamHead   → camera_params[9] → c2w + K      [inference.py:146-150]
        │      └ DPTHead   → depth, depth_conf, mask_logits   output_dim=3
        │
        └─► GaussianWorldMirror   (SplatAnything.ckpt)          "student"
               同一个 encoder（独立权重）
               ├ GSFeatHead → 128ch 特征 × N 视角
               └ DPTDecoder → 每像素 12 维 = quat4 + scale3 + opacity1 + SH_residual3 + weight1
                              means 直接取 depth 反投影（不学位置）
```

关键事实（都有代码依据）：

| 项 | 值 | 出处 |
| --- | --- | --- |
| 每像素高斯原始输出 | 12 维 `[quat4, scale3, opacity1, sh_res3, weight1]` | `rasterization.py:60` |
| 位置来源 | `depth_to_world_coords_points(depth, pose, K)`，**不参与学习** | `rasterization.py:62-64` |
| SH | 只有 DC：`sh = (rgb-0.5)/C0 + residual`，C0=0.28209 | `rasterization.py:68` |
| scale 激活 | `exp().clamp_max(0.3)` | 同上 `:67` |
| opacity / weight | 各自 `sigmoid()` | 同上 |
| 解码精度 | 全程 FP32（显式关掉 autocast） | `gaussian_worldmirror.py:31` |
| 分块解码 | `decode_chunk_size=8`（按视角切，省显存） | `rasterization.py:55` |
| 发布包能力 | **只有推理**：`rasterization.py` 注释写明 "no rasterization or training" | 文件头 |

`cond_flags`：pose 模式 `[1,0,1]`（启用 pose + rays 条件），rgb 模式 `[0,0,0]`。
`enable_cond` 与 `model.use_gt_camera` 必须同时为真或同时为假，否则 `load_config` 直接抛错。

---

## 2. Gaussian Spacing Normalization（GSN）—— v3 的核心

这是 v3 相对 v2 最实质的改动，也是"一个模型跨任意场景"能成立的原因。

### 2.1 做什么

```
spacing = depth / sqrt(fx * fy)      # 每像素在世界里的前向 footprint
factor  = spacing_target / median(spacing)      # spacing_target = 0.0013227513（硬编码）
depth  *= factor                     # 所有视角同一个 factor
c2w[:3,3] *= factor                  # 只缩平移，不缩旋转
```

代码：`rasterization.py:88-127`（`normalize_gaussian_spacing`）。

`spacing_target` 不在 YAML 里（`inference.py:68-83` 的白名单会拒绝多余字段），
是代码常量，注释说它是"per-pixel spacing and geometric mean focal"的归一目标。

### 2.2 为什么需要

不同场景深度差几个数量级（物体 0.3 m、街景 50 m、航拍 300 m），同一模型输出的
高斯 scale 也就差几个数量级。若体素合并用固定 voxel size，小场景会被合并成一个点、
大场景合并等于没做。GSN 把所有场景压到**同一个 spacing 尺度**，于是：

- 网络只需学"相对于 spacing 的 scale"，不用学绝对尺度；
- 体素合并可以用**固定** `gs_voxel_size=0.002`（≈ 1.5 × spacing_target）。

这正是 v2 做不到的事：v2 的 voxel 是 `median_scale × 1.4` 逐场景自适应
（见 `infinisplat_v2/README_wsl.md` 的口径说明），一旦换场景就得重算，且没有统一定义。

### 2.3 pose 模式下的深度尺度对齐

pose 模式给了 GT 相机，但模型自己的深度是"预测尺度"。`da3_pose_align_factor`
（`rasterization.py:8-41`）沿用了 DA3 的 Umeyama fallback：只取**相机中心两两距离的
中位数比值**

```
factor = median(‖C_ref_i − C_ref_j‖ / ‖C_pred_i − C_pred_j‖)
```

方向无关（只比距离），所以对旋转误差不敏感。三个退化保护：

1. 视角数 < 2 → factor = 1；
2. 预测基线 ≈ 0 → 跳过该对；
3. GT 基线 ≈ 0（全景/立方体，所有相机同中心）→ 跳过该对，否则比值被拉到 ~0、
   点云会整体塌到原点（注释明确写了这个坑）。

⚠️ **导出坐标不是米制**。`docs/prepare_data.md` 原文："exported coordinates are not
guaranteed to be metric"。GSN 之后一切都是归一化尺度，量尺寸要自己反算。

---

## 3. 体素合并与过滤

顺序（`rasterization.py:143-161`，全部在合并**之前**判）：

1. `filter_large_scale_infer`：`max_scale_abs`（绝对上限，默认 null）与
   `max_scale_pct`（默认 99.8，丢最大 0.2% 的长条高斯）**同时生效**；
2. `filter_flying_points_infer`（默认 false）：深度不连续处的像素；
3. sky（`--sky_seg` 才启用）：sky 像素既被丢弃，也**不参与** spacing 统计
   （`valid_mask` 只影响统计，不影响缩放本身）。

合并规则（`merge_voxels`，`:167-196`）：

```
voxel 索引 = floor(mean / voxel_size)
means/scales/sh/quats = 按 weight 加权平均（quat 再归一化）
opacity = Σ(w·w) / Σw          ← 注意不是加权平均，是"权重平方和"
```

最后导出层再按 `infer_opacity_thresh=0.05` 丢一批。

**推论**：`points.ply` 是合并前的逐像素中心（带 RGB），`gaussians.ply` 是合并后的。
两者数量差多少，直接反映该场景的重叠率。v2 的经验是丢弃率 5–9%。

### 3.1 合并实际在做什么（E-08/E-09 实测，和直觉相反）

同一场景同一分辨率，只改输入视角数：

| 视角 | 合并保留率 | 每体素均/最多样本 | 体素内 RMS 离散度 | 锐度保留 | PSNR |
| --- | --- | --- | --- | --- | --- |
| 3 | 64.1% | 1.56 / 14 | 0.380 × voxel | 85% | 22.37 |
| 24 | 40.8% | 2.45 / 55 | 0.416 × voxel | 81% | 19.71 |

两条容易搞反的结论：

1. **合并消耗样本的比例随视角数上涨（36%→59%），但体素内离散度几乎不变**（0.38→0.42
   voxel）。所以不是"多视角打架打得很凶"，而是"更多样本被换成了邻域平均"。
   高斯 footprint 四档完全一致（scale 中位数 0.0013/0.0018/0.0009 逐位相同），
   于是表面变稀（1-NN 间距 0.0023→0.0033，+42%）。
2. **合并是低通降噪器，不是单纯的糊**。把 `gs_voxel_size` 从 0.002 降到 0.001：
   锐度保留 81% → **92%**，但 PSNR 19.71 → **19.16**、SSIM 0.563 → 0.529。
   机制：体素变小 → 更少样本被合并 → 相邻视角对同一表面点的两份"略有偏差的副本"
   同时留下 → 渲染出重叠双层/斑纹。合并把它们平均成一个干净表面。
   → **PSNR 和锐度在体素尺寸上互相矛盾，选参不能只看 PSNR。**

所以自洽 PSNR **不能跨视角数比较**（E-08 里同一个视角 22→19.0 单调下降）。
要排名必须做 held-out。

---

## 4. 预处理：以主点为中心裁正方形

`preprocess_view`（`scene_io.py:28-50`）三步，DUSt3R 血统：

1. 以 `K` 的 `(cx,cy)` 为中心，裁出**最大对称方形** `(cx-mx, cy-my, cx+mx, cy+my)`；
2. `rescale_image_depthmap` 缩到 `--resolution`（默认 756×756）；
3. 用 `camera_matrix_of_crop` 再精调一次 bbox（吸收主点取整误差）。

后果：**非正方形输入会丢两边**。1920×1080 只保留中间 1080×1080（左右各丢 420 px）。
`rgb/` 目录存的就是裁完的图，PSNR 对比必须与之对齐，不能拿原图算。

`--resolution` 必须是 14 的正整数倍（patch size）。文档说支持 1K 以上。

---

## 5. 与 InfiniSplat v2 的差异（迁移经验时注意）

| 维度 | v2 | v3（SplatAnything） |
| --- | --- | --- |
| 几何基座 | DA3（Depth-Anything-3，钉死 commit） | **WorldMirror-2**（自带 camera head + depth + conf + mask） |
| 位姿模式 | provided / colmap / da3-pose / pose-free | 只有 `rgb`（自估）/ `pose`（读 JSON） |
| 尺度归一 | 无统一口径，voxel = median_scale×1.4 | **GSN**：spacing 归一到 0.0013227513，voxel 固定 0.002 |
| 输入相机格式 | `cameras.npz`（c2w OpenCV + K） | `prior_camera.json`（按 `camera_id`=图片 stem 匹配） |
| 输出 | `scene.ply` + `cameras.npz` + `result.json` | `gaussians.ply` + `points.ply` + `cameras.json` + `depth/` + `rgb/` |
| 精调 | 有（v2 仓自写 `02b`，非官方） | **无**，需要自己接 |
| 训练代码 | 无 | 无 |
| 渲染器 | 不带，需自接 gsplat | 不带，需自接 |

共同点：都是"位置 = 深度反投影、只学高斯属性"，都不含训练代码，都要自己接渲染做评测。

---

## 6. 官方发布包的已知缺陷（实测）

1. **`validation.py` 不存在**。`docs/inference.md` 写 "the lightweight `validation.py`
   entry point remains available"，但 `git ls-tree HEAD` 里没有这个文件，
   `tests/test_inference.py:17` 的 `from validation import load_config, main, parse_args`
   必然 ImportError。→ 整组 `unittest discover` 跑不了，只能单跑
   `tests.test_camera_conversion`。
2. **`examples/object/prior_camera.json` 的 `num_cameras=49`，但 extrinsics/intrinsics
   只有 24 条，images 里 23 张**。load 时按 entries 匹配、`num_cameras` 不被使用
   （`camera_io.py`），所以**不影响运行**，但说明打包脚本里那个字段是错的。
3. **仓库 418 MB**，其中 examples/ 占绝大部分（pano 24 张各 ~5 MB、rig 150 张）。
   建议 `--filter=blob:none --sparse` 克隆，只在需要时
   `git sparse-checkout add examples/<category>`。

---

## 7. 训练改造：官方缺哪三块（E-15 实测）

官方是纯推理发布包，要反传必须自己补三处。**都不需要改官方源码**，适配层
`09_train_probe.py` 就是这套做法。

| 缺什么 | 官方现状 | 补法 |
| --- | --- | --- |
| 梯度 | `GaussianWorldMirror.forward` 挂 `@torch.no_grad()`；`inference.py` 还套 `inference_mode()` | 别删装饰器——直接调 `model.encode` + `model.gs_renderer`。官方 forward 也就这两句，数值等价（实测 max\|Δ\| 4.77e-07） |
| 光栅化 | **完全没有**，`rasterization.py` 注释 "no rasterization or training" | 用 gsplat；约定见 `03_render_eval.py`（`inv(c2w)` 当 viewmats、colors 传 SH DC、**别再加 0.5 偏置**） |
| 可反传的合并 | `merge_voxels` 两处 in-place：列视图 `scatter_add_`、`quats /= norm` → autograd 报 "modified by an inplace operation ... version 5; expected 4" | 换成逐项等价的 `index_add`（合并规则不变，仍按体素中心加权，opacity 仍是 `Σw²/Σw`），输出与原实现差 4.77e-07 |

### 7.1 三个必知的坑

1. **`depth` 是 5 维 `[B,S,H,W,1]`**（DPT 末尾挂了个单通道）。`depth.shape[-2:]` 得到 `(H,1)`，
   拿去当宽高会渲出 1 像素宽。渲染尺寸**一律从输入图取**。
2. **`opacity` 分支没有梯度是正常的**。合并时 `values = weights if name == 'opacities' else ...`，
   解码出的 opacity 通道根本不参与计算（真正生效的是 merge weight）。12 个输出通道里
   只有它精确为 0，**不能用它判训练链坏了**。
3. **`scales = exp(raw).clamp_max(0.3)` 对 lr 极敏感**。1×1 卷积 256 通道累加，Adam 首步
   就能把 raw log-scale 推 ~+5 → scale 从 1.77e-3 跳到 clamp 上限 0.3（169 倍）→ 下一步
   全部被 `filter_large_scale_infer` 清空 → 渲染成黑图、loss 变成常数。**`lr=1e-3` 必炸**
   （实测），冒烟用 1e-5；真训要分通道 lr 或梯度裁剪。

## 8. 排错

| 现象 | 原因 | 处理 |
| --- | --- | --- |
| 下载 50–60 KB/s | HF 镜像 302 到 `us.aws.cdn.hf.co` / `cas-bridge.xethub.hf.co`，这两个域名没进 `no_proxy`，被塞进 v2rayN | `_env.sh` 的 `no_proxy` 加 `.hf.co,.xethub.hf.co`；`hf_download.py` 的 curl 加 `--noproxy '*'`。修完 1.7 MB/s |
| `Missing conda env` / `python: command not found` | `01` 依赖 `00` 建好的 env | 先跑 `00_setup_env.sh` |
| `pose 模式要求 prior_camera.json` | JSON 必须和 `images/` 同级 | 用 `scripts/convert_cameras.py` 生成 |
| `No camera entry for image 'xxx'` | `camera_id` 必须等于图片**文件名去扩展名** | 转换时统一命名，别手改 |
| `All Gaussians are below opacity_thresh` | 权重没对上，或 scale 过滤把点全丢了 | 先确认权重 SHA256；再试 `--opacity_thresh 0` |
| 输出目录同名覆盖 | `inference.py` 直接写 `OUT/SCENE/` | `02` 已处理：改输出根加时间戳 |
| gsplat `AttributeError: 'NoneType' object has no attribute 'CameraModelType'` | CUDA 后端没编出来（缺 nvcc） | `conda install -n $CONDA_ENV -c nvidia cuda-nvcc=12.6` |
| gsplat `_jit_compile() missing 2 required positional arguments` | gsplat 1.5.3 与 torch 2.7.1 的 JIT 接口不兼容 | 降到 `gsplat==1.5.2`（v2 实测可用的版本） |
| gsplat 编译 `#include <glm/gtc/type_ptr.hpp>` 找不到 | 1.5.2 不内置 glm（1.5.3 才自带 third_party/glm） | **已打通**：`git clone --depth 1 https://github.com/g-truc/glm.git ~/repos/glm_src`，`_gsplat_env.sh` 把它加进 `CPATH`（同时补 conda 的 `targets/*/include` 与 `TORCH_CUDA_ARCH_LIST`） |
| nvcc `gcc: No such file or directory` | 本机 WSL 没装 gcc/g++ | 借用已有 conda env 的 gcc（`envs/infinisplat_v2/bin`），或 `conda install gxx_linux-64` |

---

## 9. 人像鼻部坏例指认包：怎么出、四个必守口径

> 脚本 `10_nose_sheet.py` + `10b_verify_sheet.py`，用法见 `README_wsl.md` J 段，
> 用户侧操作说明见 `10_NOSE_PICK_GUIDE.md`。本节只记**为什么这么写**。

### 9.1 为什么需要一个"指认包"

R2-N 五个假设（N0 输入/相机不一致、N1 相机估计误差、N2 跨视角中心深度不一致、
N3 合并/footprint、N4 域与像素预算）在**没有共同靶子**时无法判读：不同假设要量的量
互相不可比。所以第一步不是量，是让用户把"鼻子变宽/重影"这个目视反馈落到
**具体 run × 视角 × 块号**。指认完成前不产任何质量结论。

### 9.2 四条口径（违反任一条，指认结果就不可用）

| # | 口径 | 违反后果 | 实测证据 |
|---|---|---|---|
| 1 | **裁窗只由 GT 决定**，绝不借渲染侧检测框 | 同一张 GT 在不同 run 被裁成不同窗，用户以为"质量随视角数变化"，其实是裁窗在变 | 早期版本借了 RD 框，view 007 在 8v 是 `face`、3v 是 `profile`，lapvar 537.7 vs 1089.7 |
| 2 | **GT 走原生分辨率**，不先降采样再裁 | GT 被插值糊掉，用户把插值模糊记成"模型模糊" | GT 1439×1919、RD 504²。早期把 GT 降到 504 再裁 133px 放大 2.7× → 白扔 2.86× 真实分辨率；改后 GT 原生 ROI 342~440px → 出图 360px，缩放 0.82~1.05× |
| 3 | **ROI 用归一化坐标存储**，裁剪时才乘各自原生边长 | 同一 ROI 在不同分辨率下对应不同像素区，cross-run 不可比 | 改后 24 个共享视角的 GT tile **逐像素一致** |
| 4 | **检测只用 `model_selection=1`** | 误检框到胸口，鼻窗整个带偏 | `sel=0` 在 view 001 给 `cy=0.297` conf 0.52，真脸在 `cy=0.127` conf 0.94；`sel=1` 与全分辨率结论 |Δcy|<0.01 |

### 9.3 漏检视角给整幅全景，不给"保守窗"

本数据 24 个视角里有 7 个是**纯侧脸/背面**（006–010、021–023），人脸检测器在全分辨率、
504²、放大到 756/1008、FaceMesh 468 点四种方式下**全部 MISS**。这不是调参能救的。

此时若给一个"保守鼻窗"，框里会是空的或衣服，用户会误以为脚本裁错了。
所以 `nose_window()` 返回 `kind='full'` + 整幅方图，sheet 上用**橙号**标注，与
`kind='nose'` 的绿号区分。**宁可给全景，也绝不假装那里有个鼻部特写。**

### 9.4 物体包默认不进指认表

物体上没有人脸，mediapipe 在物体图上只会误检——实测 13 个"鼻特写"块**全是假阳性**。
`CONTROL_RUNS` 因此默认关闭，需 `--with-controls` 显式打开；物体链路本身没问题
这件事已由 fig 系列的区域指标覆盖，不需要用户在这张表上看。

### 9.5 自纠偏窗：最后一轮只量不重定

`refine_window()` 把裁窗放大到 512 再检测，按新框中心重定窗，最多两轮。
**最后一轮必须 `break` 而不重定**——否则 `shrink` 被置 `None`，体检看不到真实偏心度，
"问题被推给下一轮"而不是被修好。改后 `shrink` max 从 0.809 降到 0.329（中位 0.080）。

### 9.6 独立体检（`10b_verify_sheet.py`）判三分类，不看图只量数

| 类别 | 判据 | 能否指认 |
|---|---|---|
| `face` | 块内能再检出人脸 | 可指认鼻宽 |
| `profile` | 检出失败但肤色占比 ≥0.06 | 鼻宽 N/A，**仍可指认糊/重影** |
| `empty` | 肤色占比极低（背面） | 不可指认 |

`skin_dxy`（**同框内** GT 与 RD 的肤色重心差）是最关键一项：它证明两图真的对齐，
否则上下对比看到的差异里混着位置偏移。本轮 16 个可见视角 |Δ|≤0.035，0 个失配。
注意这个量只能在**同框**下算——跨分辨率直接比重心是没有意义的。


## 9. 跨视角深度一致性滤波（E-19，SA 仓 f8ebce1）

- 实现位置：`~/repos/SplatAnything/src/models/models/rasterization.py` 的
  `cross_view_depth_consensus`，在 GSN 之后、DPTDecoder 提升之前调用；
  配置键 `depth_consensus_enable/tau_vis/tau_gap/min_margin`（`configs/inference.yaml`，默认关）。
- 运行方式：`sed 替换 enable:true` 生成实验配置，`--config_path` 传入（先例
  `configs/exp_depth_consensus_strict.yaml`：τvis=0.02、margin=4，是当前的更好工作点）。
- 验证口径：E-18 探针（`D:/tmp/r2n/E18_layer_ghost.py`，双拷贝拟合 + 肤色高斯深度直方图）
  + `03_render_eval.py` 自洽渲染。strict 档量化上重影 b/a 4.97→0.08（003）、PSNR/SSIM 全场最优，
  **但用户目视复核（11-17）不判优化：重影目视仍在、侧面几何仍糙、发尾新增白色条纹 → 未通过验收，
  保持默认关，不进正式链路**。签名改善 ≠ 观感改善，量化结论对外引用时必须带这句。
- **⛔ inf×bool=NaN 教训**：float 张量含 inf 投票时，`tensor * bool_mask` 会把 inf·False 算成
  NaN（bool 提升为 0.0），无声污染下游所有统计。必须 `torch.where(mask, t, 0)`。
  本次的下游放大链：NaN 深度 → 体素键 INT64_MIN → `coords - min` 溢出 →
  200 万高斯塌缩成 1974。滤波/掩码类改动，输出后立刻打印 nan/inf 占比。

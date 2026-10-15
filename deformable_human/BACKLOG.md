# BACKLOG — deformable_human 优化方案（基于 Reconstruction 成熟流水线）

> 来源：峰哥 2026-10-13 提供的服务器 Reconstruction 流水线方案文字描述（共 A~H 八案 + 背景节）。
> 本文档是**评审台账**：逐案记录原理、移植步骤、验证方法与当前状态，供峰哥逐个拍板。
> 「【本地状态】」块为小华子批注（2026-10-13 整理），与方案原文区分。

## 状态总览

| 案 | 名称 | 类型 | 预期收益 | 状态 | 依据 |
|---|---|---|---|---|---|
| A | 曝光补偿 | 质量 | +0.1~0.4dB（外推，非实测） | ❌ **som_human A/B 已测：无收益**（@10k -0.3dB） | `som_human/exposure.py` |
| B | 位姿精炼延迟启动 | 质量 | 手机位姿场景追回 | ⬜ 未实施（v1 已证伪，v2 待验） | README 实验记录 |
| C | Coarse-to-Fine 分辨率调度 | 加速 | 省 25~40% | ❌ **som_human 已实测否决**（零加速） | 探针实验 2026-10-13 |
| D | 稠密化自动调度器 | 稳定/免调参 | 防爆炸 | ✅ **som_human 已实现**（默认关，待冒烟） | commit bfa3b6d |
| E | 单目深度先验正则 | 质量（手部） | 攻残余手部模糊 | ⬜ 未实施（需下载深度模型） | — |
| F | 增强反哺监督 | 质量（目视） | 手部纹理可辨度 | ⬜ 未实施 | — |
| G | gsplat 渲染器 + 内参可学 | 加速 + 质量 | 1.5~2x 渲染 | ⬜ 未实施（优先级受新证据影响） | 见 G 节批注 |
| H | Phase 2 MHR 锚定 | 架构 | 根本解 | ⬜ 未实施（长期） | — |

---

## 0. 背景与前提

- **基线配方**（所有 A/B 的对照组）：
  `ITERATIONS=10000 WARM_UP=1000 USE_MASK_LOSS=1 FG_WEIGHT=1.0 BG_WEIGHT=0.2
  EXTRA_TRAIN_ARGS="-r 2 --densify_grad_threshold 0.0004 --densify_until_iter 2000"`
  参照数字：hand_motion 全图 **29.14** / 前景专项 **26.70**。
- 改 loss/正则一律改 `train_mask.py`（搜「MASK-LOSS」注释找差异点）；
  新开关在 `02_train.sh` 加 `${VAR:-default}` 透传。
- **已证伪、禁止重试**（README「实验记录」）：
  - 7k 迭代
  - bg 变形软正则（动态/静态两种判定都负）
  - 手机 VI 位姿直接替 COLMAP（**-4dB**）
  - POSE_REFINE 从第 0 步开始学（发散到 14.78dB）
  - VGGT 前馈位姿（-5.3dB，内参系统差）
- **评测纪律**：03_render.sh 官方 test 指标 + eval_fg_psnr.py 前景专项 + 三联图目视，
  负结果也记录。

> 【本地状态】「手机 VI 位姿直接替 COLMAP -4dB」与 som_human 2026-10-13 的
> double_wave 根因诊断互相印证：App VIO 位姿漂移（01b 重投影 inlier 摆动 250 倍）
> 导致 PSNR 钉死 13~14，点数 ×7 / 有无运动模型都无效。**位姿质量是手机采集
> 场景的第一优先级**，B 案（精炼补救）与 COLMAP 精化（治本）都指向这里。

---

## 方案 A：曝光补偿（Exposure Compensation）

**动机**：手持单目视频有 auto-exposure 逐帧波动（亮度/色温漂移），3DGS 的 loss
假设所有帧光度一致，颜色漂移系统性干扰几何学习，并可能是 train→test 泛化缺口
的一部分。当前 deformable_human 完全没处理。

**原理**（Reconstruction exposure.py 的 ExposureModule）：
每相机一个可学习 4×3 仿射矩阵 A，对渲染图做 RGB 齐次仿射校正：

- A 初始化 = [I₃ | 0]（恒等）
- 校正：`img_adj = Aᵀ @ [img_render; 1]`
- loss = L1(img_adj, gt)（替代对 img_render 的 L1）
- 正则：`reg = 0.01 * Σ|A - A_eye|`（拉回恒等，防止与 SH 颜色互相吸收震荡）
- **闭式初始化**（很有价值）：启用第一步，取 alpha>0.5 的像素列堆成 N×4 矩阵
  X=[render;1]ᵀ，`torch.linalg.lstsq(X.double(), GT.double()).solution` 解初始 A；
  可选抗外点：剔除残差 > 2×中位数像素后再解一次。

**移植步骤**（train_mask.py）：
1. 新增 ExposureBank：dict image_name -> nn.Parameter([I|0])，注册进 optimizer
   （独立 lr，1e-3 量级）。
2. 渲染后、加权 L1 前对 image 做 affine；DSSIM 也用 affine 后的图（口径一致）。
3. 门控：iteration < EXPOSURE_FROM_ITER（建议 = WARM_UP）前 A 冻结为恒等；
   首次启用时对全部训练帧跑一遍闭式初始化。
4. loss 加 `EXPOSURE_REG_WEIGHT * Σ|A - A_eye|`，默认 0.01~0.1。
5. test 口径：test 帧没训过 A → 按 fid 与相邻 train 帧的 A 线性插值，评测同样 apply。

**验证**：标准 A/B；额外看逐帧 test PSNR 方差（曝光问题特征 = 均值不低但个别帧
大幅掉点）。预期全图 +0.1~0.4dB，成本极低。

> 【本地状态】🔧 **som_human 已实现**（2026-10-13，`som_human/exposure.py` +
> train_som/render_som 集成，默认关 `USE_EXPOSURE=1` 开）：4×3 齐次仿射逐训练
> 相机、门控开启瞬间全训练帧 lstsq 闭式初始化（2×中位抗外点重解）、独立 lr
> 1e-3、拉回恒等正则、test 帧按 fid 线性插值、权重随 ckpt 存 exposure.pth。
> 与 BACKLOG 设计两点差异：渲染器无 alpha 输出 → 闭式初始化用全像素+抗外点；
> 正则为 sum 口径（按原文）。deformable_human 侧未移植。
>
> 【2026-10-15 A/B 结果】❌ **无收益**：COLMAP 精化位姿 + v1 配方下，
> 对照（无曝光）@10k test PSNR 21.27 vs 曝光组 @10k 20.95（**-0.3dB**）；
> 曝光组 20k 终点 21.63 / best 21.86@19k。闭式初始化显示曝光波动确实存在
> （max|A−I|=0.246，mean 0.020），但收益被 SH 颜色吸收/噪声吞没。服务器 AI
> 同日承认「+0.1~0.4dB 是外推预估，非 Reconstruction 实测」，与本结果一致。
> **结论：som_human 链路默认关闭即可；deformable_human 若移植先确认逐帧
> PSNR 方差是否有曝光特征，别预期有 dB 级收益。**

---

## 方案 B：位姿精炼延迟启动（Gated Pose Refine）

**动机**：train_pose_refine.py（场景侧 pose delta，pose_delta.py 有完整数学推导）
已证伪——发散根因是 warm-up 期几何未成形，位姿被垃圾梯度拖走、自我强化进坏
局部最优。README 结论「要救需 BARF 式课程调度」。

**原理**（Reconstruction 工作正常的联合位姿精炼，三个关键设计）：
1. **延迟启动**：位姿参数（可学四元数+平移+NDC 内参，nn.Parameter 初始化为
   COLMAP 初值）前段完全冻结——pose_refine 只在 epoch >= pose_refine_from_epoch
   （默认 11 / epochs=60 ≈ **18%**，几何大致成型后）才开；未启用时 w2c 用
   .detach() 版本，梯度流不进位姿。
2. ~~**reg 锚定**~~（**2026-10-15 服务器 AI 更正**：pose_refine.py 里定义了
   reg_loss 但**从未被加进主训练 loss**——grep 全 src，主循环只加了 grid 曝光的
   variation_loss，权重 10。其稳定性来源是**延迟启动 + 小 lr + 4K 下再缩 10 倍**，
   不是 reg 锚定。实现时 reg anchor 应降级为「可选保险丝」。）
3. **独立小 lr**：代码终值 `w2c_q/w2c_t = 1e-3×lr_base`、`intrinsic_ndc =
   1e-4×lr_base`，`lr_base = 1e-1 if is_4k else 1`（4K 下位姿 lr 再缩 10 倍）；
   逐相机独立 Adam。（他们的光栅化器对相机矩阵有梯度；stock 光栅化器没有，
   deformable_human 保持场景侧 delta 方案即可，数学等价。）

**test 相机口径**：评测前 `refine_test_for_evaluation` 对 test 相机优化位姿+曝光
（高斯冻结）——产品评测口径，科研式 A/B 不要混用。

**移植步骤**（train_pose_refine.py）：
1. 新增 POSE_REFINE_FROM_ITER（默认 = WARM_UP + 4000，形变 MLP 已工作一段、
   手部几何基本成形后），此前 pose delta 冻结为 0。
2. 启动后前 2000 步 lr 从 0 线性 warmup（可微课程）。
3. POSE_REFINE_LR_Q/T=1e-3；POSE_REFINE_WEIGHT：COLMAP 初值场景 0.01（轻锚），
   手机 VI 初值场景试 0.1（初值差但防漂移）。
4. 可选分级：先只放开旋转，过 1000 步再放开平移（旋转误差对手持视频更敏感
   且更稳定）。

**验证**：
- 第一步（安全性）：COLMAP 场景 + 延迟精炼 ≥ 纯 COLMAP 基线（不回退才继续）；
- 第二步（收益）：01e 手机位姿场景（transforms.json 转的 COLMAP 文本场景，
  当前 25.14dB），看追回多少；能到 27+dB 则弱纹理场景有免 COLMAP 降级路径。

**风险**：位姿与形变场耦合（时序模型可把相机误差解释成形变）——所以精炼必须
晚开、reg 必须在；A/B 时除指标外看 fork 里已有的 drift rot/trans 统计打印。

> 【本地状态】⬜ 未实施（v1「第 0 步开始学」已证伪发散到 14.78dB）。与
> double_wave 根因直接相关，但注意证伪清单「手机 VI 位姿直接替 COLMAP -4dB」
> ——精炼是**补救**，COLMAP 精化才是**治本**；两条路不矛盾（先 COLMAP 精化拿
> 好初值，再延迟精炼修残差）。

---

## 方案 C：Coarse-to-Fine 分辨率调度（加速，最大头）

**原理**（Reconstruction optim_schduler.py 的 ResolutionScheduler）：训练分三段，
低分辨率起步逐步升到全分辨率。渲染和反传成本 ∝ 像素数：

- epoch ≤ T1（约 28%）→ scale=4（像素 1/16）
- T1 < epoch ≤ T2（约 56%）→ scale=2（像素 1/4）
- epoch > T2 → scale=1
- 分界公式：T1 = (min(max(n_cam,100),200)+200) × total_epochs/600，T2 = T1/2

**移植步骤**（train_mask.py；官方 Camera 类本身支持分辨率参数，-r 2 就是这么实现的）：
1. 按 iteration 段确定 scale；对 viewpoint_cam 构造降采样副本：GT 用
   `F.interpolate(mode="area")`，内参与 w2c 平移同步缩放（fx,fy,cx,cy × 1/scale）。
   最简单实现：初始化时对每档 scale 预生成一份相机列表（内存换时间），按段切换。
2. MaskWeightCache 天然兼容（get(image_name, ref_h, ref_w) 本就双线性 resize）。
3. **稠密化交互（关键坑）**：2D 梯度阈值与分辨率相关。①首选把 densify_until_iter
   主窗口放在 scale≤2 段；②备选按 scale 补偿（Reconstruction 对**剪枝**阈值有
   ÷scale^0.5 补偿）。先 ①，出问题再 ②。
4. SH 升级（每 1000 步 +1）节奏不变；**评测渲染永远在 scale=1**。

**预期**：前 30% 便宜 10~16 倍、中段 ~4 倍，整体省 25~40%（10k 训练 49min →
30~37min）。质量损失 <0.1dB。验证：10k + c2f vs 10k 全分辨率，全图/前景 PSNR + 时间。

> 【本地状态】❌ **已在 som_human 实测否决**（2026-10-13，commit 54fe3a3）。
> A/B/C 探针（2500 步，double_wave）：调度+补偿 9:05/418k 点 vs 调度 6:48/323k
> vs 基线 6:45/307k —— **零加速**，且 grad 阈值 ÷scale 补偿有害（多 29% 冗余
> 高斯）。原因：D-3DGS+SoM 链路步时瓶颈在**随点数线性增长、与分辨率无关的开销**
> （实测静态 70 it/s vs 带形变 5.6 it/s，deform 路径占 ~92%），光栅化只占小片，
> 「渲染便宜 4~16 倍」的前提在此链路不成立。
> **deformable_human 侧若重启需先满足**：(a) profile 确认其光栅化占步时大头
> （train_mask.py 的 mask loss 按像素计价，结构与 som_human 不同）；(b) 或上
> -r 1 全分辨率。代码保留 `som_human/res_scheduler.py`，负结果数据在
> `som_human/README.md`。
>
> 【2026-10-15 前提已变 + 服务器结构性估算】som_human 的 SoM 混合已 matmul 化
> （commit 07c91c0，deform 不再占 92%），当年否决 C 的前提在 som_human 链路**已
> 失效**。服务器 AI 给出边界（0–30% scale4 / 30–60% scale2 / 60–100% scale1）与
> 收益公式：收益 ≈ 光栅化份额 × 像素节省率；光栅化占 70% 时总省 ≈35%，<30% 基本
> 白做；次级收益是 scale-4 段点数更低 → per-gaussian 成本间接下降。**重启前先做
> 一次 profile 实测当前光栅化份额，>50% 才值得做**，边界照抄 0–30%/30–60%/60–100%。
> （注：服务器无动态场景 c2f 实测加速比，这是结构性推理。）

---

## 方案 D：稠密化自动调度器（防爆炸，免调参）

**现状痛点**：AGENTS.md 不变量 5——复杂室内必须手工传
`--densify_grad_threshold 0.0004 --densify_until_iter 2000` 防高斯复利爆炸到
130 万/24GB，但这不是 02 默认值，跨场景靠人肉判断。

**原理**（Reconstruction DensificationScheduler）：
1. **自适应点数上限**：max_n = init_n + momentum；momentum 初始 = init_n × 4
   （稀疏初始化；稠密 × 0.5）。
2. **需求累计**（只增不减）：每稠密化周期后
   `momentum = max(momentum, 0.98×momentum + min(40000, 本轮超阈值新增数))`——
   爆炸场景需求会涨，但被 0.98 衰减和 cap 限成渐进曲线。
3. **topk 截断**：先按官方逻辑算 densify_mask（grad > threshold），调度器给出
   本轮目标增量 `目标 = (max_n - init_n)/current_scale`（c2f 期间上限随分辨率
   折半，天然防爆），对 (grads × ~prune_mask) 取 top-n 与 densify_mask 相交。
   效果：梯度最猛的前 k 个点优先分裂，任何场景点数曲线受控。

**移植步骤**（train_mask.py 稠密化分支 + 02_train.sh）：
1. 02_train.sh 加 USE_DENSIFY_SCHED=1 和 MAX_GAUSSIANS（默认 = 初始点数 ×5，
   或按显存设绝对值）。
2. 把 `gaussians.densify_and_prune(...)` 包一层：调用前按 momentum/target/topk
   截断 densify mask（官方仓不能改，复制成 gaussian_sched.py 引入）。
3. 曝光/分辨率方案落地后，把 scale 补偿接进来。

**验证**：hand_motion（会爆炸的室内场景）不传 EXTRA_TRAIN_ARGS 裸跑：看点数
曲线是否受控（TensorBoard）、指标不低于手工配方。目标：手工配方从「必须记的坑」
变成「默认行为」。

> 【本地状态】✅ **som_human 已实现**（`som_human/densify_sched.py`，commit
> bfa3b6d）：预过滤 grads 后调官方 clone/split，不动官方仓；单测全过（不截断
> =官方行为 / target=30 精确 +30 / target=0 安全 / topk 保大分）。默认关
> （USE_DENSIFY_SCHED=1 开），待双场景冒烟。**deformable_human 侧未移植**。
> 同日 som_human 的 v2/v3 两次复利爆炸（147 万点撞护栏）实证了该方案的必要性。
> som_human 另有硬护栏 `SOM_MAX_GAUSSIANS`（默认 150 万）已上线（commit 125ceda）。
>
> 【2026-10-15 服务器实现细节（供对齐修正）】
> - momentum 公式确认为 `max(momentum, int(0.98×momentum + min(40000,
>   densify_mask.sum())))`，单调不减 EMA，momentum_step 是 **topk 截断之前**
>   梯度超阈值的原始计数。
> - **topk 梯度口径（重要）**：每 epoch 累积 `v_means2d_accum / cnt.clamp_min(3.0)`，
>   且 use_abs_grad=True 为默认 → 实际喂的是光栅化器 extra 里的 **absgrad 变体**
>   v_means2d_abs（乘过 grad_scale），不是标准 v_means2d。topk 在
>   `(grads × ~prune_mask)` 上取 k。代码里有 todo 注释 "why use mean grad instead
>   of max grad"——**是可见次数均值归一化后的累积 2D 梯度范数**。我们实现若对
>   齐需从「原始累积」改成「均值归一化 + absgrad」口径。
> - **显式 cap 才是默认路径**：scene 600k / object·avatar·body 各 300k / salient
>   200k；momentum 自适应只在 cap≤0 时启用。c2f 折扣：`next_n = (max_n−init_n)/
>   current_scale + init_n`，rate 上限 max_densify_rate_per_step=1.0；窗口结束
>   后 next_n 直接取 max_n（只剪不增）。160 帧推演：epoch 1–18 scale4、19–36
>   scale2、37–60 scale1，densify 窗口 3–56。

---

## 方案 E：单目深度先验正则（攻残余手部模糊）

**动机**：手部残余模糊本质是单目欠定——6fps 抽帧后快速运动区域监督稀疏。
mask 加权解决「梯度往哪分」，深度先验解决「几何自由度缺约束」，两者正交。

**设计**：
1. **离线**：新脚本 01f_depth_maps.sh——每帧跑单目深度（服务器
   /data_3d/w00950754/code/ 下有 Depth-Anything-3 和 MoGe；**推荐 MoGe**，输出
   affine-invariant 深度，适合尺度不变监督），存 {stem}.png（16bit）或 npz 到
   colmap_scene/depths/。
2. **训练**（train_mask.py）：
   - 渲染深度：对**形变后**的高斯做 alpha 加权深度（官方渲染器若无 depth 输出，
     参照 T-mean depth 写法 accumulate；或换 gsplat 的 depth 输出）。
   - loss：尺度平移不变 L1（先归一化：减中值除 std 或 lstsq 解 a,b），只对前景
     mask 加权（复用 01d mask；背景深度不可靠且不是目标）。
   - 进阶变体：只用边缘感知项（L1 on ∂d/∂x, ∂d/∂y）——深度边缘对齐比绝对深度可靠。
3. **调度**：warm_up 后启用，--depth_lambda 0.05 起步，与 mask 加权叠加。

**验证**：eval_fg_psnr 前景专项 + 手部三联图目视（可走 vlm_bridge）。与
backlog #1（per-frame latent 容量路线）正交可叠加。

**风险**：深度网络对快速运动小手部误差大 → 必须 fg 加权 + 小权重；若逐帧深度
抖动大，改只用边缘项。

> 【本地状态】⬜ 未实施。**需要下载模型权重**（Depth-Anything-3 或 MoGe，HF 上
> 的小文件，届时列链接给峰哥走迅雷）。与 H 案（MHR 锚定）是攻同一症状的两条
> 路线：E 加外部监督，H 加结构先验。

---

## 方案 F：增强反哺监督（Enhance-Back / 监督信号刷新）

**原理**（Reconstruction face_enhance_post 的机制——「训练中期抬高监督信号」）：
1. 训练到指定 epoch，对每个训练相机渲染 image_pred；
2. 目标区域裁剪（人脸框长边 × upscale 扩正方形；人体取 01d mask 外接框）→
   F.interpolate 到 512×512（area）→ 增强模型推理 → interpolate 回原尺寸；
3. **渐变融合防接缝**：54px 宽 sinc 衰减 fuse_mask（中心 1.0 → 边缘 0.0），
   `img_enh × mask + img_orig × (1-mask)`；
4. `cam.reset_image(fused)`——**替换训练 GT**，后续监督信号就是这个更清晰的图。
   语义：模型某些区域已渲染得比原图好（或原图本身运动模糊），用「渲染+增强」
   回填 GT，抬高监督上限。

**移植**：目标区域 = 前景 mask 的手部/运动区；增强器 = HYPIR 或 EnhModel
（media_code 仓内 hypir 项目有完整管线和微调权重，两仓资产互通）。触发一次
（如 iteration 6000），只在 mask 内 blend。

**关键注意**：评测仍用原始 GT（不替换 test 帧），全图 PSNR 可能因增强图偏离
原始像素而小幅下降——**真实收益看目视质量（手部纹理可辨度）而非官方指标**，
A/B 结论两个口径都写。

**风险**：HYPIR 是生成式超分，有幻觉风险 → 只对运动模糊区域、小 blend 权重、
单次触发。

> 【本地状态】⬜ 未实施。与 E/H 攻同一症状（手部模糊）的第三条路线：E 加几何
> 约束、H 加结构先验、F 直接换监督信号。F 见效最快但有幻觉风险，适合作为
> E/H 落地前的快速验证。

---

## 方案 G：gsplat 渲染器升级 + 内参可学

**两条独立收益**：
1. **加速**：gsplat（nerfstudio，pip 可装）比 stock diff_gaussian_rasterization
   快约 1.5~2x，带抗锯齿。移植：fork 官方 gaussian_renderer/__init__.py 的
   render() 换 gsplat.rasterization()；形变注入语义不变（先 d_xyz/d_rot/d_scaling
   后光栅化）。注意 Deformable-GS 用到 viewspace_points_densify（稠密化梯度），
   gsplat 返回等价的 rasterize_grads（v_means2d 梯度），需做映射。
2. **内参可学**：gsplat 对 viewmats 有梯度——这正是 Reconstruction 靠自研
   光栅化器做到的事（位姿+内参可导优化）。stock 光栅化器没有此梯度，
   deformable_human 的 pose_delta.py 只能场景侧且「内参不学」。换 gsplat 后：
   内参参数化 (fx, fy, cx, cy)（初始 = COLMAP 值，reg 锚定，lr 1e-4），与方案 B
   延迟启动组合，可修掉 COLMAP BA 的残差内参误差（VGGT 路线垫底主因就是
   fx 差 17px 修不了；但注意这主要惠及 COLMAP 主链路精修，不改变前馈位姿
   替代 COLMAP 的结论）。

**工作量较大，建议拆两期**：先只换渲染器拿加速收益（评测全链路回归验证），
内参可学作为二期叠加方案 B。

> 【本地状态】⬜ 未实施。**优先级受 2026-10-13 新证据影响**：som_human 实测
> deform 路径占步时 ~92%（静态 70 it/s vs 带形变 5.6 it/s），光栅化只占小片
> → gsplat 的 1.5~2x **渲染**加速在 som_human 链路总收益预计 <10%。
> deformable_human 侧（train_mask.py + mask loss）光栅化占比未知，**上 G 之前
> 应先 profile 两仓的步时分解**。二期「内参可学」与 B 案协同价值大（手机位姿
> 场景 fx 残差可修），是真正的卖点。
>
> 【2026-10-15 服务器更正】Reconstruction **不是 gsplat**，是华为自研 CUDA
> 光栅化器（third_party/gs_render/，作者 CAI,JUN-XIONG；源码含 fast_ssim.cu /
> sparse_adam.cuh / grid_exposure.cuh / spherical_harmonics.cuh 等）。所以
> 「gsplat 1.5~2x 快于 stock」的对标在服务器侧不存在，1.5~2x 是我们从社区
> 资料的外推，非 Reconstruction 实测。若迁 gsplat：antialiased=True 即可；
> viewspace 梯度标准做法 means2d.retain_grad() 取 .grad；absgrad gsplat 有支持
> 但需对安装版本验证签名。**G 案价值重新评估：加速收益无实测背书，真正的卖点
> 只剩二期「内参可学」（依赖 gsplat 对 viewmats 的梯度）。**

---

## 方案 H：Phase 2 MHR 锚定——移植 Reconstruction 人脸链路骨架

**原理**：Reconstruction 有同构的、生产验证过的 3DMM 锚定实现（人脸版）。
Phase 2 把「3DMM 人脸模板」换成「MHR 人体模板」即可复用整套骨架：

- **逐帧 2D 观测**：ExpTracker（ViT）对每帧每人脸输出 657 点密集 2D landmark
  + 235 维表情系数（与 vggt_human 的 MHR/657 点同源，资产现成）。
- **模板对齐三阶段优化**（其 face_alignment.py）：
  - 阶段 1 全局对齐：优化模板→世界系全局 scale/quaternion/translation，目标 =
    多视角 2D landmark 重投影误差；PnP 估初始位姿 + 视线交叉法（多相机主视线
    最小二乘交点）估初始平移；Adam 300 iter，ExponentialLR γ=0.95；outlier 帧
    剔除循环（对齐误差 > 中值 × reject_extent 的帧剔除后重优化，循环到全通过）。
  - 阶段 2 局部对齐：加每帧局部 R,t，带锚定正则 λ_anchor；局部旋转偏离中值
    > 55° 的帧重置为单位元。
  - 阶段 3 系数优化：联合 global SRT + local RT + 身份系数（L1 正则）+ 逐帧
    表情系数（L1 正则），各参数组独立 lr；最终误差超阈值则该 ID 放弃锚定。
  - 代表视角选择：最正面（PnP 角最小）+ 表情最接近中值的帧做颜色初始化。
  - 初始化：对齐后模板中性姿态顶点直接作为该 subject 初始高斯点云；身体高斯
    从场景点云去掉模板包围盒内的点后初始化。
- **推荐形态**（结合 deformable_human 的时序形变 MLP）：**模板驱动低频 +
  MLP 残差高频分解**：
  `d_xyz(t) = d_template(t) + MLP(xyz, t)残差`
  （MHR 骨架/姿态系数驱动低频；MLP 只学模板解释不了的残差）
  配套：①canonical 高斯用模板顶点初始化人体部分（替换 SfM 点云初始化）；
  ②每帧 landmark/skeleton 重投影正则（小权重，warm_up 后启用）；③姿态系数
  训练中定期同步校对（对应其 fix_and_sync_exp）。
  该形态同时解决欠定（骨架先验）和容量（MLP 只学残差），是「残余手部模糊」
  的根本解。

> 【本地状态】⬜ 未实施（Phase 2 长期架构项）。与 vggt_human 的 MHR/657 点
> 资产同源，做的时候先去那边搬观测数据生成代码。E（深度先验）和 F（增强反哺）
> 是落地前的过渡止痛方案，不冲突。

---

## 附：som_human 2026-10-13 实验结论（影响上述优先级的新证据）

1. **位姿是手机采集场景的第一瓶颈**：double_wave 三轮训练（216k 点 / 147 万点 /
   纯静态消融）test PSNR 齐刷 13~14，点数 ×7 与有无运动模型都无效；01b 重投影
   诊断 inlier 摆动 250 倍（436~112750）→ App VIO 位姿漂移。与证伪清单
   「手机 VI 位姿 -4dB」互证。**B 案与 COLMAP 精化优先级最高**。
2. **deform 路径占步时 ~92%**（静态 70 it/s vs 带形变 5.6 it/s，同点数 211k）：
   - C 案（分辨率调度）在此链路零加速的根因；
   - G 案渲染器替换收益缩水，**先 profile 再决策**；
   - 若 som_motion 前向/反向存在可优化热点，修好它是比 C/G 都大的加速杠杆
     【待办：profile train_som.py 各段】。
3. **稠密化复利爆炸在本链路实证两次**（v2: thr 0.0002+窗口 10000 → ETA 10.7h；
   v3: thr 0.0004+窗口 6000 → 147 万撞护栏）：固定阈值对复杂场景无安全窗口，
   D 案是必需品而非可选项；硬护栏（MAX_GAUSSIANS）已上线。
4. **运动模型（SoM 运动基）无罪**：静态消融 PSNR 13.40 ≈ 带形变 14.04，
   模型本身工作正常，瓶颈不在它。

---

## 附2：Reconstruction 训练配方实录（2026-10-15 服务器 AI 问答，校准「5min」对标）

服务器 5 分钟重建与我们的配方差异，按代码实录：

- **总迭代数只有 ~6000**：`iterations = epochs × train_frames_ideal = 60 × 100`
  （pipeline.py:205）；每 epoch 随机采样补齐 100 帧；epochs=60 是硬编码默认值。
  我们 som_human 跑 20000 步 → 光迭代数就是 3.3×。
- **单帧/步**：epoch 循环内逐相机 render→loss→backward→gaussians.step(visible)，
  无多帧 batch。
- **fused 全家桶**（我们全没有）：fast_ssim CUDA kernel（use_sparse_ssim=True）、
  SparseGaussianOptimizer 稀疏 Adam（只 step 可见子集，betas 0.9/0.999）。
- **AMP：无**——全 src 无 autocast/GradScaler/half（不要当它是加速来源）。
- **颜色 loss 是 RMSE 非 L1**（use_rmse=True）；lambda_dssim=0.2；
  **lr_factor=2.0**（全部 3DGS 基准 lr 翻倍）。
- SH：max degree 3，从 0 起，在 epoch (20,30,40) 触发 upgrade_degree()；
  另有每 epoch sh×0.94、opacity×0.95 衰减。
- 曝光 lr：1e-2（非 4K）/ 1e-3（4K）（cameras.py:170）。

**对我们「为什么这么慢」问题的最终归因**：服务器 5min ≈ 迭代 6000（3.3×）
× 稀疏 Adam+稀疏 SSIM（节省随 N 线性增长的 dense 优化开销）× 点数压在
显式 cap 下 × A100。我们已修 SoM matmul（20×），下一步同量级的杠杆是
**D 案显式 cap + 稀疏 Adam（后者尚未在任何案里列出，可作新方案 I 评估）**。

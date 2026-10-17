# BACKLOG — deformable_human 优化方案（基于 Reconstruction 成熟流水线）

> 来源：峰哥 2026-10-13 提供的服务器 Reconstruction 流水线方案文字描述（共 A~H 八案 + 背景节）。
> 本文档是**评审台账**：逐案记录原理、移植步骤、验证方法与当前状态，供峰哥逐个拍板。
> 「【本地状态】」块为小华子批注（2026-10-13 整理），与方案原文区分。

## 状态总览

> **实现状态图例**：✅ 已落地（跑通/冒烟） ｜ 🔸 已实现（有代码在仓，但未落地或测试为负） ｜ ⬜ 未实现
> ｜ ⚠️ 已测但负结果 ｜ ❌ 已测且否决。
> **「未实现」有两义，审查时勿混**：F/G/H = 真零代码（从没写过）；B = 存在已证伪的 v1
> （发散 14.78dB），要的是**改设计的 v2**，不是从零写；I = 只在「附 2」被提名，正文无章节。

| 案 | 名称 | 类型 | 预期收益 | 实现状态 | 结论（质量/收益） | 依据 |
|---|---|---|---|---|---|---|
| A | 曝光补偿 | 质量 | +0.1~0.4dB（外推，非实测） | 🔸 已实现（som_human `exposure.py`）/ **未移植**（deformable_human） | ❌ **无收益**：@10k 严格配对 −0.31dB，SSIM/LPIPS 同向劣化；默认关 | som_human README 实验记录 ② |
| B | 位姿精炼延迟启动 | 质量 | COLMAP 好初值修残差 | ✅ **已实施**（`deformable_human/pose_delta.py` + `train_pose_refine.py`，含可微课程 `POSE_REFINE_LR_WARMUP`/`ROT_ONLY_STEPS`）<br>❌ **已测否决**（v1/v2-a 手机 VI 两度证伪；B v2 在 **COLMAP 好初值**上也 −8.29dB） | 证伪：手部弱纹理场景的光度位姿梯度不可用，正解是离线 COLMAP 精化（01c/01d，+7.29dB） | README 位姿来源消融 |
| C | Coarse-to-Fine 分辨率调度 | 加速 | 省 25~40% | 🔸 已实现，代码留仓（som_human `res_scheduler.py`）/ **未移植** | ❌ **零加速**否决（渲染前向仅占 ~5%，前提不成立） | 探针实验 2026-10-13 |
| D | 稠密化自动调度器 | 稳定/免调参 | 防爆炸 | ✅ 已落地（som_human + deformable_human 双端冒烟；两种 thr 均稳定防爆） | 防爆 + 免调参成立；⚠️ 窗口后 prune 带尺寸判据会死亡螺旋（已修）；代价 −0.26~−0.34dB 归因于严格 cap+topk 策略本身；默认关按需开 | README 冒烟节 |
| E | 单目深度先验正则 | 质量（手部） | 攻残余手部模糊 | 🔸 已实现（som_human `depth_prior.py`）/ **未接**（deformable_human） | ⚠️ λ=0.05 首轮**负结果** FG −1.04dB（量级失衡 + 手部深度不可靠）；默认关，后续要归一化/降 λ/边缘项 | som_human README |
| F | 增强反哺监督 | 质量（目视） | 手部纹理可辨度 | ⬜ **零实现** | 待实施（E/H 之间最快的验证路线，有幻觉风险） | — |
| G | gsplat 渲染器 + 内参可学 | 加速 + 质量 | 1.5~2x 渲染 | ⬜ **零实现** | 一期（换渲染器）冻结：收益上限 ~10-15%；二期「内参可学」是唯一卖点，等与 B 绑定评审 | 见 G 节批注 |
| H | Phase 2 MHR 锚定 | 架构 | 根本解 | ⬜ **零实现**（长期架构项） | 待实施（依赖 vggt_human 的 MHR/657 点资产） | — |
| I | 稀疏 Adam | 加速 | 待评估（dense Adam 占步时 9%） | ⬜ **未立案**（仅在「附 2」提名，无正文章节） | 待评估 | 附 2 |
| J | per-person 独立重建（多人拆分） | 质量/容量 | 多人场景手部细节 | ✅ **som_human 已实施并实测**（`01g_person_masks`+`01h_build_person_scenes`+`train_som` FG 监督） | ✅ hand10 双人招手：FG PSNR 同级（p02 −0.15 / p07 +0.70），点数/体积 1/6；⚠️ mask 边缘有飞点（羽化+贴身互染+init 残留） | som_human README per-person 节 |

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

**移植步骤**（train_pose_refine.py）—— ✅ **1~4 已全部实施**（见下「实施状态」）：
1. 新增 POSE_REFINE_FROM_ITER（默认 = WARM_UP + 4000，形变 MLP 已工作一段、
   手部几何基本成形后），此前 pose delta 冻结为 0。
2. 启动后前 2000 步 lr 从 0 线性 warmup（可微课程）。
3. POSE_REFINE_LR_Q/T=1e-3；POSE_REFINE_WEIGHT：COLMAP 初值场景 0.01（轻锚），
   手机 VI 初值场景试 0.1（初值差但防漂移）。
4. 可选分级：先只放开旋转，过 1000 步再放开平移（旋转误差对手持视频更敏感
   且更稳定）。

**✅ 实施状态（2026-10-16，标注清楚避免后来者重复实现）**：

| 能力 | 状态 | 代码位置 | 备注 |
|---|---|---|---|
| 可微位姿精炼（四元数+平移 delta） | ✅ 已实施 | `deformable_human/pose_delta.py`<br>`deformable_human/train_pose_refine.py` | 默认关（`USE_POSE_REFINE=0`） |
| 可微课程（延迟启动 + lr warmup + 旋转优先） | ✅ 已实施 | 同上：`PoseDeltaStore.lr_scales()` / `apply_lr_schedule()`；env `POSE_REFINE_START` / `POSE_REFINE_LR_WARMUP` / `POSE_REFINE_ROT_ONLY_STEPS` | 冒烟三机制已验证 |
| 离线 COLMAP 精化（**位姿问题的正解**） | ✅ 已实施 | `som_human/01c_colmap_refine.sh` + `som_human/01d_check_refined.py` | 实测同迭代数 **+7.29dB** |

**复用指引**：`vggt_human` / `deformable_human` / 其它链路若需要「可微位姿精炼」能力，
**直接取 `deformable_human/pose_delta.py` + `train_pose_refine.py`**（数学与课程已就绪，
接进训练循环即可，见这两个文件顶部 docstring）。⚠️ 但**默认不要开**——本页三项 A/B
（手机 VI ×2 + COLMAP 好初值 ×1）全部证伪；要动位姿优先走 `som_human/01c` 离线精化。

**验证**：
- 第一步（安全性）：COLMAP 场景 + 延迟精炼 ≥ 纯 COLMAP 基线（不回退才继续）；
- 第二步（收益）：01e 手机位姿场景（transforms.json 转的 COLMAP 文本场景，
  当前 25.14dB），看追回多少；能到 27+dB 则弱纹理场景有免 COLMAP 降级路径。

**风险**：位姿与形变场耦合（时序模型可把相机误差解释成形变）——所以精炼必须
晚开、reg 必须在；A/B 时除指标外看 fork 里已有的 drift rot/trans 统计打印。

> 【历史澄清（重要，避免重复劳动）】
> - **前两次证伪都发生在「手机 VI 烂初值」上**——10-09 v1（立即启动）发散 14.78dB
>   （drift rot 1.48°/trans 0.12m）；10-10 v2-a（`POSE_REFINE_START=3000`）17.95
>   (best@5k)→16.35@10k，仍远劣于不精炼的 25.14。根因：手机位姿噪声下「动位姿」比
>   「修几何」更易降 loss（118 个自由位姿当泄压阀），**延迟只推迟不解决**。
> - ⚠️ 但**本节「验证第一步」——COLMAP 好初值 + 延迟精炼 ≥ 纯 COLMAP 基线——从来没做过**。
>   10-09 的结论本身就写着「现有 pose_refine 只对 COLMAP 级好初值做小修」。
>   → **B v2 的靶子是 COLMAP 好初值修残差，不是再救手机 VI。**
>
> 【2026-10-16 B v2 实现】相对 v2-a（只有延迟启动）补三条，落在
> `pose_delta.py` + `train_pose_refine.py` + `02_train.sh`：
> 1. **lr 线性 warmup**：`POSE_REFINE_LR_WARMUP`（默认 2000），启动后 lr 从 0 爬到满值；
> 2. **旋转优先分级**：`POSE_REFINE_ROT_ONLY_STEPS`（默认 1000），此前只放开旋转；
> 3. **reg 保险丝化**：`POSE_REFINE_WEIGHT` 保留但降级为「可选保险丝」（服务器 10-15
>    更正：Reconstruction 的 reg 从未进主 loss，稳定性来自延迟启动 + 小 lr）。
> 实现：`PoseDeltaStore.lr_scales()` / `apply_lr_schedule()` 把课程系数每步写进
> `optimizer.param_groups` 的 lr；评测口径不变（冻结原始 COLMAP 位姿，科研 A/B 口径）。
> **冒烟（1600 步）三条机制全过**：iter 600 准点启用、iter 900 放开平移（=600+300）、
> iter 1000 `lr_scale q=0.80 t=0.80`（=400/500）。
>
> 【第一步安全性 A/B —— ❌ **负结果，路线收束**】hand_motion/colmap_scene，
> 10k + warm 1000 + `-r 2 --densify_grad_threshold 0.0004 --densify_until_iter 2000`，
> 两臂严格同参（`model_bv2_ctl` vs `model_bv2_refine`：START=5000 / LR_WARMUP=2000 /
> ROT_ONLY=1000；点数 977k vs 975k，排除容量差异）：
>
> | iter | ctl（无精炼） | refine（精炼） |
> |---|---|---|
> | 5000 | 28.171（精炼刚启用、lr=0） | 28.115 |
> | 7000 | 28.496 | **22.657** |
> | 10000 | 29.214 | **20.923** |
> | Δ@10k | — | **−8.29dB** |
>
> **根因钉死在「梯度方向」而非「lr 太大」**：课程机制逐条按预期触发（iter 5000 启用、
> 6000 放平移、lr 系数 0.50→1.00），但 **loss 在 iter 5100 就已从 0.027 抬到 0.044**
> ——那是启用后第 100 步，`scale_t=0` 平移仍冻结、`scale_q` 仅 0.05。
> **纯旋转 + 满值 5% 的 lr + 100 步 = 崩**。COLMAP 位姿已亚像素级准确，光度 loss 对
> 位姿的梯度在本场景是纯噪声，任何偏移都是净损失。
> `POSE_REFINE_WEIGHT` 的 reg 量级（delta L2²≈6e-5 × 0.01 = 6e-7）比光度项（~0.02）
> 小 **4 个数量级**，形同虚设，拉不住漂移。
> **→ 「可微精炼 COLMAP 好初值」不是调度问题，是手部弱纹理场景的光度位姿梯度不可用。
> 位姿精化的正解是离线 COLMAP 精化（01c/01d），不是可微精炼。**
> 代码保留（`USE_POSE_REFINE=0` 默认关）。
>
> 【原批注】与 double_wave 根因直接相关，但注意证伪清单「手机 VI 位姿直接替 COLMAP -4dB」
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
>
> 【2026-10-15 profile 实测（profile_step.py，三点数据）】光栅化前向只占 **~5%**
> （216k:1.47ms / 1.5M:5.4ms，TOTAL 25.5→98.7ms）→ **C 案继续冻结**（分辨率
> 调度理论上限 ~4%，服务器「光栅化 70%」前提在我们的链路不成立）。步时大头是
> 随 N 线性的 deform MLP 前向+反向（~80%）+ dense Adam（9%）→ 降 N（D 案）才是
> 剩余大杠杆。

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
> （USE_DENSIFY_SCHED=1 开），待双场景冒烟。**deformable_human 侧已于
> 2026-10-16 移植**（thin shim + 02_train.sh 五个 env，见下方 10-16 冒烟块）。
> 同日 som_human 的 v2/v3 两次复利爆炸（147 万点撞护栏）实证了该方案的必要性。
> som_human 另有硬护栏 `SOM_MAX_GAUSSIANS`（默认 150 万）已上线（commit 125ceda）。
>
> 【2026-10-15 服务器实现细节（供对齐修正）】> - momentum 公式确认为 `max(momentum, int(0.98×momentum + min(40000,
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
>
> 【2026-10-15 cap600k A/B 结果：**免费午餐成立，cap 60 万落定**】对照（同配方同位姿、
> 无曝光）三个模型 @10k/@20k test PSNR：
> | 模型 | cap | @10k | @20k (best) | 点数@20k | 20k 训练时长 |
> |---|---|---|---|---|---|
> | model_colmap | 无（护栏150万） | 21.27 | —(18.5k被杀) | — | ~30 min |
> | model_colmap_exp | 无 | 20.95 | 21.63 (21.86) | 1,502,625 | ~30 min |
> | model_colmap_cap600k | 60万 | 21.18 | **21.81 (21.81)** | 629,715 | **~17 min** |
>
> 同点 @10k 掉 0.09dB（<0.3dB 判定线，波动内持平）；@20k 反超无 cap 曝光组 +0.18dB
> （含曝光变量，不单独归功，但至少证无劣化）。点数 2.4× 缩减，训练 ~1.7× 加速
> （N 减半 → deform MLP 步时随 N 线性下降，符合 profile 外推）。
> **结论：som_human 默认 cap 改 600k**（02_train.sh MAX_GAUSSIANS 默认值）。
> D 案剩余工作只剩调度器落地（momentum 仅 cap≤0 时启用 + c2f 折扣，见上方实录），
> 优先级降——先让 cap 默认值生效即可。
>
> 【2026-10-16 调度器端到端冒烟（som_human，double_wave/colmap_refined，20k）】
> **防爆 + 免调参成立，但代价 ~0.3dB**：
> | 模型 | thr/until | 调度器 | 窗口后 prune | @10k | best(19k) | @20k | 点数@20k |
> |---|---|---|---|---|---|---|---|
> | model_colmap_cap600k（参考） | 默认 0.0007/15000 | 关 | 无（撞帽整块跳过） | 21.18 | 21.63 | **21.81** | 629,715 |
> | model_dsche_aggr（A2，修复后） | 激进 0.0002/10000 | 开 | 只按 opacity | 20.65 | — | 21.41 | 565,493 |
> | model_dsche_sched_def（C2，裸跑验收） | 默认 0.0007/15000 | 开 | 只按 opacity | 20.85 | 21.549 | 21.465 | 580,267 |
> | model_dsche_noprune（C3，隔离实验） | 默认 0.0007/15000 | 开 | **关**（完全不剪） | 20.93 | 21.557 | 21.306 | 599,319 |
>
> - 激进（0.0002/10000，v2 爆炸配置）与默认（0.0007/15000）两种 thr 都稳定收敛、
>   点数严格 ≤600k，无需人肉传 EXTRA_TRAIN_ARGS；B（momentum 模式）冒烟语义正确。
> - **⚠️ 发现并修复窗口后 prune 死亡螺旋**：densification_postfix 每次加点会把
>   `max_radii2D` 清零 → 窗口内尺寸判据形同虚设；窗口关后不再清零 → 累积成
>   「历史最大屏幕半径」→ 首次 prune 删 137k → 正反馈指数崩塌（599k→9.6k，
>   PSNR→5.34，三次重跑一致）。修复 = 窗口后只按 opacity<0.005 剪（不带尺寸判据）。
>   详见 som_human/README.md「死亡螺旋」节。
> - **代价归因（C3 已排除「窗口后剪」）**：C2 vs C3 末三点均 21.439 vs 21.384
>   （差 −0.055，@19k +0.008 / @20k −0.159，噪声 ±0.045）→ 窗口后 opacity-only
>   prune **质量中性**且 C2 少 3.2% 点（580,267 vs 599,319）→ 默认维持 `=1`。
>   代价来自「严格 cap + absgrad 均值 topk 截断」策略本身（参考组撞帽后超到
>   629.7k 冻结，实得 5% 额外容量）。综合 −0.26~−0.34dB → **调度器默认仍关**，
>   作「裸跑防爆」保险丝按需开。
> - **deformable_human 侧已移植**（train_mask.py 经 `deformable_human/densify_sched.py`
>   shim 复用 som_human 本体，`02_train.sh` 加 USE_DENSIFY_SCHED/MAX_GAUSSIANS/
>   DENSIFY_MAX_RATE/DENSIFY_CNT_CLAMP/DENSIFY_PRUNE_AFTER 一组 env，仅
>   `USE_MASK_LOSS=1` 分支接线）。运行时冒烟：hand_motion 1.5k 步、窗口压到
>   [100,600) 强制走窗口关闭分支，调度/剪枝/修复三条路径均正常执行。
> - **⚠️ 同族地雷二次发现（硬护栏路径，默认开启！）**：`n_before >= MAX_GAUSSIANS`
>   的护栏分支同为「只 prune 不 densify」，却仍带尺寸判据 → `max_radii2D` 无 densify
>   补点清零 → 退化成「历史最大屏幕半径」→ 批量误删。低 cap A/B（double_wave，
>   cap 120k，5k 步）：带判据 iter 3100 一步删 86,738 点（162k→75k，−54%）并与
>   densify 交替震荡，**@5k PSNR 8.85 崩**；去判据后 applied 只剩 opa、点数单调平稳，
>   **@5k PSNR 19.73 正常**。修复 = 护栏 prune-only 也传 `max_screen_size=None`
>   （双端同改）+ 补分量埋点。**纪律：任何「只剪不增」的 prune 调用一律不得带尺寸
>   判据**——附近没有 densify/postfix 调用的地方就不该传 `max_screen_size`。

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

> 【本地状态】⚠️ **som_human 侧已实施并完成首轮 A/B，λ=0.05 为负结果**
> （2026-10-16）。实现落点与设计有两处有意偏移：
> - **离线改用 UniDepth V1 ViT-L14**（`01f_depth_maps.sh`，gavs env，米制深度
>   存 `<scene>/depths/*.npz`）——不是 MoGe；且修了 torch.hub 离线加载
>   （`source="local"`）与「静默失败仍打印 Done」两个坑。
> - **训练侧做点级锚定而非渲染深度 loss**：本仓 fork 光栅化器的 depth 是
>   前向-only 输出（backward 无深度梯度通道），图像级渲染深度 loss 是摆设；
>   改为把形变后高斯中心投影采样单目深度做加权 SSI-L1（`depth_prior.py`，
>   梯度只往位置流）。== deformable_human 侧未接（它走图像级 mask loss）。
>
> 【2026-10-16 首轮 A/B，负结果】double_wave/colmap_refined，20k，同代码逐字
> 同参、唯一变量 λ 0→0.05：
> | 模型 | λ | ALL | FG | BG | 点数@20k |
> |---|---|---|---|---|---|
> | model_depth_ctl（对照） | 0 | **21.872** | **23.474** | **21.670** | 613,855 |
> | model_depth005 | 0.05 | 21.345 | 22.433 | 21.140 | — |
> | Δ | | **−0.527** | **−1.041** | **−0.530** | |
> （FG/BG 由 `som_human/03b_fg_metrics.py` 在 01d mask 内/外分别算；ALL 口径与
> 训练期 Evaluating test 逐位一致）
>
> **根因（两条叠加）**：① 量级失衡——残差是绝对米制未归一化（实测 ~3.5~6.7，
> ×0.05=0.18~0.33 ≈ 光度项 ~0.09 的 2~4 倍，深度项反客为主主导优化）；② 单目
> 深度对手部不可靠（上方风险命中），把逐点几何硬拉向错误深度、专伤前景。
> **过程还顺带挖出一个 gy 垂直镜像 bug**（投影采样 y 符号错，`a<=0` 门限让损失
> 静默失效 10k 步；用光栅化器自渲染深度逐点核对裁决 +0.68~0.88 vs −0.48~−0.50，
> 已修 `depth_prior.py` 与源头 `train_mask.py sample_fg_prob`）。
>
> **结论与后续**：当前 λ=0.05 不可用，**默认保持 DEPTH_LAMBDA=0**。若要再做：
> ① 残差尺度归一化（除对齐尺度/用相对深度）否则 λ 随米制尺度漂移不可移植；
> ② λ 降到 0.005 量级；③ 换边缘感知项（只对 ∂d/∂x,∂d/∂y 加 L1）——正是上方
> 「进阶变体」。代码与评测工具留仓备查。

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
>
> 【2026-10-15 profile 实测】渲染前向占步时仅 ~5%（@1.5M 点），即使 backward 里
> 的光栅器部分全部白送，G 案一期渲染器替换的总收益上限 ~10-15% 量级 → 一期
> **冻结**；二期「内参可学」仍是唯一卖点，与 B 案绑定评审。

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

## 方案 J：per-person 独立重建（多人场景拆分）

> 【本地状态】✅ **已实施并实测**（2026-10-17，som_human）。链路：
> `01g_person_masks.sh`（SAM3 video prompt=person）→ `01h_build_person_scenes.sh`
> （按 pid 拆场景）→ `02_train.sh FG_MASK_DIR=<persons/pXX>/masks`。

**动机**：多人采集包（双人招手、三人聚餐）整场景一起重建有三个结构性问题，
都会吃掉手部细节：①容量竞争（600k 上限被背景+他人先占满）；②运动基竞争
（10 条 SE(3) 基同时表达多人）；③评测被背景稀释（fg 只占 10%）。

**关键实现点**（都是实测踩出来的，详见 som_human README per-person 节）：
- SAM3 必须 **video 模式**（obj_id 跨帧一致）；image 模式 ID 逐帧乱序。
- text prompt 会把背景物体误检成 person：hand10 实测 27 个 track 里只有
  2 个真人，其余与真人 IoU=0。过滤靠**帧覆盖率 ≥50% + 平均面积 ≥0.5%**。
- **初始点必须按 mask 过滤**（否则背景点永远留在模型里），实测 53,676→13,925。
- loss 按 mask 加权后要**面积归一化**（乘 numel/area），否则 densify 阈值失准。
- DG 的 `max_radii2D` 在 `create_from_pcd` 里按原点数建，过滤后必须重建。

**实测（hand10 双人招手，10k + -r 2）**：

| 模型 | 点数 | 耗时 | FG PSNR（本人 mask 内） |
|---|---|---|---|
| 整场景（基线） | 608,400 | 635 s | p02: 24.492 / p07: 24.497 |
| per-person p02 | 103,669 | ~320 s | 24.343（−0.15，噪声内） |
| per-person p07 | 115,896 | ~320 s | 25.193（+0.70） |

**结论**：可行。点数/体积 1/6，训练速度 3→32 it/s，FG PSNR 持平或略升，
且逐视角最差帧更好（+1.1dB）。⚠️ 遗留缺陷：mask 边缘有黑色飞溅状飞点
（羽化区梯度 + 双人贴身互染 + 初始残留点）——后续可做 mask 腐蚀、训练中
按 mask 周期性 prune、提高 init ratio。

**与 F（增强反哺）的关系**：J 解决「容量与基竞争」，F 解决「手部纹理不足」，
两者**正交可叠加**（per-person 模型里再跑 F）。J 优先做（更根本）。

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

**2026-10-15 步时分解实测（som_human profile_step.py，-r 2，每点 200 迭代均值）**：

| N_gauss | deform | render | loss | backward | opt | TOTAL | it/s | 峰值显存 |
|---|---|---|---|---|---|---|---|---|
| 215,900 (v1@20k) | 5.7ms | 1.5ms | 2.4ms | 13.8ms | 2.2ms | **25.5ms** | 39.2 | 2.2GB |
| 1,378,956 (@7k) | 28.5ms | 4.9ms | 2.5ms | 49.9ms | 8.1ms | **93.9ms** | 10.7 | 6.0GB |
| 1,502,625 (@20k) | 31.0ms | 5.4ms | 2.5ms | 50.7ms | 8.9ms | **98.7ms** | 10.1 | 6.4GB |

- deform MLP 前向+反向 ≈ 80% 且随 N 线性 → **降 N（D 案 cap）是唯一大杠杆**；
- render 前向仅 ~5% → C/G 两案维持冻结；
- dense Adam 9% → 稀疏 Adam 定级小杠杆（服务器 sparse Adam 的真实收益需在其
  点数/帧数口径下重新评估，勿外推）。

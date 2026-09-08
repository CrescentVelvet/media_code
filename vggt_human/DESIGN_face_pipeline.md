# 人脸链路 v2 — 设计方案与决策记录

> 2026-09-08 设计方案评审（grill-me）产出。记录**八阶段全链路的技术决策、论证过程与遗留项**。
> 与现有实现的关系：现有 `03f_fit_head_3dmm.py` / `03g_build_head_gs.py` / `03h_split_person_scene.py`
> 是「无表情的中性脸 + 每帧刚体位姿」路线；本方案改为「FLAME + 表情驱动 + 重心绑定」，
> 差距较大，**可能新起代码仓**（未定）。
>
> 运行命令看 README.md；实验结论看 EXPERIMENTS.md；原理与排错看 NOTES.md。

---

## 0. 八阶段总览

```
1 人脸检测 → 2 人脸匹配 → 3 人脸重建 → 4 3DMM 三阶段对齐
                                                  ↓
8 训练 ← 7 Body 初始化 ← 6 Avatar 初始化 ← 5 人脸增强(HYPIR)
```

---

## 1. 阶段一：人脸检测

| 项 | 决定 |
|---|---|
| 输出 | face bbox + 5 点 landmark |
| 5 点 landmark | **副产物，不参与任何下游**（不用于仿射对齐，不用于时序匹配） |

---

## 2. 阶段二：人脸匹配（face ↔ body ID 关联）

### 定稿：三层判据

1. **主判据** point-in-mask：人脸框中心（或鼻尖）是否落在 person mask 内
2. **次判据** IoM = \|face ∩ person\| / \|face\|
3. **全局分配** 逐帧累计 → **匈牙利**一对一（不是贪心）+ 时序约束（允许缺帧，不允许 ID 跳变）

### 为什么不用字面 IoU(face_box, person_box)

人脸框基本完全包含在人体框内，`IoU = \|face\| / \|person\|`，量级仅 0.02~0.1。
更致命的是两人并排/遮挡时，A 的脸框会同时落在 A、B 的人体框内，两个 IoU 只差分母
`\|person\|`，**区分度趋近于零**——会得到一串看起来合理、实际串号的 pid。

> 现 `split_person_scene.py:112 map_person_to_face` 用的是 IoM + 贪心一对一。
> 迁移时需改为匈牙利 + 时序约束。

### 时序跟踪

由 **SegTrack 用 mask IoU** 完成（mask vs mask，同类同尺度，此处 IoU 才是有意义的）。
与阶段二的 face↔body 关联是**两件事**，不要混淆。

### 漏检帧策略

**不兜底**——该帧静默跳过人脸链路，但**保留 body/scene 训练**（用 SegTrack mask 分割）。
容错靠成功率阈值 + 降级，而不是逐帧补洞：

- `face_alignment.py:384-387` 有效帧数 < num_limit → 抛异常退出人像模式
- 漏检过多 → `cnt_hasface < face_align_num_limit` → `human_mode=False` 降级为静态重建

---

## 3. 阶段三：人脸重建（bbox → 512 crop → dense landmark）

| 项 | 决定 |
|---|---|
| 窗口 | `cal_center_winlen()`：`winlen = max(w,h)·0.5·scale_exp`，x/y 同 winlen → **正方形窗口** |
| 仿射 | `cv2.estimateAffinePartial2D(LMEDS)` —— 4 自由度相似变换（旋转+等比缩放+平移），**天然排除各向异性缩放** |
| 输出尺寸 | 512×512 |
| scale_exp | **2.5**（ExpTracker 输入裁剪，含上胸）/ **1.25**（人脸增强 `affine_l2s`）|
| 越界 | 填黑 `border_value=0`，`flag_outbound` 有标记但**未用于拒帧** |

### ⚠️ 遗留项

1. **scale_exp=2.5 锁死了人脸的像素预算**：方形边长 = 2×winlen = 2.5×maxdim，
   人脸在 512 canvas 里恒占 `512/2.5 ≈ 205 px`，**与源分辨率无关**。近景帧（脸 500~600 px，
   如 06e 注入的 closeup 帧）被降采样 2.5~3×，657 点逆变换回原图后量化误差 ≈ ±3 px。
   且 4.1 的 reject 用「中值误差 × reject_extent」，**中值是相对量，地板抬高不会被发现**。
   → 2.5 可调，近景帧建议单独走 1.25 或 1.6。
2. **`flag_outbound` 打了但没人用**：越界填黑的帧，黑区上的 landmark 大概率是垃圾，
   现在没有机制拦截。→ 在 4.1 的 reject 里直接剔除或单独压低阈值（改动两三行）。
3. **增强与重建的尺度不匹配**：增强 crop 用 1.25（脸占 ~410 px），
   **增强 GT 的细节尺度比 landmark 精度尺度细 2 倍**——3DMM 对齐精度会成为增强收益的瓶颈。

---

## 4. 阶段四：3DMM 三阶段对齐

### 4.0 模型选型（重大变更：自研 3DMM → FLAME）

| | 自研 3DMM | FLAME 2020 |
|---|---|---|
| 顶点 | 20971 | 5023 |
| 身份 | 532 (id_base) | **300** (shape) |
| 表情 | 235 (exp_base) | **100** (expr) |
| landmark | 657 | **468**（MediaPipe）|
| 顶点索引 | `lm_tid` + `lm_abc` | FLAME 自带 landmark face index 映射 |
| 权重 | `face_bs_info.bin` | `FLAME2020.pkl`（直接下载，smplx 加载）|
| 重建器 | ExpTracker（ViT 自研）| **DECA / EMOCA**（预训练）|

**已排除 MHR**：MHR (Meta Momentum Human Rig, arXiv 2511.15586) 实际维度是
**45 identity**（20 body + 20 head + 5 hands）+ **72 FACS expression**（艺术家雕刻 blendshape，
非 PCA）+ 204 pose，与 532/235 对不上。

**DECA 的职责边界**：**只提供 shape / expr 初值，不提供 landmark**。
landmark 仍用 MediaPipe 468（复用现有 `detect_face_landmarks.py`）。

### ⚠️ 468 点 vs 68 点：不是偏好，是可解性

| landmark | 每帧观测 | 每帧自由度（local_SRT 6 + exp 100） | 观测/自由度 |
|---|---|---|---|
| 68 点 | 136 | 106 | **1.26** |
| 468 点 | 936 | 106 | **8.83** |

比值**与帧数无关**（观测与自由度同步增长），**加帧救不了 68 点**。1.26 基本是插值方程，
L1 正则扛下全部约束，exp 学出来的东西不可信。DECA 只输出 68 点，所以 landmark 必须另配。

### 4.1 optimize_global

- 优化变量：全局 `global_s` + `global_r` + `global_t`
- Adam 300 iter，`ExponentialLR(γ=0.95)`

**⚠️ 对原设计的修正**：原设计 4.1 只有全局 SRT，但人头在视频里会转——残差大小主要由
「该帧偏离平均朝向多少」决定，**而不是该帧 landmark 的质量**。
`reject_mask = err ≥ median × reject_extent(=5.0)` 会**系统性砍掉大角度转头帧**，
而这些帧恰恰是约束 3D 形状最有用的（极端姿态对 shape/pose 解耦贡献最大）。
（`_global_init` 用 PnP 旋转的中位数作全局旋转初值，进一步印证 4.1 拟的是「平均朝向」。）

修正方案：

```
4.1 之前：DECA（优先）或 PnP 给出 per-frame local_SRT 初值
          local_r[frame] = global_R.mT @ w2c_r.mT @ pnp_r[frame]
4.1      ：local_SRT 固定（requires_grad=False），不参与优化
          变换链 p_world = Global_SRT( local_SRT[f] @ p_canonical )
          → 残差真正反映 landmark 质量
4.2      ：local_SRT 放开（requires_grad=True），从 DECA/PnP 初值微调
4.2 之后 ：才做 reject（此时每帧有独立 local_SRT，残差干净）
```

DECA pose 优于 PnP：从图像语义学到，对遮挡和模糊更鲁棒；PnP 纯靠 2D-3D 点对应，
landmark 有噪声时旋转估计会飘。

### 4.2 optimize_local

- 新增：每帧 `local_r` + `local_t`，`requires_grad=True`
- 全局参数 lr × 0.1；Adam 300 iter

**anchor 正则**（限制 local_SRT 幅度，防止局部解跑飞）：

```python
# canonical 空间一个固定 3D 点（原实现：两对称顶点 9091/19407 的中点，头部中后方）
self.anchor = (mean_shape[9091] + mean_shape[19407]) * 0.5
anchor_local   = quat_to_mat(local_r) @ anchor + local_t
anchor_reg_loss = ||anchor_local - anchor||²
```

- 权重：`optimize_local` **0.005** / `optimize_coeff` **0.01** —— 量级很小，软约束，投影 loss 仍主导
- FLAME 下的 anchor：改用 FLAME mesh 的头部中心点（head joint 位置，或两耳顶点中点）

**55° 校验与重置**：局部旋转与中位旋转夹角 > 55° 的帧重置。

| 方案 | 评价 |
|---|---|
| 归零 identity（原实现）| ❌ **最差**。重置后该帧只剩全局平均朝向，4.3 要从零学 >55° 旋转，很难收敛 |
| **重置为 PnP 初值** | ✅ **采纳**，最小改动（`pnp_r`/`pnp_t` 在 `_pnp_init` 里已算好） |
| 重置为 DECA pose | 可行，与上一项同源 |
| 邻帧插值 | 可行，但视频抽帧的场景下帧间姿态未必连续 |

### 4.3 optimize_coeff

- 联合优化：Global SRT + Local SRT + `id_coeff`(300) + `local_exp`(100)
- Loss = 2D landmark 重投影误差 + `λ_id·‖id‖₁` + `λ_exp·‖exp‖₁` + `λ_anchor·anchor_reg`
- Adam 300 iter
- 待定：`λ_id` / `λ_exp` 取值；是否需要 exp 的时间平滑（见 §9）

---

## 5. 阶段五：AvatarGaussian 初始化

### 中性脸的定义

```python
neutral = mean_shape + id_base @ id_coeff      # exp = 0
```

**「平均表情」不是对表情系数取平均，而是直接将 exp 置零**。3DMM 线性模型中表情基底是
相对中性脸的偏移量，`exp=0` 就是中性状态。

### ⚠️ 重心坐标绑定（原设计缺失，必须补）

`exp_base` 形状是 `(100, 5023, 3)`，**只能驱动 5023 个顶点位置**。而 3DGS 训练默认开
densify，点数会长到几万——新点没有 `exp_base` 索引，形变时头部会**分层撕裂**。

| 方案 | 后果 |
|---|---|
| 关 densify | 点数只有自研 20971 的 1/4，且无 densify 补细节，质量倒退 |
| 开 densify 不绑定 | ❌ 新点驱动不到，撕裂 |
| **开 densify + 重心绑定** | ✅ **采纳** |

```
每个高斯绑定一个 FLAME 三角形，存 (face_id, bary)
每帧位置： p_f = bary @ V_f[face_id]
          V_f = mean + id_base@id + exp_base@exp_f
densify： clone 直接继承父点 (face_id, bary)；split 继承 + 微小扰动
```

> ⚠️ 现 `build_head_gs.py:71-94 sample_surface()` **已经算出** `face_ids` 和 `bary`，
> 但 `build_head_gs.py:432` 是 `pts, nrm, _, _, spacing = sample_surface(...)` —— **bary 被丢了**。

**绑定方式**：mesh 表面走**硬绑定**（`p = bary @ V_f`，xyz 不作为独立优化参数）。

### 自由高斯（头发 / 睫毛）

另挂一批**不绑定**的高斯，专门吃 mesh 表达不了的结构。分两类：

| 区域 | 变换链 |
|---|---|
| 头发生（头皮区）| 复用 face 的 `local_SRT` + `global_SRT`，**不叠 exp** —— 参照 glass 的 `vert_ids` 虚拟顶点机制（`gaussian_avatar.py:128-129` 把 `bs_offset` 填零）|
| 躯干 / 肩膀 | 保持独立 `BodyGaussianModel`（自有 `local_r`/`local_t`，**无 global_SRT，无 blendshape**）——躯干不该跟着头部转 |

---

## 6. 阶段六：训练

- `local_exp` `(N_frames, 100)` 作为**可训练参数**，与高斯属性（opacity / scale / rotation / SH）联合优化
- 每次前向：`means3D = mean_shape + id_base @ id + exp_base @ local_exp[frame]`
- 梯度来自渲染 loss（L1 + LPIPS + SSIM）
- 迭代是**必须的**：exp 初值来自 DECA 预测，精度有限，需要渲染 loss 持续 refine

---

## 7. 阶段七：后处理增强

```python
# gaussian_avatar.py — fix_and_sync_exp()
# 只改 exp，不碰 SRT
self.local_exp.data.copy_(rep_exp.unsqueeze(0).expand(N, -1))
# global_s, global_r, global_t, local_r, local_t 全部不变
```

- 代表视角帧 = 最正面（yaw/pitch 最小）且表情最接近中值
- 统一表情后重新渲染 → 超分辨率增强

### SRT 与 exp 的独立性（论证成立，但要区分两个层面）

```
顶点 = mean_shape + id_base@id + exp_base@exp   ← 表情（中性空间）
→ local_R @ p + local_t                         ← 局部刚体
→ global_s * (global_R @ p + global_t)          ← 全局刚体
```

- **表示层面**：改 exp 只影响顶点局部形变，SRT 参数不变 → **独立**（成立，无需重跑 SRT）
- **优化层面**：landmark 残差对 `exp` 与 `local_SRT` 的雅可比**不正交**
  （jaw-open 与「微微低头」在正面视角的 2D 投影几乎不可分）→ 仍存在耦合。
  这正是 L1 正则与三阶段（先固定 exp 调 SRT，再放开）存在的理由。
  **表示独立 ≠ 优化可分离。**

---

## 8. HYPIR 增强（前处理 / 后处理）

### 实测基线（EXPERIMENTS.md）

| 结论 | 出处 |
|---|---|
| masked-L1（增强图仅作人脸区 GT）对人脸区锐度**无可测量贡献**（+4.04% vs +3.98%）| L81 |
| 06e 整视角注入（增强近景作**完整训练视角** + 真实相机位姿 + COLMAP 注入）**实测有效** | L96-103 |
| 「masked-L1 无效 ≠ 增强图无用；整视角注入才是正确用法」 | L103 |

### ⚠️ 06e 的结论不能直接迁移

06e 的增强图配的是**注入的近景相机**——一个原本没有观测的**新视角**，没有旧观测就没有
冲突，纯增益。而「前处理增强原训练帧」是拿一个新的外观去**覆盖一个已存在的一致观测**。
这是两件事。

### 定稿：三个入口全部启用（不是二选一）

#### 入口 1 — 前处理增强（训练前，全帧替换 GT 人脸区域）

- 时机：`_face_preprocess()`，训练开始前
- 函数：`face_enhance_hy_with_sam_multi(is_post_enhance=False)`
- 覆盖：**所有检测到人脸的帧**（全帧，非子集）
- 流程（`face_enhance.py:63-104`）：`src = tgt = 原图` → 裁人脸 box → 512² 增强 →
  `fused = crop_enh·mask + crop_out·(1−mask)`（中心=增强，边缘=原图，**54px 渐变边框**）
  → `cam.reset_image(enh_image)` 永久替换 GT

#### 入口 2 — 后处理增强（epoch 48，伪 GT / 自蒸馏）

- 函数：`face_enhance_hy_with_sam_multi(is_post_enhance=True)`；前置 `fix_and_sync_exp()`
- **关键区别**：`src_image = render(cam, gaussians)` —— 裁剪的是**当前模型渲染图**，不是原图
- 人脸中心 = 增强渲染图，边缘 = 当前 GT → `cam.reset_image()`
- 目的：自蒸馏。告诉模型「你渲染的样子增强后长这样」。
  原始 GT 人脸可能模糊/低质量；渲染图是 3D 模型多视角聚合的结果，**天然多视角一致**
  （这一点正好规避了入口 1 的跨视角不一致问题）

#### 入口 3 — 新注相机（epoch 48，增强渲染图作额外训练数据）

- 函数：`get_novel_views_portrait()` + `enhance_novel_cameras()`
- 采样 20 个输入相机 → 沿「相机→人脸中心」推进直到 avatar mask 覆盖率 ≥ 40%
  → 绕人脸中心 ±5° pitch → 每相机 2 个新视角
- 整图增强（`enhance_512`，不需要 SAM 检测框）→ `cam.reset_image()` → 追加 `novel_cameras`
- 后续 epoch 由 `set_train_cameras()` 自动追加（`pipeline.py:248-249`）
- **loss 权重 0.5**（`pipeline.py:356-357`，L1 与 SSIM 同乘）

> ⚠️ 数字待核对：「入口 3」写 20 相机 × 2 = **40 张**；「触发时机」一节写 **160 张**。

### 跨帧一致性：seed 改进（定稿）

现 `face_enhance.py:312` `set_seed(231)` 在 `main()` 里只调**一次**，各帧顺序消耗同一个
随机流，因此**每帧拿到不同的噪声样本**（注释 L225-228 已指出输出对「第几次调用」敏感）。
固定 seed 保证了可复现，但**不保证跨帧一致**。

**改为每帧处理前重新 `set_seed(231)`**，让每帧用同一个初始噪声。

加成：增强 crop 经 `estimateAffinePartial2D` 对齐（`affine_l2s`, scale_exp=1.25），
五官在 crop 内大致对齐 → 同一噪声图案落在同一面部位置，一致性收益比未对齐 crop 大得多。
一行改动，建议先做 5 帧 A/B 验证。

---

## 9. 头 / 身切分与接缝

### 现状（`__gaussian_init_human_body`, pipeline.py:1294-1331）

```python
avatar_points = gaussians_avatar.compute_neutral_global().detach()   # FLAME mesh 顶点
max_dist = (avatar_points.max - avatar_points.amin)².sum() * 0.1     # 对角线² × 10%
knn_dists = knn_points(body_points, avatar_points, K=1).dists        # body 点 → 最近 face 顶点
body_mask = knn_dists > max_dist                                     # 只保留离 face 远的
body_points = body_points[body_mask]
```

策略：**暴力删近邻点，不补缝**。

- face mesh 覆盖：脸 + 脖子（FLAME 含 neck 顶点；自研 20971 顶点覆盖到脖子下方）
- body 点云覆盖：躯干、四肢、头发等（离 face 远的所有点）
- **gap**：脖子到肩膀的过渡区——离 face mesh 太近被删除，又不在 face mesh 覆盖范围内

gap **无人显式覆盖**，当前靠三个隐式机制兜底（均不可靠，大角度转头时易暴露）：

1. 高斯 alpha blending 重叠（要求两侧高斯 scale 足够大）
2. densification 往 gap 生长新高斯
3. `max_dist` 阈值不够大时，部分脖子/肩膀点被保留

## 10. 训练期触发时机

| 项 | 值 |
|---|---|
| `epochs` | 60 |
| `face_enhance_epoch` | 48（= epochs − 12）|
| `portrait_novel_epoch` | 48（与后处理同 epoch）|
| 触发方式 | **单次**（`==` 不是 `>=`）|

epoch 48 末尾顺序（`pipeline.py:595-600`）：

1. `face_enhance_post()` → `fix_and_sync_exp()` + 增强渲染图替换 GT 人脸区
2. `portrait_novel_enhance()` → 生成 novel view → 增强 → 追加训练列表

epoch 49~60：`set_train_cameras()` 自动追加 `novel_cameras`；GT 与 novel 集合**都不再变**。

**为什么是 48**：留 12 个 epoch 在增强 GT 上收敛；前 48 epoch 让几何与外观基本成型。
太早（如 20）渲染质量差，伪 GT 引入伪影；太晚（如 58）只剩 2 epoch 来不及收敛。

## 11. 未决项 / 下一步

| # | 项 | 说明 |
|---|---|---|
| 1 | **换 FLAME 会让 KNN 切分行为翻转** | 顶点密度 20971→5023（1/4），body→最近顶点距离变大 → 保留的点增多 → 问题从「gap 空洞」变「head/body 重叠 → 重影」。`max_dist` 未归一化顶点密度 |
| 2 | **自蒸馏后人脸区失去真实锚定** | epoch 48 后人脸 box 内 GT 与 novel GT 都是「增强渲染图」，后 12 epoch 无真实观测，偏差会被固化且无回滚 |
| 3 | novel view 张数 | 入口 3 写 40 张，触发时机节写 160 张，需核对 |
| 4 | 新仓 vs 继续 vggt_human | 未定 |
| 5 | `λ_id` / `λ_exp` 取值 | 4.3 的 L1 正则权重 |
| 6 | exp 时间平滑 | 视频帧间连续，当前 loss 只有 L1 幅度正则，无帧间平滑 |
| 7 | 4.3 收敛性 | 变量 = 7 + 300 + F×(6+100)，Adam 300 iter 是否足够 |
| 8 | 增强区 mask 的 3D 一致性 | 现为逐帧 2D bbox crop + feather；同一 3D 点在不同帧可能落在增强区内/外 |

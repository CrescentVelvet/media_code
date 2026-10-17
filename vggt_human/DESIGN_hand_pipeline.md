# 手部链路 v1 — 审计与可行方案

> 2026-10-17 立。输入是 som_human 侧「梯度学运动已证伪 / 改走先验给运动」的路线结论
> （`som_human/EXPERIMENTS.md` E-06/E-07），本文做两件事：
> ① 拿结论逐条核对 vggt_human 现状；② 给出可落地的分阶段方案与判据。
>
> 与 `DESIGN_face_pipeline.md` 的关系：人脸链路的「3DMM 对齐 → 固定系数 → 静态重建」
> 是手部路线的骨架，本文只写差异项，不重复八阶段推导。

---

## 0. 结论摘要（先给判断）

1. **vggt_human 目前没有手部能力**：全仓 `hand`/`mano` 命中的只有 `09_hand_micro_batch.sh`
   与 `09b_collect_preview.sh`（批量跑静态 3DGS 的出图脚本），**没有任何 mesh 绑定、
   形变驱动、手模拟合代码**。所谓「搬过来就有骨架」不成立——骨架在 `flame_human`。
2. **09 的位姿来源与路线约束直接冲突**：`09_hand_micro_batch.sh:8` 明写「transforms.json
   自带位姿本步骤不使用（走 VGGT-Omega 前馈）」。这是必须先改的一行级问题（§3.1）。
3. **`−5.3dB`（VGGT 前馈位姿 vs COLMAP）在本仓找不到出处**：全仓 md 检索无该数字，
   `EXPERIMENTS.md` 只有 03b「固定内参只精修外参」的记录（L276-281）。
   → 该结论在 vggt_human 里是**外部结论**，必须补一次仓内 A/B 才可当基线引用（§3.2）。
4. **「手部必须外挂 MANO」这条约束可以放松一档**：`sam_3d_body` 已带手部解码器
   （`02_run_inference.sh:43` `INFERENCE_TYPE=full|body|hand`），且逐帧落盘
   `hand_pose_params` / `lhand_bbox` / `rhand_bbox`（`run_inference.py:131,139-141`）。
   MHR 的 **5 维只是手部 identity**（`DESIGN_face_pipeline.md:101-103`），
   **不等于手指姿态只有 5 维**——实测手部 pose 是 **27 DOF/只**（2026-10-17 探针）。
   先探 MHR 手，MANO 降为 fallback。
5. **真正可复用的骨架在 flame_human**，不在 vggt_human：
   `init_avatar_gs.py:68` 面积加权采样出 `(face_id, bary)`、
   `train_avatar.py:179` 绑定形变 `p = bary @ V_f[face_id]` + local/global SRT、
   `train_avatar.py:198` densify 继承绑定、
   `align_3dmm.py:162` `init_from_pnp` 用最小二乘解 **米 ↔ SfM 任意单位**的全局尺度。
   最后这条正好是单目手模落世界坐标的核心难点。

---

## 1. 结论 ↔ 仓内代码 逐条核对

| 结论条目 | 仓内实况 | 判定 |
|---|---|---|
| 位姿不能用 VGGT 前馈（−5.3dB，内参系统差） | 09 走 VGGT；03 的 `cameras.txt` 就是 VGGT 预测内参（`03_npz_to_colmap.sh:13-14`）；03b 只是在内参固定的前提下 BA 精修外参（`EXPERIMENTS.md:276-281`），**结构上治不了内参偏差** | ⚠️ 结论方向成立，**数字缺失**，需补测 |
| 位姿走 COLMAP（01c，+7.29dB） | vggt_human **没有 from-scratch SfM**；01c 在 som_human，且 `01c:141-147` 要求「注册帧数 == 总帧数」否则直接退出 | ✅ 可搬，但有硬门槛（§4.2） |
| MHR 手只有 5 维 identity，做不了手指形变 | `DESIGN_face_pipeline.md:101-103` 属实：45 id = 20 body + 20 head + **5 hands**，72 FACS，204 pose | ✅ 前半属实；**「做不了手指形变」已证伪**：手部 pose = **27 DOF/只**、scale 另有 10 维（探针 2026-10-17，见 `EXPERIMENTS.md`） |
| 可复用资产：01g/01h/01i/01j、01c | 成立，且 `01g:14` 已有跨目录复用先例（调 `vggt_human/sam3_face_masks_worker.py`） | ✅ |
| vggt_human 侧有等价 per-person 资产 | `split_person_scene.py:32-40` 依赖 `head_fit.json` + head mesh 包围盒，且是**训练后点云投票**，不是训练前拆场景 | ❌ 不能替代 01h，别混用 |
| 提容量、SoM、B 案已证伪 | `EXPERIMENTS.md:262-269`：本仓 rasterizer 对 viewmatrix 不求梯度，可微位姿在本后端**根本不成立**（与 B 案 −8.29dB 是不同层面的同一结论） | ✅ 无需再试 |
| pdfgs_human 的 PDF-GS 只能救背景 | 成立；但反过来用是优点：把「动的手」当 distractor 滤掉，正好给静态分支（§3.5） | ✅ 且可正向利用 |

---

## 2. 三个必须先接受的物理事实

1. **手部像素预算极小**：E-06 实测手占画面 1.61%（0.79%~2.39%）。720p 下两只手合计
   约 1.5 万像素 ≈ 每只 ~85×85。任何「整帧 256² 输入」的网络（含 SAM 3D Body）
   在手部区域只看到 ~85 px 的有效分辨率。
2. **单目手模没有绝对尺度**：SAM 3D Body 输出的是相机坐标下的米制网格，
   COLMAP 是任意单位。两者之间只差**一个全局标量 s**（同一人同一模型），
   但**逐帧不可观测**（同一时刻只有一视图）。必须借静止的身体去解 s。
3. **手部运动不是刚体**：招手=手臂刚体大位移（两态冲突，删姿态段有效）；
   写字/弹琴=手部局部形变且全程持续（`EXPERIMENTS.md` E-04）。方案必须能同时吃这两种，
   所以驱动量必须是**逐帧顶点**，不能只是一条刚体轨迹。

---

## 3. 方案

### 3.1 P0-1 位姿源切换（改动最小、收益最确定）

`09_hand_micro_batch.sh` 增加 `POSE_SOURCE=colmap|vggt`，默认 `colmap`：

```
colmap：01(采集包→COLMAP 场景) → 01c(SfM 精化，内参固定为 transforms.json 标定)
        → 04_train_3dgs.sh SOURCE_DIR=<colmap_refined>
vggt  ：现状 02→03→04（保留作对照，不作默认）
```

要点：
- 复用 som_human 的 01/01c（跨目录调用，先例见 `01g:14`），**不新写 SfM**。
- `04_train_3dgs.sh` 只要求 `SOURCE_DIR/{images,sparse/0}`，01c 产物天然满足，
  **训练侧零改动**（已核 `04:56-60`）。
- vggt_human 的 02/03 仍保留：它给的是**密集点云初始化**（scene.ply），
  与「位姿来源」是两件事，别一起砍掉。若要两者兼得：位姿用 01c、
  点云从 `03_source/sparse/0/points3D.txt` 取（需一个合并小脚本，约 80 行）。

### 3.1b P0-1b 01c 的硬门槛要补回退路径

`01c:141-147` 要求全帧注册，失败只打印排查方向、**没有实现**注释里写的
「point_triangulator 固定 VIO 位姿 + bundle_adjuster 只精化」。
手部数据运动大、易掉帧，这条回退是必需品而非可选项：

```
mapper 成功 → 现状
mapper 掉帧 → point_triangulator(--pose 用 transforms.json 的 VIO 位姿) → bundle_adjuster(固定内参)
```

另加一条成本极低但常被漏掉的开关：**COLMAP 特征提取屏蔽手部**
（`--ImageReader.mask_path`，mask 为 0 的像素不参与 SIFT）。
手在运动 → 其特征是纯外点，屏蔽后 on-person 匹配质量上升。
【待核实】COLMAP 4.2 的 mask 目录命名约定（`<image_name>.png`，是否需与 images 同名同层）。

### 3.2 P0-2 把 −5.3dB 变成仓内数字

同一 5~10 个采集包，两条链路各训 10k iter（`-r 2` 省时），只比位姿来源：

| 组 | 位姿 | 点云初始化 |
|---|---|---|
| A | VGGT（现状 03） | VGGT |
| B | COLMAP 01c | VGGT |
| C | COLMAP 01c | COLMAP 三角化 |

A/B 分离「位姿误差」与「点云误差」，这正是当前结论里混在一起的两件事。
判据：B−A 若 ≥ +1dB，则 09 默认钉死 colmap 并在 README 写明禁止用 VGGT 位姿做手部。

### 3.3 P1 主线：手部先验运动（MHR hand → 绑定高斯）

四步，每步独立可验：

**Step 1 — 手部专用 SAM 3D Body 推理（`01k_hand_mhr.sh` + `01k_hand_mhr.py`）**

- 输入：`persons/masks/{stem}.pXX.mask.png`（01g，prompt=hand）→ 取手 bbox，外扩 1.4×
  → **裁图放大到 512²** 再喂 SAM 3D Body（`INFERENCE_TYPE=hand` 或 `full`）。
  这一步是把物理事实①补回来：网络看到的手从 ~85 px 变 ~256 px（有效分辨率 ×3）。
- 输出：逐帧 `mhr_hand.json`——每只手 `hand_pose_params`(a)、`pred_vertices`(b)、
  `pred_cam_t`、`lhand/rhand_bbox`、重投影残差。**不做任何时序处理**，原样落盘。
- ✅ 已验（2026-10-17 探针，`EXPERIMENTS.md`）：`hand_pose_params` 是 **27 DOF/只手**
  （16 关节，54 维两只），MHR70 带 **21 关键点/只手**（ratio 42/6 = 7.0，够解 SRT）；
  网格 18439 顶点 / 127 关节；手部解码器输出的是**整网格**（`-2*18439` × 2 手），
  不是手部子网格。
- ⏳ 仍未决：18439 个顶点里**哪些属于手**，源码不暴露（`hand_joint_idxs_*` 是关节索引）。
  先走免资产近似：用 `pred_keypoints_3d` 的 21 个手部点 + KD-tree 半径取顶点并集；
  要精确解再读 ckpt 的 skinning 权重（需 gated 资产）。

**Step 2 — 逐帧 SRT 对齐（`01l_align_hand.py`）**

只优化**每帧的相似变换**（s 全局固定、R/t 逐帧），**pose 完全不优化**——
这是与已经被证伪的「梯度学运动」的根本区别：运动是给的，这里只解「给的运动怎么落到
COLMAP 世界系」。

- 初值：`init_from_pnp` 照搬（`align_3dmm.py:162-220`）——用 `hand_pose_params` 出的 3D 关键点
  与 2D 观测做 PnP，再最小二乘解 `R_cam·p + t_cam = s·t_pnp` 的全局 s。
  s 用**身体**（静止段）估：同一个人的 MHR body 网格 vs COLMAP 人体点云，一次解出、全程复用。
- 精化：2D 重投影（手部关键点 + mask 轮廓）Adam 200 iter，只放开逐帧 R/t。
- 时序：Savitzky-Golay（窗 5）平滑 R/t；残差 > median×3 的帧标记 `bad`（不删，训练跳过）。
- 输出：`hand_align.json`——每帧 `{s, R, t, rms, bad}`，与 `head_align.json` 同构。

**Step 3 — 绑定高斯初始化（`06x_init_hand_gs.py`）**

- 复用 `flame_human/init_avatar_gs.py:68` `area_weighted_sample`（改输入为手部网格子集），
  面积加权采样出 `(face_id, bary)`；按手部实际面积定采样数（手掌 8cm 尺度，
  参考头部 5023 顶点的密度，先给 8k~15k/只手，用 `--n_points` 扫）。
- `is_free=True` 的自由高斯留给袖口/手腕接缝（与 `DESIGN_face_pipeline.md` §5 同理）。
- **不改 densify 语义**：`train_avatar.py:198` 的「clone 继承、split 加扰动」直接可用。

**Step 4 — 训练（`08_train.sh` 变体）**

- 直接以 `flame_human/train_avatar.py:179` 的 `deform()` 为模板，把
  「FLAME 前向 + 可训练 local_exp」换成「逐帧外部顶点 + 固定 SRT」：
  `p_world = R_f @ (s·V_f[tri]·bary) + t_f`，V_f 来自 Step 1 的 npz（只读缓存）。
- 三条损失项：手部 mask 内 L1 加权 W=3（手占 1.6%，不加权等于没监督）、
  DSSIM 全图、可选「mask 外泄漏惩罚」（见 §3.4）。
- 训练规模对齐 som_human 的结论：per-person 场景（01g/01h）+ `-r 2` 起跑，
  不要一上来全分辨率（`02_train.sh` 默认全分辨率，单跑必须显式 `-r 2`）。

### 3.4 P0-3 评估口径先补齐（不做这步，后面所有结论都不可信）

E-03 的教训是硬约束：**FG PSNR 抓不到幽灵/透明**，因为幽灵落在 GT mask 之外。
`som_human/03b_fg_metrics.py:1-16` 只有 fg/bg PSNR，**没有泄漏项**。补三项：

| 指标 | 定义 | 治什么 |
|---|---|---|
| `leak_out` | `mean(alpha)` 在「GT 手 mask 膨胀 5px − GT mask」环带内 | 半透明光晕 / 多出来的手 |
| `ghost_in` | 该环带内的 |render−GT| 均值（GT 无手处出现手） | 两态冲突残留 |
| `sharp_ratio` | mask 内 log10(Laplacian var) 比（render/GT） | 糊（沿用 E-02 口径，便于跨仓对比） |

评估集固定为「姿态分段后的主段 + 全覆盖视角」，避免用被删帧的视角刷分。

### 3.5 P2/P3 备选与分工

- **P2-A HaMeR/MANO fallback**：若 Step 1 探针发现 MHR 手部顶点/姿态不可用。
  代价是 MANO 需注册下载（mano.is.tue.mpg.de）——要峰哥手动做一次。
- **P2-B 2D 对齐路线**：手部关键点（MediaPipe Hands 21 点）+ 光流驱动一层手部高斯，
  不做 3D 拟合。精度低、但几乎不可能失败，可作为 P1 失败时的交付保底。
- **P3 PDF-GS 分工**：`pdfgs_human` 的 DINOv3 distractor filtering 把「动的手」当干扰滤掉
  → 训 body/scene 静态分支；手部另起绑定分支。**这不是替代方案，是拆分架构里的分工**，
  正好把「只能救背景」变成优点。

### 3.6 明确不做

| 不做 | 理由 |
|---|---|
| 往 vggt_human 移植 SoM / HiMoR | som_human E-06 三档全退化；本仓 rasterizer 连位姿梯度都没有（`EXPERIMENTS.md:262-269`） |
| 提容量 / 提分辨率 / 加迭代 | E-02：24% → 14%，反向 |
| 用 MHR 的手型维度做骨相级个性化 | 5 identity + 10 scale 只能改胖瘦长短；手型差异靠高斯外观吸收 |
| 用 `03h_split_person_scene.py` 拆手部 | 依赖 head_fit.json + head bbox，语义不适用；per-person 走 01h |

---

## 4. 风险与判据

| 风险 | 触发信号 | 应对 | 止损判据 |
|---|---|---|---|
| COLMAP 注册不全 | `01c:141` 报 `N_REG != N_IMG` | §3.1b 回退路径；放宽 min_num_matches 15→10 | 掉帧 >10% 就退回 VGGT 位姿并记录 |
| MHR 手部姿态不可用 | Step 1 探针：pose 维度不含手指 / 手部顶点区间缺失 | 转 P2-A MANO | 探针 30 分钟内出结论，不拖 |
| 单目手模尺度漂 | 逐帧 PnP 解出的 s 方差 > 5% | s 全局固定（身体段估）而非逐帧解 | s 方差 >20% → 该序列放弃 P1 |
| 时序抖动 | 相邻帧 R 夹角中位 > 5° | Savitzky-Golay + `bad` 帧跳过 | 平滑后仍 >10% 帧为 bad → 降级 P2-B |
| 手部仍糊（绑定正确但高频缺失） | `sharp_ratio` < 30% | 检查 GT 是否已被 3DGS 平均；考虑 06e 式「手部近景整视角注入」 | 两轮注入无增益 → 接受静态上限，转交付 |

**回滚点**：每个 Step 产物独立落盘（json/npz），训练只读；任何一步变差可只换那一步的文件重训，
不需要重跑前序。

---

## 5. 脚本清单（遵循 AGENTS.md 数字编号）

| 脚本 | 作用 | 复用来源 |
|---|---|---|
| `09_hand_micro_batch.sh`（改） | 加 `POSE_SOURCE`，默认走 COLMAP | 现有 |
| `01k_hand_mhr.sh/.py`（新） | 手部 mask→crop 放大→SAM 3D Body→`mhr_hand.json` | `sam_3d_body/02`、`01g` |
| `01l_align_hand.py`（新） | 逐帧 SRT 对齐 + 全局 s + 平滑/坏帧标记 | `flame_human/align_3dmm.py:162` |
| `06x_init_hand_gs.py`（新） | 手部网格面积采样 → `(face_id,bary)` 绑定 npz | `flame_human/init_avatar_gs.py:68` |
| `08h_train_hand.sh`（新） | 绑定手部高斯训练（外部顶点驱动） | `flame_human/train_avatar.py:179,198` |
| `03b_fg_metrics.py`（搬+扩） | 加 `leak_out`/`ghost_in`/`sharp_ratio` | `som_human/03b_fg_metrics.py` |
| `01c_colmap_refine.sh`（搬+补） | 加 point_triangulator 回退 + 手部特征屏蔽 | `som_human/01c` |

---

## 6. 未决项

1. ✅ 已答（2026-10-17，`EXPERIMENTS.md`）：手部 pose = 27 DOF/只、identity 5 + scale 10 维、
   2D 观测 21 点/只手（ratio 7.0）→ **不必先上 MANO**。
   ⏳ 剩一项：手部**顶点索引集合**（需 gated 资产，先走 KD-tree 几何近似）。
2. 【待核实】COLMAP 4.2 `ImageReader.mask_path` 的确切命名/取值约定。
3. 【待核实】`−5.3dB` 的原始实验条件（帧数/分辨率/迭代），决定 §3.2 能否直接复现。
4. `09` 是否要保留 VGGT 分支：建议保留但默认关，只在 §3.2 的 A 组使用。
5. 手部高斯采样数、mask 加权 W 的取值——留给 Step 3/4 的小规模扫描，不预设。

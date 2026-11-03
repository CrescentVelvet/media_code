# 人像数据询价信（可直接发送）

> 2026-11-03。配套 `PORTRAIT_DATA_SOURCING.md`。
> 三封英文信 + 一封中文 pilot 询价。**方括号 `[...]` 为必改项**，其余可直接发。
> 发送前唯一要定的数字：pilot 买多少人（建议 50）、规模档位（500 / 2000）。

---

## 使用说明

| 信 | 收件人 | 目的 | 优先级 |
|---|---|---|---|
| **一** | humandataset.com 官网表单 / 其销售邮箱 | 买"多视角原图 + 相机参数 + 可发布权重的训练授权" | **最高，今天就发** |
| **二** | `info@renderpeople.com` | 问零售资产能否单独开 §4.2a 训练许可（少量试水用） | 与信一同日发 |
| **三** | Volucap（`info@volucap.de`） | 确认体积捕捉棚能否交付"原始多视角帧 + 标定" | 可选，路径 D 才需要 |
| **附** | 数据堂 `services@datatang.com`｜海天瑞声商务入口 | 中国面孔定制采集 pilot 报价 | 与信一同日发，用中文 |

**三封信里最不能省的一句话**：`the right to publish trained model weights`。
行业里最常见的坑是对方回一句 "yes, you may use our data for AI training"，但你训练完**不能发布权重**——那模型只能烂在自己机器里。这句话必须让对方**书面确认**。

---

## 信一：humandataset.com（主信，B2B 批量）

**Subject:**
```
Inquiry — multi-view capture data license for training a feed-forward 3D human reconstruction model
```

**Body:**

```
Dear [Name / HumanDataset Sales Team],

We are [公司/团队名称], a [团队定位，例如: computer-vision R&D team] based in Shenzhen, China.
We are building a feed-forward model that reconstructs a photorealistic 3D human avatar from a
small number of input views, and we are looking for a licensed training-data source.

We are specifically interested in your multi-view capture data, not in finished 3D assets.
Finished meshes cannot be used for our task, because the supervision signal we need is
"several calibrated photographs of the same subject at the same instant".

## What we need

| Item | Requirement |
|---|---|
| Views per synchronized frame | >= 16 (8 acceptable for a pilot) |
| Camera parameters | per-view intrinsics (focal length, principal point, distortion) at native resolution, plus extrinsics; delivered as a calibration file or COLMAP |
| Images | original camera frames, not reconstructed or retouched renders; long side >= 2K; JPG or RAW acceptable |
| Content | same-pose synchronized multi-view; not frames cut from a moving video |
| Poses per subject | >= 10, covering frontal, profile and back |
| Subjects | 50 for the pilot; 500 and 2,000 for production phases |
| Lighting | uniform studio lighting; please do not deliver textures with baked lighting |
| Background | plain backdrop, with alpha or matte |
| Annotation | SMPL-X registration parameters (preferred, not blocking) |
| Delivery format | COLMAP or transforms.json (NeRF/3DGS convention) |

## Licensing requirements

This is the part that decides whether we can proceed, so we would rather raise it upfront:

1. The right to use the data to **train machine-learning / deep-learning models**.
2. The right to **publish, open-source and commercially distribute the trained model weights**,
   including weights trained on your data. We understand and accept that we may not redistribute
   the source scans or meshes themselves.
3. The license must be **irrevocable, perpetual, worldwide and transferable**, including to
   downstream customers of our product.
4. Model releases from every captured person that cover machine-learning use under
   **GDPR Art. 6(1) and Art. 9(2)**, and that can be passed through to us.
5. No restriction on the derived model or on its commercial use.

If any of these five points cannot be granted, please tell us now rather than later — it will
save both sides a lot of time.

## Questions

1. Of the 34,000+ scans, for how many do you still hold the **raw multi-view photographs**
   (as opposed to only the refined meshes)? What is the per-scan view-count distribution
   (minimum / median / maximum)?
2. Do you retain per-view intrinsics and extrinsics from the photogrammetry rig, and can they
   be included in the delivery?
3. Your site mentions 35+ poses per person. Were these captured in a single session under
   identical lighting and calibration, or across multiple sessions?
4. Can you provide **1-3 sample scans (paid)** for a technical evaluation before we commit to a
   larger order? We would verify camera parameters by re-projecting the scans.
5. Pricing for: (a) pilot of 50 subjects, (b) 500 subjects, (c) 2,000 subjects — with and
   without SMPL-X registrations, and with and without RAW.
6. Delivery: format, transfer method (S3 / download link / physical HDD), and expected timeline.
7. For SMPL-X: do you deliver parameters only, or also registered meshes? Please note whether
   the underlying body-model license is included or must be obtained separately.
8. Please confirm in writing whether publishing trained model weights is permitted under your
   standard data license.

## Next step

If the above is workable, we would like to start with a paid pilot (50 subjects) to validate the
data against our pipeline, and scale up once the pilot passes acceptance.

Happy to sign an NDA first if that helps. Thank you for your time.

Best regards,
[Name]
[Title], [Company]
[Email] | [Phone] | [Website]
```

---

## 信二：RenderPeople（零售 + 单独申请 §4.2a 训练许可）

**Subject:**
```
Separate license request under T&C 4.2(a) — machine-learning training use of 3D scan data
```

**Body:**

```
Dear RenderPeople Team,

We are [公司/团队名称], based in Shenzhen, China. We are developing a 3D human reconstruction
model and would like to purchase a small number of your assets — approximately 10 Posed and
10 4D People items — for a technical pilot.

We have read your General Terms, and we understand that Section 4.2(a) lists "Use of the 3D data
for Computer Vision Research", including the development and training of machine-learning models,
as a use that requires separate written permission. We are therefore writing to request exactly
that permission, rather than assuming it is covered.

Specifically, we would need:

1. Permission to use the purchased assets to **train machine-learning models**.
2. Permission to **publish and commercially distribute the trained model weights** (not the
   source scans or meshes — we will not redistribute those).
3. Confirmation that such permission is **irrevocable and perpetual** for the assets purchased.

Three practical questions:

1. Is a separate license for this purpose available for retail purchases, and how is it priced
   (flat fee, percentage uplift, or per-asset)?
2. Does your answer differ between Posed / Rigged / Animated and 4D People? The 4D line appears
   to be captured by Volucap — would a separate agreement with them be required?
3. For the 4D People items: how many synchronized camera views are behind each sequence, and are
   the original per-view images with camera calibration available, or only the final mesh
   sequence? We ask because our task needs calibrated multi-view frames, not the reconstructed
   result.

We are also in parallel contact with your HumanDataset team regarding bulk multi-view data, so if
this request is better handled there, please feel free to redirect us.

Thank you,

[Name]
[Title], [Company]
[Email] | [Phone]
```

---

## 信三：Volucap（确认能否交付原始多视角帧）

**Subject:**
```
Technical inquiry — availability of raw multi-view frames with camera calibration
```

**Body:**

```
Dear Volucap Team,

We are [公司/团队名称] in Shenzhen, China, working on feed-forward 3D human reconstruction.
We understand that the 4D People assets sold by RenderPeople are captured in your studio, and we
are evaluating whether volumetric-capture data can serve as training data for our model.

Our requirement is unusual in one respect: we do not need the final mesh or volumetric video.
We need the **raw synchronized camera frames together with per-camera calibration** — that is,
for a given instant, N images plus the intrinsics and extrinsics of those N cameras.

Could you tell us:

1. How many synchronized cameras does your rig use, and at what resolution and frame rate?
2. Are the raw per-camera frames retained and deliverable, or is only the fused mesh sequence
   archived?
3. Is per-camera intrinsic and extrinsic calibration available and deliverable?
4. Is the studio lighting uniform enough to avoid baked-in illumination in the delivered imagery?
5. Commercial terms: is data licensing available for machine-learning training, including the
   right to publish trained model weights? Indicative pricing and lead time?

If a short paid test capture is possible, we would prefer to start there and evaluate the data
before discussing volume.

Best regards,
[Name]
[Title], [Company]
[Email] | [Phone]
```

---

## 附：国内 pilot 询价（中文，发数据堂 / 海天瑞声）

**主题：**
```
多视角人像采集 pilot 询价（1-2 人 × 10 姿态 × 16 视角，带标定）
```

**正文：**

```
您好，

我们在做三维人体/人像重建的算法研发，需要采购一批"同一时刻多视角拍摄 + 相机标定"的人像
数据用于模型训练。先做一个小规模 pilot 验证产线，通过后再放量。需求如下：

一、采集规格
1. 视角数：同一时刻 ≥16 台相机同步拍摄（最低 8 台）
2. 相机参数：逐视角内参（焦距、主点、畸变）+ 外参，按原始分辨率给出；提供标定文件或 COLMAP
3. 图像：原始拍摄帧（不要重建/修图后的结果），长边 ≥2K，JPG 或 RAW
4. 内容：同一姿态的同步多视角，不是视频里截的跨帧
5. 每人姿态数：≥10 个，含正面/侧面/背面
6. 光照：均匀白光；背景纯色并提供 alpha 或 mask
7. 交付格式：COLMAP 或 transforms.json

二、pilot 规模
2 名模特 × 10 个姿态 × 16 视角，共 320 张（如按人计价请按此折算）

三、请在报价中一并说明
1. 单价口径：按"人/姿态/视角"如何计价，是否含标定、是否含标注
2. 相机阵列规格：机型、数量、同步方式、标定精度（重投影误差）
3. 交付周期
4. 授权范围：数据能否用于训练机器学习模型，**训练所得模型权重能否对外发布/商用**；
   肖像权授权能否转授给我们的下游客户；授权是否不可撤回
5. 合规：是否提供个人信息保护影响评估、单独同意的记录

四、验收标准
按上述第 1-7 项逐条验收；标定精度以重投影误差 < 0.5 px 为准，不达标需补采。

如有现成的多视角人像样例（哪怕是旧项目的一小段），也请一并发来，我们可先做格式评估。

顺祝商祺，
[姓名]
[公司] [职位]
[手机] [邮箱]
```

---

## 收到回复后怎么判（红旗清单）

| 对方说法 | 判定 | 应对 |
|---|---|---|
| "research use only" | 🔴 **红线** | 直接终止，这份数据进不了产品 |
| "yes, training is fine" 但对权重不表态 | 🟡 追问 | 回信原文点出 "publish trained model weights"，要书面确认 |
| "we don't retain camera parameters" | 🔴 **红线** | 对前馈无用，只给 mesh 的数据等于没给 |
| "we only have the refined meshes for most scans" | 🟡 缩量 | 问清"有原图"的那部分到底多少个身份，按实际数砍价 |
| "permission may be revoked at our discretion" | 🟡 谈 | 要求改为 irrevocable，否则模型随时变侵权 |
| "SMPL license is separate" | 🟢 正常 | 正常，SMPL/SMPL-X 商业许可走 Meshcapade/Epic 另购 |
| 报价只给总价不给单价口径 | 🟡 追问 | 要"每人每姿态含多少视角、含不含标定"，否则无法横比 |

## 跟进节奏

```
Day 0   三封信同发（信一、信二、附表中文）
Day 3   未回 → 跟进一次，只问一句"any update on the licensing question in point 2?"
Day 7   仍未回 → humandataset 换 LinkedIn 找销售；国内两家直接电话
Day 7-14 并行：花 €500-1500 买 10 个 4D + 10 个 Posed，不等回信先解析格式
```

**为什么建议并行买小样**：三封信最快也要一两周才有实质回复，而"多视角原图是否真带逐视角内外参"这件事，**花 €800 买 20 个资产就能自己验证**。如果连零售资产都不带相机参数，B2B 那批大概率也不带——这笔小钱买的是决策依据。

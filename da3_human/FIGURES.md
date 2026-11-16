# da3_human 配图台账

> 版式规格：`../figkit/SPEC.md`（§5.6 单方法页 / §5.7 计划页）。
> **成品图与一次性出图脚本不入库**（AGENTS §14）；入库的只有本台账。

## 实验方案六页（2026-11-13，`q01`–`q06`）

把 `EXPERIMENT_PLAN.md` §0 / §0.0 的 R0～R4 排期与判据拆成六页，供汇报 / 答辩直接铺满整页。
位置：`D:/da3_human_figures/figures/`；脚本：`D:/da3_human_figures/_build/make_plan_pages.py`。

| 文件 | 这页的论点 | 内容来源（可回查） |
|---|---|---|
| `q01_plan_overview.png` | 主线转向 SplatAnything，R0→R4 五步；本轮只回答三个问题 | `EXPERIMENT_PLAN` §0 / §0.0 / §1.2 |
| `q02_da3_diagnosis.png` | 三个症状各有结论，没有一个指向「相机没给对」 | `EXPERIMENTS` 2026-11-11 D1/D2 + Phase 1.5 |
| `q03_eval_protocol.png` | 先修口径再排名：self 与 held-out 方向相反（+ 六项修正） | `ROUTE_VALIDATION_REPORT` §0 / §3.1 |
| `q04_r1_paired.png` | 固定三图，用配对门槛判定而不是「看着更好」 | `EXPERIMENT_PLAN` §0.0 R1、路线附录 §6 |
| `q05_r2_breaks.png` | 先拆成 N0–N4 五个可分辨假设，再动手改一处 | `EXPERIMENT_PLAN` §0.0 R2 / R2-N |
| `q06_r3_cost.png` | 先证明「能训」再谈域训练；T0→T2 差一次 100 步 | `EXPERIMENT_PLAN` §0.0 R3；报告 §5 |

再生成：

```bash
python D:/da3_human_figures/_build/make_plan_pages.py
FIG_ROOT=/path/to/dir python D:/da3_human_figures/_build/make_plan_pages.py   # 换输出根
```

## 口径纪律（这几页最容易出错的地方）

1. `q01` 的 15.00/17.76/18.39 dB 是 SA 单方法 held-out 水位（旧底色协议、11 个共同 target），
   **不是 SA 相对 DA3 的提升**；页脚已写明，不要在讲稿里说成「赢了」。
2. `q02` 三条归因只适用于 DA3，不能外推成「SplatAnything 必须重训几何」。
3. `q03` 的 self 22.37→19.71 与 held-out 15.00→18.39 目标像素集不同，只作方向对照，不可相减。
4. `q04` 的门槛是**预注册值**，不是实测；22 GiB 是建议上限。
5. `q05` 鼻部「变宽 / 重影」目前只有用户观察，没有逐视角定量证据，图内不得写成结论。
6. `q06` 的 T0/T1/T2 是可行性分级，T2 只说明小规模属性头微调可行。
7. `q01` 的「正式达标」行原写「缺 DA3 配对」，因 2026-11-13 的 E-12 / E-13 已给出首轮同帧横评，
   改为「缺独立 mask；首轮同帧横评口径未定」。E-13 收口后须复核该行与 `q04` 的现状描述。

## 历史配图

- `make_ppt_figures.py`（matplotlib 自制版式，1920×1080，未套 figkit 卡片/表格组件）；
  存量状态见 `figkit/SPEC.md` §9，暂不重排。

## 数据集规范五页（2026-11-16，`d01`–`d05`）

把 `24_ue_render_spec.md` V1.1（UE 人物多视角数据集构建规范）拆成五页汇报图。
位置：`D:/da3_human_figures/figures/`；脚本：`D:/da3_human_figures/_build/make_dataset_spec_pages.py`。

| 文件 | 这页的论点 | 内容来源（可回查） |
|---|---|---|
| `d01_dataset_overview.png` | 一次渲染三个环（A 13 张 / B 三摄 / C 125 张双角色）+ 目录即接口 + V1 边界 | spec §1 / §2 / §6 |
| `d02_camera_projection.png` | 三摄 6DoF 独立 + 竖幅 filmback 4/3 陷阱 + w2c = inv(D@M_ue@C) + 闭环 <0.5px/<1px | spec §2.3 / §3.2–§3.4 |
| `d03_eval_ring.png` | C 环 360° 顶视示意图：train_target 50（弧外监督）+ eval 75（held-out）+ 防泄漏反向规则 | spec §2.4 / §7 |
| `d04_acceptance.png` | 自动验收六格 + 人工拼图 + 深度一致性红线（中位 <2cm 且同报 P95）+ 输入组合 | spec §8 / §7 |
| `d05_release_roadmap.png` | 放行 8 条 + 数据量（282 张只够验证）+ 版本路线 V1.1→V3 | spec §12 / §9 |

再生成：

```bash
python D:/da3_human_figures/_build/make_dataset_spec_pages.py
```

口径纪律：

1. `d03` 的 az 序号 k ↔ 方位角 k×14.4°、0=正面；角色点集 {6,7,9,10,12,13,15,16,18,19} /
   {0~5,20~24} / {8,11,14,17} 与 §2.4 表逐一核对过（10+11+4 = 25 方位 × 5 俯仰 = 125 张）。
2. `d02` 的 fx 差 7.4× = 10923/1479，算法可复验；filmback 4/3 陷阱与 >1% 阈值是 spec 原文。
3. `d05` 的「282 张不足以支持泛化结论」是 spec §9 原文，汇报勿夸大。
4. 雅黑无 ✅/↔ 字形（SPEC §6），统一用 √ ·——本批出图时又踩了一次，已按规范修。

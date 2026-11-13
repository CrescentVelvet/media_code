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

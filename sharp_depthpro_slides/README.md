# sharp_depthpro_slides — 汇报用图（SHARP / Depth Pro 数据构建）

给领导汇报用的 6 张 16:9 图，内容是 Apple 两条前馈路线的**数据构建方式**对比：
SHARP 自建数据工厂 vs Depth Pro 拼装 19 个公开数据集。

所有数字来自 `../FEEDFORWARD_HUMAN_DATASETS.md` 的核实结果（Depth Pro 附录 Table 15/17 逐集核对、
SHARP 附录核实），图上未核实的标【待核实】、推断的标【推断】。

## 出图

```bash
# 需要 Edge（已内置在 Windows），不需要装 matplotlib
python 01_make_slides.py          # 出 slides/*.html + slides/*.png（3200x1800）
python 01_make_slides.py --html   # 只出 html，用于改版
```

## 六张图

| 文件 | 内容 | 建议讲法 |
|---|---|---|
| `01_overview.png` | 两条路线总览对比 | 开场：同样的目标，完全相反的建库策略 |
| `02_depthpro_corpus.png` | 19 个训练集按相机形态分三组 | 关键页：只有 3 个真给多视角，人体 0 个 |
| `03_depthpro_protocol.png` | Depth Pro 两阶段协议 | 能抄的部分：250ep 打底 → 100ep 提锐度 |
| `04_sharp_pipeline.png` | SHARP 数据流水线 + 合成/真实比例 | 为什么它 0.3 秒出片、为什么我们学不了 |
| `05_depthpro_table.png` | 19 集逐项清单（内容/体量/门槛/用途） | 答疑页：到底哪些能下、下了干什么 |
| `06_takeaways.png` | 三条动作：照抄 / 取数 / 不做 | 收尾：落到执行 |

## 改版注意

- 版式全在 `01_make_slides.py` 的 CSS 常量里；改配色只动顶部那几个色值。
- 每页用 flex 撑满 1600×900，**内容不足时不要居中，否则卡片中间会出现空洞**——
  优先补一条有信息量的内容（见第 2 页的「注意」「建议取用」块）。
- 截图靠 `--force-device-scale-factor=2` 拿 2 倍图；PPT 里按 16:9 铺满即可。

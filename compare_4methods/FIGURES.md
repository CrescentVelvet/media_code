# 出图台账（PPT 用）

> 图不纳入仓库（体积 + 一次性）。成品在 **`D:/ppt/figures/compare_4methods/figures/`**，
> 出图脚本在同目录 `_build/`（一次性脚本，不入库）。
> 版式规格见 [`figkit/SPEC.md`](../figkit/SPEC.md) §5.6（单方法页）/ §2（画布栅格）。

## 方法一页纸 p01–p05（2026-11-13）

1920×1080 满版，插入 PPT 铺满整页。每页固定五段：它怎么走（横向 5 步）→ 三个关键设计 →
关键数字（论文口径 / 本仓实测）→ 一句话结论 → 页脚口径。

| 文件 | 方法 | 这张图的论点 | 机制来源 | 数字来源 |
|---|---|---|---|---|
| `p01_da3gs.png` | DA3-GS | 冻结 backbone + 一个 GS-DPT 头，一次前向出高斯，全程没有一步在训练高斯 | `media_paper/3D-20251113-Depth-Anything-3.html` §5 / §11 | 论文表 2/3/5/8；本仓 FINDINGS §2 |
| `p02_querysplat.png` | QuerySplat | 几何 / 外观双查询解耦 + 冻结 VGM，因此能不要位姿；SOTA 只在 DL3DV 成立 | `media_paper/3D-20261105-QuerySplat.html` | 论文 Table 1 / Table 14；本仓 FINDINGS §2 |
| `p03_infinisplat_v2.png` | InfiniSplat v2 | 位置钉死只学属性；精度优势里含回炉，比数字先问"哪一档" | `media_paper/3D-20261014-InfiniSplat-v2.html`；[`infinisplat_v2/README_wsl.md`](../infinisplat_v2/README_wsl.md) | 论文表 1 / 表 9；本仓 FINDINGS §2 |
| `p04_splatanything.png` | SplatAnything（v3） | 换的不是画质，是"跨场景不用调参"（体素可固定 0.002） | `media_paper/3D-20261112-SplatAnything.html`；[`splatanything/NOTES.md`](../splatanything/NOTES.md) | 论文 Table 1 / Table 11；本仓 E-04 / E-06 / E-08 |
| `p05_vggt.png` | VGGT / VGGT-Ω | 前馈几何全能王但不出高斯；本仓当几何源（点云 init + 3DGS 精化），渲染质量是下游的事 | `media_paper/3D-20250314-VGGT.html` / `3D-20260514-VGGT-Omega.html` | 论文 Sintel 位姿 AUC；本仓 FINDINGS §2 / §8（35 s + 1189 s、175 万高斯、3 图 self3 9.30 dB） |

身份色：`C_VGGT = #207A58`（已在 `figlib.py` 与 SPEC §5 登记；Ω 的 Sintel 数字是 10B，
开源仅 1B——页脚已注明）。

## 失败模式对照 c01（2026-11-13）

FINDINGS §3 的「现象 / 已证实 / 仍是推断」三列归因表（`planpage.matrix`），
配复现正确性锚点（DTU 示例与论文逐位一致）。给"效果不佳的原因"一节用：
**「仍是推断」列不是结论，汇报里能当事实讲的只有「已证实」列。**

| 文件 | 内容 | 来源 |
|---|---|---|
| `c01_failure_modes.png` | 五方法失败模式对照 + 结论条 + 正确性锚点 | `FINDINGS.md` §3 / §2 |

## SA vs DA3 改进专页 s01（2026-11-16）

汇报第 2 篇的收尾页：四处改动（归一化空间 RDNorm→GSNorm / 几何教师解耦 / 高斯骨干可微调 /
AnySplat 式体素合并）+ 没改的部分（仍是前馈、GSNorm 依赖几何教师、归一化常数论文与代码不一致），
结论条把话头交给第 4 篇改进方案。**RDNorm 4.5× 漂移是 SA 论文 Fig.4 图注口径（对方自我报告），
不是本仓实测；本页不含跨方法 PSNR 横比。**

| 文件 | 内容 | 来源 |
|---|---|---|
| `s01_sa_vs_da3.png` | 四处改动 4 列对照表 + 「没改的部分」三条 + 结论条 | `media_paper/3D-20261112-SplatAnything.html`（GSNorm/架构/§5 失效条件）；`media_paper/3D-20251113-Depth-Anything-3.html`（DA3 三任务/GS-DPT 头）；DA3 高斯数 = `da3_human/EXPERIMENTS.md` E2 实测（190,512/层、32v 6.10M） |

## 三条容易讲错（已写进图内卡片或页脚）

1. **DA3 的三个提升口径互不一致**：摘要 35.7% / 23.6%，正文 §7.1 是 25.1% / 21.5%，
   abs 页还有 44.3%。图上写摘要口径并注明，引用时直接引表 2 / 表 3。
2. **v2 的分数含回炉**：前馈 33.9 s 的图与加 3000 步后差 +7.30 dB（自洽口径）。
   拿它的前馈数字去跟别家比，两边都会失真。
3. **v3 换的是"跨场景不用调参"**，不是画质：ETH3D LPIPS 它不在前二、PSNR 也非最优；
   与 v2 的场景 / 分辨率 / 体素都不同，只能比机制不能比分数。
4. **VGGT-Ω 的 Sintel 40.0 是 10B**：开源权重只有 1B；它不出高斯，本仓「VGGT-Ω+3DGS」
   含 30k 步 3DGS 精化，与纯前馈方法比耗时/质量都不可比。

## LGTM 四方法调研独立 PPT（2026-11-24）

- 成品：`D:/ppt/20261124-四方法LGTM高分辨率纹理改造调研.pptx`（16 页）。
- 可增量修改工程：`D:/ppt/_src/20261124-四方法LGTM高分辨率纹理改造调研/`，含 `DESIGN.md`、`slides/01–16.slide` 与论文图 assets；slide 均使用 `assets/...` 相对路径。
- 论文原图：`media_paper/images/2603.25745/architecture.png`、`lgtm_teaser.jpg`；只归属 LGTM 的静态场景与硬件口径。
- 其他视觉均为页内结构化 SVG，标明候选机制、非实验结果；不生成虚构的四方法纹理重建效果。
- 统一结论与源码行号见 `FINDINGS.md` §10；N2 训练继续冻结，本 PPT 不表示新训练已批准。

## 重新出图

```bash
cd D:/ppt/figures/compare_4methods/_build
python make_method_pages.py            # p01–p05，FIGKIT / FIG_ROOT 可用环境变量覆盖
python make_failure_page.py            # c01
python make_sa_vs_da3_page.py          # s01
```

依赖 pillow + numpy；模板与色板来自 `figkit/`（`import figlib` + `import onepager`），
中文字体走 `C:/Windows/Fonts/msyh*.ttc`。

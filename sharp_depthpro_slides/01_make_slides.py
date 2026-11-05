#!/usr/bin/env python3
"""生成 SHARP / Depth Pro 数据构建汇报图（1600x900 CSS，2x 输出 3200x1800）。

做法：写 HTML → 用 Edge 无头模式截图 → 出 PNG。
为什么不用 matplotlib：中文字重/字号/卡片布局在 CSS 里更可控，且不需要额外装包。

用法：
  python 01_make_slides.py            # 生成 html + png
  python 01_make_slides.py --html     # 只出 html，不截图

事实来源：FEEDFORWARD_HUMAN_DATASETS.md 第 5 节（Depth Pro 附录 Table 15/17 逐集核实）、
          SHARP arXiv 2512.10685 附录。数字均可回原文核对，未核实的标【待核实】，推断标【推断】。
"""
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "slides")
EDGE = r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"

INK = "#0F172A"
SUB = "#475569"
MUTED = "#8A97A8"
LINE = "#E4E8EF"
BG = "#F4F6FA"
SHARP = "#C2410C"
SHARP_BG = "#FFF4E8"
DP = "#1D4ED8"
DP_BG = "#EDF3FF"
OK = "#047857"
OK_BG = "#EAFBF2"
BAD = "#B91C1C"
BAD_BG = "#FEF1F1"
WARN = "#A9600A"
WARN_BG = "#FFF9EA"

CSS = f"""
* {{ box-sizing: border-box; }}
html, body {{ margin: 0; padding: 0; }}
body {{
  width: 1600px; height: 900px; overflow: hidden; background: {BG};
  font-family: "Microsoft YaHei", "Segoe UI", sans-serif;
  color: {INK}; -webkit-font-smoothing: antialiased;
}}
.slide {{ display: flex; flex-direction: column; width: 1600px; height: 900px;
          padding: 50px 64px 0 64px; }}
.head {{ flex: 0 0 auto; }}
.kicker {{ font-size: 15px; letter-spacing: .16em; color: {MUTED}; font-weight: 700; }}
h1 {{ font-size: 42px; line-height: 1.16; margin: 8px 0 10px 0; font-weight: 700; letter-spacing: -.012em; }}
.sub {{ font-size: 19px; color: {SUB}; line-height: 1.55; max-width: 1400px; }}
.rule {{ height: 3px; width: 76px; background: {INK}; margin: 16px 0 24px 0; border-radius: 2px; }}
.wrap {{ flex: 1 1 auto; display: flex; flex-direction: column; min-height: 0; padding-bottom: 56px; }}
.foot {{
  position: absolute; left: 64px; right: 64px; bottom: 20px;
  display: flex; justify-content: space-between; align-items: center;
  font-size: 13.5px; color: {MUTED}; border-top: 1px solid {LINE}; padding-top: 11px;
}}
.card {{ background: #fff; border: 1px solid {LINE}; border-radius: 16px; }}
.tag {{ display: inline-block; font-size: 13px; font-weight: 700; padding: 3px 10px; border-radius: 6px; }}
.num {{ font-variant-numeric: tabular-nums; font-weight: 700; }}
"""


def page(body: str, kicker: str, title: str, sub: str, source: str, idx: str) -> str:
    return f"""<!doctype html><html><head><meta charset="utf-8"><style>{CSS}</style></head><body>
<div class="slide">
  <div class="head">
    <div class="kicker">{kicker}</div>
    <h1>{title}</h1>
    <div class="sub">{sub}</div>
    <div class="rule"></div>
  </div>
  <div class="wrap">{body}</div>
  <div class="foot"><span>{source}</span><span>{idx}</span></div>
</div></body></html>"""


# ───────────────────────── 1 · 总览对比 ─────────────────────────
def s1() -> str:
    def row(k, v, first=False):
        bt = "" if first else f"border-top:1px solid {LINE};"
        return (f'<div style="display:flex;gap:16px;padding:14px 0;{bt}">'
                f'<div style="width:104px;flex:0 0 104px;font-size:15px;color:{MUTED};font-weight:700">{k}</div>'
                f'<div style="font-size:16.5px;line-height:1.6;color:{INK}">{v}</div></div>')

    sharp = f"""
    <div class="card" style="flex:1;display:flex;flex-direction:column;padding:26px 30px;
         border-top:4px solid {SHARP}">
      <div style="display:flex;align-items:center;gap:12px;margin-bottom:4px">
        <div style="font-size:27px;font-weight:700;color:{SHARP}">SHARP</div>
        <div class="tag" style="background:{SHARP_BG};color:{SHARP}">Apple · arXiv 2512.10685</div>
      </div>
      <div style="font-size:15.5px;color:{SUB}">自己「造」数据：先搭渲染工厂，再补真实照片</div>
      <div style="flex:1;display:flex;flex-direction:column;justify-content:space-evenly;margin-top:6px">
        {row("数据来源", "自建程序化场景 <span style='color:%s'>＋</span> 商业授权图库，两条腿并行" % MUTED, True)}
        {row("Stage 1 合成", "<b class='num'>700K</b> 场景 × <b class='num'>11</b> 视角 ≈ <b class='num'>8M</b> 张图，V-Ray 自建，1536² / 2048²")}
        {row("Stage 2 真实", "<b class='num'>2.65M</b> 张：OpenScene + Shutterstock / Getty / Flickr（付费授权）")}
        {row("开源情况", "官方仓库<b>只开源推理</b>，训练流程与训练数据均未发布")}
        {row("可复现性", "<b style='color:%s'>无法复现</b> —— 私有 700K 场景 ＋ 需商业图库授权" % BAD)}
      </div>
    </div>"""

    dp = f"""
    <div class="card" style="flex:1;display:flex;flex-direction:column;padding:26px 30px;
         border-top:4px solid {DP}">
      <div style="display:flex;align-items:center;gap:12px;margin-bottom:4px">
        <div style="font-size:27px;font-weight:700;color:{DP}">Depth Pro</div>
        <div class="tag" style="background:{DP_BG};color:{DP}">Apple · ICLR 2025 · arXiv 2410.02073</div>
      </div>
      <div style="font-size:15.5px;color:{SUB}">「拼装」公开数据：19 个现成数据集凑成一锅</div>
      <div style="flex:1;display:flex;flex-direction:column;justify-content:space-evenly;margin-top:6px">
        {row("数据来源", "全部为公开数据集，清单在论文附录 Table 15（URL / 许可 / 用途）与 Table 17（损失 × 数据集）", True)}
        {row("规模", "<b class='num'>19</b> 个训练集，全量约 <b class='num'>10–12 TB</b>")}
        {row("公开可下", "<b class='num' style='color:%s'>18 / 19</b>（仅 SmartPortraits 已失效）" % OK)}
        {row("开源情况", "推理代码 <b>＋</b> 预训练权重均开源")}
        {row("可复现性", "<b style='color:%s'>可复现</b> —— 但多半要签协议，剩下的是纯工程量" % OK)}
      </div>
    </div>"""

    band = f"""
    <div style="margin-top:22px;background:{INK};color:#fff;border-radius:14px;padding:20px 28px;
                display:flex;align-items:center;gap:24px;flex:0 0 auto">
      <div style="font-size:13px;letter-spacing:.14em;color:#94A3B8;font-weight:700;flex:0 0 auto">一句结论</div>
      <div style="font-size:18px;line-height:1.55">
        SHARP 的壁垒是<b>数据工厂</b>（8M 张自渲染 ＋ 2.65M 张付费图），不是模型结构；
        Depth Pro 的壁垒已经不存在，剩下的只是<b>10 TB 的下载与清洗工程量</b>。
      </div>
    </div>"""

    return page(f'<div style="flex:1;display:flex;gap:24px;min-height:0">{sharp}{dp}</div>{band}',
                "APPLE 的两条前馈路线", "同样的目标，完全相反的建库策略",
                "两家都是 Apple、都想做<b>无需相机位姿</b>的前馈重建，但一个自建数据工厂、一个拼装公开数据。"
                "这个差别直接决定了哪条路我们能走。",
                "来源：SHARP arXiv 2512.10685 附录；Depth Pro arXiv 2410.02073 v2 附录 Table 15 / 17", "1 / 6")


# ───────────────────────── 2 · 19 集按相机形态 ─────────────────────────
def s2() -> str:
    def col(title, count, color, bg, items, scale, note, pick, warn):
        chips = "".join(
            f'<span style="display:inline-block;font-size:15.5px;padding:8px 13px;margin:0 7px 9px 0;'
            f'border-radius:8px;background:{bg};color:{color};font-weight:600">{n}</span>'
            for n in items)
        return f"""
        <div style="flex:1;display:flex;flex-direction:column;justify-content:space-between;padding:22px 20px 18px 20px;
                    background:#fff;border:1px solid {LINE};border-radius:16px;border-top:4px solid {color}">
          <div style="display:flex;align-items:baseline;gap:9px;margin-bottom:14px">
            <div style="font-size:20px;font-weight:700;color:{color}">{title}</div>
            <div style="font-size:15px;color:{MUTED};font-weight:700">{count} 个</div>
          </div>
          <div style="align-content:flex-start">{chips}</div>
          <div style="background:#FAFBFD;border:1px solid {LINE};border-radius:9px;
                      padding:10px 12px;font-size:13.5px;color:{SUB};line-height:1.55">
            <b style="color:{WARN}">注意</b>　{warn}
          </div>
          <div style="background:{bg};border-radius:9px;padding:11px 13px;
                      font-size:14px;color:{SUB};line-height:1.6">
            <b style="color:{color}">建议取用</b>　{pick}
          </div>
          <div style="font-size:13.5px;color:{MUTED};line-height:1.5">{scale}</div>
          <div style="border-top:1px dashed {LINE};padding-top:11px;font-size:14.5px;color:{SUB};line-height:1.55">{note}</div>
        </div>"""

    mono = col("单目 / 单机位", "10", DP, DP_BG,
               ["Hypersim", "Synscapes", "UrbanSyn", "Virtual KITTI 2", "BEDLAM", "SAIL-VOS 3D",
                "ARKitScenes", "ScanNet v2", "3D Ken Burns", "SmartPortraits ✕"],
               "体量最大：Hypersim 1.9 TB、ARKitScenes ~1.5 TB、ScanNet v2 1.3 TB",
               "只能提供<b>深度监督</b>，学不到视角之间的一致性。人体只有 BEDLAM 一个，且是单目合成。",
               "Hypersim 子集 ＋ BEDLAM（6fps 档）—— 这组只有这两个值得动",
               "Hypersim 是 CC BY-SA 3.0（<b>传染性</b>），Synscapes <b>禁止再分发</b>")
    stereo = col("立体对（2 视点）", "6", WARN, WARN_BG,
                 ["TartanAir", "Dynamic Replica", "IRS", "DIML Indoor", "UnrealStereo4K", "HRWSI"],
                 "TartanAir ~3 TB、UnrealStereo4K 570 GB、HRWSI 7.6 GB",
                 "两点视差够算深度、练边缘锐度，但<b>做不了环绕约束</b>；Dynamic Replica 为 2–3 视角。",
                 "TartanAir（唯一可商用 ＋ 逐帧位姿）＋ UnrealStereo4K（边缘锐度）",
                 "UnrealStereo4K 单场景 67–76 GB，先下 HF 上的 <b>1/4 分辨率版</b>试")
    multi = col("真多视角（≥3 视点）", "3", OK, OK_BG,
                ["BlendedMVS", "MVS-Synth", "EDEN"],
                "BlendedMVS 113 场景、MVS-Synth 120 序列、EDEN 38 GB–1.4 TB",
                "唯一能提供<b>多视角一致性</b>监督的一类。代价：全是物体 / 场景，<b>没有一个人体</b>。",
                "三个全取（低清档合计约 75 GB），先用 MVS-Synth 打通 dataloader",
                "MVS-Synth 的 JSON 里 <b>R 行列式为 −1</b>，直接当旋转矩阵用会镜像")

    band = f"""
    <div style="margin-top:20px;display:flex;gap:16px;flex:0 0 auto">
      <div style="flex:1;background:{BAD_BG};border:1px solid #F9C9C9;border-radius:12px;padding:16px 22px">
        <div style="font-size:14.5px;font-weight:700;color:{BAD};margin-bottom:5px">⚠️ 对人像前馈直接可用的：0 个</div>
        <div style="font-size:15px;color:{SUB};line-height:1.55">
          人像需要「同一时刻 ≥4 视角 ＋ 标定内外参」。清单里唯一的人体集 BEDLAM 是单目合成，
          多视角的三个全是物体场景 —— 两者<b>不重叠</b>。
        </div>
      </div>
      <div style="flex:1;background:{OK_BG};border:1px solid #B7EED1;border-radius:12px;padding:16px 22px">
        <div style="font-size:14.5px;font-weight:700;color:{OK};margin-bottom:5px">✅ 它真正的用途</div>
        <div style="font-size:15px;color:{SUB};line-height:1.55">
          给 backbone 加<b>深度 / 法线辅助头</b>，用 ARKitScenes / ScanNet 补<b>室内背景分布</b>；
          当主训练集用会白费 10 TB 下载。
        </div>
      </div>
    </div>"""

    return page(f'<div style="flex:1;display:flex;gap:16px;min-height:0">{mono}{stereo}{multi}</div>{band}',
                "DEPTH PRO · 附录 TABLE 15", "19 个训练集，只有 3 个真给多视角",
                "把 19 个集按<b>相机形态</b>重排，问题一眼就看出来：能提供多视角一致性监督的只有 3 个，"
                "能提供人体的是 0 个 —— 而前馈人像重建恰恰同时需要这两件事。",
                "来源：Depth Pro 附录 Table 15（19 集逐项核实，形态依据各数据集官方文档）；SmartPortraits 仓库已 archive、下载链接为 TODO", "2 / 6")


# ───────────────────────── 3 · Depth Pro 训练协议 ─────────────────────────
def s3() -> str:
    def stage(no, tag_txt, ep, lines, color):
        lis = "".join(f'<div style="font-size:16px;color:{SUB};line-height:1.75">{x}</div>' for x in lines)
        return f"""
        <div style="flex:1;display:flex;flex-direction:column;background:#fff;border:1px solid {LINE};
                    border-left:4px solid {color};border-radius:14px;padding:24px 26px 22px 26px">
          <div style="font-size:13px;letter-spacing:.12em;color:{MUTED};font-weight:700">{no} · {tag_txt}</div>
          <div style="font-size:34px;font-weight:700;margin:2px 0 14px 0;color:{color}"><span class="num">{ep}</span>
            <span style="font-size:19px;color:{SUB};font-weight:600"> epoch</span></div>
          <div style="flex:1;display:flex;flex-direction:column;justify-content:center">{lis}</div>
        </div>"""

    stages = f"""
    <div style="display:flex;align-items:stretch;flex:0 0 auto">
      {stage("STAGE 1", "打底", "250", [
          "全部 <b>19 个标注集联合训练</b>",
          "损失：尺度不变 MAE（SSI-MAGE）",
          "目标：先把<b>度量尺度</b>学对"], DP)}
      <div style="flex:0 0 56px;display:flex;align-items:center;justify-content:center;font-size:28px;color:{MUTED}">→</div>
      {stage("STAGE 2", "提锐度", "100", [
          "在 Stage 1 权重上继续，同一批数据",
          "损失：<b>追加一阶 / 二阶导数损失</b>",
          "目标：把边缘从「糊」拉回「锐」"], SHARP)}
    </div>"""

    branch = f"""
    <div style="margin-top:14px;background:{WARN_BG};border:1px dashed #F5D48A;border-radius:12px;
                padding:13px 22px;display:flex;align-items:center;gap:16px;flex:0 0 auto">
      <div style="font-size:14.5px;font-weight:700;color:{WARN};flex:0 0 auto">旁支 · 焦距头</div>
      <div style="font-size:15.5px;color:{SUB};line-height:1.5">
        <b>单独训练</b>，不参与上面两阶段。论文只说用了「多个数据集 ＋ 仅焦距监督的大规模图像集」，
        <b>未逐项列明</b>是哪些集 —— 复现这一步得自己配。
      </div>
    </div>"""

    def k(t, b, c):
        return f"""<div style="flex:1;display:flex;flex-direction:column;background:#fff;
          border:1px solid {LINE};border-radius:12px;padding:20px 20px">
          <div style="font-size:16px;font-weight:700;margin-bottom:8px;color:{c}">{t}</div>
          <div style="font-size:15px;color:{SUB};line-height:1.65">{b}</div></div>"""

    cards = f"""
    <div style="flex:1;display:flex;gap:14px;margin-top:16px;min-height:0">
      {k("为什么 19 个源能拼在一起", "损失是<b>尺度不变</b>形式，度量与非度量数据可以混训；"
         f"代价是每集仍需单独做尺度归一。<span style='color:{WARN}'>【推断】</span><br>"
         f"<span style='color:{MUTED}'>风险：混训会偏向数据量最大的那几集，采样权重得手工调。</span>", DP)}
      {k("值得直接抄的两点", "① <b>两阶段</b>：先学全局度量，再用高阶导数损失专门拉边缘；"
         "② <b>辅助头解耦</b>（焦距头独立训练），互不干扰。<br>"
         f"<span style='color:{MUTED}'>这两点与数据集无关，换任何 backbone 都成立。</span>", OK)}
      {k("不能直接搬的一点", "监督目标是<b>单张图的像素深度</b>，没有多视角几何一致性约束 —— "
         f"论文里也没有 NVS / 3DGS 实验。<br><span style='color:{MUTED}'>要用它做多视角，"
         "必须自己加跨视角损失与位姿监督。</span>", BAD)}
    </div>

    <div style="margin-top:14px;background:#fff;border:1px solid {LINE};
                border-left:4px solid {INK};border-radius:0 12px 12px 0;padding:16px 22px;flex:0 0 auto">
      <div style="font-size:15.5px;color:{SUB};line-height:1.65">
        <b style="color:{INK}">映射到我们的 pipeline（建议）</b>　　Stage 1 用合成 / 伪 GT 数据把尺度与结构打底
        → Stage 2 换真实人体序列、用导数损失提锐度。投影到人像就是：<b>先解决「几何对不对」，再解决「脸糊不糊」</b>。
      </div>
    </div>"""

    return page(stages + branch + cards,
                "DEPTH PRO · 训练协议", "两阶段：先学度量，再学锐度",
                "这套协议的价值不在数据，在<b>怎么用异构数据</b>：19 个来源、尺度定义各不相同的集能被联合训练，"
                "靠的是损失设计与阶段切分。这部分是能抄的。",
                "来源：Depth Pro 论文 Stage 1 / Stage 2 定义与附录 Table 17（损失 × 数据集）；焦距头训练集未列明", "3 / 6")


# ───────────────────────── 4 · SHARP 流水线 ─────────────────────────
def s4() -> str:
    def box(tagtxt, tagnote, tagbg, tagfg, color, bg, items, flex="1"):
        lis = "".join(f'<div style="font-size:15.5px;color:{SUB};line-height:1.75">{x}</div>' for x in items)
        return f"""
        <div style="flex:{flex};display:flex;flex-direction:column;background:{bg};
                    border:1px solid {LINE};border-radius:14px;padding:22px 24px 20px 24px">
          <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:12px">
            <div style="font-size:13px;letter-spacing:.12em;color:{color};font-weight:700">{tagtxt}</div>
            <span class="tag" style="background:{tagbg};color:{tagfg}">{tagnote}</span>
          </div>
          {lis}
        </div>"""

    flow = f"""
    <div style="display:flex;align-items:stretch;flex:0 0 auto">
      {box("STAGE 1 · 自建合成", "未发布", "#FDE4E4", BAD, SHARP, SHARP_BG, [
          "自研<b>程序化场景生成系统</b>：<b class='num'>2K+</b> 户外程序化地形 ＋ <b class='num'>5K+</b> 室内艺术家手工场景",
          "V-Ray 渲染，<b>每个场景 11 个视角</b>，分辨率 1536² 或 2048²",
          "合计 <b class='num'>≈700K</b> 场景 / <b class='num'>≈8M</b> 张图"], flex="1.18")}
      <div style="flex:0 0 52px;display:flex;align-items:center;justify-content:center;font-size:26px;color:{MUTED}">→</div>
      {box("STAGE 2 · 真实照片微调", "需付费授权", "#FDECC8", WARN, DP, DP_BG, [
          "OpenScene ＋ Shutterstock / Getty / Flickr，合计 <b class='num'>2.65M</b> 张",
          "<b>自监督 warp-back</b>：模型先预测 → 渲染伪新视角 → 回原图算损失",
          "→ <b>不需要深度 GT</b>，这是它能吃海量无标注图的根本原因"], flex="1.14")}
    </div>"""

    ev = f"""
    <div style="margin-top:14px;background:#fff;border:1px solid {LINE};border-radius:14px;
                padding:16px 24px;flex:0 0 auto">
      <div style="display:flex;align-items:center;gap:12px;margin-bottom:6px">
        <div style="font-size:13px;letter-spacing:.12em;color:{OK};font-weight:700">评估集 · 唯一公开可复现的部分</div>
        <span class="tag" style="background:{OK_BG};color:{OK}">全部公开</span>
      </div>
      <div style="font-size:15.5px;color:{SUB};line-height:1.6">
        Middlebury · Booster · ScanNet++ · WildRGBD · ETH3D · Tanks and Temples ——
        每场景取 <b>10 个视角</b>，每集不超过 <b>512 对</b>；另用 Unsplash 做定性展示。
      </div>
    </div>"""

    ratio = f"""
    <div style="margin-top:14px;background:#fff;border:1px solid {LINE};border-radius:12px;
                padding:14px 24px 16px 24px;flex:0 0 auto">
      <div style="display:flex;justify-content:space-between;align-items:baseline;margin-bottom:9px">
        <div style="font-size:13px;letter-spacing:.1em;color:{MUTED};font-weight:700">训练数据构成（按张数，合计约 10.6M）</div>
        <div style="font-size:13.5px;color:{SUB}">合成占四分之三 —— 它的主力是「造」出来的</div>
      </div>
      <div style="display:flex;height:30px;border-radius:7px;overflow:hidden;font-size:13.5px;font-weight:700;color:#fff">
        <div style="flex:75.1;background:{SHARP};display:flex;align-items:center;padding-left:14px">
          Stage 1 · 合成　≈8M（75%）</div>
        <div style="flex:24.9;background:{DP};display:flex;align-items:center;padding-left:14px">
          Stage 2 · 真实 2.65M（25%）</div>
      </div>
    </div>"""

    call = f"""
    <div style="flex:1;display:flex;gap:16px;margin-top:14px;min-height:0">
      <div style="flex:1;display:flex;flex-direction:column;justify-content:center;background:{OK_BG};
                  border:1px solid #B7EED1;border-radius:12px;padding:18px 24px">
        <div style="font-size:16px;font-weight:700;color:{OK};margin-bottom:8px">能学的：warp-back 自监督</div>
        <div style="font-size:15.5px;color:{SUB};line-height:1.7">
          把「有深度的合成数据」换成「大量无标注真实图」—— 这是我们手上人体序列（无 mesh GT）
          最现实的放大手段，且不需要额外采购。
        </div>
        <div style="font-size:15px;color:{SUB};line-height:1.7;margin-top:9px;padding-top:9px;
                    border-top:1px dashed #B7EED1">
          落到执行：给定同一帧的 N 个视角，先渲染伪新视角做监督 —— 既不依赖 GT 深度，
          也不要求 COLMAP 在每一帧都收敛。
        </div>
      </div>
      <div style="flex:1;display:flex;flex-direction:column;justify-content:center;background:{BAD_BG};
                  border:1px solid #F9C9C9;border-radius:12px;padding:18px 24px">
        <div style="font-size:16px;font-weight:700;color:{BAD};margin-bottom:8px">学不了的：700K 场景的工厂</div>
        <div style="font-size:15.5px;color:{SUB};line-height:1.7">
          8M 张图 = V-Ray 渲染数月 ＋ 艺术家建模，不在人力预算内；2.65M 张的商业图库授权也拿不到。
          这两条决定我们只能「改造现有数据」，不能「新建数据工厂」。
        </div>
        <div style="font-size:15px;color:{SUB};line-height:1.7;margin-top:9px;padding-top:9px;
                    border-top:1px dashed #F9C9C9">
          量级对照：8M 张 ≈ 700K 场景 × 11 视角；单场景离线渲染按分钟计，总机时是万小时级
          <span style="color:{WARN}">【推断】</span>。
        </div>
      </div>
    </div>"""

    return page(flow + ev + ratio + call,
                "SHARP · arXiv 2512.10685", "SHARP 的数据是「造」出来的，不是「找」出来的",
                "它没有去拼公开数据集，而是搭了一条渲染流水线自己生产 —— 这正是它亚秒级出片的前提，"
                "也正是我们无法复制的部分。",
                "来源：SHARP 论文附录（Stage 1 场景构成与渲染设置、Stage 2 数据来源与 warp-back 描述、评估集协议）", "4 / 6")


# ───────────────────────── 5 · 19 集清单表 ─────────────────────────
ROWS = [
    ("Hypersim", "单目", "366 场景 / 77.4K 图，室内，深度 + 法线", "1.9 TB", "脚本直下", "深度 / 法线辅助头", OK),
    ("TartanAir", "立体", "30+ 环境 / 100 万+ 图，逐帧位姿", "~3 TB", "直下", "几何先验（可商用）", OK),
    ("Synscapes", "单目", "25,000 张街景，语义 + 深度", "337 GB", "直下", "街景深度（禁再分发）", DP),
    ("UrbanSyn", "单目", "7,534 帧街景，深度 EXR", "10–40 GB", "直下", "可商用深度", OK),
    ("Dynamic Replica", "立体 2–3", "1,451 段合成动态，带 SMPL", "1.8 TB", "签协议", "动态一致性", WARN),
    ("BEDLAM", "单目", "10,450 段人体，SMPL-X + 深度", "数 TB", "MPI 注册", "唯一的「人体 + 深度」", SHARP),
    ("IRS", "立体", "10 万室内立体样本", "【待核实】", "OneDrive", "室内视差", DP),
    ("Virtual KITTI 2", "单目", "5 序列 / 21,260 帧街景", "15 GB", "直下", "轻量深度", DP),
    ("SAIL-VOS 3D", "单目", "237,611 帧 GTA-V，SMPL + 深度", "【待核实】", "邮件 + 自购游戏", "人体深度（门槛高）", MUTED),
    ("ARKitScenes", "单目 RGB-D", "1,661 室内场景，iPhone 采集", "~1.5 TB", "同意许可", "室内背景", DP),
    ("DIML Indoor", "双目", "1,500 帧室内 RGB-D", "2.8 GB", "链接可疑", "低优先", MUTED),
    ("ScanNet v2", "单目 RGB-D", "707 场景 / 2.5M 视图", "1.3 TB", "签协议", "室内背景", DP),
    ("SmartPortraits", "单目", "真实人像 + 逐像素深度", "—", "已失效", "本可最有用 ✕", BAD),
    ("UnrealStereo4K", "立体", "8 场景 4K 合成立体", "570 GB", "直下", "边缘锐度", DP),
    ("3D Ken Burns", "单目 + 视差", "合成深度，分场景", "~200 GB", "直下", "视差深度", DP),
    ("EDEN", "多视角", "合成花园场景，位姿精确", "38 GB–1.4 TB", "直下", "多视角（合成物体）", OK),
    ("MVS-Synth", "多视角", "120 序列 × 100 帧", "34 GB", "直下", "打通 dataloader", OK),
    ("HRWSI", "立体", "21K 真实立体对", "7.6 GB", "直下", "真实视差", DP),
    ("BlendedMVS", "多视角", "113 场景 / ~17K 图", "27.5 GB", "直下", "可商用 MVS", OK),
]


def s5() -> str:
    body_rows = ""
    for i, (name, form, content, size, get, use, color) in enumerate(ROWS, 1):
        bg = "#FBFCFE" if i % 2 else "#fff"
        body_rows += f"""
        <tr style="background:{bg}">
          <td style="color:{MUTED};font-size:13px;padding:0 8px">{i}</td>
          <td style="font-weight:700;font-size:14.5px;color:{color};padding:0 8px">{name}</td>
          <td style="font-size:14px;color:{SUB};padding:0 8px">{form}</td>
          <td style="font-size:14px;color:{SUB};padding:0 8px">{content}</td>
          <td style="font-size:14px;color:{INK};font-weight:600;padding:0 8px">{size}</td>
          <td style="font-size:13.5px;color:{SUB};padding:0 8px">{get}</td>
          <td style="font-size:14px;color:{SUB};padding:0 8px">{use}</td>
        </tr>"""

    table = f"""
    <table style="width:100%;height:100%;border-collapse:collapse;background:#fff;border:1px solid {LINE};
                  border-radius:12px;overflow:hidden">
      <thead>
        <tr style="background:{INK};color:#fff;font-size:13px;letter-spacing:.05em">
          <th style="padding:11px 8px;text-align:left;width:34px">#</th>
          <th style="padding:11px 8px;text-align:left;width:196px">数据集</th>
          <th style="padding:11px 8px;text-align:left;width:112px">相机形态</th>
          <th style="padding:11px 8px;text-align:left">内容</th>
          <th style="padding:11px 8px;text-align:left;width:136px">体量</th>
          <th style="padding:11px 8px;text-align:left;width:136px">获取</th>
          <th style="padding:11px 8px;text-align:left;width:196px">对我们的用途</th>
        </tr>
      </thead>
      <tbody>{body_rows}</tbody>
    </table>"""

    return page(table,
                "DEPTH PRO · 19 集逐项清单", "逐集清单：内容、体量、门槛、用途",
                "这张表是上一页的展开，用于回答「到底哪些能下、下了干什么」。<b>颜色 = 建议动作</b>："
                f"<span style='color:{OK};font-weight:700'>绿 = 优先取</span>　"
                f"<span style='color:{DP};font-weight:700'>蓝 = 按需取</span>　"
                f"<span style='color:{SHARP};font-weight:700'>橙 = 唯一人体但有短板</span>　"
                f"<span style='color:{BAD};font-weight:700'>红 = 拿不到</span>",
                "来源：各数据集官方页面逐项核实（2026-11）；【待核实】为官方未公布具体体量", "5 / 6")


# ───────────────────────── 6 · 结论与建议 ─────────────────────────
def s6() -> str:
    def card(title, tagtxt, color, bg, items, tail):
        lis = "".join(f'<div style="display:flex;gap:10px;margin-bottom:12px;font-size:15.5px;color:{SUB};line-height:1.65">'
                      f'<span style="color:{color};font-weight:700;flex:0 0 auto">·</span><div>{x}</div></div>'
                      for x in items)
        return f"""
        <div style="flex:1;display:flex;flex-direction:column;background:#fff;border:1px solid {LINE};
                    border-radius:16px;padding:24px 22px 20px 22px;border-top:4px solid {color}">
          <div style="display:flex;align-items:center;gap:10px;margin-bottom:16px">
            <div style="font-size:20px;font-weight:700">{title}</div>
            <span class="tag" style="background:{bg};color:{color}">{tagtxt}</span>
          </div>
          <div style="flex:1">{lis}</div>
          <div style="padding-top:13px;border-top:1px dashed {LINE};font-size:14px;color:{MUTED};line-height:1.55">{tail}</div>
        </div>"""

    c1 = card("照抄：训练协议", "零成本", OK, OK_BG, [
        "两阶段：全局度量 → 高阶导数损失提边缘",
        "<b>warp-back 自监督</b>：不需要深度 GT，直接吃无标注序列",
        "尺度不变损失 ＋ 辅助头解耦（焦距 / 深度 / 法线各管一块）",
        "训练顺序：先合成后真实 —— 反过来容易被真实数据的尺度带偏<span style='color:%s'>【推断】</span>" % WARN,
    ], "这两条对「无 GT 的人体序列」尤其关键 —— 正是我们手头数据的短板。")

    c2 = card("已有 / 可采购：数据", "工程量", DP, DP_BG, [
        "Depth Pro 清单 <b>18 / 19 可下</b>，但真正值得取的不到 1/3",
        "人体主训练集走 S 级：<b>HuMMan 724 GB 同事已下</b>、SynBody 379 GB",
        "跑通三件套：THUman4.0 ＋ AvatarReX ＋ <b>PKU-DyMVHumans（格式现成）</b>",
        "要中国面孔就走定制采集：数据堂 / 海天瑞声，4–12 周，先交 pilot 验收",
    ], "Depth Pro 那 19 集当 backbone 的深度 / 法线辅助头，不当人像主训练集。")

    c3 = card("别碰：拿不到的部分", "省时间", BAD, BAD_BG, [
        "SHARP 的 <b>700K 私有场景 / 8M 张 V-Ray 渲染图</b>",
        "SHARP Stage 2 的 <b>2.65M 张图库授权</b>（Shutterstock / Getty）",
        "SmartPortraits：真实人像 ＋ 深度，仓库已 archive、链接为 TODO",
        "3D 商店资产（Sketchfab / Fab / RenderHub）也不覆盖 AI 训练，别当替代品",
    ], "这三项投入产出比最低，明确写进「不做清单」，避免反复评估。")

    band = f"""
    <div style="margin-top:22px;background:{INK};color:#fff;border-radius:14px;padding:22px 28px;
                display:flex;gap:24px;align-items:center;flex:0 0 auto">
      <div style="font-size:13px;letter-spacing:.12em;color:#94A3B8;font-weight:700;flex:0 0 92px;line-height:1.5">结论<br>与建议</div>
      <div style="font-size:17.5px;line-height:1.6">
        我们抄不了 SHARP 的<b>数据规模</b>，但可以抄它的<b>数据构造方法</b>：用 CC0 资产（Poly Haven / ambientCG）
        给已有的人体序列<b>重新打光、换背景</b>，再用 BlenderProc 批量渲染多视角 ——
        这条路的成本比自建 8M 张 V-Ray 工厂低几个数量级，且授权干净。
      </div>
    </div>"""

    return page(f'<div style="flex:1;display:flex;gap:16px;min-height:0">{c1}{c2}{c3}</div>{band}',
                "对我们的启示", "三条动作：照抄协议、按需取数、明确不做",
                "把上面四页落到执行：能抄的是<b>训练协议</b>，能买的是<b>部分公开数据</b>，"
                "拿不到的要提前划定<b>不做清单</b>，避免在这上面反复投入评估成本。",
                "配套文档：FEEDFORWARD_HUMAN_DATASETS.md（44 项分级台账）、PORTRAIT_DATA_SOURCING.md（采购路径）", "6 / 6")


SLIDES = [
    ("01_overview", s1),
    ("02_depthpro_corpus", s2),
    ("03_depthpro_protocol", s3),
    ("04_sharp_pipeline", s4),
    ("05_depthpro_table", s5),
    ("06_takeaways", s6),
]


def main() -> None:
    os.makedirs(OUT, exist_ok=True)
    only_html = "--html" in sys.argv
    for name, fn in SLIDES:
        html_path = os.path.join(OUT, name + ".html")
        # newline="\n"：Windows 默认会翻 CRLF，而仓库约定 LF
        with open(html_path, "w", encoding="utf-8", newline="\n") as f:
            f.write(fn())
        if only_html:
            print(f"📄 {html_path}")
            continue
        png_path = os.path.join(OUT, name + ".png")
        cmd = [EDGE, "--headless=new", "--disable-gpu", "--hide-scrollbars", "--no-first-run",
               "--force-device-scale-factor=2", "--window-size=1600,900",
               "--user-data-dir=" + os.path.join(OUT, ".edgeprof"),
               f"--screenshot={png_path}", "file:///" + html_path.replace("\\", "/")]
        r = subprocess.run(cmd, capture_output=True, text=True)
        ok = os.path.exists(png_path)
        print(("✅ " if ok else "❌ ") + png_path + ("" if ok else f"\n{r.stderr[-400:]}"))
    print("🎉 done")


if __name__ == "__main__":
    main()

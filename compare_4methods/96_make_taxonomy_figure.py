#!/usr/bin/env python3
"""生成「三条技术路线」算法思路图（PPT 用，1600×900 @2x）。

为什么用 HTML：这台机器上没有可用的中文排版方案能像 CSS 一样排版卡片，
仓里已有先例（sharp_depthpro_slides 用 Edge 无头截图）。零额外依赖。

产出：
  compare_4methods/ppt_assets/fig7_taxonomy.html   （源，可手改）
  compare_4methods/ppt_assets/fig7_taxonomy.png    （3200×1800）

用法（Windows 侧，Git Bash / PowerShell 均可）：
  python compare_4methods/96_make_taxonomy_figure.py
"""
from pathlib import Path
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "ppt_assets"
OUT.mkdir(exist_ok=True)
EDGE = r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"

# ---------------- 内容（数字与 91_make_ppt_assets.py 同源） ----------------
CARDS = [
    dict(
        route="路线 A",
        name="前馈直出高斯",
        en="Feed-forward Gaussian Head",
        color="#2E8B7F",
        reps=["DA3 原生高斯头", "QuerySplat"],
        mech="一次前向，逐像素或逐查询解码 位置 / 尺度 / 旋转 / 不透明度 / SH，直接写出 PLY",
        time="13.5 s（DA3）　38.5 s（QuerySplat）",
        size="610 万高斯 · 314 MB　｜　37 万 · 33 MB",
        issue="视角之间质量不稳（视角 8 拖影、视角 16 人物消失）；细节被 504 px 输入与查询预算卡住",
        verdict="速度唯一达标，保真度不达标",
    ),
    dict(
        route="路线 B",
        name="前馈 + 短精调",
        en="Feed-forward then short refine",
        color="#5B7FB5",
        reps=["InfiniSplat v2"],
        mech="前馈定几何 → 3000 步只调 SH / scale / opacity / quat，位置完全冻结",
        time="34 s + 163 s ≈ 3.3 min",
        size="594 万高斯 · 386 MB",
        issue="重影：同一空间出现多份几何副本。精调 +7.30 dB 说明几何大体对、外观覆盖不足",
        verdict="性价比最好，但几何错误优化不了",
    ),
    dict(
        route="路线 C",
        name="前馈几何 + 逐场景优化",
        en="Two-stage: predict then optimize",
        color="#C77C2E",
        reps=["DA3 + 3DGS", "VGGT-Ω + 3DGS"],
        mech="前馈出深度 / 位姿 / 点云 → 组装 COLMAP 场景 → 30k it 光度优化",
        time="20.4 min　22.9 min（前馈只占 3~10%）",
        size="175~189 万高斯 · 414~448 MB",
        issue="漂浮物、转动闪烁；20 分钟的优化把前馈的秒级优势完全抵消",
        verdict="质量最稳，代价 100 倍",
    ),
]


def hex2rgb(h):
    h = h.lstrip("#")
    return tuple(int(h[k:k + 2], 16) for k in (0, 2, 4))


def tint(h, keep):
    """把颜色按 keep 比例混白（keep=0.12 → 12% 原色 + 88% 白）。"""
    r, g, b = hex2rgb(h)
    f = lambda c: int(round(c * keep + 255 * (1 - keep)))
    return "#%02X%02X%02X" % (f(r), f(g), f(b))


def card_html(c):
    reps = "".join(f'<span class="chip">{r}</span>' for r in c["reps"])
    rows = [
        ("机制", c["mech"]),
        ("耗时", c["time"]),
        ("产物", c["size"]),
        ("实测问题", c["issue"]),
    ]
    dl = "".join(f'<div class="row"><dt>{k}</dt><dd>{v}</dd></div>' for k, v in rows)
    style = (f"--c:{c['color']};--c-chip:{tint(c['color'], 0.12)};"
             f"--c-line:{tint(c['color'], 0.30)};--c-bg:{tint(c['color'], 0.08)}")
    return f"""
    <section class="card" style="{style}">
      <div class="route">{c['route']}</div>
      <div class="name">{c['name']}</div>
      <div class="en">{c['en']}</div>
      <div class="chips">{reps}</div>
      <div class="body">{dl}</div>
      <div class="verdict">{c['verdict']}</div>
    </section>"""


CSS = """
* { box-sizing: border-box; margin: 0; padding: 0; }
body { width: 1600px; height: 900px; background: #ffffff;
       font-family: "Microsoft YaHei", "Segoe UI", sans-serif; color: #222;
       -webkit-font-smoothing: antialiased; }
.page { width: 1600px; height: 900px; padding: 42px 48px 34px; display: flex;
        flex-direction: column; }
header h1 { font-size: 34px; font-weight: 700; letter-spacing: .5px; }
header .sub { margin-top: 8px; font-size: 15px; color: #7a7a7a; }
header .sub b { color: #444; font-weight: 600; }
.cards { margin-top: 26px; display: grid; grid-template-columns: repeat(3, 1fr);
         gap: 22px; align-items: stretch; }
.card { position: relative; background: #fbfbfa; border: 1px solid #e8e6e2;
        border-top: 5px solid var(--c); border-radius: 6px; padding: 24px 24px 20px;
        display: flex; flex-direction: column; }
.route { font-size: 12px; letter-spacing: 2px; color: var(--c); font-weight: 700; }
.name { margin-top: 8px; font-size: 26px; font-weight: 700; }
.en { margin-top: 3px; font-size: 12.5px; color: #9a9a9a; letter-spacing: .3px; }
.chips { margin-top: 12px; display: flex; flex-wrap: wrap; gap: 6px; }
.chip { font-size: 12px; padding: 3px 9px; border-radius: 11px;
        background: var(--c-chip); color: var(--c);
        border: 1px solid var(--c-line); }
.body { margin-top: 18px; }
.row { display: grid; grid-template-columns: 62px 1fr; gap: 10px; padding: 10px 0;
       border-top: 1px dashed #e4e2de; }
.row:first-child { border-top: none; }
dt { font-size: 12px; color: #9a9a9a; letter-spacing: .5px; padding-top: 2px; }
dd { font-size: 14.5px; line-height: 1.6; color: #333; }
.verdict { margin-top: auto; padding: 11px 13px; border-radius: 4px; font-size: 14.5px;
           font-weight: 600; color: var(--c); background: var(--c-bg); }
.concl { margin-top: 30px; display: grid; grid-template-columns: repeat(3, 1fr); gap: 26px;
         border-top: 1px solid #ebe8e3; padding-top: 20px; }
.cn { display: grid; grid-template-columns: 24px 1fr; gap: 9px; align-items: start; }
.cn .num { width: 24px; height: 24px; border-radius: 50%; background: #2f2f2f; color: #fff;
           font-size: 12.5px; font-weight: 700; display: flex; align-items: center;
           justify-content: center; margin-top: 1px; }
.cn .t { font-size: 14.5px; line-height: 1.65; color: #444; }
.cn .t b { color: #161616; }
footer { margin-top: auto; padding-top: 18px; display: grid; grid-template-columns: 1fr; gap: 22px; }
.note { font-size: 14px; line-height: 1.65; color: #444; padding-left: 14px;
        border-left: 3px solid #d8d4cc; }
.note b { color: #1f1f1f; }
.note .tag { font-size: 11.5px; letter-spacing: 1.5px; color: #999; display: block;
             margin-bottom: 3px; }
"""

HTML = f"""<!DOCTYPE html>
<html lang="zh-CN"><head><meta charset="utf-8"><style>{CSS}</style></head>
<body><div class="page">
  <header>
    <h1>前馈重建的三条技术路线：多快、多好、差在哪</h1>
    <div class="sub">同一数据（佳佳抱枕 · 32 视角 · RTX 3090）同一批帧，端到端实测。
      核心差别不在骨干网络，而在<b>「要不要做逐场景优化」</b>。</div>
  </header>
  <div class="cards">{''.join(card_html(c) for c in CARDS)}</div>
  <div class="concl">
    <div class="cn"><div class="num">1</div><div class="t"><b>速度已经分出胜负。</b>
      纯前馈 13.5 秒就能出可渲染的 PLY；接上 3DGS 变成 20 分钟以上，
      多出来的时间换来的是"能看"，不是"好用"。</div></div>
    <div class="cn"><div class="num">2</div><div class="t"><b>质量谁都没赢，但失败模式不一样。</b>
      视角不稳 / 重影 / 漂浮物分别指向 位姿一致性、几何副本、深度噪声——
      换一个方法解决不了另一个方法的问题。</div></div>
    <div class="cn"><div class="num">3</div><div class="t"><b>现在还排不了名。</b>
      只有定性和耗时，缺同口径 NVS holdout 与 Sim(3) Chamfer。
      这个洞不补，启动大规模训练就没有衡量标尺。</div></div>
  </div>
  <footer>
    <div class="note"><span class="tag">下一步（低成本验证先做）</span>
      ① 输入分辨率 504 → 1008，看细节是否随分辨率走；② 各方法用同一 COLMAP 相机重渲染，
      消掉构图差异；③ 前馈 init + <b>3~5k 步短优化</b>，验证能否用 2~4 分钟换掉 20 分钟；
      ④ 再谈改结构（换 2DGS 原语 / 加高分辨率外观头）——那是训练项目，不是推理改动。</div>
  </footer>
</div></body></html>"""

html_path = OUT / "fig7_taxonomy.html"
html_path.write_text(HTML, encoding="utf-8", newline="\n")
print("✅", html_path)

png_path = OUT / "fig7_taxonomy.png"
tmp = tempfile.mkdtemp(prefix="edge_shot_")
cmd = [
    EDGE,
    "--headless=new", "--disable-gpu", "--no-first-run", "--hide-scrollbars",
    "--force-device-scale-factor=2",
    "--window-size=1600,900",
    f"--user-data-dir={tmp}",
    f"--screenshot={png_path}",
    html_path.resolve().as_uri(),
]
r = subprocess.run(cmd, capture_output=True, timeout=120)
if not png_path.exists():
    raise SystemExit(f"❌ 截图失败 code={r.returncode}\n{r.stderr.decode('utf-8', 'replace')[:800]}")
print("✅", png_path, png_path.stat().st_size // 1024, "KB")

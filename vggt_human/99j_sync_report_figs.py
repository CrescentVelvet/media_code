# -*- coding: utf-8 -*-
"""把 report_figs/*.svg 最新内容同步进 camera_pose_tool_report.html 的内联 figure。

按 HTML 中 `<svg viewBox="0 0 1180 ...">` 出现顺序，依次替换为 fig1..fig4 的新 SVG。
替换时去掉 XML 声明与根标签 width/height（保留 viewBox，CSS `figure svg{width:100%}` 自适应）。
"""
import re
import pathlib

BASE = pathlib.Path(r"C:\code\media_code\vggt_human")
HTML = BASE / "camera_pose_tool_report.html"
FIGS = [
    "fig1_problem_mapping",
    "fig2_solution_architecture",
    "fig3_toolchain_dataflow",
    "fig4_before_after",
]


def extract_svg(path):
    text = path.read_text(encoding="utf-8")
    i = text.index("<svg")
    j = text.rindex("</svg>") + len("</svg>")
    seg = text[i:j]
    # 去掉根标签 width/height（保留 viewBox 自适应）
    seg = re.sub(r'(<svg[^>]*?)\s+width="\d+"\s+height="\d+"', r"\1", seg, count=1)
    return seg


def main():
    html = HTML.read_text(encoding="utf-8")
    svgs = [extract_svg(BASE / "report_figs" / (name + ".svg")) for name in FIGS]
    it = iter(svgs)
    pattern = re.compile(r'<svg\b[^>]*viewBox="0 0 \d+ \d+"[^>]*>.*?</svg>', re.S)
    new_html, n = pattern.subn(lambda m: next(it), html)
    print(f"✅ replaced {n} inline svg blocks")
    assert n == 4, f"expected 4, got {n}"
    HTML.write_text(new_html, encoding="utf-8")
    # 报告新 viewBox
    for name in FIGS:
        vb = re.search(r'viewBox="([^"]+)"', (BASE / "report_figs" / (name + ".svg")).read_text(encoding="utf-8"))
        print(f"  📐 {name}: viewBox {vb.group(1)}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Insert two independent-single-frame PCD comparison pages into the 42-page deck.

Use PPTX_INPUT=the validated 42-page backup and PPTX_PATH=the formal output.
This script is not idempotent: never run it on an already extended deck.
All existing pictures, video bytes and PLY links are preserved byte-for-byte;
only subsequent page numbers change.
"""
import copy
import json
import os
import re
from pathlib import Path

from pptx import Presentation
from pptx.enum.shapes import MSO_SHAPE_TYPE
from pptx.util import Pt

from build_full_view_splat_depth_ppt import card, text


def main():
    results = Path(os.environ["RESULTS_DIR"])
    assets = Path(os.environ["ASSET_DIR"])
    source_deck = Path(os.environ["PPTX_INPUT"])
    output_deck = Path(os.environ["PPTX_PATH"])
    assert source_deck.is_file() and not source_deck.is_symlink()
    assert output_deck.is_file() and not output_deck.is_symlink()
    assert source_deck.resolve() != output_deck.resolve()
    prs = Presentation(str(source_deck))
    assert len(prs.slides) == 42, len(prs.slides)
    old_pages = list(prs.slides)
    blank_layout = prs.slide_layouts[0]
    assert not blank_layout.placeholders
    groups = []
    for scene, title, anchor in [("human", "佳佳抱枕", 13), ("router", "路由器", 30)]:
        folder = results / scene / "single_frame" / "own_camera_splat_depth"
        audit = json.loads((folder / "view_render_audit.json").read_text(encoding="utf-8"))
        assert audit["sampling"].startswith("none;")
        assert audit["inference"].startswith("three independent model.infer(T=1)")
        slide = prs.slides.add_slide(blank_layout)
        slide.shapes._spTree.insert(2, copy.deepcopy(old_pages[10].shapes[0]._element))
        text(slide, "单帧对照 / 独立推理", 45, 35, 870, 18, 12, "8FBCE4")
        text(slide, f"{title} · 单帧输入 PCD splat / depth", 45, 57, 870, 33, 27, "FFFFFF", True)
        text(slide, "每帧单独 model.infer(T=1) · 无跨帧注意力 · 原相机投影 · 全部有效点", 45, 94, 870, 20, 12, "63A7D4", True)
        for row, kind in enumerate(("splat", "pcd_depth")):
            y = 125 + 193 * row
            text(slide, "PCD splat · RGB 点圆盘" if row == 0 else "PCD depth · 投影轴向 Z",
                 45, y, 565, 18, 12, "A9CFE7", True)
            for index, record in enumerate(audit["frames"]):
                image = assets / f"{scene}_single_{index:02d}_{kind}.png"
                assert image.is_file()
                x = 45 + 190 * index
                card(slide, x, y + 24, 176, 157)
                slide.shapes.add_picture(str(image), Pt(x + 37), Pt(y + 26), Pt(102), Pt(136))
                text(slide, f"Frame {index:02d} · {record['point_count'] / 1e6:.2f}M 点",
                     x, y + 166, 176, 13, 9, "C9CCCE", align=PP_ALIGN.CENTER)
        card(slide, 630, 145, 285, 349)
        text(slide, "怎么看这组对照", 650, 167, 245, 22, 17, "FFFFFF", True)
        text(slide, "• 三帧互不共享上下文，不是把联合 batch 拆成三张图。", 650, 207, 244, 45, 12, "E7E8E9")
        text(slide, "• 每帧使用原始 K / c2w 反投影，再投回自己的相机。", 650, 267, 244, 45, 12, "E7E8E9")
        text(slide, "• 路由器沿用联合实验的整组 SfM 尺度；两组共用各自联合实验深度色阶。", 650, 327, 244, 59, 12, "E7E8E9")
        text(slide, "• 自视角近满覆盖不代表几何正确；观察轮廓、薄结构与边缘错层。", 650, 402, 244, 59, 12, "E7E8E9")
        text(slide, "没有跨视角融合或新视角推理；本页不证明单帧更一致。PCD splat 不是训练得到的 3DGS。",
             45, 507, 790, 20, 9, "C9CCCE")
        text(slide, "单帧深度：single_frame/depth/*.npz · 保留原始输出，不做逐帧尺度配准",
             45, 530, 790, 16, 9, "63A7D4")
        slide_id = prs.slides._sldIdLst[-1]
        prs.slides._sldIdLst.remove(slide_id)
        prs.slides._sldIdLst.insert(anchor, slide_id)
        groups.append((anchor, slide))
    assert prs.slides[13].part == groups[0][1].part
    assert prs.slides[30].part == groups[1][1].part
    assert any(s.has_text_frame and "08 / SA 当前瓶颈" in s.text for s in prs.slides[40].shapes)
    for page_index, slide in enumerate(prs.slides, 1):
        for shape in slide.shapes:
            if shape.shape_type == MSO_SHAPE_TYPE.MEDIA or not shape.has_text_frame:
                continue
            for paragraph in shape.text_frame.paragraphs:
                for run in paragraph.runs:
                    if re.fullmatch(r"\d{2} / (?:12|18|24|42|44)", run.text.strip()):
                        run.text = f"{page_index:02d} / 44"
        if 11 < page_index < 44:
            expected = f"{page_index:02d} / 44"
            if not any(shape.has_text_frame and shape.text.strip() == expected for shape in slide.shapes):
                text(slide, expected, 884, 513, 50, 15, 10.5, "C9CCCE", align=PP_ALIGN.RIGHT)
    prs.save(str(output_deck))
    print(json.dumps({"pptx": str(output_deck), "slides": len(prs.slides),
                      "new_pages": {"human": 13, "router": 29}}, ensure_ascii=False))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Add nine full-point PCD splat/depth pages per scene to the 24-page deck.

Use PPTX_INPUT=the validated 24-page backup and PPTX_PATH=the formal output.
Every scene uses every valid point, never a random subset. The existing images,
videos, PLY links and sparse-view pages remain unchanged except for page numbers.
"""
import copy
import hashlib
import json
import os
import re
from pathlib import Path

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE, MSO_SHAPE_TYPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.opc.constants import RELATIONSHIP_TYPE as RT
from pptx.util import Pt


def text(slide, value, x, y, w, h, size, color="E7E8E9", bold=False, align=PP_ALIGN.LEFT):
    shape = slide.shapes.add_textbox(Pt(x), Pt(y), Pt(w), Pt(h))
    shape.text_frame.clear()
    shape.text_frame.word_wrap = True
    shape.text_frame.margin_left = shape.text_frame.margin_right = Pt(0)
    shape.text_frame.margin_top = shape.text_frame.margin_bottom = Pt(0)
    shape.text_frame.vertical_anchor = MSO_ANCHOR.TOP
    paragraph = shape.text_frame.paragraphs[0]
    paragraph.alignment = align
    run = paragraph.add_run()
    run.text = value
    run.font.name = "Noto Sans SC"
    run.font.size = Pt(size)
    run.font.bold = bold
    run.font.color.rgb = RGBColor.from_string(color)
    return shape


def card(slide, x, y, w, h):
    shape = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Pt(x), Pt(y), Pt(w), Pt(h))
    shape.fill.solid()
    shape.fill.fore_color.rgb = RGBColor.from_string("182532")
    shape.line.color.rgb = RGBColor.from_string("43505E")
    shape.line.width = Pt(0.75)
    shape.adjustments[0] = 0.025


def main():
    results = Path(os.environ["RESULTS_DIR"])
    assets = Path(os.environ["ASSET_DIR"])
    source_deck = Path(os.environ.get("PPTX_INPUT", os.environ["PPTX_PATH"]))
    output_deck = Path(os.environ["PPTX_PATH"])
    assert source_deck.is_file() and not source_deck.is_symlink()
    assert output_deck.is_file() and not output_deck.is_symlink()
    prs = Presentation(str(source_deck))
    assert len(prs.slides) == 24, len(prs.slides)
    old_pages = list(prs.slides)
    blank_layout = prs.slide_layouts[0]
    assert not blank_layout.placeholders
    groups = {}
    for scene, title, anchor in [("human", "佳佳", 14), ("router", "路由器", 29)]:
        audit = json.loads((results / scene / "view_splat_depth_full_9views" / "view_render_audit.json").read_text(encoding="utf-8"))
        assert audit["sampling"].startswith("none;")
        count = audit["point_count"]
        assert count == sum(audit["source_counts"])
        pages = []
        for index, record in enumerate(audit["views"]):
            view = record["view"]
            kind = record["kind"]
            kind_title = {"input": "输入视角", "interpolated": "内插视角", "extrapolated": "外插视角"}[kind]
            splat = assets / f"{scene}_full_{view}_splat.png"
            depth = assets / f"{scene}_full_{view}_pcd_depth.png"
            for image in (splat, depth):
                assert image.is_file()
                assert hashlib.sha256(image.read_bytes()).hexdigest() == hashlib.sha256(
                    (results / scene / "view_splat_depth_full_9views" / image.name.replace(f"{scene}_full_", "")).read_bytes()).hexdigest()
            slide = prs.slides.add_slide(blank_layout)
            slide.shapes._spTree.insert(2, copy.deepcopy(old_pages[10].shapes[0]._element))
            text(slide, f"全点投影 / {kind_title}", 45, 40, 870, 19, 12, "8FBCE4")
            short_view = {
                "input_00": "输入 Frame 00", "input_01": "输入 Frame 01", "input_02": "输入 Frame 02",
                "interpolated_00_01": "内插 00 → 01", "interpolated_01_02": "内插 01 → 02",
                "interpolated_00_02": "内插 00 → 02（全弧中点）",
                "extrapolated_before_00": "外插 00 之前", "extrapolated_after_02": "外插 02 之后",
                "extrapolated_full_arc": "外插 00 → 02 全弧前端",
            }[view]
            text(slide, f"{title} · {short_view}", 45, 64, 870, 34, 27, "FFFFFF", True)
            text(slide, f"全部有效点 {count:,} · 不抽样 · 三帧融合 PCD", 45, 103, 870, 20, 13, "63A7D4", True)
            text(slide, "PCD splat · RGB 点圆盘", 45, 137, 420, 20, 15, "A9CFE7", True)
            text(slide, "PCD depth · 投影轴向 Z", 505, 137, 410, 20, 15, "A9CFE7", True)
            w, h = record["image_width"], record["image_height"]
            image_h = 300
            image_w = image_h * w / h
            for column, image in enumerate((splat, depth)):
                x = 45 + 460 * column
                card(slide, x, 162, 420, 304)
                pic = slide.shapes.add_picture(str(image), Pt(x + (420 - image_w) / 2), Pt(164), Pt(image_w), Pt(image_h))
                pic.name = f"{scene}/full/{view}/{'splat' if column == 0 else 'pcd_depth'}: all {count} valid points"
                text(slide, f"{w} × {h} · 原图比例", x, 470, 420, 14, 9, "C9CCCE", align=PP_ALIGN.CENTER)
            description = record["description"].replace("Real input camera", "真实输入相机 Frame")
            description = description.replace("Virtual midpoint of cameras", "虚拟内插相机").replace("; t=0.5", " · t=0.5")
            description = description.replace("Virtual extrapolation along cameras", "虚拟外插相机").replace("; t=", " · t=")
            text(slide, f"{description} · 圆盘覆盖 {record['disc_coverage']:.1%}", 45, 494, 870, 18, 11, "63A7D4", True)
            if kind == "input":
                detail = "三帧全部有效点投回该真实输入相机；这是点圆盘/投影深度，不是原始 RGB 或原始单帧深度。"
            elif kind == "interpolated":
                detail = "虚拟相机：相对旋转 SLERP、相机中心线性插值；使用第二端点的内参，无新推理或新几何。"
            else:
                detail = "虚拟相机：沿相邻或全弧相机轨迹外推；离开观测范围，遮挡与错层可能更明显。"
            detail += " 覆盖率不是精度；空洞可来自遮挡、几何/位姿误差或运动。"
            text(slide, detail, 45, 521, 870, 36, 9.5, "C9CCCE")
            text(slide, "本批没有抽样稀疏化；PCD splat 不是训练得到的 3DGS。", 45, 565, 870, 18, 10, "C9CCCE")
            pages.append(slide)
            # Move this slide immediately after the preceding page in its group.
            slide_id = prs.slides._sldIdLst[-1]
            prs.slides._sldIdLst.remove(slide_id)
            prs.slides._sldIdLst.insert(anchor + index, slide_id)
        groups[scene] = (anchor, pages)
    current_slides = list(prs.slides)
    assert [s.part for s in current_slides[14:23]] == [s.part for s in groups["human"][1]]
    assert [s.part for s in current_slides[29:38]] == [s.part for s in groups["router"][1]]
    assert any(s.has_text_frame and "08 / SA 当前瓶颈" in s.text for s in current_slides[38].shapes)
    for page_index, slide in enumerate(prs.slides, 1):
        for shape in slide.shapes:
            if shape.shape_type == MSO_SHAPE_TYPE.MEDIA or not shape.has_text_frame:
                continue
            for paragraph in shape.text_frame.paragraphs:
                for run in paragraph.runs:
                    if re.fullmatch(r"\d{2} / (?:12|18|24|42)", run.text.strip()):
                        run.text = f"{page_index:02d} / 42"
        if 11 < page_index < 42:
            expected = f"{page_index:02d} / 42"
            if not any(shape.has_text_frame and shape.text.strip() == expected for shape in slide.shapes):
                text(slide, expected, 884, 513, 50, 15, 10.5, "C9CCCE", align=PP_ALIGN.RIGHT)
    for page, scene in [(9, "human"), (24, "router")]:
        slide = prs.slides[page - 1]
        for shape in list(slide.shapes):
            if shape.shape_type == MSO_SHAPE_TYPE.MEDIA:
                shape._element.getparent().remove(shape._element)
        for rel in list(slide.part.rels.values()):
            if rel.reltype in (RT.VIDEO, RT.MEDIA):
                slide.part.drop_rel(rel.rId)
        slide.shapes.add_movie(str(results / scene / "rgb_depth_pcd_comparison.mp4"),
            Pt(185), Pt(309), Pt(590), Pt(184),
            poster_frame_image=str(assets / f"{scene}_rgb_depth_pcd_comparison_poster.jpg"), mime_type="video/mp4")
    for page, scene in [(10, "human"), (25, "router")]:
        slide = prs.slides[page - 1]
        labels = {"PLY 00": 0, "PLY 01": 1, "PLY 02": 2, "融合 PLY": None}
        found = []
        for shape in slide.shapes:
            if not shape.has_text_frame or shape.text.strip() not in labels:
                continue
            index = labels[shape.text.strip()]
            target = results / scene / ("fused_no_registration.ply" if index is None else f"pcd/{index:02d}.ply")
            for paragraph in shape.text_frame.paragraphs:
                for run in paragraph.runs:
                    run.hyperlink.address = str(target).replace("\\", "/")
            found.append(shape.text.strip())
        assert sorted(found) == sorted(labels), (page, found)
    prs.save(str(output_deck))
    print(json.dumps({"pptx": str(output_deck), "slides": len(prs.slides),
                      "new_pages": {"human": list(range(15, 24)), "router": list(range(30, 39))},
                      "point_counts": {scene: json.loads((results / scene / "view_splat_depth_full_9views" / "view_render_audit.json").read_text(encoding="utf-8"))["point_count"] for scene in ("human", "router")}},
                     ensure_ascii=False))


if __name__ == "__main__":
    main()

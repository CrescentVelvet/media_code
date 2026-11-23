#!/usr/bin/env python3
"""Insert six large PCD splat/depth view pages into the formal delivery deck.

Run after render_view_splat_depth.py for both scenes. This script creates editable
vector/text elements and embeds the actual PNGs. Run it only once on the 18-page
input deck; it is not an idempotent page-content editor.

Env vars:
  PPTX_PATH, ASSET_DIR, RESULTS_DIR: explicit delivery input/output locations.
  PPTX_BACKUP: optional backup destination; existing backup is never overwritten.
"""
import copy
import hashlib
import json
import os
import re
import shutil
from pathlib import Path

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE, MSO_SHAPE_TYPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.opc.constants import RELATIONSHIP_TYPE as RT
from pptx.util import Pt


def add_text(slide, text, x, y, w, h, size, color, bold=False, align=PP_ALIGN.LEFT):
    shape = slide.shapes.add_textbox(Pt(x), Pt(y), Pt(w), Pt(h))
    frame = shape.text_frame
    frame.clear()
    frame.word_wrap = True
    frame.vertical_anchor = MSO_ANCHOR.TOP
    frame.margin_left = frame.margin_right = frame.margin_top = frame.margin_bottom = Pt(0)
    paragraph = frame.paragraphs[0]
    paragraph.alignment = align
    run = paragraph.add_run()
    run.text = text
    run.font.name = "Noto Sans SC"
    run.font.size = Pt(size)
    run.font.bold = bold
    run.font.color.rgb = RGBColor.from_string(color)
    return shape


def add_card(slide, x, y, w, h):
    shape = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Pt(x), Pt(y), Pt(w), Pt(h))
    shape.fill.solid()
    shape.fill.fore_color.rgb = RGBColor.from_string("182532")
    shape.line.color.rgb = RGBColor.from_string("43505E")
    shape.line.width = Pt(0.75)
    shape.adjustments[0] = 0.025
    return shape


def main():
    deck_path = Path(os.environ["PPTX_PATH"])
    asset_dir = Path(os.environ["ASSET_DIR"])
    results_dir = Path(os.environ["RESULTS_DIR"])
    for path in (deck_path, asset_dir, results_dir):
        if not path.exists():
            raise FileNotFoundError(path)
    if deck_path.is_symlink():
        raise ValueError("Refuse to replace a symlinked PPTX")
    backup = os.environ.get("PPTX_BACKUP")
    if backup:
        backup_path = Path(backup)
        if not backup_path.exists():
            shutil.copyfile(deck_path, backup_path)
    prs = Presentation(str(deck_path))
    if len(prs.slides) != 18:
        raise ValueError(f"Expected the original 18-page deck, got {len(prs.slides)}")
    old_pages = list(prs.slides)
    blank_layout = prs.slide_layouts[0]
    assert len(blank_layout.placeholders) == 0, "The deck's DEFAULT layout must remain blank"
    expected_images = [
        (1, 0, asset_dir / "human_rgb.png"), (1, 1, asset_dir / "human_depth.png"),
        (3, 0, asset_dir / "architecture.png"), (5, 0, asset_dir / "human_fourpanel.jpg"),
        (6, 0, asset_dir / "router_rgb.png"), (6, 1, asset_dir / "router_depth.png"),
        (7, 0, asset_dir / "human_crossview_last.jpg"), (8, 0, asset_dir / "router_crossview.jpg"),
        (9, 0, asset_dir / "human_video_frames_triptych.jpg"), (10, 0, asset_dir / "human_pcd_world_views.png"),
        (12, 0, asset_dir / "router_video_frames_triptych.jpg"), (13, 0, asset_dir / "router_pcd_world_views.png"),
        (15, 0, asset_dir / "sa_target.png"), (15, 1, asset_dir / "sa_rgb.png"), (15, 2, asset_dir / "sa_zip.png"),
        (11, 6, asset_dir / "human_splat_disc_diagnostic.png"),
    ]
    rebound = []
    for page, picture_index, source in expected_images:
        pictures = [shape for shape in old_pages[page - 1].shapes
                    if shape.shape_type == MSO_SHAPE_TYPE.PICTURE]
        picture = pictures[picture_index]
        expected_hash = hashlib.sha256(source.read_bytes()).hexdigest()
        if hashlib.sha256(picture.image.blob).hexdigest() != expected_hash:
            _, rel_id = old_pages[page - 1].part.get_or_add_image_part(str(source))
            picture._element.blipFill.blip.rEmbed = rel_id
            assert hashlib.sha256(picture.image.blob).hexdigest() == expected_hash
            rebound.append([page, source.name])
    content_pages = []
    specs = [
        ("human", "佳佳", "人像", 11, 4),
        ("router", "路由器", "路由器", 14, 10),
    ]
    view_specs = [
        ("input", "输入视角", "真实 Frame 01 相机"),
        ("interpolated", "内插视角", "虚拟相机 · t = 0.5"),
        ("extrapolated", "外插视角", "虚拟相机 · t = -0.5"),
    ]
    for scene, title_scene, category, anchor, section_start in specs:
        audit_path = results_dir / scene / "view_splat_depth_100k" / "view_render_audit.json"
        audit = json.loads(audit_path.read_text(encoding="utf-8"))
        assert audit["point_count"] == 100000
        for offset, (view, title_view, view_description) in enumerate(view_specs):
            record = next(item for item in audit["views"] if item["view"] == view)
            assert record["is_real_camera"] == (view == "input")
            splat = asset_dir / f"{scene}_{view}_splat.png"
            depth = asset_dir / f"{scene}_{view}_pcd_depth.png"
            for image in (splat, depth):
                if not image.is_file():
                    raise FileNotFoundError(image)
            slide = prs.slides.add_slide(blank_layout)
            # New slides are appended; final ordering is established below.
            background = old_pages[10].shapes[0]
            slide.shapes._spTree.insert(2, copy.deepcopy(background._element))
            add_text(slide, f"成果 {section_start + offset:02d} / {category}点云投影 · {title_view}",
                     45, 43, 870, 19, 12, "8FBCE4")
            add_text(slide, f"{title_scene} · {title_view} / PCD splat 与 PCD depth",
                     45, 66, 870, 38, 27, "FFFFFF", True)
            add_text(slide, "PCD splat · RGB 点圆盘", 45, 122, 420, 22, 16, "A9CFE7", True)
            add_text(slide, "PCD depth · 投影轴向 Z", 505, 122, 410, 22, 16, "A9CFE7", True)
            width, height = record["image_width"], record["image_height"]
            image_height = 280
            image_width = image_height * width / height
            for column, image in enumerate((splat, depth)):
                x = 45 + column * 460
                add_card(slide, x, 150, 420, 284)
                picture = slide.shapes.add_picture(str(image), Pt(x + (420 - image_width) / 2),
                                                   Pt(152), Pt(image_width), Pt(image_height))
                picture.name = f"{scene}/{view}/{'splat' if column == 0 else 'pcd_depth'}: formal 100k projection"
                add_text(slide, f"{width} × {height} 原图 · 未拉伸", x, 438, 420, 14, 9, "C9CCCE",
                         align=PP_ALIGN.CENTER)
            coverage = record["disc_coverage"]
            add_text(slide, f"{view_description} · 固定 100,000 点 · 圆盘覆盖 {coverage:.1%}",
                     45, 462, 870, 20, 13, "63A7D4", True)
            if view == "input":
                detail = "同一稀疏点集投回中间输入相机；圆盘与深度空洞清楚可见，不是原始 RGB / 原始深度。"
            elif view == "interpolated":
                detail = "相机 00→01 的相对旋转 SLERP、中心线性插值；无新推理、无新几何，不是采集图像。"
            else:
                detail = "沿相机 00→01 向 00 外延伸半个基线；离开观测弧，遮挡与错层更明显。"
            detail += " 空洞混合抽样、遮挡和几何/位姿误差，覆盖率不是精度。"
            add_text(slide, detail, 45, 490, 870, 31, 9.5, "C9CCCE")
            content_pages.append({"slide": slide, "anchor": anchor, "offset": offset})

    # Insert each group after its original anchor, then leave the SA decision
    # pages together at the end. Their original media relationships are untouched.
    slide_ids = list(prs.slides._sldIdLst)
    groups = [content_pages[i:i + 3] for i in range(0, len(content_pages), 3)]
    order = old_pages[:11] + [item["slide"] for item in groups[0]]
    order += old_pages[11:14] + [item["slide"] for item in groups[1]] + old_pages[14:]
    assert len(order) == 24
    part_to_id = {slide.part: slide_id for slide, slide_id in zip(prs.slides, slide_ids)}
    sorted_ids = [part_to_id[slide.part] for slide in order]
    for element in list(prs.slides._sldIdLst):
        prs.slides._sldIdLst.remove(element)
    for element in sorted_ids:
        prs.slides._sldIdLst.append(element)
    assert [slide.part for slide in prs.slides] == [slide.part for slide in order]
    for page_index, slide in enumerate(prs.slides, 1):
        for item in content_pages:
            if item["slide"] is not slide:
                continue
            scene = next(scene for scene, _, _, anchor, _ in specs if anchor == item["anchor"])
            label_shape = next(shape for shape in slide.shapes
                               if shape.has_text_frame and shape.text.startswith("成果 "))
            replacement = f"成果 {7 + item['offset'] if scene == 'human' else 10 + item['offset']:02d} / {'人像' if scene == 'human' else '路由器'}点云投影 · {['输入视角', '内插视角', '外插视角'][item['offset']]}"
            label_shape.text_frame.paragraphs[0].runs[0].text = replacement
        for shape in slide.shapes:
            if shape.shape_type == MSO_SHAPE_TYPE.MEDIA:
                continue
            if not shape.has_text_frame:
                continue
            for paragraph in shape.text_frame.paragraphs:
                for run in paragraph.runs:
                    match = re.fullmatch(r"\d{2} / (?:12|18|24)", run.text.strip())
                    if match:
                        run.text = f"{page_index:02d} / 24"
        if 11 < page_index < 21:
            expected = f"{page_index:02d} / 24"
            if not any(shape.has_text_frame and shape.text.strip() == expected for shape in slide.shapes):
                add_text(slide, expected, 884, 513, 50, 15, 10.5, "C9CCCE", align=PP_ALIGN.RIGHT)
    # Reorder may detach the editor-created media shape. Always restore both
    # videos after the final order is known, never reopen/save via the editor.
    for page, scene in [(9, "human"), (15, "router")]:
        slide = prs.slides[page - 1]
        for shape in list(slide.shapes):
            if shape.shape_type == MSO_SHAPE_TYPE.MEDIA:
                shape._element.getparent().remove(shape._element)
        for rel in list(slide.part.rels.values()):
            if rel.reltype in (RT.VIDEO, RT.MEDIA):
                slide.part.drop_rel(rel.rId)
        slide.shapes.add_movie(
            str(results_dir / scene / "rgb_depth_pcd_comparison.mp4"),
            Pt(185), Pt(309), Pt(590), Pt(184),
            poster_frame_image=str(asset_dir / f"{scene}_rgb_depth_pcd_comparison_poster.jpg"),
            mime_type="video/mp4")
    for page, scene in [(10, "human"), (16, "router")]:
        slide = prs.slides[page - 1]
        labels = {"PLY 00": 0, "PLY 01": 1, "PLY 02": 2, "融合 PLY": None}
        found = []
        for shape in slide.shapes:
            if not shape.has_text_frame or shape.text.strip() not in labels:
                continue
            index = labels[shape.text.strip()]
            target = (results_dir / scene / "fused_no_registration.ply" if index is None
                      else results_dir / scene / "pcd" / f"{index:02d}.ply")
            address = str(target).replace("/mnt/d/", "D:/").replace("\\", "/")
            for paragraph in shape.text_frame.paragraphs:
                for run in paragraph.runs:
                    run.hyperlink.address = address
            found.append(shape.text.strip())
        assert sorted(found) == sorted(labels), (page, found)
    prs.save(str(deck_path))
    print(json.dumps({"pptx": str(deck_path), "slides": len(prs.slides),
                      "new_pages": {"human": [12, 13, 14], "router": [18, 19, 20]},
                      "source_count_per_scene": 100000, "media_reembedded": [9, 15],
                      "source_images_rebound_by_hash": rebound}, ensure_ascii=False))


if __name__ == "__main__":
    main()

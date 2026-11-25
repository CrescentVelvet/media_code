#!/usr/bin/env python3
"""Add the 82-camera depth/PCD experiment to the current 47-page report.

Env vars: PPTX_INPUT (clean backup), PPTX_PATH, RESULTS_DIR, ASSET_DIR.
Run on Windows with python-pptx available. All existing slide content is kept;
only footers and new-page content change. Movie bytes are embedded exactly.
"""
import copy
import json
import os
import re
import zipfile
from pathlib import Path

from PIL import Image
from pptx import Presentation
from pptx.enum.shapes import MSO_SHAPE_TYPE
from pptx.enum.text import PP_ALIGN
from pptx.opc.constants import RELATIONSHIP_TYPE as RT
from pptx.util import Pt

from build_full_view_splat_depth_ppt import card, text
from io_utils import write_json

def video_poster(source, target, width):
    import subprocess
    import tempfile
    ffmpeg = os.environ.get("FFMPEG_PATH", "ffmpeg")
    with tempfile.TemporaryDirectory(prefix="dp2_video_poster_") as directory:
        frame = Path(directory) / "poster.png"
        # Do not assume frame count here: the final validator checks the exact
        # 82-frame video. The ordered midpoint is only a representative cover.
        subprocess.run([ffmpeg, "-y", "-ss", "3.4166666667", "-i", str(source),
                        "-frames:v", "1", str(frame)], check=True,
                       stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        with Image.open(frame) as image:
            image = image.convert("RGB")
            image.resize((width, round(width * image.height / image.width)),
                         Image.Resampling.LANCZOS).save(target)


def main():
    results = Path(os.environ["RESULTS_DIR"])
    scene = results / "human_sequence_82"
    assets = Path(os.environ["ASSET_DIR"])
    source, output = Path(os.environ["PPTX_INPUT"]), Path(os.environ["PPTX_PATH"])
    assert source.resolve() != output.resolve() and source.is_file() and not source.is_symlink()
    assert output.is_file() and not output.is_symlink()
    assets.mkdir(parents=True, exist_ok=True)
    inference = json.loads((scene / "inference.json").read_text(encoding="utf-8"))
    render = json.loads((scene / "render_audit.json").read_text(encoding="utf-8"))
    manifest = json.loads((scene / "input/manifest.json").read_text(encoding="utf-8"))
    assert inference["processed_frames"] == manifest["frame_count"] == 82
    assert render["fused_ply"]["points"] == sum(render["per_frame_pcd"]["counts"])
    assert (scene / render["fused_ply"]["file"]).is_file()
    for video in render["videos"]:
        assert (scene / video["file"]).is_file()
        poster = assets / f"sequence_{video['file'].replace('.mp4', '_poster.png')}"
        if os.environ.get("GENERATE_POSTERS", "0") == "1":
            video_poster(scene / video["file"], poster, 1280)
        assert poster.is_file() and not poster.is_symlink(), poster
    if os.environ.get("RESTORE_LEGACY_MEDIA", "1") == "1":
        with zipfile.ZipFile(source) as package:
            # The recovery input already has four valid MP4s. Never discard
            # valid legacy media while adding this experiment's two new videos.
            assert len([name for name in package.namelist()
                        if name.startswith("ppt/media/") and name.endswith(".mp4")]) == 4
    prs = Presentation(str(source))
    assert len(prs.slides) == 47
    cloudcompare = prs.slides[40]
    assert any(shape.has_text_frame and "CloudCompare" in shape.text
               for shape in cloudcompare.shapes)
    # The recovered page originally used the depth image dimensions. Keep the
    # real 2560x1372 screenshot aspect ratio and its text inside the slide.
    screenshot = assets / "cloudcompare_original.jpg"
    with Image.open(screenshot) as image:
        assert image.size == (2560, 1372)
    pictures = [shape for shape in cloudcompare.shapes
                if shape.shape_type == MSO_SHAPE_TYPE.PICTURE]
    assert len(pictures) == 1
    picture = pictures[0]
    assert picture.image.size == (2560, 1372)
    picture.crop_left = picture.crop_right = 0
    picture.crop_top = picture.crop_bottom = 0
    # Fit the whole landscape screenshot into the 304 pt content region.
    picture.width = Pt(304 * 2560 / 1372)
    picture.height = Pt(304)
    picture.left = Pt((960 - 304 * 2560 / 1372) / 2)
    picture.top = Pt(164)
    for shape in cloudcompare.shapes:
        if not shape.has_text_frame:
            continue
        if shape.text.strip() == "1152 × 1536 · 原图比例":
            shape.text_frame.paragraphs[0].runs[0].text = "2560 × 1372 · 原图比例"
            shape.top = Pt(474)
        elif "PCD splat 不是训练得到的 3DGS" in shape.text:
            # This duplicate note was already covered on the preceding PLY page
            # and was below the 540 pt canvas in the malformed legacy package.
            shape._element.getparent().remove(shape._element)
    layout = prs.slide_layouts[0]
    assert not layout.placeholders
    old_pages = list(prs.slides)
    new_ids = []

    def base(kicker, title, subtitle):
        slide = prs.slides.add_slide(layout)
        slide.shapes._spTree.insert(2, copy.deepcopy(old_pages[40].shapes[0]._element))
        text(slide, kicker, 45, 32, 870, 18, 12, "8FBCE4")
        text(slide, title, 45, 54, 870, 35, 27, "FFFFFF", True)
        text(slide, subtitle, 45, 94, 870, 20, 12, "63A7D4", True)
        new_ids.append(prs.slides._sldIdLst[-1])
        return slide

    def movie(slide, name, poster, x=45, y=126, w=870, h=348):
        card(slide, x, y, w, h)
        slide.shapes.add_movie(str(scene / name), Pt(x + 10), Pt(y + 10), Pt(w - 20), Pt(h - 20),
                               poster_frame_image=str(assets / poster), mime_type="video/mp4")

    slide = base("连续机位 / 82 帧官方 buffered 滑窗", "佳佳抱枕 · RGB / 全场景深度",
                 f"82 帧 · 12 fps 浏览 · {inference['inference_seconds']:.1f}s 推理及导出 · 峰值 {inference['peak_vram_gib']:.2f} GiB")
    movie(slide, "rgb_depth_comparison.mp4", "sequence_rgb_depth_comparison_poster.png")
    card(slide, 45, 487, 870, 48)
    text(slide, "16 帧窗口 / 4 帧重叠；显式使用去畸变后的标定 fx。深度近红远蓝，未输入外参。",
         62, 495, 836, 15, 10.5, "E7E8E9")
    text(slide, "有序静态图片按 12 fps 浏览；无采集时间戳，不把播放速度解释为人物运动速度。",
         62, 516, 836, 14, 9, "C9CCCE")

    slide = base("连续机位 / 世界坐标 PCD 邻域投影", "佳佳抱枕 · PCD Splat / PCD Depth",
                 "每帧前后各 8 帧 · 真实 c2w 后处理变换 · 不做 ICP / 逐帧尺度")
    movie(slide, "pcd_splat_depth_comparison.mp4", "sequence_pcd_splat_depth_comparison_poster.png")
    card(slide, 45, 487, 870, 48)
    text(slide, "预览每 2×2 像素取 1 点；左 RGB 点圆盘，右投影轴向 Z。圆盘不是训练的 3DGS。",
         62, 495, 836, 15, 10.5, "E7E8E9")
    text(slide, "密机位没有自动消除重影：头发、抱枕边缘仍见多层，遮挡与人物微动不能归于位姿缺失。",
         62, 516, 836, 14, 9, "C9CCCE")

    slide = base("同一滑窗深度 / 稀疏三帧 vs 全 82 帧", "增加机位不等于改善几何",
                 "固定目标相机 41 · 同一批 DP2 深度 · 仅改变参与融合的点云帧数")
    for column, kind in enumerate(("splat", "pcd_depth")):
        for row, group in enumerate(("three", "dense")):
            image = scene / "three_vs_82" / f"{group}_{kind}.png"
            x, y = 45 + 460 * column, 132 + 179 * row
            card(slide, x, y, 420, 170)
            label = ("三帧 00 / 41 / 81" if row == 0 else "全部 82 帧") + (" · Splat" if column == 0 else " · Depth")
            text(slide, label, x + 12, y + 8, 396, 16, 11, "FFFFFF", True)
            slide.shapes.add_picture(str(image), Pt(x + (420 - 100) / 2), Pt(y + 31), Pt(100), Pt(133))
    card(slide, 45, 491, 870, 42)
    text(slide, "实测：全 82 帧直接叠加把多时刻人物姿态全部保留，头发、衣服、抱枕重影反而更明显。",
         62, 497, 790, 13, 9.5, "E7E8E9")
    text(slide, "同 buffered 深度，仅改变融合成员；不是旧 T=3 推理复刻。",
         62, 515, 780, 12, 8.5, "C9CCCE")

    near_medians = [r["all_projected"]["median"] for r in render["near_pairs"]]
    slide = base("连续机位 / 实测判断", "视角变密有帮助，但不能单独解决动态几何",
                 "旧三帧：64.6°～163.7°；相邻：中位 3.35°；相邻 Z 中位误差 "
                 f"{min(near_medians):.2%}～{max(near_medians):.2%}（无遮挡筛选）")
    for column, frame in enumerate((10, 70)):
        for row, kind in enumerate(("rgb", "splat", "pcd_depth")):
            image = scene / "previews" / f"{frame:06d}_{kind}.png"
            x, y = 45 + 300 * column, 132 + row * 125
            card(slide, x, y, 280, 118)
            text(slide, f"Frame {frame:02d} · {dict(rgb='RGB', splat='邻域 Splat', pcd_depth='邻域 Depth')[kind]}",
                 x + 10, y + 6, 260, 15, 10.5, "E7E8E9", True)
            slide.shapes.add_picture(str(image), Pt(x + (280 - 67.5) / 2), Pt(y + 26), Pt(67.5), Pt(90))
    card(slide, 660, 132, 255, 368)
    text(slide, "判断", 680, 150, 215, 24, 17, "63A7D4", True)
    text(slide, "✓ 短基线观测更连续\n\n✓ 局部邻域可补充覆盖\n\n✕ 头发 / 抱枕仍多层\n\n✕ 全弧叠加保留多时刻姿态\n\n需要运动分区、可见性筛选与时间邻域选择，而不是简单堆更多帧。",
         680, 190, 215, 265, 13, "E7E8E9")
    text(slide, f"82 份逐帧全点 PLY；融合 {render['fused_ply']['points']:,} 点。无体素合并，未拟合 target。",
         45, 508, 780, 13, 8.5, "C9CCCE")

    # Insert as a block before the existing monkey page (old P42, index 41).
    for slide_id in new_ids:
        prs.slides._sldIdLst.remove(slide_id)
    for offset, slide_id in enumerate(new_ids):
        prs.slides._sldIdLst.insert(41 + offset, slide_id)
    assert len(prs.slides) == 51
    for page, slide in enumerate(prs.slides, 1):
        for shape in slide.shapes:
            if shape.shape_type == MSO_SHAPE_TYPE.MEDIA or not shape.has_text_frame:
                continue
            for paragraph in shape.text_frame.paragraphs:
                for run in paragraph.runs:
                    if re.fullmatch(r"\d{2} / (?:12|18|24|42|44|46)", run.text.strip()):
                        run.text = f"{page:02d} / 51"
        if 1 < page < 51:
            footer = f"{page:02d} / 51"
            if not any(s.has_text_frame and s.text.strip() == footer for s in slide.shapes):
                text(slide, footer, 884, 513, 50, 15, 10.5, "C9CCCE", align=PP_ALIGN.RIGHT)
    if os.environ.get("RESTORE_LEGACY_MEDIA", "1") == "1":
        legacy = [
            (9, "human", "rgb_depth_pcd_comparison.mp4", "human_rgb_depth_pcd_comparison_poster.jpg"),
            (25, "router", "rgb_depth_pcd_comparison.mp4", "router_rgb_depth_pcd_comparison_poster.jpg"),
            (46, "video_7ef19637", "rgb_depth_comparison.mp4", "video_7ef19637_rgb_depth_poster.png"),
            (47, "video_a964d156", "rgb_depth_comparison.mp4", "video_a964d156_rgb_depth_poster.png"),
        ]
        for page, group, filename, poster_name in legacy:
            slide = prs.slides[page - 1]
            shapes = [shape for shape in slide.shapes if shape.shape_type == MSO_SHAPE_TYPE.MEDIA]
            assert len(shapes) == 1, (page, len(shapes))
            left, top, width, height = shapes[0].left, shapes[0].top, shapes[0].width, shapes[0].height
            for shape in shapes:
                shape._element.getparent().remove(shape._element)
            for rel in list(slide.part.rels.values()):
                if rel.reltype in (RT.VIDEO, RT.MEDIA):
                    slide.part.drop_rel(rel.rId)
            for node in list(slide._element.findall(".//{http://schemas.openxmlformats.org/presentationml/2006/main}video")):
                node.getparent().remove(node)
            poster = assets / poster_name
            assert poster.is_file(), poster
            slide.shapes.add_movie(str(results / group / filename), left, top, width, height,
                                   poster_frame_image=str(poster), mime_type="video/mp4")
    prs.save(str(output))
    with zipfile.ZipFile(output) as package:
        seen, items = set(), []
        for item in package.infolist():
            if item.filename not in seen:
                seen.add(item.filename)
                items.append((item, package.read(item.filename)))
    with zipfile.ZipFile(output, "w") as package:
        for item, data in items:
            package.writestr(item, data)
    write_json(results / "human_sequence_82/ppt_build_audit.json", {
        "input": str(source), "output": str(output), "old_pages": 47, "new_pages": [42, 43, 44, 45],
        "slides": 51,
        "unresolved_media": {"page": 41, "action": "Retained original screenshot as a static image; original malformed package preserved separately",
                             "note": "Original CloudCompare recording unavailable; no replacement fabricated."},
        "notes": "Preserve the pre-existing CloudCompare page; its unavailable original recording is not replaced with a new result."})
    print(json.dumps({"pptx": str(output), "slides": 51, "new_pages": [42, 43, 44, 45]}, ensure_ascii=False))


if __name__ == "__main__":
    main()

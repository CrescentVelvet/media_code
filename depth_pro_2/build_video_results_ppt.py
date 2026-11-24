#!/usr/bin/env python3
"""Insert two full-video RGB/full-scene-depth pages into the validated 44-page deck.

Env vars: PPTX_INPUT (44-page backup), PPTX_PATH, RESULTS_DIR, ASSET_DIR.
The two original MP4s are embedded byte-for-byte; old media and PLY links remain.
"""
import copy
import hashlib
import json
import os
import re
import zipfile
from pathlib import Path, PurePosixPath

from pptx import Presentation
from pptx.enum.shapes import MSO_SHAPE_TYPE
from pptx.enum.text import PP_ALIGN
from pptx.util import Pt

from build_full_view_splat_depth_ppt import card, text


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    results = Path(os.environ["RESULTS_DIR"])
    assets = Path(os.environ["ASSET_DIR"])
    source = Path(os.environ["PPTX_INPUT"])
    output = Path(os.environ["PPTX_PATH"])
    assert source.is_file() and not source.is_symlink()
    assert output.is_file() and not output.is_symlink()
    assert source.resolve() != output.resolve()
    with zipfile.ZipFile(source) as package:
        media = [name for name in package.namelist() if name.startswith("ppt/media/") and name.endswith(".mp4")]
        assert len(media) == 2
        assert sorted(package.read(name) for name in media) == sorted(
            (results / scene / "rgb_depth_pcd_comparison.mp4").read_bytes() for scene in ("human", "router"))
    prs = Presentation(str(source))
    assert len(prs.slides) == 44
    old_pages = list(prs.slides)
    layout = prs.slide_layouts[0]
    assert not layout.placeholders
    specs = [
        ("video_7ef19637", "猴子视频 · 原 RGB / 全场景深度", "362 帧 · 29.97 fps · 12.08 秒",
         "主体前后层次和轮廓基本合理；海面有空洞、远景动态范围大。",
         "天空零逆深度／无限远保留为无有限深度；不补造背景。"),
        ("video_a964d156", "鸵鸟视频 · 原 RGB / 全场景深度", "302 帧 · 25 fps · 12.08 秒",
         "鸵鸟头颈、喙部开口及细羽毛轮廓清楚；竹林前后层次可见。",
         "眼喙内部较平滑；估计焦距不是实测标定，不能推断绝对尺度。"),
    ]
    new_slide_ids = []
    for scene, title, frame_text, finding, limit in specs:
        folder = results / scene
        video = folder / "rgb_depth_comparison.mp4"
        poster = assets / f"{scene}_rgb_depth_poster.png"
        inference = json.loads((folder / "inference.json").read_text(encoding="utf-8"))
        audit = json.loads((folder / "render_audit.json").read_text(encoding="utf-8"))
        assert inference["processed_frames"] == len(inference["frames"])
        assert inference["output_width"] == 1280 and inference["output_height"] == 720
        assert audit["video"]["columns"] == 2 and audit["video"]["near_preview_range_m"] is None
        assert sha(video) == audit["video"]["sha256"]
        assert poster.is_file() and not poster.is_symlink()
        slide = prs.slides.add_slide(layout)
        slide.shapes._spTree.insert(2, copy.deepcopy(old_pages[10].shapes[0]._element))
        text(slide, "整段视频 / 官方 buffered 滑窗", 45, 32, 870, 18, 12, "8FBCE4")
        text(slide, title, 45, 54, 870, 35, 27, "FFFFFF", True)
        text(slide, f"{frame_text} · 16 帧窗口 / 4 帧重叠 · 原视频逐帧推理", 45, 94, 870, 20, 12, "63A7D4", True)
        card(slide, 45, 129, 870, 354)
        slide.shapes.add_movie(str(video), Pt(65), Pt(144), Pt(830), Pt(320),
                               poster_frame_image=str(poster), mime_type="video/mp4")
        card(slide, 45, 475, 870, 59)
        text(slide, f"观察：{finding}", 62, 481, 836, 15, 10, "E7E8E9")
        text(slide, f"边界：{limit}", 62, 499, 836, 14, 9, "C9CCCE")
        text(slide, "左：原 RGB ｜ 右：全场景深度，近红远蓝；视频内嵌，点击播放。",
             62, 517, 750, 13, 9, "63A7D4")
        slide_id = prs.slides._sldIdLst[-1]
        new_slide_ids.append(slide_id)
    # Remove both appended entries first. Removing and reinserting each entry
    # immediately can produce duplicate slide parts in python-pptx on this deck.
    for slide_id in new_slide_ids:
        prs.slides._sldIdLst.remove(slide_id)
    for offset, slide_id in enumerate(new_slide_ids):
        prs.slides._sldIdLst.insert(40 + offset, slide_id)
    assert len(prs.slides) == 46
    for i, (_, title, *_rest) in enumerate(specs, 41):
        actual = next(shape.text for shape in prs.slides[i - 1].shapes if shape.has_text_frame and "· 原 RGB / 全场景深度" in shape.text)
        assert actual == title, (i, actual, title)
    assert any(shape.has_text_frame and "08 / SA 当前瓶颈" in shape.text for shape in prs.slides[42].shapes)
    for page, slide in enumerate(prs.slides, 1):
        for shape in slide.shapes:
            if shape.shape_type == MSO_SHAPE_TYPE.MEDIA or not shape.has_text_frame:
                continue
            for paragraph in shape.text_frame.paragraphs:
                for run in paragraph.runs:
                    if re.fullmatch(r"\d{2} / (?:12|18|24|42|44)", run.text.strip()):
                        run.text = f"{page:02d} / 46"
        if 1 < page < 46:
            footer = f"{page:02d} / 46"
            if not any(shape.has_text_frame and shape.text.strip() == footer for shape in slide.shapes):
                text(slide, footer, 884, 513, 50, 15, 10.5, "C9CCCE", align=PP_ALIGN.RIGHT)
    prs.save(str(output))
    # python-pptx can append duplicate parts when two new slides are reordered as
    # a block. Keep the first package entry, matching its relationship reader.
    with zipfile.ZipFile(output) as package:
        seen = set()
        items = []
        for item in package.infolist():
            if item.filename not in seen:
                seen.add(item.filename)
                items.append((item, package.read(item.filename)))
    with zipfile.ZipFile(output, "w") as package:
        for item, data in items:
            package.writestr(item, data)
    with zipfile.ZipFile(output) as package:
        assert len(package.namelist()) == len(set(package.namelist()))
        assert package.testzip() is None
    print(json.dumps({"pptx": str(output), "slides": len(prs.slides),
                      "new_pages": {"video_7ef19637": 41, "video_a964d156": 42}}, ensure_ascii=False))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Validate the 46-page deck, two full-video pages, and all legacy content.

Run on Windows with the PPT and SciPy dependencies available. A temporary package
only restores the previous page order and footers before running the 44-page check.
"""
import hashlib
import json
import os
import re
import zipfile
from pathlib import Path, PurePosixPath
import xml.etree.ElementTree as etree

from verify_single_frame_splat_depth_ppt import main as verify_44
from verify_view_splat_depth_ppt import R, image_sha, sha, slide_parts, slide_rel_path, xml_findall


def main():
    results = Path(os.environ["RESULTS_DIR"])
    assets = Path(os.environ["ASSET_DIR"])
    deck = Path(os.environ["PPTX_PATH"])
    with zipfile.ZipFile(deck) as package:
        assert package.testzip() is None
        parts = slide_parts(package)
        assert len(parts) == 46
        presentation = etree.fromstring(package.read("ppt/presentation.xml"))
        ns = {"p": "http://schemas.openxmlformats.org/presentationml/2006/main"}
        slide_ids = presentation.find("p:sldIdLst", ns)
        removed_parts = []
        for page in (42, 41):
            slide_id = list(slide_ids)[page - 1]
            rid = slide_id.get(R + "id")
            rels = etree.fromstring(package.read("ppt/_rels/presentation.xml.rels"))
            target = next(rel.get("Target") for rel in rels if rel.get("Id") == rid)
            part = "ppt/" + target.replace("../", "")
            removed_parts.append(part)
            slide_ids.remove(slide_id)
        for part in list(removed_parts):
            slide_rels = etree.fromstring(package.read(slide_rel_path(part)))
            for rel in slide_rels:
                if rel.get("Type", "").endswith(("/video", "/media")):
                    removed_parts.append("ppt/media/" + PurePosixPath(rel.get("Target")).name)
        rewritten = {}
        for page, part in parts.items():
            if page in (41, 42):
                continue
            assert part not in removed_parts
            old_page = page - 2 * int(page > 42)
            root = etree.fromstring(package.read(part))
            changed = False
            for node in xml_findall(root, "shape_texts"):
                for text_node in node.iter():
                    if text_node.tag.endswith("}t") and text_node.text and re.fullmatch(r"\d{2} / 46", text_node.text.strip()):
                        text_node.text = f"{old_page:02d} / 44"
                        changed = True
            if changed:
                rewritten[part] = etree.tostring(root, encoding="utf-8", xml_declaration=True)
        base = results / "_video_validation_base_44.pptx"
        with zipfile.ZipFile(base, "w") as restored:
            for item in package.infolist():
                if item.filename in removed_parts:
                    continue
                if item.filename == "ppt/presentation.xml":
                    data = etree.tostring(presentation, encoding="utf-8", xml_declaration=True)
                else:
                    data = rewritten.get(item.filename, package.read(item.filename))
                restored.writestr(item, data)
    previous = os.environ["PPTX_PATH"]
    os.environ["PPTX_PATH"] = str(base)
    try:
        verify_44()
    finally:
        os.environ["PPTX_PATH"] = previous
    with zipfile.ZipFile(deck) as package:
        parts = slide_parts(package)
        report = json.loads((results / "ppt_delivery_validation.json").read_text(encoding="utf-8"))
        report["pptx"] = str(deck)
        report["slide_count"] = 46
        report["full_video_checks"] = []
        for page, part in parts.items():
            root = etree.fromstring(package.read(part))
            texts = ["".join(node.itertext()).strip() for node in xml_findall(root, "shape_texts")]
            footers = [t for t in texts if re.fullmatch(r"\d{2} / (?:12|18|24|42|44|46)", t)]
            if 1 < page < 46:
                assert footers == [f"{page:02d} / 46"], (page, footers)
        media = [name for name in package.namelist() if name.startswith("ppt/media/") and name.endswith(".mp4")]
        assert len(media) == 4, media
        for scene, page, expected_title in [
                ("video_7ef19637", 41, "猴子视频"),
                ("video_a964d156", 42, "鸵鸟视频")]:
            folder = results / scene
            video = folder / "rgb_depth_comparison.mp4"
            inference = json.loads((folder / "inference.json").read_text(encoding="utf-8"))
            audit = json.loads((folder / "render_audit.json").read_text(encoding="utf-8"))
            assert audit["video"]["columns"] == 2 and audit["video"]["near_preview_range_m"] is None
            assert audit["video"]["frames"] == inference["processed_frames"]
            assert sha(video.read_bytes()) == audit["video"]["sha256"]
            root = etree.fromstring(package.read(parts[page]))
            texts = ["".join(node.itertext()).strip() for node in xml_findall(root, "shape_texts")]
            assert any(expected_title in t and "全场景深度" in t for t in texts)
            rels = etree.fromstring(package.read(slide_rel_path(parts[page])))
            videos = [rel for rel in rels if rel.get("Type", "").endswith("/video")]
            medias = [rel for rel in rels if rel.get("Type", "").endswith("/media")]
            assert len(videos) == len(medias) == 1
            name = "ppt/media/" + PurePosixPath(videos[0].get("Target")).name
            assert package.read(name) == video.read_bytes()
            assert xml_findall(root, "timing_videos")
            poster = assets / f"{scene}_rgb_depth_poster.png"
            assert image_sha(package, parts[page], 0) == sha(poster.read_bytes())
            report["videos"].append({"scene": scene, "page": page, "sha256": sha(video.read_bytes()),
                                     "frames": inference["processed_frames"], "fps": inference["output_fps"],
                                     "columns": 2, "source_sha256": audit["video"]["sha256"]})
            report["image_checks"].append({"slide": page, "picture": 0, "asset": str(poster), "match": True})
            report["full_video_checks"].append({"scene": scene, "page": page, "columns": 2,
                                                "frame_count": inference["processed_frames"],
                                                "video_and_poster_match": True})
        assert len(report["image_checks"]) == 90
        assert len(report["videos"]) == 4
        assert len(report["ply_links"]) == 8
        base.unlink()
    report["warnings"] = []
    (results / "ppt_delivery_validation.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8", newline="\n")
    print(json.dumps({"slides": 46, "verified_images": len(report["image_checks"]),
                      "videos": report["videos"], "ply_links": len(report["ply_links"]),
                      "full_video_checks": report["full_video_checks"], "warnings": []}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

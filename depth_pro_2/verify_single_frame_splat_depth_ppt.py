#!/usr/bin/env python3
"""Validate the 44-page deck including two independent-single-frame PCD pages.

The temporary 42-page base is validated on Windows so the original D:/ PLY
hyperlinks need no path rewriting. Only slide order and footer text are restored.
"""
import json
import os
import re
import zipfile
from pathlib import Path
import xml.etree.ElementTree as etree

from verify_full_view_splat_depth_ppt import main as verify_42
from verify_view_splat_depth_ppt import image_sha, sha, slide_parts, xml_findall


def main():
    results = Path(os.environ["RESULTS_DIR"])
    assets = Path(os.environ["ASSET_DIR"])
    deck = Path(os.environ["PPTX_PATH"])
    with zipfile.ZipFile(deck) as package:
        parts = slide_parts(package)
        assert len(parts) == 44
        assert package.testzip() is None
        # Restore the original display order and page-number texts in a temporary
        # package, then run every pre-existing image/media/link assertion on it.
        presentation = etree.fromstring(package.read("ppt/presentation.xml"))
        ns = {"p": "http://schemas.openxmlformats.org/presentationml/2006/main"}
        slide_ids = presentation.find("p:sldIdLst", ns)
        for page in sorted((14, 31), reverse=True):
            slide_ids.remove(list(slide_ids)[page - 1])
        output = results / "_single_frame_validation_base_42.pptx"
        rewritten = {}
        for page, part in parts.items():
            if page in (14, 31):
                continue
            old_page = page - int(page > 14) - int(page > 31)
            root = etree.fromstring(package.read(part))
            changed = False
            for node in xml_findall(root, "shape_texts"):
                for text_node in node.iter():
                    if text_node.tag.endswith("}t") and text_node.text and re.fullmatch(r"\d{2} / 44", text_node.text.strip()):
                        text_node.text = f"{old_page:02d} / 42"
                        changed = True
            if changed:
                rewritten[part] = etree.tostring(root, encoding="utf-8", xml_declaration=True)
        with zipfile.ZipFile(output, "w") as restored:
            for item in package.infolist():
                if item.filename == "ppt/presentation.xml":
                    data = etree.tostring(presentation, encoding="utf-8", xml_declaration=True)
                else:
                    data = rewritten.get(item.filename, package.read(item.filename))
                restored.writestr(item, data)
    previous = os.environ["PPTX_PATH"]
    os.environ["PPTX_PATH"] = str(output)
    try:
        verify_42()
    finally:
        os.environ["PPTX_PATH"] = previous
    with zipfile.ZipFile(deck) as package:
        parts = slide_parts(package)
        report = json.loads((results / "ppt_delivery_validation.json").read_text(encoding="utf-8"))
        report["pptx"] = str(deck)
        report["slide_count"] = 44
        report["single_frame_checks"] = []
        for page, part in parts.items():
            root = etree.fromstring(package.read(part))
            texts = ["".join(node.itertext()).strip() for node in xml_findall(root, "shape_texts")]
            footers = [t for t in texts if re.fullmatch(r"\d{2} / (?:12|18|24|42|44)", t)]
            if 1 < page < 44:
                assert footers == [f"{page:02d} / 44"], (page, footers)
        for scene, page, expected_title in [("human", 14, "佳佳抱枕"), ("router", 31, "路由器")]:
            folder = results / scene / "single_frame" / "own_camera_splat_depth"
            audit = json.loads((folder / "view_render_audit.json").read_text(encoding="utf-8"))
            inference = json.loads((results / scene / "single_frame" / "inference.json").read_text(encoding="utf-8"))
            assert inference["joint_frames"] is False and inference["window"] == 1
            assert len(inference["frames"]) == len(audit["frames"]) == 3
            assert sha((results / scene / "manifest.json").read_bytes()) == inference["input_manifest_sha256"]
            root = etree.fromstring(package.read(parts[page]))
            texts = ["".join(node.itertext()).strip() for node in xml_findall(root, "shape_texts")]
            assert any(expected_title in t and "单帧输入" in t for t in texts)
            for index, record in enumerate(audit["frames"]):
                depth_path = results / scene / "single_frame" / "depth" / f"{index:02d}.npz"
                assert record["source_depth_sha256"] == sha(depth_path.read_bytes())
                assert record["source_depth_sha256"] == inference["frames"][index]["depth_npz_sha256"]
                for column, kind in enumerate(("splat", "pcd_depth")):
                    source = folder / f"{index:02d}_{kind}.png"
                    asset = assets / f"{scene}_single_{index:02d}_{kind}.png"
                    assert sha(source.read_bytes()) == sha(asset.read_bytes())
                    picture = column * 3 + index
                    assert image_sha(package, parts[page], picture) == sha(source.read_bytes()), (scene, index, kind)
                    report["image_checks"].append({"slide": page, "picture": picture, "asset": str(asset), "match": True})
            report["single_frame_checks"].append({"scene": scene, "page": page, "independent_calls": 3,
                                                 "point_counts": [f["point_count"] for f in audit["frames"]]})
        assert len(report["image_checks"]) == 88
        output.unlink()
    report["warnings"] = []
    (results / "ppt_delivery_validation.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8", newline="\n")
    print(json.dumps({"slides": 44, "verified_images": len(report["image_checks"]), "videos": report["videos"],
                      "ply_links": len(report["ply_links"]), "single_frame_checks": report["single_frame_checks"],
                      "warnings": []}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Validate PCD view sampling, images, camera geometry and final PPT media/assets."""
import hashlib
import json
import os
import posixpath
import re
import zipfile
from pathlib import Path, PurePosixPath

import numpy as np
import xml.etree.ElementTree as etree
from PIL import Image

NS = {
    "p": "http://schemas.openxmlformats.org/presentationml/2006/main",
    "a": "http://schemas.openxmlformats.org/drawingml/2006/main",
    "r": "http://schemas.openxmlformats.org/officeDocument/2006/relationships",
}
R = "{" + NS["r"] + "}"


def sha(data):
    return hashlib.sha256(data).hexdigest()


def xml_findall(root, query):
    if query == "pictures":
        return root.findall(".//p:pic", NS)
    if query == "blips":
        return root.findall(".//a:blip", NS)
    if query == "shape_texts":
        return root.findall(".//p:sp", NS)
    if query == "timing_videos":
        return root.findall("./p:timing//p:video", NS)
    if query == "hyperlinks":
        return root.findall(".//a:r[a:t]/a:rPr/a:hlinkClick", NS)
    raise ValueError(query)


def slide_rel_path(part):
    directory, filename = posixpath.split(part)
    return f"{directory}/_rels/{filename}.rels"


def image_sha(package, part, picture_index):
    root = etree.fromstring(package.read(part))
    rels = etree.fromstring(package.read(slide_rel_path(part)))
    rel_map = {rel.get("Id"): rel for rel in rels}
    picture = xml_findall(root, "pictures")[picture_index]
    rid = xml_findall(picture, "blips")[0].get(R + "embed")
    target = posixpath.normpath(posixpath.join(posixpath.dirname(part), rel_map[rid].get("Target")))
    return sha(package.read(target))


def slide_parts(package):
    """Map display-page numbers to actual package parts after slide reordering."""
    root = etree.fromstring(package.read("ppt/presentation.xml"))
    rels = etree.fromstring(package.read("ppt/_rels/presentation.xml.rels"))
    rel_map = {rel.get("Id"): rel.get("Target") for rel in rels}
    return {index: posixpath.normpath(posixpath.join("ppt", rel_map[node.get(R + "id")]))
            for index, node in enumerate(root.findall("./p:sldIdLst/p:sldId", NS), 1)}


def main():
    results = Path(os.environ["RESULTS_DIR"])
    deck = Path(os.environ["PPTX_PATH"])
    assets = Path(os.environ["ASSET_DIR"])
    report = {"pptx": str(deck), "scene_checks": [], "image_checks": [],
              "videos": [], "ply_links": [], "warnings": []}
    with zipfile.ZipFile(deck) as package:
        assert package.testzip() is None
        report["zip_test"] = None
        parts = slide_parts(package)
        assert len(parts) == 24, len(parts)
        report["slide_count"] = 24
        for index, name in parts.items():
            root = etree.fromstring(package.read(name))
            texts = ["".join(node.itertext()).strip()
                     for node in xml_findall(root, "shape_texts")]
            footer = [text for text in texts if re.fullmatch(r"\d{2} / (?:18|24)", text)]
            if 1 < index < 24:
                assert footer == [f"{index:02d} / 24"], (index, footer)
            rels = etree.fromstring(package.read(slide_rel_path(name)))
            rel_map = {rel.get("Id"): rel for rel in rels}
            for hlink in xml_findall(root, "hyperlinks"):
                rid = hlink.get(R + "id")
                assert rid, (index, "empty hyperlink relationship")
                target = rel_map[rid].get("Target")
                if target.lower().endswith(".ply"):
                    assert not target.startswith(("0000e", "../")), target
                    path = Path(target) if Path(target).is_absolute() else Path("/mnt/d") / target[3:]
                    assert path.is_file() and path.stat().st_size > 1000000, target
                    report["ply_links"].append({"slide": index, "target": target,
                                                 "bytes": path.stat().st_size})
        assert len(report["ply_links"]) == 8, report["ply_links"]
        for scene, page in [("human", 10), ("router", 16)]:
            expected = [f"D:/output/results/depth_pro_2_results/{scene}/pcd/{i:02d}.ply" for i in range(3)]
            expected.append(f"D:/output/results/depth_pro_2_results/{scene}/fused_no_registration.ply")
            actual = sorted({link["target"] for link in report["ply_links"] if link["slide"] == page})
            assert actual == sorted(expected), (scene, actual)
        media = [name for name in package.namelist()
                 if name.startswith("ppt/media/") and name.lower().endswith(".mp4")]
        assert len(media) == 2, media
        report["embedded_mp4"] = media
        for scene, page in [("human", 9), ("router", 15)]:
            root = etree.fromstring(package.read(parts[page]))
            rels = etree.fromstring(package.read(slide_rel_path(parts[page])))
            videos = [rel for rel in rels if rel.get("Type", "").endswith("/video")]
            medias = [rel for rel in rels if rel.get("Type", "").endswith("/media")]
            assert len(videos) == 1 and len(medias) == 1, (scene, len(videos), len(medias))
            name = "ppt/media/" + PurePosixPath(videos[0].get("Target")).name
            source = results / scene / "rgb_depth_pcd_comparison.mp4"
            assert package.read(name) == source.read_bytes(), scene
            assert xml_findall(root, "timing_videos"), scene
            report["videos"].append({"scene": scene, "page": page, "part": name,
                                     "sha256": sha(package.read(name)), "timing": True})
        expected_pictures = [
            (1, 0, assets / "human_rgb.png"), (1, 1, assets / "human_depth.png"),
            (3, 0, assets / "architecture.png"),
            (6, 0, assets / "router_rgb.png"), (6, 1, assets / "router_depth.png"),
            (7, 0, assets / "human_crossview_last.jpg"), (8, 0, assets / "router_crossview.jpg"),
            (9, 0, assets / "human_video_frames_triptych.jpg"),
            (10, 0, assets / "human_pcd_world_views.png"),
            (12, 0, assets / "human_input_splat.png"),
            (15, 0, assets / "router_video_frames_triptych.jpg"),
            (16, 0, assets / "router_pcd_world_views.png"),
            (21, 0, assets / "sa_target.png"), (21, 1, assets / "sa_rgb.png"),
            (21, 2, assets / "sa_zip.png"), (11, 6, assets / "human_splat_disc_diagnostic.png"),
        ]
        for page, scene in [(11, "human"), (17, "router")]:
            for picture in range(6):
                kind = "splat" if picture < 3 else "pcd_depth"
                expected_pictures.append((page, picture, results / scene / kind / f"{picture % 3:02d}.png"))
        for scene, pages in [("human", [12, 13, 14]), ("router", [18, 19, 20])]:
            audit_path = results / scene / "view_splat_depth_100k" / "view_render_audit.json"
            audit = json.loads(audit_path.read_text(encoding="utf-8"))
            assert audit["point_count"] == 100000
            assert audit["seed"] == 20261123
            assert audit["registration"] == "none; no ICP; no per-frame scale fitting"
            assert len(audit["sampled_point_indices_sha256"]) == 64
            for key, digest in audit["source_depth_npz_sha256"].items():
                index = int(key.split("_")[-1])
                assert sha((results / scene / "depth" / f"{index:02d}.npz").read_bytes()) == digest
            for page, record in zip(pages, audit["views"]):
                view = record["view"]
                camera = np.asarray(record["c2w"])
                assert np.allclose(camera[:3, :3].T @ camera[:3, :3], np.eye(3), atol=1e-6)
                assert np.isclose(np.linalg.det(camera[:3, :3]), 1, atol=1e-6)
                assert record["is_real_camera"] == (view == "input")
                if view == "input":
                    manifest = json.loads((results / scene / "manifest.json").read_text(encoding="utf-8"))
                    assert np.allclose(camera, np.asarray(manifest["frames"][1]["c2w"]), atol=1e-7)
                for picture, kind in enumerate(("splat", "pcd_depth")):
                    path = assets / f"{scene}_{view}_{kind}.png"
                    with Image.open(path) as image:
                        assert image.size == (record["image_width"], record["image_height"])
                        assert len(np.unique(np.asarray(image).reshape(-1, 3), axis=0)) > 100
                    expected_pictures.append((page, picture, path))
            report["scene_checks"].append({"scene": scene, "points": 100000,
                                           "coverage": {item["view"]: item["disc_coverage"] for item in audit["views"]},
                                           "camera_pages": pages})
        for page, picture, source in expected_pictures:
            actual = image_sha(package, parts[page], picture)
            expected = sha(source.read_bytes())
            assert actual == expected, (page, picture, str(source), actual, expected)
            report["image_checks"].append({"slide": page, "picture": picture,
                                           "asset": str(source), "sha256": actual, "match": True})
    output = results / "ppt_delivery_validation.json"
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8", newline="\n")
    print(json.dumps({"slides": report["slide_count"], "verified_images": len(report["image_checks"]),
                      "videos": report["videos"], "ply_links": len(report["ply_links"]),
                      "scene_checks": report["scene_checks"], "warnings": report["warnings"],
                      "report": str(output)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

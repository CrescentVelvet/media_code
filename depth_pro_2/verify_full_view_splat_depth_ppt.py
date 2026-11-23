#!/usr/bin/env python3
"""Validate the 42-page deck, full-point cameras, image sources and media."""
import hashlib
import json
import os
import re
import zipfile
from pathlib import Path, PurePosixPath
import xml.etree.ElementTree as etree

import sys
import types

import numpy as np
from PIL import Image

# Import camera construction without requiring OpenCV in the file-validation env.
if "render_geometry" not in sys.modules:
    geometry_stub = types.ModuleType("render_geometry")
    geometry_stub.depth_color = geometry_stub.depth_to_world = None
    geometry_stub.render_splat = geometry_stub.write_json = geometry_stub.save_ply = None
    sys.modules["render_geometry"] = geometry_stub
from render_full_view_splat_depth import camera_specs
from verify_view_splat_depth_ppt import (
    R, image_sha, sha, slide_parts, slide_rel_path, xml_findall,
)


def main():
    results = Path(os.environ["RESULTS_DIR"])
    assets = Path(os.environ["ASSET_DIR"])
    deck = Path(os.environ["PPTX_PATH"])
    report = {"pptx": str(deck), "slide_count": 42, "image_checks": [],
              "full_scene_checks": [], "videos": [], "ply_links": [], "warnings": []}
    with zipfile.ZipFile(deck) as package:
        assert package.testzip() is None
        parts = slide_parts(package)
        assert len(parts) == 42
        for page, part in parts.items():
            root = etree.fromstring(package.read(part))
            texts = ["".join(node.itertext()).strip() for node in xml_findall(root, "shape_texts")]
            footers = [t for t in texts if re.fullmatch(r"\d{2} / (?:12|18|24|42)", t)]
            if 1 < page < 42:
                assert footers == [f"{page:02d} / 42"], (page, footers)
            rels = etree.fromstring(package.read(slide_rel_path(part)))
            rel_map = {rel.get("Id"): rel for rel in rels}
            for link in xml_findall(root, "hyperlinks"):
                target = rel_map[link.get(R + "id")].get("Target")
                if target.lower().endswith(".ply"):
                    path = Path(target)
                    assert path.is_file() and path.stat().st_size > 1000000, target
                    report["ply_links"].append({"slide": page, "target": target})
        assert len(report["ply_links"]) == 8
        for scene, page in [("human", 10), ("router", 25)]:
            expected = [f"D:/output/results/depth_pro_2_results/{scene}/pcd/{i:02d}.ply" for i in range(3)]
            expected.append(f"D:/output/results/depth_pro_2_results/{scene}/fused_no_registration.ply")
            actual = sorted(link["target"] for link in report["ply_links"] if link["slide"] == page)
            assert actual == sorted(expected), (scene, actual)
        media = [name for name in package.namelist() if name.startswith("ppt/media/") and name.endswith(".mp4")]
        assert len(media) == 2
        for scene, page in [("human", 9), ("router", 24)]:
            root = etree.fromstring(package.read(parts[page]))
            rels = etree.fromstring(package.read(slide_rel_path(parts[page])))
            videos = [rel for rel in rels if rel.get("Type", "").endswith("/video")]
            medias = [rel for rel in rels if rel.get("Type", "").endswith("/media")]
            assert len(videos) == len(medias) == 1, scene
            name = "ppt/media/" + PurePosixPath(videos[0].get("Target")).name
            assert package.read(name) == (results / scene / "rgb_depth_pcd_comparison.mp4").read_bytes()
            assert xml_findall(root, "timing_videos")
            report["videos"].append({"scene": scene, "page": page, "sha256": sha(package.read(name))})
        expected = [
            (1, 0, assets / "human_rgb.png"), (1, 1, assets / "human_depth.png"),
            (3, 0, assets / "architecture.png"), (5, 0, assets / "human_fourpanel.jpg"),
            (6, 0, assets / "router_rgb.png"), (6, 1, assets / "router_depth.png"),
            (7, 0, assets / "human_crossview_last.jpg"), (8, 0, assets / "router_crossview.jpg"),
            (9, 0, assets / "human_video_frames_triptych.jpg"), (10, 0, assets / "human_pcd_world_views.png"),
            (24, 0, assets / "router_video_frames_triptych.jpg"), (25, 0, assets / "router_pcd_world_views.png"),
            (39, 0, assets / "sa_target.png"), (39, 1, assets / "sa_rgb.png"), (39, 2, assets / "sa_zip.png"),
            (11, 6, assets / "human_splat_disc_diagnostic.png"),
        ]
        for page, scene in [(11, "human"), (26, "router")]:
            for picture in range(6):
                kind = "splat" if picture < 3 else "pcd_depth"
                expected.append((page, picture, results / scene / kind / f"{picture % 3:02d}.png"))
        for scene, pages in [("human", [12, 13, 14]), ("router", [27, 28, 29])]:
            audit = json.loads((results / scene / "view_splat_depth_100k" / "view_render_audit.json").read_text(encoding="utf-8"))
            assert audit["point_count"] == 100000
            for page, record in zip(pages, audit["views"]):
                for picture, kind in enumerate(("splat", "pcd_depth")):
                    expected.append((page, picture, assets / f"{scene}_{record['view']}_{kind}.png"))
        for scene, pages in [("human", list(range(15, 24))), ("router", list(range(30, 39)))]:
            folder = results / scene / "view_splat_depth_full_9views"
            audit = json.loads((folder / "view_render_audit.json").read_text(encoding="utf-8"))
            provenance = json.loads((folder / "full_valid_point_provenance.json").read_text(encoding="utf-8"))
            geometry = json.loads((results / scene / "geometry.json").read_text(encoding="utf-8"))
            manifest = json.loads((results / scene / "manifest.json").read_text(encoding="utf-8"))
            count = audit["point_count"]
            assert count == sum(audit["source_counts"]) == sum(geometry["point_count"])
            assert audit["sampling"].startswith("none;")
            assert provenance["point_count"] == count
            assert provenance["source_fused_ply_sha256"] == sha((results / scene / "fused_no_registration.ply").read_bytes())
            for key, digest in audit["source_depth_npz_sha256"].items():
                index = int(key.split("_")[-1])
                assert sha((results / scene / "depth" / f"{index:02d}.npz").read_bytes()) == digest
            specs = camera_specs(manifest["frames"])
            assert len(audit["views"]) == len(pages) == 9
            for page, record, spec in zip(pages, audit["views"], specs):
                assert record["view"] == spec["view"] and record["point_count"] == count
                assert record["is_real_camera"] == (record["kind"] == "input")
                assert np.allclose(record["c2w"], spec["c2w"], atol=1e-7)
                assert np.allclose(record["K"], spec["K"], atol=1e-7)
                assert np.allclose(np.asarray(record["c2w"])[:3, :3].T @ np.asarray(record["c2w"])[:3, :3], np.eye(3), atol=1e-6)
                assert np.isclose(np.linalg.det(np.asarray(record["c2w"])[:3, :3]), 1, atol=1e-6)
                if record["kind"] == "input":
                    assert record["fraction"] is None
                    assert np.allclose(record["c2w"], manifest["frames"][record["reference_frame"]]["c2w"], atol=1e-7)
                for picture, kind in enumerate(("splat", "pcd_depth")):
                    path = assets / f"{scene}_full_{record['view']}_{kind}.png"
                    with Image.open(path) as image:
                        assert image.size == (record["image_width"], record["image_height"])
                        # Discrete depth maps can have few colors but valid geometry.
                        # Require non-empty pixels and at least two distinct colors.
                        pixels = np.asarray(image).reshape(-1, 3)
                        assert len(np.unique(pixels, axis=0)) > 1
                        assert np.any(np.any(pixels != 244, axis=1))
                    assert sha(path.read_bytes()) == sha((folder / f"{record['view']}_{kind}.png").read_bytes())
                    expected.append((page, picture, path))
            report["full_scene_checks"].append({"scene": scene, "points": count, "source_counts": audit["source_counts"],
                                                "pages": pages, "coverage": {v["view"]: v["disc_coverage"] for v in audit["views"]}})
        for page, picture, source in expected:
            digest = image_sha(package, parts[page], picture)
            assert digest == sha(source.read_bytes()), (page, picture, str(source))
            report["image_checks"].append({"slide": page, "picture": picture, "asset": str(source), "match": True})
        assert len(expected) == 76, len(expected)
    output = results / "ppt_delivery_validation.json"
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8", newline="\n")
    print(json.dumps({"slides": 42, "verified_images": len(report["image_checks"]), "videos": report["videos"],
                      "ply_links": len(report["ply_links"]), "full_scene_checks": report["full_scene_checks"],
                      "warnings": report["warnings"], "report": str(output)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

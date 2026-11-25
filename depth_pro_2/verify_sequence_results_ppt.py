#!/usr/bin/env python3
"""Validate continuous-sequence experiment and current 51-page PPT delivery.

Env vars: RESULTS_DIR, ASSET_DIR, PPTX_PATH, EXPECT_MEDIA_REPAIRED (0/1).
Set FFMPEG_PATH for full null-sink decoding, or FFPROBE_PATH for ffprobe.
This validator never saves the deck and never rewrites its videos or linked PLY files.
"""
import json
import os
import posixpath
import re
import subprocess
import zipfile
from fractions import Fraction
from io import BytesIO
from pathlib import Path, PurePosixPath
import xml.etree.ElementTree as etree

from PIL import Image

from io_utils import sha256, write_json
from verify_view_splat_depth_ppt import R, slide_parts, slide_rel_path, xml_findall


def video_probe(path):
    """Decode the complete video or raise; never trust the container frame count."""
    ffprobe = os.environ.get("FFPROBE_PATH")
    if ffprobe:
        if not Path(ffprobe).is_file():
            raise FileNotFoundError(f"FFPROBE_PATH is not a file: {ffprobe}")
        return json.loads(subprocess.check_output([
            ffprobe, "-v", "error", "-select_streams", "v:0", "-count_frames",
            "-show_entries", "stream=width,height,r_frame_rate,nb_read_frames",
            "-of", "json", str(path)], text=True, encoding="utf-8"))["streams"][0]
    if os.environ.get("FFMPEG_PATH"):
        # FFmpeg returns successfully only after decoding the complete stream;
        # the progress line reports the final frame, dimensions and frame rate.
        completed = subprocess.run([
            os.environ["FFMPEG_PATH"], "-v", "info", "-i", str(path),
            "-map", "0:v:0", "-f", "null", "-"], check=True,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, encoding="utf-8")
        header = re.search(r"Video:.*?, (\d+)x(\d+).*?, ([0-9.]+) fps", completed.stderr)
        progress = re.findall(r"frame=\s*(\d+).*?fps=\s*([0-9.]+)", completed.stderr)
        if not header or not progress:
            raise ValueError(f"Cannot parse full FFmpeg decode: {path}\n{completed.stderr}")
        return {"width": int(header[1]), "height": int(header[2]),
                "r_frame_rate": header[3], "nb_read_frames": int(progress[-1][0])}
    raise ValueError("Set FFMPEG_PATH or FFPROBE_PATH for independent full-video validation")


def main():
    results = Path(os.environ["RESULTS_DIR"])
    scene = results / "human_sequence_82"
    deck = Path(os.environ["PPTX_PATH"])
    inference = json.loads((scene / "inference.json").read_text(encoding="utf-8"))
    audit = json.loads((scene / "render_audit.json").read_text(encoding="utf-8"))
    manifest = json.loads((scene / "input/manifest.json").read_text(encoding="utf-8"))
    assert inference["processed_frames"] == manifest["frame_count"] == audit["frames"] == 82
    assert len(inference["frames"]) == len(manifest["frames"]) == 82
    assert len(audit["per_frame_pcd"]["counts"]) == 82
    assert audit["fused_ply"]["points"] == sum(audit["per_frame_pcd"]["counts"]) == 145096704
    assert sha256(scene / audit["fused_ply"]["file"]) == audit["fused_ply"]["sha256"]
    for index, count in enumerate(audit["per_frame_pcd"]["counts"]):
        with (scene / "pcd" / f"{index:06d}.ply").open("rb") as stream:
            header = stream.read(4096)
        assert header.startswith(b"ply\n") and b"end_header\n" in header
        assert int(re.search(rb"element vertex (\d+)", header)[1]) == count
    for record in audit["videos"]:
        path = scene / record["file"]
        assert sha256(path) == record["sha256"]
        stream = video_probe(path)
        assert int(stream["nb_read_frames"]) == record["frames"] == 82
        assert abs(float(Fraction(str(stream["r_frame_rate"]))) - record["fps"]) < .001
        assert (stream["width"], stream["height"]) == (record["width"], record["height"]) == (2304, 1574)
    near = [row["all_projected"]["median"] for row in audit["near_pairs"]]
    assert len(near) == 81 and all(row["all_projected"]["n"] > 10000 for row in audit["near_pairs"])
    with zipfile.ZipFile(deck) as package:
        assert package.testzip() is None
        assert len(package.namelist()) == len(set(package.namelist()))
        for name in package.namelist():
            if name.endswith((".xml", ".rels")):
                etree.fromstring(package.read(name))
        parts = slide_parts(package)
        assert len(parts) == 51
        titles = {}
        media = [name for name in package.namelist() if name.startswith("ppt/media/") and name.endswith(".mp4")]
        for page, part in parts.items():
            root = etree.fromstring(package.read(part))
            texts = ["".join(node.itertext()).strip() for node in xml_findall(root, "shape_texts")]
            titles[page] = [t for t in texts if len(t) > 10 and not re.fullmatch(r"\d{2} / \d{2}", t)]
            footers = [t for t in texts if re.fullmatch(r"\d{2} / (?:12|18|24|42|44|46|51)", t)]
            if 1 < page < 51:
                assert footers == [f"{page:02d} / 51"], (page, footers)
        for page, expected in {
            41: "CloudCompare", 42: "RGB / 全场景深度", 43: "PCD Splat / PCD Depth",
            44: "增加机位不等于改善几何", 45: "视角变密有帮助", 46: "猴子", 47: "鸵鸟",
        }.items():
            assert any(expected in text for text in titles[page]), (page, titles[page])
        cloud = etree.fromstring(package.read(parts[41]))
        picture = xml_findall(cloud, "pictures")[0]
        rid = xml_findall(picture, "blips")[0].get(R + "embed")
        rels = etree.fromstring(package.read(slide_rel_path(parts[41])))
        target = next(rel.get("Target") for rel in rels if rel.get("Id") == rid)
        image_part = posixpath.normpath(posixpath.join("ppt/slides", target))
        with Image.open(Path(os.environ["ASSET_DIR"]) / "cloudcompare_original.jpg") as source_image:
            with Image.open(BytesIO(package.read(image_part))) as embedded_image:
                assert source_image.size == embedded_image.size == (2560, 1372)
                assert source_image.tobytes() == embedded_image.tobytes()
        transform = picture.find("./p:spPr/a:xfrm", {"p": "http://schemas.openxmlformats.org/presentationml/2006/main",
                                                     "a": "http://schemas.openxmlformats.org/drawingml/2006/main"})
        offset = transform.find("./{http://schemas.openxmlformats.org/drawingml/2006/main}off")
        extent = transform.find("./{http://schemas.openxmlformats.org/drawingml/2006/main}ext")
        x, y = (int(offset.get(k)) / 12700 for k in ("x", "y"))
        width, height = (int(extent.get(k)) / 12700 for k in ("cx", "cy"))
        assert x >= 0 and y >= 0 and x + width <= 960 and y + height <= 540
        assert abs(width / height - 2560 / 1372) < .001
        assert not any(rel.get("Type", "").endswith("/video") for rel in rels)
        assert "2560 × 1372 · 原图比例" in titles[41]
        checked = []
        for page, filename in ((42, "rgb_depth_comparison.mp4"), (43, "pcd_splat_depth_comparison.mp4")):
            part = parts[page]
            root = etree.fromstring(package.read(part))
            texts = ["".join(node.itertext()).strip() for node in xml_findall(root, "shape_texts")]
            assert any("佳佳抱枕" in t for t in texts)
            rels = etree.fromstring(package.read(slide_rel_path(part)))
            videos = [rel for rel in rels if rel.get("Type", "").endswith("/video")]
            medias = [rel for rel in rels if rel.get("Type", "").endswith("/media")]
            assert len(videos) == len(medias) == 1
            name = "ppt/media/" + PurePosixPath(videos[0].get("Target")).name
            assert package.read(name) == (scene / filename).read_bytes()
            assert xml_findall(root, "timing_videos")
            checked.append({"page": page, "file": filename, "embedded": True,
                            "sha256": sha256(scene / filename), "frames": 82, "fps": 12})
        report = {"pptx": str(deck), "slides": 51, "new_video_checks": checked,
                  "embedded_mp4_count": len(media), "sequence_frames": 82,
                  "full_ply_points": audit["fused_ply"]["points"],
                  "near_pair_z_median_range": [float(min(near)), float(max(near))],
                  "near_pair_metric_boundary": "2x2 display-grid source-to-target projection; no visibility filter; not GT accuracy",
                  "warnings": []}
        if os.environ.get("EXPECT_MEDIA_REPAIRED", "0") == "1":
            assert len(media) == 6, media
            for page, scene_name in ((9, "human"), (25, "router"), (46, "video_7ef19637"), (47, "video_a964d156")):
                rels = etree.fromstring(package.read(slide_rel_path(parts[page])))
                videos = [rel for rel in rels if rel.get("Type", "").endswith("/video")]
                assert len(videos) == 1
                name = "ppt/media/" + PurePosixPath(videos[0].get("Target")).name
                filename = "rgb_depth_pcd_comparison.mp4" if page in (9, 25) else "rgb_depth_comparison.mp4"
                assert package.read(name) == (results / scene_name / filename).read_bytes()
            report["legacy_video_repairs"] = [9, 25, 46, 47]
            report["unresolved_legacy_media"] = [{"page": 41, "reason": "Original CloudCompare recording not found in project output or PPT backups; no substitute fabricated."}]
            report["warnings"].append("Pre-existing CloudCompare recording on P41 remains unavailable; its page is preserved, not passed as playable.")
    write_json(scene / "ppt_validation.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

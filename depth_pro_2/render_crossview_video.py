#!/usr/bin/env python3
"""Render each view plus the other two PCDs in the same camera for comparison."""
import json
import os
from pathlib import Path

import av
import numpy as np
from PIL import Image

from render_geometry import label, render_splat


def main():
    root = Path(os.environ["SCENE_DIR"])
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    scale_path = root / "sfm_scale.json"
    scale = json.loads(scale_path.read_text(encoding="utf-8"))["metric_to_sfm_scale"] if scale_path.is_file() else 1.0
    frames = manifest["frames"]
    worlds, colors, rgbs = [], [], []
    for fr in frames:
        i = fr["index"]
        rgb = np.asarray(Image.open(root / "rgb" / f"{i:02d}.png").convert("RGB"))
        depth = np.load(root / "depth" / f"{i:02d}.npz")["depth"] * scale
        xyz = np.stack(((np.arange(fr["width"])[None, :] - fr["K"][0][2]) / fr["K"][0][0] * depth,
                        (np.arange(fr["height"])[:, None] - fr["K"][1][2]) / fr["K"][1][1] * depth,
                        depth), axis=-1).reshape(-1, 3)
        worlds.append(xyz @ np.asarray(fr["c2w"])[:3, :3].T + np.asarray(fr["c2w"])[:3, 3])
        colors.append(rgb.reshape(-1, 3))
        rgbs.append(rgb)
    images = []
    for i, fr in enumerate(frames):
        others = [j for j in range(3) if j != i]
        rendered, _, coverage = render_splat(np.concatenate([worlds[j] for j in others]),
                                             np.concatenate([colors[j] for j in others]), fr, rgbs[i].shape[:2], 1.2)
        canvas = np.concatenate([
            label(rgbs[i], f"frame {i} | RGB"),
            label(rendered, "other 2 PCDs | same camera", f"original poses, coverage={coverage:.1%}"),
        ], axis=1)
        images.append(canvas)
        Image.fromarray(canvas).save(root / "pcd_reprojection" / f"comparison_{i:02d}.jpg", quality=94)
    output = root / "crossview_pcd_comparison.mp4"
    with av.open(str(output), "w") as container:
        stream = container.add_stream("libx264", rate=1)
        stream.width, stream.height = images[0].shape[1], images[0].shape[0]
        stream.pix_fmt = "yuv420p"
        stream.options = {"crf": "18", "preset": "medium"}
        for image in images:
            for _ in range(3):
                frame = av.VideoFrame.from_ndarray(image, format="rgb24")
                for packet in stream.encode(frame):
                    container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    print(f"🎬 Cross-view PCD video: {output}")


if __name__ == "__main__":
    main()

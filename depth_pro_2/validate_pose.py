#!/usr/bin/env python3
"""Visual-only sanity check: project the optional capture PCD back to RGB.

Env var: SCENE_DIR. Capture PCD is not an independent metric ground truth.
"""
import json
import os
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from plyfile import PlyData


def main():
    root = Path(os.environ["SCENE_DIR"])
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    reference = manifest.get("reference_pcd")
    if not reference or not Path(reference).is_file():
        print("⏭️ No capture PCD; pose check not applicable")
        return
    v = PlyData.read(reference)["vertex"]
    points = np.stack([v["x"], v["y"], v["z"]], axis=1)
    colors = np.stack([v["red"], v["green"], v["blue"]], axis=1)
    (root / "pose_check").mkdir(exist_ok=True)
    for fr in manifest["frames"]:
        i = fr["index"]
        rgb = np.asarray(Image.open(root / "rgb" / f"{i:02d}.png").convert("RGB"))
        h, w = rgb.shape[:2]
        K, w2c = np.asarray(fr["K"]), np.asarray(fr["w2c"])
        camera = points @ w2c[:3, :3].T + w2c[:3, 3]
        z = camera[:, 2]
        u = np.round(K[0, 0] * camera[:, 0] / z + K[0, 2]).astype(np.int32)
        y = np.round(K[1, 1] * camera[:, 1] / z + K[1, 2]).astype(np.int32)
        valid = (z > 0) & (u >= 0) & (u < w) & (y >= 0) & (y < h)
        z, u, y, cols = z[valid], u[valid], y[valid], colors[valid]
        order = np.argsort(z)[::-1]
        ref = np.zeros((h, w, 3), dtype=np.uint8)
        overlay = rgb.copy()
        for idx in order:
            cv2.circle(ref, (int(u[idx]), int(y[idx])), 2, cols[idx].tolist(), -1)
            cv2.circle(overlay, (int(u[idx]), int(y[idx])), 2, (255, 40, 40), -1)
        canvas = Image.new("RGB", (w*3, h+48), (248, 248, 248))
        canvas.paste(Image.fromarray(rgb), (0, 48))
        canvas.paste(Image.fromarray(ref), (w, 48))
        canvas.paste(Image.fromarray(overlay), (w*2, 48))
        d = ImageDraw.Draw(canvas)
        font = ImageFont.truetype("DejaVuSans.ttf", 16)
        d.text((12, 12), "RGB", fill=(25, 25, 25), font=font)
        d.text((w+12, 12), "Capture PCD (not ground truth)", fill=(25, 25, 25), font=font)
        d.text((w*2+12, 12), "Pose-convention overlay (red)", fill=(25, 25, 25), font=font)
        canvas.save(root / "pose_check" / f"{i:02d}.jpg", quality=94)
        print(f"✅ Pose convention check frame {i}: {valid.sum()} projected points")


if __name__ == "__main__":
    main()

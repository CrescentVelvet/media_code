#!/usr/bin/env python3
"""cmp_official.py — 我的渲染管线 vs nn 官方渲染（按 cameras.json 正确索引）。"""
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
from composite_check import load_gaussian_ply, MergedGS  # noqa: E402
from train_avatar import read_cameras  # noqa: E402


def main():
    results_dir = Path(os.environ.get("RESULTS_DIR", ""))
    upstream = Path(os.environ.get("UPSTREAM_DIR", ""))
    source_dir = os.environ.get("SOURCE_DIR", f"{upstream}/source")
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    full = load_gaussian_ply(
        upstream / "model_3dgs_nn/point_cloud/iteration_7000/point_cloud.ply")

    sys.path.insert(0, os.environ.get("GS_DIR", ""))
    from gaussian_renderer import render
    from scene.cameras import Camera as GSCamera
    from types import SimpleNamespace
    pipe = SimpleNamespace(compute_cov3D_python=False,
                           convert_SHs_python=False, debug=False,
                           antialiasing=False)

    views = read_cameras(source_dir)
    align = json.loads(
        (results_dir / "05_align/head_align.json").read_text())
    stems = align["persons"]["0"]["frames"]
    bg = torch.zeros(3, device=dev)

    cams = json.load(open(upstream / "model_3dgs_nn/cameras.json"))
    ridx = {c["img_name"].rsplit(".", 1)[0]: i for i, c in enumerate(cams)}

    from PIL import Image
    for i in (0, 14, 29, 44, 58):
        s = stems[i]
        v = next(v for v in views if s in v["name"])
        gt_pil = Image.open(
            f"{source_dir}/images/{s}.jpg").convert("RGB").resize(
            (v["W"], v["H"]))
        gt = torch.tensor(np.asarray(gt_pil), dtype=torch.float32,
                          device=dev) / 255.0
        gt = gt.permute(2, 0, 1)
        cam = GSCamera(resolution=(v["W"], v["H"]), colmap_id=0,
                       R=v["R"].T, T=v["T"],
                       FoVx=2 * np.arctan(v["W"] / (2 * v["fx"])),
                       FoVy=2 * np.arctan(v["H"] / (2 * v["fy"])),
                       depth_params=None, invdepthmap=None,
                       image=gt_pil, image_name=s, uid=i,
                       data_device=dev)
        m = MergedGS(full["xyz"].to(dev),
                     torch.cat([full["f_dc"].unsqueeze(1),
                                full["f_rest"]], 1).to(dev),
                     full["opacity"].to(dev), full["scaling"].to(dev),
                     full["rotation"].to(dev), 3)
        with torch.no_grad():
            pkg = render(cam, m, pipe, bg, 1.0)
        mine = pkg["render"].clamp(0, 1)

        idx = ridx[s]
        official = torch.tensor(np.asarray(Image.open(
            upstream / f"model_3dgs_nn/train/ours_7000/renders/{idx:05d}.png"
        ).convert("RGB")), dtype=torch.float32, device=dev) / 255.0
        official = official.permute(2, 0, 1)
        mse2 = ((mine - official) ** 2).mean()
        mse3 = ((mine - gt) ** 2).mean()
        print(f"frame {i:3d} ({s}) ridx={idx}: "
              f"mine-vs-official={-10 * np.log10(mse2.item()):.2f} dB  "
              f"mine-vs-GT={-10 * np.log10(mse3.item()):.2f} dB", flush=True)


if __name__ == "__main__":
    main()

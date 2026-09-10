#!/usr/bin/env python3
"""cmp_render2.py — 同相机同 ply：MergedGS vs 官方 GaussianModel 渲染对比。"""
import os
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
from composite_check import load_gaussian_ply, MergedGS  # noqa: E402


def main():
    upstream = Path(os.environ.get("UPSTREAM_DIR", ""))
    source_dir = os.environ.get("SOURCE_DIR", f"{upstream}/source")
    dev = torch.device("cuda")

    ply = upstream / "model_3dgs_nn/point_cloud/iteration_7000/point_cloud.ply"

    sys.path.insert(0, os.environ.get("GS_DIR", ""))
    from scene import Scene, GaussianModel
    from gaussian_renderer import render
    from scene.cameras import Camera as GSCamera
    from types import SimpleNamespace
    from argparse import Namespace

    args = Namespace(sh_degree=3,
                     source_path=str(upstream / "source"),
                     model_path=str(upstream / "model_3dgs_nn"),
                     images="images", depths="", resolution=-1,
                     white_background=False, data_device="cuda",
                     eval=False, train_test_exp=False, shs_lr=None)
    gs = GaussianModel(3)
    scene = Scene(args, gs, load_iteration=-1)
    cams = (scene.train_cameras[0.0] if 0.0 in scene.train_cameras
            else list(scene.train_cameras.values())[0])
    cam_by_name = {c.image_name: c for c in cams}
    pipe = SimpleNamespace(compute_cov3D_python=False,
                           convert_SHs_python=False, debug=False,
                           antialiasing=False)
    bg = torch.zeros(3, device=dev)

    # 官方 GaussianModel 已由 Scene.load 加载（iteration 7000）
    t = load_gaussian_ply(ply)
    merged = MergedGS(
        t["xyz"].to(dev),
        torch.cat([t["f_dc"].unsqueeze(1), t["f_rest"]], 1).to(dev),
        t["opacity"].to(dev), t["scaling"].to(dev),
        t["rotation"].to(dev), 3)

    # 张量级对比（官方全是 property，直接属性访问）
    print("== 张量级对比 ==")
    try:
        print("xyz  diff:", (gs._xyz - merged.get_xyz).abs().max().item())
        print("opa  diff:", (gs.get_opacity - merged.get_opacity).abs().max().item())
        print("sc   diff:", (gs.get_scaling - merged.get_scaling).abs().max().item())
        print("rot  diff:", (gs.get_rotation - merged.get_rotation).abs().max().item())
        print("feat diff:", (gs.get_features - merged.get_features).abs().max().item())
        print("feat shape:", gs.get_features.shape, merged.get_features.shape)
        print("official active_sh:", gs.active_sh_degree,
              " max_sh:", gs.max_sh_degree)
    except Exception as e:
        print("tensor cmp skip:", e)
    # 渲染级对比（帧 29 的 stem）
    stem = "679455776749000"
    for name, c in cam_by_name.items():
        if stem in name:
            cam = c
            break
    with torch.no_grad():
        pkg_o = render(cam, gs, pipe, bg, 1.0)
        pkg_m = render(cam, merged, pipe, bg, 1.0)
    o = pkg_o["render"].clamp(0, 1)
    m = pkg_m["render"].clamp(0, 1)
    mse = ((o - m) ** 2).mean()
    print(f"\n== 渲染对比 ({stem}) ==")
    print(f"official-model vs MergedGS: {-10 * np.log10(mse.item()):.2f} dB")
    # 与官方训练时落盘的渲染对比
    from PIL import Image
    idx = int(cam.uid)
    # 从 cameras.json 找 idx
    import json
    cams_j = json.load(open(upstream / "model_3dgs_nn/cameras.json"))
    ridx = next(i for i, c in enumerate(cams_j)
                if stem in c["img_name"])
    disk = torch.tensor(np.asarray(Image.open(
        upstream / f"model_3dgs_nn/train/ours_7000/renders/{ridx:05d}.png"
    ).convert("RGB")), dtype=torch.float32, device=dev) / 255.0
    disk = disk.permute(2, 0, 1)
    mse2 = ((o - disk) ** 2).mean()
    mse3 = ((m - disk) ** 2).mean()
    print(f"official-model vs 落盘渲染: {-10 * np.log10(mse2.item()):.2f} dB")
    print(f"MergedGS vs 落盘渲染     : {-10 * np.log10(mse3.item()):.2f} dB")


if __name__ == "__main__":
    main()

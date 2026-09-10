#!/usr/bin/env python3
"""cmp_camera.py — 官方 Scene 相机 vs 我手工构造的 GSCamera 逐项对比。"""
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
from train_avatar import read_cameras  # noqa: E402


def main():
    upstream = Path(os.environ.get("UPSTREAM_DIR", ""))
    source_dir = os.environ.get("SOURCE_DIR", f"{upstream}/source")

    sys.path.insert(0, os.environ.get("GS_DIR", ""))
    # 官方 Scene 加载（不走 NeuralRendererDataset 那套也行，直接用 CamerasBuilder?
    # 最稳：用 scene.Scene 加载 model_3dgs_nn 的 cameras.json + gaussians ply）
    from scene import Scene, GaussianModel
    from argparse import Namespace

    # cfg_args 从模型目录读
    cfg = eval(open(upstream / "model_3dgs/cfg_args").read())
    print("cfg.source_path =", cfg.source_path)

    # Scene 需要 args: sh_degree, source_path, model_path, images, resolution...
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
    print("official train cams:", len(cams))

    views = read_cameras(source_dir)
    vmap = {v["stem"]: v for v in views}

    from scene.cameras import Camera as GSCamera
    from PIL import Image as PILImage
    dummy = PILImage.new("RGB", (16, 16))

    for oc in cams[:3]:
        stem = oc.image_name
        if stem not in vmap:
            # image_name 可能带扩展名
            stem = stem.rsplit(".", 1)[0]
        v = vmap.get(stem)
        if v is None:
            print("no view for", oc.image_name)
            continue
        mine = GSCamera(resolution=(v["W"], v["H"]), colmap_id=0,
                        R=v["R"].T, T=v["T"],
                        FoVx=2 * np.arctan(v["W"] / (2 * v["fx"])),
                        FoVy=2 * np.arctan(v["H"] / (2 * v["fy"])),
                        depth_params=None, invdepthmap=None,
                        image=dummy.resize((v["W"], v["H"])),
                        image_name=v["stem"], uid=0,
                        data_device="cpu")
        wvt_official = oc.world_view_transform
        wvt_mine = mine.world_view_transform
        diff = (wvt_official - wvt_mine).abs().max().item()
        print(f"{v['stem']}: world_view_transform max diff = {diff:.6f}"
              f"  (R param: official {'?'} vs mine v['R'].T)")
        # 逐矩阵对比
        print("  official R:", oc.R.round(3).tolist()[0])
        print("  sparse   R:", v["R"].round(3).tolist()[0])
        print("  official T:", oc.T.round(3).tolist())
        print("  sparse   T:", v["T"].round(3).tolist())


if __name__ == "__main__":
    main()

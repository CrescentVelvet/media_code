#!/usr/bin/env python3
"""composite_psnr_all.py — 全帧 composite PSNR 分布（不只 5 帧）。"""
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
from composite_check import load_gaussian_ply, MergedGS, CkptAvatar  # noqa: E402
from train_avatar import N_SHAPE, N_EXPR, read_cameras  # noqa: E402


def main():
    results_dir = Path(os.environ.get("RESULTS_DIR", ""))
    upstream = Path(os.environ.get("UPSTREAM_DIR", ""))
    source_dir = os.environ.get("SOURCE_DIR", f"{upstream}/source")
    images_dir = Path(os.environ.get("IMAGES_DIR", f"{source_dir}/images"))
    pid = os.environ.get("PID", "0")
    dev = torch.device("cuda")

    body = load_gaussian_ply(results_dir / "07_body_gs_src" / f"body_gs_p{pid}.ply")
    scene = load_gaussian_ply(results_dir / "07_body_gs_src" / "scene_gs.ply")

    ck = torch.load(results_dir / "08_train" / f"avatar_p{pid}_final.pth",
                    map_location="cpu")
    import smplx
    flame = smplx.create(model_path=os.path.dirname(os.environ["FLAME_MODEL"]),
                         model_type="flame",
                         num_betas=N_SHAPE, num_expression_coeffs=N_EXPR,
                         use_face_contour=False).to(dev)
    for p in flame.parameters():
        p.requires_grad_(False)
    head = CkptAvatar(ck, flame, dev)

    sys.path.insert(0, os.environ.get("GS_DIR", ""))
    from gaussian_renderer import render
    from scene.cameras import Camera as GSCamera
    from types import SimpleNamespace
    pipe = SimpleNamespace(compute_cov3D_python=False,
                           convert_SHs_python=False, debug=False,
                           antialiasing=False)

    views = read_cameras(source_dir)
    view_by = {v["stem"]: v for v in views}
    align = json.loads((results_dir / "05_align/head_align.json").read_text())
    stems = align["persons"][pid]["frames"]
    bg = torch.zeros(3, device=dev)

    # 静态分支预先 cat（body+scene）
    def to_branch(t):
        return (t["xyz"], torch.cat([t["f_dc"].unsqueeze(1), t["f_rest"]], 1),
                t["opacity"], t["scaling"], t["rotation"])

    bs = [to_branch(body), to_branch(scene)]
    s_xyz = torch.cat([b[0] for b in bs]).to(dev)
    s_feat = torch.cat([b[1] for b in bs]).to(dev)
    s_opa = torch.cat([b[2] for b in bs]).to(dev)
    s_sc = torch.cat([b[3] for b in bs]).to(dev)
    s_rot = torch.cat([b[4] for b in bs]).to(dev)

    from PIL import Image
    psnrs = []
    worst = []
    for i, s in enumerate(stems):
        v = view_by.get(s)
        if v is None:
            continue
        img_path = images_dir / f"{s}.jpg"
        if not img_path.exists():
            img_path = images_dir / f"{s}.png"
        gt_pil = Image.open(img_path).convert("RGB").resize((v["W"], v["H"]))
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
        head.set_frame(i)
        h = head.tensors()
        m = MergedGS(torch.cat([s_xyz, h[0]]),
                     torch.cat([s_feat, h[1]]),
                     torch.cat([s_opa, h[2]]),
                     torch.cat([s_sc, h[3]]),
                     torch.cat([s_rot, h[4]]), 3)
        with torch.no_grad():
            pkg = render(cam, m, pipe, bg, 1.0)
        pred = pkg["render"].clamp(0, 1)
        mse = ((pred - gt) ** 2).mean()
        p = float(-10 * np.log10(mse.item()))
        psnrs.append((i, s, p))

    arr = np.array([p for _, _, p in psnrs])
    print(f"n={len(arr)}  mean={arr.mean():.2f}  median={np.median(arr):.2f}  "
          f"min={arr.min():.2f}  max={arr.max():.2f}")
    print(f"<10dB: {(arr < 10).sum()} 帧   <15dB: {(arr < 15).sum()} 帧   "
          f">=20dB: {(arr >= 20).sum()} 帧")
    worst = sorted(psnrs, key=lambda x: x[2])[:10]
    print("最差 10 帧:")
    for i, s, p in worst:
        print(f"  [{i:3d}] {s}  {p:.2f} dB")
    out = results_dir / "composite_psnr_all.json"
    out.write_text(json.dumps(
        [{"i": i, "stem": s, "psnr": p} for i, s, p in psnrs], indent=1))
    print("saved", out)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""对照实验：分别渲染 scene-only / scene+body / scene+body+head，定位帧 29 掉分来源。"""
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent))
from composite_check import load_gaussian_ply, MergedGS  # noqa: E402
from train_avatar import N_SHAPE, N_EXPR  # noqa: E402


def log(m):
    print(m, flush=True)


def main():
    results_dir = Path(os.environ.get("RESULTS_DIR", ""))
    upstream = Path(os.environ.get("UPSTREAM_DIR", ""))
    source_dir = os.environ.get("SOURCE_DIR", f"{upstream}/source")
    pid = os.environ.get("PID", "0")
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    body_ply = results_dir / "07_body_gs_src" / f"body_gs_p{pid}.ply"
    scene_ply = results_dir / "07_body_gs_src" / "scene_gs.ply"
    ckpt = results_dir / "08_train" / f"avatar_p{pid}_final.pth"

    body = load_gaussian_ply(body_ply)
    scene = load_gaussian_ply(scene_ply)

    # head 分支（复用 composite_check.CkptAvatar）
    from composite_check import CkptAvatar
    ck = torch.load(ckpt, map_location="cpu")
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

    from train_avatar import read_cameras
    views = read_cameras(source_dir)
    view_by = {v["stem"]: v for v in views}
    align = json.loads((results_dir / "05_align" / "head_align.json").read_text())
    stems = align["persons"][pid]["frames"]
    bg = torch.zeros(3, device=dev)

    def to_branch(t):
        feat = torch.cat([t["f_dc"].unsqueeze(1), t["f_rest"]], dim=1)
        return (t["xyz"], feat, t["opacity"], t["scaling"], t["rotation"],
                t["max_sh"])

    def render_psnr(cam, model, gt):
        with torch.no_grad():
            pkg = render(cam, model, pipe, bg, 1.0)
        pred = pkg["render"].clamp(0, 1)
        mse = ((pred - gt) ** 2).mean()
        return float(-10 * torch.log10(mse).item()), pred

    def mk_cam(i):
        s = stems[i]
        v = view_by[s]
        from PIL import Image
        img_dir = Path(os.environ.get(
            "IMAGES_DIR", f"{source_dir}/images"))
        img_path = img_dir / f"{s}.jpg"
        if not img_path.exists():
            img_path = img_dir / f"{s}.png"
        gt_pil = Image.open(img_path).convert("RGB").resize((v["W"], v["H"]))
        gt = torch.tensor(np.asarray(gt_pil), dtype=torch.float32,
                          device=dev) / 255.0
        cam = GSCamera(resolution=(v["W"], v["H"]), colmap_id=0,
                       R=v["R"].T, T=v["T"],
                       FoVx=2 * np.arctan(v["W"] / (2 * v["fx"])),
                       FoVy=2 * np.arctan(v["H"] / (2 * v["fy"])),
                       depth_params=None, invdepthmap=None,
                       image=gt_pil, image_name=s, uid=i,
                       data_device=dev)
        return cam, gt.permute(2, 0, 1), s

    out_dir = results_dir / "composite_ablation"
    out_dir.mkdir(exist_ok=True)
    from PIL import Image

    for i in (0, 29):
        cam, gt, s = mk_cam(i)
        results = {}

        # scene only
        m = MergedGS(scene["xyz"].to(dev),
                     torch.cat([scene["f_dc"].unsqueeze(1),
                                scene["f_rest"]], 1).to(dev),
                     scene["opacity"].to(dev), scene["scaling"].to(dev),
                     scene["rotation"].to(dev), 3)
        p, img = render_psnr(cam, m, gt)
        results["scene"] = p

        # scene+body
        bs = [to_branch(scene), to_branch(body)]
        xs = torch.cat([b[0] for b in bs]).to(dev)
        fs = torch.cat([b[1] for b in bs]).to(dev)
        os_ = torch.cat([b[2] for b in bs]).to(dev)
        ss = torch.cat([b[3] for b in bs]).to(dev)
        rs = torch.cat([b[4] for b in bs]).to(dev)
        m = MergedGS(xs, fs, os_, ss, rs, 3)
        p, img2 = render_psnr(cam, m, gt)
        results["scene+body"] = p

        # scene+body+head
        head.set_frame(i)
        h = head.tensors()
        xs = torch.cat([xs, h[0]])
        fs = torch.cat([fs, h[1]])
        os_ = torch.cat([os_, h[2]])
        ss = torch.cat([ss, h[3]])
        rs = torch.cat([rs, h[4]])
        m = MergedGS(xs, fs, os_, ss, rs, 3)
        p, img3 = render_psnr(cam, m, gt)
        results["full"] = p

        log(f"frame {i} ({s}):")
        for k, v_ in results.items():
            log(f"  {k:14s} {v_:.2f} dB")

        # 三列对比图: scene | scene+body | full
        combo = torch.cat([img, img2, img3], dim=-1)
        arr = (combo.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
        Image.fromarray(arr).save(out_dir / f"{i:03d}_{s}_abl.png")
    log(f"✅ {out_dir}")


if __name__ == "__main__":
    main()

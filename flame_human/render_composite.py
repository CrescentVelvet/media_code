#!/usr/bin/env python3
"""render_composite.py — 渲染三分支合成帧并落盘（09 enhance_post 的输入侧）。

与 composite_psnr_all.py 同一套加载/渲染逻辑（head ckpt + body ply + scene ply
cat 成 MergedGS 单次渲染，跨分支遮挡由深度排序正确处理），但**逐帧存 PNG**，
供后续 HYPIR 2D 后处理。

Env: RESULTS_DIR / BODY_PLY / SCENE_PLY / HEAD_CKPT / PID / SOURCE_DIR /
     IMAGES_DIR / OUT_DIR / FRAMES(可选,逗号分隔的帧号)
"""
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
    results_dir = Path(os.environ["RESULTS_DIR"])
    source_dir = os.environ.get("SOURCE_DIR", "")
    images_dir = Path(os.environ.get("IMAGES_DIR", f"{source_dir}/images"))
    pid = os.environ.get("PID", "0")
    out_dir = Path(os.environ.get("OUT_DIR", f"{results_dir}/09_render/raw"))
    out_dir.mkdir(parents=True, exist_ok=True)
    dev = torch.device("cuda")

    body = load_gaussian_ply(Path(os.environ.get(
        "BODY_PLY", results_dir / "07_body_gs_src" / f"body_gs_p{pid}.ply")))
    scene = load_gaussian_ply(Path(os.environ.get(
        "SCENE_PLY", results_dir / "08e_pruned" / f"scene_pruned_p{pid}.ply")))
    ck_path = Path(os.environ.get(
        "HEAD_CKPT", results_dir / "08e_pruned" / f"avatar_pruned_p{pid}.pth"))
    ck = torch.load(ck_path, map_location="cpu")

    import smplx
    flame = smplx.create(model_path=os.path.dirname(os.environ["FLAME_MODEL"]),
                         model_type="flame", num_betas=N_SHAPE,
                         num_expression_coeffs=N_EXPR,
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
    stems = json.loads(
        (results_dir / "05_align/head_align.json").read_text()
    )["persons"][pid]["frames"]
    sel = os.environ.get("FRAMES", "")
    idxs = ([int(x) for x in sel.split(",") if x.strip()] if sel
            else list(range(len(stems))))
    bg = torch.zeros(3, device=dev)

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
    print(f"🎬 渲染 {len(idxs)} 帧 → {out_dir}")
    for i in idxs:
        s = stems[i]
        v = view_by.get(s)
        if v is None:
            continue
        img_path = images_dir / f"{s}.jpg"
        if not img_path.exists():
            img_path = images_dir / f"{s}.png"
        gt_pil = Image.open(img_path).convert("RGB").resize((v["W"], v["H"]))
        cam = GSCamera(resolution=(v["W"], v["H"]), colmap_id=0,
                       R=v["R"].T, T=v["T"],
                       FoVx=2 * np.arctan(v["W"] / (2 * v["fx"])),
                       FoVy=2 * np.arctan(v["H"] / (2 * v["fy"])),
                       depth_params=None, invdepthmap=None,
                       image=gt_pil, image_name=s, uid=i, data_device=dev)
        head.set_frame(i)
        h = head.tensors()
        m = MergedGS(torch.cat([s_xyz, h[0].to(dev)]),
                     torch.cat([s_feat, h[1].to(dev)]),
                     torch.cat([s_opa, h[2].to(dev)]),
                     torch.cat([s_sc, h[3].to(dev)]),
                     torch.cat([s_rot, h[4].to(dev)]), 3)
        with torch.no_grad():
            pkg = render(cam, m, pipe, bg, 1.0)
        pred = pkg["render"].clamp(0, 1)
        arr = (pred.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
        Image.fromarray(arr).save(out_dir / f"{i:03d}_{s}.png")
        if (i + 1) % 10 == 0:
            print(f"  [{i+1}] {s}")
    print(f"✅ 落盘 {len(idxs)} 帧 → {out_dir}")


if __name__ == "__main__":
    main()

"""_debug_cmp_08e.py — 08e 三分支剪枝前后 composite 对比。

布局: GT | before(scene_ft + 原body/head) | after(08e 三分支剪枝) 三联。
帧: 15/29（大 yaw 段）+ 53（新暴露的尾部差帧）。
"""
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from composite_check import load_gaussian_ply, MergedGS, CkptAvatar  # noqa: E402
from train_avatar import N_SHAPE, N_EXPR, read_cameras  # noqa: E402


def main():
    results_dir = Path(os.environ["RESULTS_DIR"])
    source_dir = os.environ["SOURCE_DIR"]
    images_dir = Path(os.environ.get("IMAGES_DIR", f"{source_dir}/images"))
    pid = "0"
    dev = torch.device("cuda")

    cfg = {
        "before": (results_dir / "07_body_gs_src/body_gs_p0.ply",
                   results_dir / "08d_finetune_scene/scene_ft_p0.ply",
                   results_dir / "08_train/avatar_p0_final.pth"),
        "after": (results_dir / "07_body_gs_src/body_gs_p0.ply",
                  results_dir / "08e_pruned/scene_pruned_p0.ply",
                  results_dir / "08e_pruned/avatar_pruned_p0.pth"),
    }
    loaded = {}
    import smplx
    for tag, (bp, sp, hp) in cfg.items():
        flame = smplx.create(
            model_path=os.path.dirname(os.environ["FLAME_MODEL"]),
            model_type="flame", num_betas=N_SHAPE,
            num_expression_coeffs=N_EXPR,
            use_face_contour=False).to(dev)
        for p in flame.parameters():
            p.requires_grad_(False)
        ck = torch.load(hp, map_location="cpu")
        loaded[tag] = (load_gaussian_ply(bp), load_gaussian_ply(sp),
                       CkptAvatar(ck, flame, dev))

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
    bg = torch.zeros(3, device=dev)

    def to_branch(t):
        return (t["xyz"], torch.cat([t["f_dc"].unsqueeze(1), t["f_rest"]], 1),
                t["opacity"], t["scaling"], t["rotation"])

    from PIL import Image
    out_dir = results_dir / "composite_08e"
    out_dir.mkdir(exist_ok=True)
    for i in (15, 29, 53):
        s = stems[i]
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
        row = [gt]
        for tag in ("before", "after"):
            body, scene, head = loaded[tag]
            head.set_frame(i)
            h = head.tensors()
            bs = [to_branch(body), to_branch(scene)]
            m = MergedGS(
                torch.cat([torch.cat([b[0] for b in bs]).to(dev), h[0].to(dev)]),
                torch.cat([torch.cat([b[1] for b in bs]).to(dev), h[1].to(dev)]),
                torch.cat([torch.cat([b[2] for b in bs]).to(dev), h[2].to(dev)]),
                torch.cat([torch.cat([b[3] for b in bs]).to(dev), h[3].to(dev)]),
                torch.cat([torch.cat([b[4] for b in bs]).to(dev), h[4].to(dev)]), 3)
            with torch.no_grad():
                pkg = render(cam, m, pipe, bg, 1.0)
            pred = pkg["render"].clamp(0, 1)
            mse = ((pred - gt) ** 2).mean()
            print(f"  [{i}] {tag} PSNR={float(-10*np.log10(mse.item())):.2f}")
            row.append(pred)
        combo = torch.cat(row, dim=-1)
        arr = (combo.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
        Image.fromarray(arr).save(out_dir / f"cmp_{i:03d}_{s}.png")
    print("saved to", out_dir)


if __name__ == "__main__":
    main()

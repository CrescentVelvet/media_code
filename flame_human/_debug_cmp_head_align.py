"""_debug_cmp_head_align.py — 对齐修复前后 head 区域专项对比。

全图 PSNR 被背景稀释（head 只占小部分像素），故只统计 head 框内 PSNR，
并输出裁剪放大图（GT | 旧对齐head | 新对齐head）看锐度是否真提升。
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
from split_body_scene import head_box  # noqa: E402


def main():
    results_dir = Path(os.environ["RESULTS_DIR"])
    source_dir = os.environ["SOURCE_DIR"]
    images_dir = Path(os.environ.get("IMAGES_DIR", f"{source_dir}/images"))
    pid = "0"
    dev = torch.device("cuda")

    body = load_gaussian_ply(
        results_dir / "07_body_gs_src" / f"body_gs_p{pid}.ply")
    scene = load_gaussian_ply(results_dir / "08e_pruned/scene_pruned_p0.ply")

    import smplx
    # A/B 两个 head ckpt（env 可指定；默认 对齐修复版 vs 小scale版）
    ck_a = Path(os.environ.get(
        "CKPT_A", results_dir / "08e_pruned/avatar_pruned_align7px.pth"))
    ck_b = Path(os.environ.get(
        "CKPT_B", results_dir / "08e_pruned/avatar_pruned_scale05.pth"))
    lab_a = os.environ.get("LABEL_A", "A")
    lab_b = os.environ.get("LABEL_B", "B")
    heads = {}
    for tag, hp in ((lab_a, ck_a), (lab_b, ck_b)):
        flame = smplx.create(
            model_path=os.path.dirname(os.environ["FLAME_MODEL"]),
            model_type="flame", num_betas=N_SHAPE,
            num_expression_coeffs=N_EXPR, use_face_contour=False).to(dev)
        for p in flame.parameters():
            p.requires_grad_(False)
        heads[tag] = CkptAvatar(torch.load(hp, map_location="cpu"), flame, dev)
    tags = [lab_a, lab_b]

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

    # head 框（用当前 05 的 mesh 投影，作为裁剪与统计区域）
    lo, hi, _, _ = head_box(results_dir / "06_avatar_gs/avatar_mesh_p0.npz",
                            results_dir / "06_avatar_gs/avatar_bind_p0.npz",
                            0.35)

    def to_branch(t):
        return (t["xyz"], torch.cat([t["f_dc"].unsqueeze(1), t["f_rest"]], 1),
                t["opacity"], t["scaling"], t["rotation"])

    from PIL import Image
    out_dir = results_dir / "head_align_cmp"
    out_dir.mkdir(exist_ok=True)
    totals = {t: [] for t in tags}
    for i in (0, 15, 29):
        s = stems[i]
        v = view_by.get(s)
        img_path = images_dir / f"{s}.jpg"
        if not img_path.exists():
            img_path = images_dir / f"{s}.png"
        gt_pil = Image.open(img_path).convert("RGB").resize((v["W"], v["H"]))
        gt = torch.tensor(np.asarray(gt_pil), dtype=torch.float32,
                          device=dev).permute(2, 0, 1) / 255.0
        cam = GSCamera(resolution=(v["W"], v["H"]), colmap_id=0,
                       R=v["R"].T, T=v["T"],
                       FoVx=2 * np.arctan(v["W"] / (2 * v["fx"])),
                       FoVy=2 * np.arctan(v["H"] / (2 * v["fy"])),
                       depth_params=None, invdepthmap=None,
                       image=gt_pil, image_name=s, uid=i, data_device=dev)
        # 2D head 框
        from finetune_body import head_mask_2d
        hm = head_mask_2d(lo, hi, v, dilate=20)
        ys, xs = np.nonzero(hm > 0.5)
        y0, y1, x0, x1 = ys.min(), ys.max(), xs.min(), xs.max()

        row = [gt[:, y0:y1, x0:x1]]
        bs = [to_branch(body), to_branch(scene)]
        for tag in tags:
            head = heads[tag]
            head.set_frame(i)
            h = head.tensors()
            m = MergedGS(
                torch.cat([torch.cat([b[0] for b in bs]).to(dev)] + [h[0].to(dev)]),
                torch.cat([torch.cat([b[1] for b in bs]).to(dev)] + [h[1].to(dev)]),
                torch.cat([torch.cat([b[2] for b in bs]).to(dev)] + [h[2].to(dev)]),
                torch.cat([torch.cat([b[3] for b in bs]).to(dev)] + [h[3].to(dev)]),
                torch.cat([torch.cat([b[4] for b in bs]).to(dev)] + [h[4].to(dev)]), 3)
            with torch.no_grad():
                pred = render(cam, m, pipe, bg, 1.0)["render"].clamp(0, 1)
            crop = pred[:, y0:y1, x0:x1]
            mse = ((crop - row[0]) ** 2).mean().item()
            ps = -10 * np.log10(mse)
            totals[tag].append(ps)
            print(f"  [{i:2d}] head-crop PSNR {tag} = {ps:.2f} dB")
            row.append(crop)
        combo = torch.cat(row, dim=-1)
        arr = (combo.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
        Image.fromarray(arr).save(out_dir / f"hc_{i:03d}_{s}.png")
    for tag in tags:
        print(f"head-crop 平均 PSNR {tag} = {np.mean(totals[tag]):.2f} dB")
    print("saved to", out_dir)


if __name__ == "__main__":
    main()

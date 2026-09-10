"""_debug_branch_split.py — 分支分解渲染：定位人物周围黑雾来源。

对帧 29/26/15 渲染五联图: GT | scene-only | body-only | head-only | full。
人物区域若在某分支单独渲染时出现暗色团块 = 该分支是黑雾源。
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
    scene_ply = Path(os.environ.get(
        "SCENE_PLY",
        results_dir / "08d_finetune_scene/scene_ft_p0.ply"))
    pid = "0"
    dev = torch.device("cuda")

    body = load_gaussian_ply(
        results_dir / "07_body_gs_src" / f"body_gs_p{pid}.ply")
    scene = load_gaussian_ply(scene_ply)

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
    stems = json.loads(
        (results_dir / "05_align/head_align.json").read_text()
    )["persons"][pid]["frames"]
    bg = torch.zeros(3, device=dev)

    def to_branch(t):
        return (t["xyz"], torch.cat([t["f_dc"].unsqueeze(1), t["f_rest"]], 1),
                t["opacity"], t["scaling"], t["rotation"])

    from PIL import Image
    out_dir = results_dir / "composite_branch"
    out_dir.mkdir(exist_ok=True)
    for i in (29, 26, 15):
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
        head.set_frame(i)
        h = head.tensors()
        B = to_branch(body)
        S = to_branch(scene)

        def render_set(names, bgv):
            xs, fs, os_, ss, rs = [], [], [], [], []
            for nm in names:
                b = h if nm == "head" else (B if nm == "body" else S)
                xs.append(b[0].to(dev)); fs.append(b[1].to(dev))
                os_.append(b[2].to(dev)); ss.append(b[3].to(dev))
                rs.append(b[4].to(dev))
            m = MergedGS(torch.cat(xs), torch.cat(fs),
                         torch.cat(os_), torch.cat(ss),
                         torch.cat(rs), 3)
            with torch.no_grad():
                return render(cam, m, pipe, bgv, 1.0)["render"].clamp(0, 1)

        # 单分支用白底：暗雾若为真高斯覆盖则会显形（黑=背景=无高斯）
        wht = torch.ones(3, device=dev)
        row = [gt, render_set(["scene"], wht), render_set(["body"], wht),
               render_set(["head"], wht),
               render_set(["body", "scene", "head"], bg)]
        combo = torch.cat(row, dim=-1)
        arr = (combo.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
        Image.fromarray(arr).save(out_dir / f"br_{i:03d}_{s}.png")
        print(f"  [{i}] saved br_{i:03d}_{s}.png")
    print("saved to", out_dir)


if __name__ == "__main__":
    main()

"""_debug_head_free.py — head 分支 bound / free 高斯分别渲染什么？

背景：train_avatar.py:181 deform() 里
  p_bound = (V[self.tri] * bary).sum(1)     # 绑定点：贴 FLAME 网格表面
  p_can[is_free] += self._free_can          # 自由点：+ 全局固定偏移
且 __init__ 里 self.tri[is_free] = 0（自由点不用 tri）→ 自由点的 anchor
恒为顶点 V[0] + free_can。若 free_can≈0，则 20000 个自由点全堆在 V[0]。
本脚本白底分别渲染 bound-only / free-only / all，验证头发帽子归属。
"""
import os
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from composite_check import MergedGS, CkptAvatar  # noqa: E402
from train_avatar import N_SHAPE, N_EXPR, read_cameras  # noqa: E402


def main():
    results_dir = Path(os.environ["RESULTS_DIR"])
    source_dir = os.environ["SOURCE_DIR"]
    pid = "0"
    dev = torch.device("cuda")
    ck = torch.load(results_dir / "08_train" / f"avatar_p{pid}_final.pth",
                    map_location="cpu")
    isf = ck["is_free"].bool().to(dev)
    print(f"bound={int((~isf).sum()):,}  free={int(isf.sum()):,}")

    import smplx
    flame = smplx.create(model_path=os.path.dirname(os.environ["FLAME_MODEL"]),
                         model_type="flame", num_betas=N_SHAPE,
                         num_expression_coeffs=N_EXPR,
                         use_face_contour=False).to(dev)
    for p in flame.parameters():
        p.requires_grad_(False)
    head = CkptAvatar(ck, flame, dev)

    # free 点世界坐标统计（验证是否堆在一点）
    head.set_frame(0)
    xyz0 = head.inner.get_xyz.detach()
    f = xyz0[isf]
    print("free xyz: mean=%s  std=%s" % (np.round(f.mean(0).cpu().numpy(), 4),
                                         np.round(f.std(0).cpu().numpy(), 5)))
    b = xyz0[~isf]
    print("bound xyz: std=%s" % np.round(b.std(0).cpu().numpy(), 4))

    sys.path.insert(0, os.environ.get("GS_DIR", ""))
    from gaussian_renderer import render
    from scene.cameras import Camera as GSCamera
    from types import SimpleNamespace
    pipe = SimpleNamespace(compute_cov3D_python=False,
                           convert_SHs_python=False, debug=False,
                           antialiasing=False)

    views = read_cameras(source_dir)
    view_by = {v["stem"]: v for v in views}
    import json
    stems = json.loads(
        (results_dir / "05_align/head_align.json").read_text()
    )["persons"][pid]["frames"]

    from PIL import Image
    out_dir = results_dir / "head_free_check"
    out_dir.mkdir(exist_ok=True)
    wht = torch.ones(3, device=dev)
    for i in (29, 15):
        s = stems[i]
        v = view_by.get(s)
        gt_pil = Image.open(Path(source_dir) / "images" / f"{s}.jpg").convert(
            "RGB").resize((v["W"], v["H"]))
        gt = torch.tensor(np.asarray(gt_pil), dtype=torch.float32,
                          device=dev).permute(2, 0, 1) / 255.0
        cam = GSCamera(resolution=(v["W"], v["H"]), colmap_id=0,
                       R=v["R"].T, T=v["T"],
                       FoVx=2 * np.arctan(v["W"] / (2 * v["fx"])),
                       FoVy=2 * np.arctan(v["H"] / (2 * v["fy"])),
                       depth_params=None, invdepthmap=None,
                       image=gt_pil, image_name=s, uid=i, data_device=dev)
        head.set_frame(i)
        xyz, feat, opa, sc, rot = head.tensors()
        row = [gt]
        for tag, m in (("bound", ~isf), ("free", isf)):
            mg = MergedGS(xyz[m], feat[m], opa[m], sc[m], rot[m], 3)
            with torch.no_grad():
                row.append(render(cam, mg, pipe, wht, 1.0)["render"].clamp(0, 1))
        combo = torch.cat(row, dim=-1)
        arr = (combo.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
        Image.fromarray(arr).save(out_dir / f"hf_{i:03d}.png")
        print(f"  saved hf_{i:03d}.png")
    print("saved to", out_dir)


if __name__ == "__main__":
    main()

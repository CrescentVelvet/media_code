"""渲染验证：加载 08 训练产物，渲染若干帧与 GT 对比拼图。

用法（在 flame_human conda 环境里）:
    python render_check.py
环境变量同 08_train.sh（RESULTS_DIR / AVATAR_PLY / BIND_NPZ / ALIGN_JSON /
PID / SOURCE_DIR / IMAGES_DIR / GS_DIR / FLAME_MODEL）。
输出: $RESULTS_DIR/08_train/render_check/*.png
"""
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent))
from train_avatar import (AvatarModel, read_cameras, quat_to_mat,  # noqa: E402
                          N_SHAPE, N_EXPR)


def load_model(ckpt_path, flame, ap, dev):
    """从 checkpoint 重建 AvatarModel（不走 PLY，直接用 ckpt 张量）。"""
    import train_avatar as ta

    class CkptAvatar(ta.AvatarModel):
        def __init__(self, ck, flame, ap, dev):
            # 不调 super().__init__（那要 PLY）；手动重建同构属性
            self.device = dev
            self.flame = flame
            self.id_coeff = torch.nn.Parameter(
                ck["id_coeff"].to(dev), requires_grad=False)
            self.exp = torch.nn.Parameter(
                ck["exp"].to(dev), requires_grad=False)
            self.local_q = torch.nn.Parameter(
                ck["local_q"].to(dev), requires_grad=False)
            self.local_t = torch.nn.Parameter(
                ck["local_t"].to(dev), requires_grad=False)
            self.global_q = torch.nn.Parameter(
                ck["global_q"].to(dev), requires_grad=False)
            self.global_t = torch.nn.Parameter(
                ck["global_t"].to(dev), requires_grad=False)
            self.log_s = torch.nn.Parameter(
                ck["log_s"].to(dev), requires_grad=False)
            self._bary_raw = torch.nn.Parameter(
                ck["bary_raw"].to(dev), requires_grad=False)
            self._free_can = torch.nn.Parameter(
                ck["free_can"].to(dev), requires_grad=False)
            self._opacity = torch.nn.Parameter(
                ck["opacity"].to(dev), requires_grad=False)
            self._scaling = torch.nn.Parameter(
                ck["scaling"].to(dev), requires_grad=False)
            self._rotation = torch.nn.Parameter(
                ck["rotation"].to(dev), requires_grad=False)
            self._features_dc = torch.nn.Parameter(
                ck["features_dc"].to(dev), requires_grad=False)
            self._features_rest = torch.nn.Parameter(
                ck["features_rest"].to(dev), requires_grad=False)
            self.tri = ck["tri"].to(dev)
            self.is_free = ck["is_free"].to(dev)
            self.max_sh = int(round(
                (ck["features_rest"].shape[1] + 1) ** 0.5)) - 1
            self.active_sh = self.max_sh
            self.grad_accum = torch.zeros(len(self._opacity), device=dev)
            self.denom = torch.zeros(len(self._opacity), device=dev)
            self._frame = 0

    return CkptAvatar(ckpt_path, flame, ap, dev)


def main():
    results_dir = os.environ.get("RESULTS_DIR", "")
    ckpt = Path(os.environ.get(
        "CKPT", f"{results_dir}/08_train/avatar_p{os.environ.get('PID', '0')}_final.pth"))
    align_json = Path(os.environ.get(
        "ALIGN_JSON", f"{results_dir}/05_align/head_align.json"))
    out_dir = Path(os.environ.get("OUT_DIR", f"{results_dir}/08_train/render_check"))
    pid = os.environ.get("PID", "0")
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"🖼️  render_check  ckpt={ckpt.name}  device={dev}")
    ck = torch.load(ckpt, map_location="cpu")
    print(f"  高斯数={len(ck['opacity']):,}  帧数={ck['exp'].shape[0]}  "
          f"SH={int(round((ck['features_rest'].shape[1] + 1) ** 0.5)) - 1}")

    import smplx
    flame = smplx.create(model_path=os.path.dirname(os.environ["FLAME_MODEL"]),
                         model_type="flame",
                         num_betas=N_SHAPE, num_expression_coeffs=N_EXPR,
                         use_face_contour=False).to(dev)
    for p in flame.parameters():
        p.requires_grad_(False)

    align = json.loads(align_json.read_text())
    ap = align["persons"][pid]

    model = load_model(ck, flame, ap, dev)

    sys.path.insert(0, os.environ.get("GS_DIR", ""))
    from gaussian_renderer import render
    from scene.cameras import Camera as GSCamera
    from types import SimpleNamespace
    pipe = SimpleNamespace(compute_cov3D_python=False,
                           convert_SHs_python=False, debug=False,
                           antialiasing=False)

    views = read_cameras(os.environ.get("SOURCE_DIR", ""))
    view_by = {v["stem"]: v for v in views}
    stems = ap["frames"]
    n = len(stems)
    # 首帧 / 1/4 / 代表帧(中) / 3/4 / 尾帧，共 5 帧
    picks = sorted({0, n // 4, n // 2, 3 * n // 4, n - 1})
    bg = torch.zeros(3, device=dev)

    out_dir.mkdir(parents=True, exist_ok=True)
    imgs_dir = Path(os.environ.get("IMAGES_DIR", ""))
    psnrs = []
    for i in picks:
        s = stems[i]
        v = view_by.get(s)
        if v is None:
            print(f"  ⚠️ {s} 无相机，跳过")
            continue
        model.set_frame(i)
        img_path = imgs_dir / f"{s}.jpg"
        if not img_path.exists():
            img_path = imgs_dir / f"{s}.png"
        from PIL import Image
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
        with torch.no_grad():
            pkg = render(cam, model, pipe, bg, 1.0)
        pred = pkg["render"].clamp(0, 1)
        mse = ((pred - gt) ** 2).mean()
        psnr = float(-10 * torch.log10(mse).item())
        psnrs.append(psnr)
        # 拼图: GT | 渲染 | |diff|×5
        diff = ((pred - gt).abs() * 5).clamp(0, 1)
        combo = torch.cat([gt, pred, diff], dim=-1)  # (3,H,3W)
        arr = (combo.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
        Image.fromarray(arr).save(out_dir / f"{i:03d}_{s}.png")
        print(f"  [{i:3d}] {s}  PSNR={psnr:.2f} dB")

    if psnrs:
        print(f"\n✅ 均值 PSNR={np.mean(psnrs):.2f} dB  "
              f"({len(psnrs)} 帧, 输出在 {out_dir})")


if __name__ == "__main__":
    main()

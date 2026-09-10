#!/usr/bin/env python3
"""composite_check.py — head+body+scene 合成渲染验证。

三分支来源：
  head  = 08 ckpt（AvatarModel，FLAME 变形，per-frame）
  body  = 07_body_gs_src/body_gs_p{pid}.ply（07a 投票提取，自带外观）
  scene = 07_body_gs_src/scene_gs.ply（其余全部点）

三者同处 SfM 世界系（COLMAP）。渲染方式：三分支的 xyz/features/opacity/
scaling/rotation 激活后 cat 成一个 merged model 单次 render——跨分支的
遮挡关系由同一次渲染的深度排序正确处理（逐分支渲染再 alpha 合成会把
"头在身体后"的视角做错）。

SH 阶数不一致时截断到 min(head, body, scene)。

输出 $RESULTS_DIR/composite/*.png 三联拼图（GT | 合成 | 5×diff）。

Env: RESULTS_DIR / UPSTREAM_DIR / SOURCE_DIR / IMAGES_DIR / PID /
     GS_DIR / FLAME_MODEL / CKPT / ALIGN_JSON / BODY_PLY / SCENE_PLY
"""
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent))
from train_avatar import N_SHAPE, N_EXPR  # noqa: E402


def log(m):
    print(m, flush=True)


def load_gaussian_ply(path):
    """3DGS ply → 已激活的渲染张量（xyz/f_dc/f_rest/opacity/scaling/rotation）。

    f_rest 布局按官方 restore：np (N, 3*K) → view(N,3,K) → transpose(1,2)
    → (N,K,3)；与 train_avatar 的 (N,K,3) 一致，可 cat。
    """
    from plyfile import PlyData
    v = PlyData.read(str(path))["vertex"]
    d = {p.name: np.asarray(v[p.name]) for p in v.properties}
    xyz = np.stack([d["x"], d["y"], d["z"]], 1).astype(np.float32)
    f_dc = np.stack([d[f"f_dc_{i}"] for i in range(3)], 1).astype(np.float32)
    n_rest = sum(1 for k in d if k.startswith("f_rest_"))
    K = n_rest // 3
    f_rest = (np.stack([d[f"f_rest_{i}"] for i in range(n_rest)], 1)
              .reshape(-1, 3, K).transpose(0, 2, 1).astype(np.float32))
    opa = torch.sigmoid(torch.tensor(d["opacity"]).float()).cpu()
    # 官方 ply 存 log(scale)，激活是 exp（不是 softplus！）
    sc = torch.exp(torch.tensor(
        np.stack([d[f"scale_{i}"] for i in range(3)], 1)).float())
    rot = F.normalize(torch.tensor(
        np.stack([d[f"rot_{i}"] for i in range(4)], 1)).float(), dim=-1)
    return dict(
        xyz=torch.tensor(xyz),
        f_dc=torch.tensor(f_dc), f_rest=torch.tensor(f_rest),
        opacity=opa.unsqueeze(-1), scaling=sc, rotation=rot,
        max_sh=int(round(K ** 0.5)) - 1)


class CkptAvatar:
    """head 分支：从 08 ckpt 重建（接口同 render_check.load_model）。"""

    def __init__(self, ck, flame, dev):
        import train_avatar as ta

        class _A(ta.AvatarModel):
            def __init__(self):
                self.device = dev
                self.flame = flame
                for k_src, k_attr in (
                        ("id_coeff", "id_coeff"), ("exp", "exp"),
                        ("local_q", "local_q"), ("local_t", "local_t"),
                        ("global_q", "global_q"), ("global_t", "global_t"),
                        ("log_s", "log_s"), ("bary_raw", "_bary_raw"),
                        ("free_can", "_free_can"), ("opacity", "_opacity"),
                        ("scaling", "_scaling"), ("rotation", "_rotation"),
                        ("features_dc", "_features_dc"),
                        ("features_rest", "_features_rest")):
                    setattr(self, k_attr,
                            ck[k_src].to(dev).clone().requires_grad_(False))
                self.tri = ck["tri"].to(dev)
                self.is_free = ck["is_free"].to(dev)
                self.max_sh = int(round(
                    (ck["features_rest"].shape[1] + 1) ** 0.5)) - 1
                self.active_sh = self.max_sh
                self.grad_accum = torch.zeros(len(self._opacity), device=dev)
                self.denom = torch.zeros(len(self._opacity), device=dev)
                self._frame = 0

        self.inner = _A()

    @property
    def max_sh(self):
        return self.inner.max_sh

    def set_frame(self, f):
        self.inner._frame = f

    def tensors(self):
        """→ 已激活的 (xyz, features(K,3), opacity, scaling, rotation)。"""
        m = self.inner
        return (m.get_xyz.detach(),
                m.get_features.detach(), m.get_opacity.detach(),
                m.get_scaling.detach(), m.get_rotation.detach())


class MergedGS:
    """cat 后的单次渲染模型（全部已激活）。"""

    def __init__(self, xyz, features, opacity, scaling, rotation, sh):
        self._xyz, self._feat = xyz, features
        self._opa, self._sc, self._rot = opacity, scaling, rotation
        self.active_sh_degree = sh

    @property
    def get_xyz(self):
        return self._xyz

    @property
    def get_features(self):
        return self._feat

    @property
    def get_opacity(self):
        return self._opa

    @property
    def get_scaling(self):
        return self._sc

    @property
    def get_rotation(self):
        return self._rot


def cat_branches(branches, sh_min, dev):
    """branches: list of (xyz, features, opacity, scaling, rotation, sh)。

    SH 截断到 sh_min（features (N,K,3) → 取前 (sh_min+1)^2 列）。
    """
    K_min = (sh_min + 1) ** 2
    xs, fs, os_, ss, rs = [], [], [], [], []
    for xyz, feat, opa, sc, rot, sh in branches:
        feat = feat[:, :K_min].to(dev)
        # DC + rest 布局: train_avatar features 已是 (N,K,3) 完整 cat
        xs.append(xyz.to(dev))
        fs.append(feat)
        os_.append(opa.to(dev))
        ss.append(sc.to(dev))
        rs.append(rot.to(dev))
    return MergedGS(torch.cat(xs), torch.cat(fs), torch.cat(os_),
                    torch.cat(ss), torch.cat(rs), sh_min)


def main():
    results_dir = Path(os.environ.get("RESULTS_DIR", ""))
    upstream = Path(os.environ.get("UPSTREAM_DIR", ""))
    source_dir = os.environ.get("SOURCE_DIR", f"{upstream}/source")
    images_dir = Path(os.environ.get("IMAGES_DIR", f"{source_dir}/images"))
    pid = os.environ.get("PID", "0")
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt = Path(os.environ.get(
        "CKPT", f"{results_dir}/08_train/avatar_p{pid}_final.pth"))
    align_json = Path(os.environ.get(
        "ALIGN_JSON", f"{results_dir}/05_align/head_align.json"))
    body_ply = Path(os.environ.get(
        "BODY_PLY", f"{results_dir}/07_body_gs_src/body_gs_p{pid}.ply"))
    scene_ply = Path(os.environ.get(
        "SCENE_PLY", f"{results_dir}/07_body_gs_src/scene_gs.ply"))
    out_dir = Path(os.environ.get("OUT_DIR", f"{results_dir}/composite"))

    log(f"🧩 [composite] head={ckpt.name} body={body_ply.name} "
        f"scene={scene_ply.name} device={dev}")

    # ── head 分支 ─────────────────────────────────────────────────────
    ck = torch.load(ckpt, map_location="cpu")
    import smplx
    flame = smplx.create(model_path=os.path.dirname(os.environ["FLAME_MODEL"]),
                         model_type="flame",
                         num_betas=N_SHAPE, num_expression_coeffs=N_EXPR,
                         use_face_contour=False).to(dev)
    for p in flame.parameters():
        p.requires_grad_(False)
    head = CkptAvatar(ck, flame, dev)
    log(f"  head : {len(ck['opacity']):,} 高斯, SH{head.max_sh}, "
        f"{ck['exp'].shape[0]} 帧")

    # ── body / scene 分支（静态）──────────────────────────────────────
    body = load_gaussian_ply(body_ply) if body_ply.exists() else None
    scene = load_gaussian_ply(scene_ply) if scene_ply.exists() else None
    if body:
        log(f"  body : {len(body['xyz']):,} 高斯, SH{body['max_sh']}")
    if scene:
        log(f"  scene: {len(scene['xyz']):,} 高斯, SH{scene['max_sh']}")

    # ── 相机 ─────────────────────────────────────────────────────────
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
    align = json.loads(align_json.read_text())
    ap = align["persons"][pid]
    stems = ap["frames"]
    n = len(stems)
    picks = sorted({0, n // 4, n // 2, 3 * n // 4, n - 1})
    bg = torch.zeros(3, device=dev)

    shs = [head.max_sh] + [b["max_sh"] for b in (body, scene) if b]
    sh_min = min(shs)
    if sh_min < max(shs):
        log(f"  ⚠️ SH 阶数不一致 {shs}，截断到 SH{sh_min}")

    out_dir.mkdir(parents=True, exist_ok=True)
    psnrs = []
    for i in picks:
        s = stems[i]
        v = view_by.get(s)
        if v is None:
            continue
        img_path = images_dir / f"{s}.jpg"
        if not img_path.exists():
            img_path = images_dir / f"{s}.png"
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

        head.set_frame(i)
        h_xyz, h_feat, h_opa, h_sc, h_rot = head.tensors()
        branches = [(h_xyz, h_feat, h_opa, h_sc, h_rot, head.max_sh)]
        for b in (body, scene):
            if b is None:
                continue
            branches.append((b["xyz"].to(dev), b["f_dc"], None,
                             b["scaling"], b["rotation"], b["max_sh"]))
            # body/scene 的 features 需补 DC 维度拼成 (N,K,3)
            b["features"] = torch.cat(
                [b["f_dc"].unsqueeze(1), b["f_rest"]], dim=1).to(dev)
            branches[-1] = (b["xyz"].to(dev), b["features"],
                            b["opacity"].to(dev), b["scaling"], b["rotation"],
                            b["max_sh"])
        merged = cat_branches(branches, sh_min, dev)

        with torch.no_grad():
            pkg = render(cam, merged, pipe, bg, 1.0)
        pred = pkg["render"].clamp(0, 1)
        mse = ((pred - gt) ** 2).mean()
        psnr = float(-10 * torch.log10(mse).item())
        psnrs.append(psnr)
        diff = ((pred - gt).abs() * 5).clamp(0, 1)
        combo = torch.cat([gt, pred, diff], dim=-1)
        arr = (combo.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
        Image.fromarray(arr).save(out_dir / f"{i:03d}_{s}.png")
        log(f"  [{i:3d}] {s}  PSNR={psnr:.2f} dB")

    if psnrs:
        log(f"\n✅ 均值 PSNR={np.mean(psnrs):.2f} dB  "
            f"({len(psnrs)} 帧, 输出在 {out_dir})")


if __name__ == "__main__":
    main()

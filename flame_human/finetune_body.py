#!/usr/bin/env python3
"""finetune_body.py — 阶段 08b：body 高斯在 59 帧上 finetune。

背景（composite 诊断结论，2026-09-10）：
  body 点云来自上游 model_3dgs_nn 7k 的多视角投票提取——本质是**静态**
  高斯，而人在 59 帧里有动态姿态变化（手臂/肩膀位移）。且 nn 模型在
  大 yaw 段（帧 24-33，相机 yaw -31°~-42°）欠拟合，投票提取继承了这个
  偏差。落盘渲染是旧版模型产物，掩盖了这一点。

做法（参考 vggt_human train_face_finetune 的 region 监督思路）：
  1. 全属性自由 3DGS 优化（xyz/scale/rot/opacity/SH）——body 没有参数化
     绑定，动态姿态靠自由位置吸收（59 帧全视角联合优化，位置只能收敛
     到「多帧平均」，但至少能修 yaw 欠拟合带来的外观/形状偏差）；
  2. person mask 区域监督（p0 alpha）——场景区域不参与 loss，防止把
     body 往背景拉；
  3. densify（官方标准流程，grad 屏幕空间阈值 + opacity prune）；
  4. **mask 外亮度惩罚**（LAMBDA_OUTSIDE，首轮负结果后补）：body-only
     渲染在 mask 外本应全黑，惩罚泄漏亮度把膨胀出 mask 的高斯压回。
     首轮无此项时 composite 17.51→14.72 变差——区域监督对「长到 mask
     外的高斯」梯度为零（监督盲区），densify 无约束膨胀污染背景。

与 08 train_avatar 的区别：无 FLAME 绑定、无表情驱动，就是标准 3DGS
微调，只是 loss 限 mask 区域。

Env: BODY_PLY / MASKS_DIR / ALIGN_JSON / SOURCE_DIR / IMAGES_DIR /
     OUT_DIR / PID / EPOCHS / LR_* / DENSIFY_* / LAMBDA_DSSIM
"""
import os
import sys
import json
import time
import random
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from colmap_io import read_cameras  # noqa: E402
from composite_check import load_gaussian_ply  # noqa: E402


def log(m):
    print(m, flush=True)


def head_mask_2d(head_lo, head_hi, v, dilate=8):
    """3D head_box 投影到帧 v 的图像空间 → 2D 头部排除 mask（float 0/1）。

    投影 8 个角点取 2D 包围矩形（再外扩 dilate px）。头部由 head 分支
    渲染，body 监督必须排除它，否则 body 被 mask 内头部 GT 拉着往头部
    生长 → composite 头部重影（08b 两轮负结果的根因）。
    """
    from colmap_io import proj_matrix
    P, _ = proj_matrix(v)
    xs = [head_lo[0], head_hi[0]]
    ys = [head_lo[1], head_hi[1]]
    zs = [head_lo[2], head_hi[2]]
    corners = np.array([[x, y, z, 1.0]
                        for x in xs for y in ys for z in zs]).T   # (4,8)
    proj = P @ corners                                            # (3,8)
    u = proj[0] / np.clip(proj[2], 1e-6, None)
    w = proj[1] / np.clip(proj[2], 1e-6, None)
    front = proj[2] > 0
    if not front.any():
        return np.zeros((v["H"], v["W"]), dtype=np.float32)
    u0 = max(0, int(np.floor(u[front].min())) - dilate)
    u1 = min(v["W"], int(np.ceil(u[front].max())) + dilate)
    w0 = max(0, int(np.floor(w[front].min())) - dilate)
    w1 = min(v["H"], int(np.ceil(w[front].max())) + dilate)
    m = np.zeros((v["H"], v["W"]), dtype=np.float32)
    m[w0:w1, u0:u1] = 1.0
    return m


class BodyGS:
    """自由 3DGS（无绑定），属性直存 log/raw 空间，激活同官方。"""

    def __init__(self, ply_path, device):
        b = load_gaussian_ply(ply_path)
        self.device = device
        # 存 raw 参数（激活的逆）：xyz 原值、scale 取 log、opacity logit
        self._xyz = torch.nn.Parameter(b["xyz"].to(device).float())
        self._scaling = torch.nn.Parameter(
            torch.log(b["scaling"].to(device).clamp_min(1e-9)))
        self._rotation = torch.nn.Parameter(
            b["rotation"].to(device).float())
        self._opacity = torch.nn.Parameter(
            torch.logit(b["opacity"].to(device).clamp(1e-4, 1 - 1e-4)))
        self._features = torch.nn.Parameter(torch.cat(
            [b["f_dc"].unsqueeze(1), b["f_rest"]], dim=1
        ).to(device).float())                      # (N,K,3) DC+rest
        self.max_sh = b["max_sh"]
        self.active_sh = 0
        # densify 统计
        self.grad_accum = torch.zeros(len(self._xyz), device=device)
        self.denom = torch.zeros(len(self._xyz), device=device)

    @property
    def get_xyz(self):
        return self._xyz

    @property
    def get_opacity(self):
        return torch.sigmoid(self._opacity)

    @property
    def get_scaling(self):
        return torch.exp(self._scaling)

    @property
    def get_rotation(self):
        return torch.nn.functional.normalize(self._rotation, dim=-1)

    @property
    def get_features(self):
        return self._features

    @property
    def active_sh_degree(self):
        return self.active_sh

    # ── densify（自由高斯：无绑定表，官方标准 clone/split/prune）─────
    @torch.no_grad()
    def densify(self, grad_thresh, min_opacity, extent, scale_factor=0.2):
        grads = self.grad_accum / self.denom.clamp(min=1)
        grads[~torch.isfinite(grads)] = 0.0
        smax = self.get_scaling.max(1).values
        op = self.get_opacity.squeeze(-1)
        prune = op < min_opacity
        clone_idx = torch.where((grads > grad_thresh) &
                                (smax <= scale_factor * extent))[0]
        split_idx = torch.where((grads > grad_thresh) &
                                (smax > scale_factor * extent))[0]
        clone_idx = clone_idx[~prune[clone_idx]]
        split_idx = split_idx[~prune[split_idx]]
        keep = ~prune

        new_src = torch.cat([clone_idx, split_idx], 0)
        is_split = torch.zeros(len(new_src), dtype=torch.bool,
                               device=self.device)
        is_split[len(clone_idx):] = True

        def rebuild(p, extra):
            if extra is None or len(extra) == 0:
                return torch.nn.Parameter(p[keep].clone())
            return torch.nn.Parameter(torch.cat([p[keep], extra], 0))

        # 位置：clone 继承，split 沿父点方向加扰动（官方用 N(0,1)*scale）
        pos_new = self._xyz[new_src].clone()
        if is_split.any():
            sc_p = self.get_scaling.detach()[new_src]
            jitter = torch.randn_like(pos_new) * sc_p
            pos_new = torch.where(is_split.unsqueeze(-1),
                                  pos_new + jitter, pos_new)
        self._xyz = rebuild(self._xyz, pos_new)
        self._opacity = rebuild(self._opacity,
                                self._opacity[new_src].clone())
        # split 的 scale 缩小 1.6（官方）
        sc_new = self._scaling[new_src].clone()
        sc_new[is_split] = sc_new[is_split] - np.log(1.6)
        self._scaling = rebuild(self._scaling, sc_new)
        self._rotation = rebuild(self._rotation,
                                 self._rotation[new_src].clone())
        self._features = rebuild(self._features,
                                 self._features[new_src].clone())

        self.grad_accum = torch.zeros(len(self._xyz), device=self.device)
        self.denom = torch.zeros(len(self._xyz), device=self.device)
        return int(keep.sum()), int(len(new_src))

    def save(self, path):
        from plyfile import PlyData, PlyElement
        n = len(self._xyz)
        xyz = self._xyz.detach().cpu().numpy()
        K = self._features.shape[1]
        f_dc = self._features[:, :1].detach().cpu().numpy()      # (N,1,3)
        f_rest = self._features[:, 1:].detach().cpu().numpy()    # (N,K-1,3)
        # 官方 ply 布局：f_rest (N,3,K-1)（存 (N, 3*(K-1)) 展平）
        f_rest = f_rest.transpose(0, 2, 1).reshape(n, -1)
        props = [("x", "f4"), ("y", "f4"), ("z", "f4"),
                 ("nx", "f4"), ("ny", "f4"), ("nz", "f4")]
        arr = np.empty(n, dtype=props + [
            (f"f_dc_{i}", "f4") for i in range(3)] +
            [(f"f_rest_{i}", "f4") for i in range(f_rest.shape[1])] +
            [("opacity", "f4")] +
            [(f"scale_{i}", "f4") for i in range(3)] +
            [(f"rot_{i}", "f4") for i in range(4)])
        arr["x"], arr["y"], arr["z"] = xyz.T
        arr["nx"] = arr["ny"] = arr["nz"] = 0.0
        arr["f_dc_0"], arr["f_dc_1"], arr["f_dc_2"] = \
            f_dc[:, 0, :].T
        for i in range(f_rest.shape[1]):
            arr[f"f_rest_{i}"] = f_rest[:, i]
        arr["opacity"] = self._opacity.detach().cpu().numpy().ravel()
        sc = self._scaling.detach().cpu().numpy()
        arr["scale_0"], arr["scale_1"], arr["scale_2"] = sc.T
        rot = self.get_rotation.detach().cpu().numpy()
        arr["rot_0"], arr["rot_1"], arr["rot_2"], arr["rot_3"] = rot.T
        PlyData([PlyElement.describe(arr, "vertex")],
                text=False).write(str(path))


def main():
    results_dir = Path(os.environ.get("RESULTS_DIR", ""))
    body_ply = Path(os.environ.get(
        "BODY_PLY", f"{results_dir}/07_body_gs/body_p00.ply"))
    masks_dir = Path(os.environ.get(
        "MASKS_DIR", f"{results_dir}/01c_sam3_person_masks"))
    align_json = Path(os.environ.get(
        "ALIGN_JSON", f"{results_dir}/05_align/head_align.json"))
    source_dir = os.environ.get("SOURCE_DIR", "")
    images_dir = os.environ.get("IMAGES_DIR", f"{source_dir}/images")
    out_dir = Path(os.environ.get(
        "OUT_DIR", f"{results_dir}/08b_finetune_body"))
    pid = os.environ.get("PID", "0")
    epochs = int(os.environ.get("EPOCHS", "60"))
    lr = float(os.environ.get("LR", "1e-3"))
    densify_from = int(os.environ.get("DENSIFY_FROM", "5"))
    densify_until = int(os.environ.get("DENSIFY_UNTIL", "40"))
    densify_every = int(os.environ.get("DENSIFY_EVERY", "5"))
    grad_thresh = float(os.environ.get("DENSIFY_GRAD", "2e-4"))
    min_opacity = float(os.environ.get("MIN_OPACITY", "0.005"))
    lam_dssim = float(os.environ.get("LAMBDA_DSSIM", "0.2"))
    lam_outside = float(os.environ.get("LAMBDA_OUTSIDE", "0.5"))
    sh_every = int(os.environ.get("SH_EVERY", "10"))
    seed = int(os.environ.get("SEED", "0"))

    if not body_ply.exists():
        sys.exit(f"❌ 缺少 body ply: {body_ply}")
    if not align_json.exists():
        sys.exit(f"❌ 缺少 05 对齐结果: {align_json}")

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir.mkdir(parents=True, exist_ok=True)
    log(f"🏋️  [08b] body finetune  device={dev}")
    log(f"  📦 {body_ply.name}")

    model = BodyGS(body_ply, dev)
    log(f"  🔢 {len(model._xyz):,} 高斯, SH{model.max_sh}")

    sys.path.insert(0, os.environ.get("GS_DIR", ""))
    try:
        from gaussian_renderer import render
        from utils.loss_utils import l1_loss, ssim
        from scene.cameras import Camera as GSCamera
    except Exception as e:
        sys.exit(f"❌ 无法导入 gaussian-splatting: {e}")
    from types import SimpleNamespace
    pipe = SimpleNamespace(compute_cov3D_python=False,
                           convert_SHs_python=False, debug=False,
                           antialiasing=False)

    views = read_cameras(source_dir)
    log(f"  📷 {len(views)} views")

    # person mask（p0 alpha）——body 监督区域
    from PIL import Image
    pid2 = pid.zfill(2)

    # 头盒（06 头网格，07a 同款）：头部由 head 分支渲染，body 监督必须
    # 排除，否则 body 被 mask 内头部 GT 拉着往头部生长 → composite 重影。
    # 首轮实验证据：ft 后 22.2% body 高斯落进头盒（old=0%），头部内
    # opacity 0.334（整体 p50 才 0.181）。
    from split_body_scene import head_box
    mesh_dir = Path(os.environ.get(
        "MESH_DIR", f"{results_dir}/06_avatar_gs"))
    head_lo, head_hi, _, _ = head_box(
        mesh_dir / f"avatar_mesh_p{pid}.npz",
        mesh_dir / f"avatar_bind_p{pid}.npz",
        float(os.environ.get("HEAD_BBOX_MARGIN", "0.15")))
    log(f"  🗃️ head_box 排除: lo={np.round(head_lo,3)} hi={np.round(head_hi,3)}")

    mask_cache = {}
    n_mask = 0
    for v in views:
        s = v["stem"]
        mp = masks_dir / f"{s}.p{pid2}.alpha.png"
        if not mp.exists():
            mp = masks_dir / f"{s}.p{pid2}.mask.png"
        if not mp.exists():
            continue
        m = Image.open(mp).convert("L")
        if m.size != (v["W"], v["H"]):
            m = m.resize((v["W"], v["H"]), Image.LANCZOS)
        m = np.asarray(m, dtype=np.float32) / 255.0
        # 监督区域 = person mask ∩ ~head_box 投影（头部交给 head 分支）
        m = m * (1.0 - head_mask_2d(head_lo, head_hi, v))
        mask_cache[s] = torch.from_numpy(m).unsqueeze(0)
        n_mask += 1
    log(f"  🎭 person mask: {n_mask}/{len(views)} 帧")
    if n_mask == 0:
        sys.exit("❌ 无 person mask，body finetune 无法限定监督区域")

    # 训练相机 = 该人全部帧（05 align frames）
    align = json.loads(align_json.read_text())
    ap = align["persons"][pid]
    stems = ap["frames"]
    view_by = {v["stem"]: v for v in views}
    cams = []
    for s in stems:
        v = view_by.get(s)
        if v is None:
            continue
        cams.append((s, v))
    log(f"  🎬 训练相机 {len(cams)} 帧")

    def build_params():
        return [
            {"params": [model._xyz], "lr": lr},
            {"params": [model._opacity], "lr": lr},
            {"params": [model._scaling], "lr": lr},
            {"params": [model._rotation], "lr": lr},
            {"params": [model._features], "lr": lr},
        ]

    opt = torch.optim.Adam(build_params(), lr=lr)

    bg = torch.zeros(3, device=dev)
    t0 = time.time()
    for epoch in range(1, epochs + 1):
        model.active_sh = min(model.max_sh, (epoch - 1) // max(sh_every, 1))
        order = list(range(len(cams)))
        random.shuffle(order)
        running = 0.0
        for ci in order:
            stem, v = cams[ci]
            img_path = Path(images_dir) / f"{stem}.png"
            if not img_path.exists():
                img_path = Path(images_dir) / f"{stem}.jpg"
            if not img_path.exists():
                continue
            gt_pil = Image.open(img_path).convert("RGB").resize(
                (v["W"], v["H"]))
            gt = torch.tensor(np.asarray(gt_pil), dtype=torch.float32,
                              device=dev) / 255.0
            gt = gt.permute(2, 0, 1)
            cam = GSCamera(resolution=(v["W"], v["H"]), colmap_id=0,
                           R=v["R"].T, T=v["T"],
                           FoVx=2 * np.arctan(v["W"] / (2 * v["fx"])),
                           FoVy=2 * np.arctan(v["H"] / (2 * v["fy"])),
                           depth_params=None, invdepthmap=None,
                           image=gt_pil, image_name=stem, uid=ci,
                           data_device=dev)
            pkg = render(cam, model, pipe, bg, 1.0)
            m = mask_cache.get(stem)
            if m is not None:
                m_d = m.to(dev)
                area = m_d.sum() * 3 + 1e-6
                img_m = pkg["render"] * m_d + gt.detach() * (1.0 - m_d)
                Ll1 = (torch.abs(img_m - gt) * m_d).sum() / area
                loss = (1 - lam_dssim) * Ll1 + \
                    lam_dssim * (1.0 - ssim(img_m, gt))
                # 🚧 mask 外惩罚（堵监督盲区）：body-only 渲染（bg=黑）在
                # mask 外本应全黑；泄漏到背景的高斯会让这里变亮。直接惩罚
                # mask 外的渲染亮度 → 把膨胀出 mask 的高斯压回透明/收缩。
                # 这是 08b 首轮 composite 变差（17.51→14.72）的根因修复：
                # 区域监督对「长到 mask 外的高斯」梯度为零，densify 无约束
                # 膨胀污染背景。lam_outside 控制惩罚强度。
                if lam_outside > 0:
                    out_area = (1.0 - m_d).sum() * 3 + 1e-6
                    L_out = (pkg["render"] * (1.0 - m_d)).sum() / out_area
                    loss = loss + lam_outside * L_out
            else:
                loss = (1 - lam_dssim) * l1_loss(pkg["render"], gt) + \
                    lam_dssim * (1.0 - ssim(pkg["render"], gt))
            loss.backward()
            opt.step()
            opt.zero_grad(set_to_none=True)
            running += float(loss.item())
            vp = pkg.get("viewspace_points", None)
            vf = pkg.get("visibility_filter", None)
            if (vp is not None and vp.grad is not None and vf is not None
                    and len(vf) > 0 and len(vf) <= model.grad_accum.shape[0]):
                with torch.no_grad():
                    g = torch.norm(vp.grad[vf, :2], dim=-1)
                    model.grad_accum[vf] += g
                    model.denom[vf] += 1

        n = max(len(order), 1)
        log(f"  epoch {epoch}/{epochs} loss={running/n:.5f} "
            f"n={len(model._xyz):,} ({time.time()-t0:.0f}s)")

        if densify_from <= epoch <= densify_until and epoch % densify_every == 0:
            # extent：body bbox 对角线（官方 scene_extent 语义的近似）
            with torch.no_grad():
                bbox = model._xyz
                extent = float(torch.norm(bbox.max(0).values -
                                          bbox.min(0).values).item())
            kept, added = model.densify(grad_thresh, min_opacity, extent)
            log(f"     🌱 densify: 保留 {kept:,} 新增 {added:,} "
                f"→ {len(model._xyz):,}")
            opt = torch.optim.Adam(build_params(), lr=lr)
            log("     🔄 optimizer 已重建（densify 换参后重挂）")

    final = out_dir / f"body_ft_p{pid}.ply"
    model.save(final)
    log(f"💾 {final}")
    log("🎉 08b body finetune 完成")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""train_avatar.py — 阶段六/八：表情驱动的 AvatarGaussian 训练。

与标准 3DGS 的三处不同：
  1. 位置**不是自由参数**（硬绑定）：p = bary @ V_f[face_id]，V_f 由 FLAME 的
     id/exp 决定，再过 local_SRT → global_SRT。所以每帧位置不同，几何随表情动。
     bary 用 softmax 参数化，保证恒为合法重心坐标且可训练（densify 后能微调）。
  2. `local_exp` 是 `nn.Parameter`，与 opacity/scale/rot/SH 一起被渲染 loss 优化
     ——DECA 给的初值精度有限，必须靠渲染 loss refine。
  3. densify 时子点**继承父点的 face_id 与 bary**（clone 直接继承，split 加扰动），
     否则新点没有 exp_base 索引，形变时头部撕裂。
     注意：硬绑定下 clone 出来的两点初始完全重合，靠 optimizer 从 bary/scale 分开。

epoch 48（= EPOCHS-12）触发后处理增强：
  fix_and_sync_exp() 把所有帧表情统一为代表帧 → 存 checkpoint → 增强 → 跑指标，
  变差可回滚（设计文档 §10）。

渲染直接复用官方 gaussian_renderer.render，本模型只需提供
get_xyz / get_opacity / get_scaling / get_rotation / get_features 接口。

Env: AVATAR_PLY / BIND_NPZ / ALIGN_JSON / SOURCE_DIR / OUT_DIR
     EPOCHS / FACE_ENHANCE_EPOCH / LR_* / DENSIFY_* / LAMBDA_DSSIM
"""
import os
import sys
import json
import time
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
from colmap_io import read_cameras  # noqa: E402

N_SHAPE = 300
N_EXPR = 100


def log(m):
    print(m, flush=True)


def quat_to_mat(q):
    q = F.normalize(q, dim=-1)
    w, x, y, z = q.unbind(-1)
    return torch.stack([
        1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y),
        2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
        2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y),
    ], dim=-1).reshape(q.shape[:-1] + (3, 3))


def median_quat_np(qs):
    from scipy.spatial.transform import Rotation
    q = Rotation.from_quat(np.c_[qs[:, 1:4], qs[:, 0]]).mean().as_quat()
    return np.concatenate([q[3:4], q[:3]])


class AvatarModel:
    """绑定版高斯。位置由 FLAME 形变 + 绑定决定，其余属性是标准 3DGS 参数。"""

    def __init__(self, ply_path, bind_path, flame, align_p, device, max_sh=3):
        from plyfile import PlyData
        v = PlyData.read(str(ply_path))["vertex"]
        z = np.load(bind_path)
        self.device = device
        self.flame = flame
        self.face_id = torch.as_tensor(z["face_id"], device=device)
        self.is_free = torch.as_tensor(z["is_free"].astype(bool), device=device)
        faces = torch.as_tensor(
            np.asarray(flame.faces, dtype=np.int64), device=device)
        self.tri = faces[self.face_id.clamp(min=0)]          # (N,3) 顶点索引
        self.tri[self.is_free] = 0                            # 自由点不用

        bary0 = torch.as_tensor(z["bary"], dtype=torch.float32, device=device)
        bary0 = bary0.clamp_min(1e-6)
        bary0 = bary0 / bary0.sum(-1, keepdim=True)
        # softmax 参数化：恒为合法重心坐标，且可训练
        self._bary_raw = torch.nn.Parameter(torch.log(bary0))
        n_free = int(self.is_free.sum())
        self._free_can = torch.nn.Parameter(
            torch.zeros(n_free, 3, device=device))

        # 形变参数（阶段四的结果作为初值）
        self.id_coeff = torch.nn.Parameter(
            torch.tensor(align_p["id_coeff"], dtype=torch.float32, device=device))
        nf = len(align_p["local_q"])
        self.local_q = torch.nn.Parameter(
            torch.tensor(np.asarray(align_p["local_q"], dtype=np.float32),
                         device=device))
        self.local_t = torch.nn.Parameter(
            torch.tensor(np.asarray(align_p["local_t"], dtype=np.float32),
                         device=device))
        self.exp = torch.nn.Parameter(
            torch.tensor(np.asarray(align_p["exp_coeff"], dtype=np.float32),
                         device=device))
        self.log_s = torch.nn.Parameter(torch.tensor(
            np.log(float(align_p["scale"])), dtype=torch.float32, device=device))
        self.global_q = torch.nn.Parameter(torch.tensor(
            np.asarray(align_p["global_q"], dtype=np.float32), device=device))
        self.global_t = torch.nn.Parameter(torch.tensor(
            np.asarray(align_p["global_t"], dtype=np.float32), device=device))

        # 标准 3DGS 属性
        def arr(names):
            return np.stack([np.asarray(v[n]) for n in names], axis=1)
        self._opacity = torch.nn.Parameter(
            torch.tensor(np.asarray(v["opacity"]), dtype=torch.float32,
                         device=device).reshape(-1, 1))
        self._scaling = torch.nn.Parameter(
            torch.tensor(arr([f"scale_{i}" for i in range(3)]),
                         dtype=torch.float32, device=device))
        self._rotation = torch.nn.Parameter(
            torch.tensor(arr([f"rot_{i}" for i in range(4)]),
                         dtype=torch.float32, device=device))
        self._features_dc = torch.nn.Parameter(
            torch.tensor(arr([f"f_dc_{i}" for i in range(3)]),
                         dtype=torch.float32, device=device).unsqueeze(-1))
        self._features_rest = torch.nn.Parameter(torch.zeros(
            len(self._opacity), 3, (max_sh + 1) ** 2 - 1,
            dtype=torch.float32, device=device))
        self.max_sh = max_sh
        self.active_sh = 0

        # densify 统计
        self.grad_accum = torch.zeros(len(self._opacity), device=device)
        self.denom = torch.zeros(len(self._opacity), device=device)
        self._frame = 0

    # ── 属性接口（给官方 gaussian_renderer.render 用）──────────────────
    @property
    def get_opacity(self):
        return torch.sigmoid(self._opacity)

    @property
    def get_scaling(self):
        return torch.exp(self._scaling)

    @property
    def get_rotation(self):
        return F.normalize(self._rotation, dim=-1)

    @property
    def get_features(self):
        return torch.cat([self._features_dc, self._features_rest], dim=2) \
            if self.active_sh > 0 else self._features_dc

    @property
    def get_covariance(self):
        return 1.0

    def get_xyz(self):
        return self.deform(self._frame)

    def set_frame(self, f):
        self._frame = int(f)

    # ── 形变 ─────────────────────────────────────────────────────────
    def deform(self, f):
        """帧 f 的世界坐标位置 (N,3)。"""
        exp_f = self.exp[f].unsqueeze(0)
        out = self.flame(betas=self.id_coeff.unsqueeze(0), expression=exp_f)
        V = out.vertices[0]                                  # (nv,3) canonical
        bary = torch.softmax(self._bary_raw, dim=-1)
        p_bound = (V[self.tri] * bary[:, :, None]).sum(1)    # (N,3)
        p_can = p_bound
        if int(self.is_free.sum()) > 0 and self._free_can.shape[0] > 0:
            p_can = p_can.clone()
            p_can[self.is_free] = p_can[self.is_free] + self._free_can
        Rl = quat_to_mat(self.local_q[f])
        p_local = (Rl @ p_can.T).T + self.local_t[f]
        Rg = quat_to_mat(self.global_q)
        p_world = (Rg @ p_local.T).T + self.global_t
        return p_world * torch.exp(self.log_s)

    # ── densify（关键：继承绑定）────────────────────────────────────
    @torch.no_grad()
    def densify(self, grad_thresh, min_opacity, extent, scale_factor=0.2):
        n0 = len(self._opacity)
        grads = self.grad_accum / self.denom.clamp(min=1)
        grads[~torch.isfinite(grads)] = 0.0
        big = (grads > grad_thresh) & (self.get_scaling.max(1).values >
                                       scale_factor * extent)
        op = self.get_opacity.squeeze(-1)
        prune = op < min_opacity
        clone_idx = torch.where(big & (self.get_scaling.max(1).values <=
                                       scale_factor * extent))[0]
        split_idx = torch.where(big & (self.get_scaling.max(1).values >
                                       scale_factor * extent))[0]

        keep = ~prune
        new_bary, new_free_can = [], []
        idx_map = torch.arange(n0, device=self.device)[keep]

        def grow(idx, perturb):
            """返回新增点的 (bary_raw, free_can)。"""
            b = self._bary_raw[idx].clone()
            if perturb:
                b = b + torch.randn_like(b) * 0.15
            fc = None
            if self._free_can.shape[0] > 0:
                sel = self.is_free[idx]
                fc = self._free_can[
                    torch.cumsum(self.is_free, 0)[idx] - 1].clone()
                fc[~sel] = 0.0
            return b, fc

        for idx, perturb in ((clone_idx, False), (split_idx, True)):
            idx = idx[keep[idx]]
            if len(idx) == 0:
                continue
            b, fc = grow(idx, perturb)
            new_bary.append(b)
            if fc is not None:
                new_free_can.append(fc)

        if not new_bary:
            self._prune_only(keep, idx_map)
            return int(keep.sum()), 0

        nb = torch.cat(new_bary, 0)
        nfc = (torch.cat(new_free_can, 0) if new_free_can else
               torch.zeros(0, 3, device=self.device))

        def cat_param(p, extra):
            return torch.nn.Parameter(torch.cat([p[keep], extra], 0))

        self._bary_raw = cat_param(self._bary_raw, nb)
        self._opacity = cat_param(self._opacity, self._opacity[keep].clone())
        self._scaling = cat_param(self._scaling,
                                  self._scaling[keep].clone() - np.log(1.6))
        self._rotation = cat_param(self._rotation, self._rotation[keep].clone())
        self._features_dc = cat_param(self._features_dc,
                                      self._features_dc[keep].clone())
        self._features_rest = cat_param(self._features_rest,
                                        self._features_rest[keep].clone())
        if self._free_can.shape[0] > 0 and nfc.shape[0] > 0:
            self._free_can = torch.nn.Parameter(
                torch.cat([self._free_can, nfc], 0))
        self.is_free = torch.cat([self.is_free[keep], self.is_free[keep].clone()])
        self.tri = torch.cat([self.tri[keep], self.tri[keep].clone()])
        self.grad_accum = torch.zeros(len(self._opacity), device=self.device)
        self.denom = torch.zeros(len(self._opacity), device=self.device)
        return int(keep.sum()), len(nb)

    @torch.no_grad()
    def _prune_only(self, keep, idx_map):
        self._bary_raw = torch.nn.Parameter(self._bary_raw[keep].clone())
        self._opacity = torch.nn.Parameter(self._opacity[keep].clone())
        self._scaling = torch.nn.Parameter(self._scaling[keep].clone())
        self._rotation = torch.nn.Parameter(self._rotation[keep].clone())
        self._features_dc = torch.nn.Parameter(self._features_dc[keep].clone())
        self._features_rest = torch.nn.Parameter(
            self._features_rest[keep].clone())
        self.is_free = self.is_free[keep].clone()
        self.tri = self.tri[keep].clone()
        self.grad_accum = torch.zeros(len(self._opacity), device=self.device)
        self.denom = torch.zeros(len(self._opacity), device=self.device)

    @torch.no_grad()
    def fix_and_sync_exp(self):
        """所有帧表情统一为代表帧（最正面 + 表情最接近中值）。

        只改 exp，不碰 SRT —— 变换链顺序保证了二者在表示层面独立。
        """
        e = self.exp.detach()
        med = e.median(0).values
        d = (e - med).abs().sum(1)
        rep = int(d.argmin())
        self.exp.data.copy_(e[rep].unsqueeze(0).expand_as(e))
        self.exp.requires_grad_(False)   # 冻结：不再逐帧优化
        return rep

    def save(self, path):
        torch.save({
            "bary_raw": self._bary_raw.detach().cpu(),
            "free_can": self._free_can.detach().cpu(),
            "opacity": self._opacity.detach().cpu(),
            "scaling": self._scaling.detach().cpu(),
            "rotation": self._rotation.detach().cpu(),
            "features_dc": self._features_dc.detach().cpu(),
            "features_rest": self._features_rest.detach().cpu(),
            "tri": self.tri.detach().cpu(),
            "is_free": self.is_free.detach().cpu(),
            "id_coeff": self.id_coeff.detach().cpu(),
            "exp": self.exp.detach().cpu(),
            "local_q": self.local_q.detach().cpu(),
            "local_t": self.local_t.detach().cpu(),
            "global_q": self.global_q.detach().cpu(),
            "global_t": self.global_t.detach().cpu(),
            "log_s": self.log_s.detach().cpu(),
        }, path)


def main():
    results_dir = os.environ.get("RESULTS_DIR", "")
    ply_path = Path(os.environ.get(
        "AVATAR_PLY", f"{results_dir}/06_avatar_gs/avatar_p00.ply"))
    bind_path = Path(os.environ.get(
        "BIND_NPZ", str(ply_path).replace(".ply", "_bind.npz")))
    align_json = Path(os.environ.get(
        "ALIGN_JSON", f"{results_dir}/05_align/head_align.json"))
    out_dir = Path(os.environ.get("OUT_DIR", f"{results_dir}/08_train"))
    pid = os.environ.get("PID", "00")
    epochs = int(os.environ.get("EPOCHS", "60"))
    enh_epoch = int(os.environ.get("FACE_ENHANCE_EPOCH", str(max(epochs - 12, 1))))
    lr = float(os.environ.get("LR", "1e-3"))
    lr_exp = float(os.environ.get("LR_EXP", "1e-3"))
    densify_from = int(os.environ.get("DENSIFY_FROM", "5"))
    densify_until = int(os.environ.get("DENSIFY_UNTIL", "40"))
    densify_every = int(os.environ.get("DENSIFY_EVERY", "5"))
    grad_thresh = float(os.environ.get("DENSIFY_GRAD", "2e-4"))
    min_opacity = float(os.environ.get("MIN_OPACITY", "0.005"))
    lam_dssim = float(os.environ.get("LAMBDA_DSSIM", "0.2"))
    sh_every = int(os.environ.get("SH_EVERY", "10"))
    seed = int(os.environ.get("SEED", "0"))

    if not bind_path.exists():
        # init_avatar_gs.py 存的是 avatar_bind_p{pid}.npz，PLY 是 avatar_p{pid}.ply
        alt = ply_path.parent / f"avatar_bind_p{pid}.npz"
        if alt.exists():
            bind_path = alt
        else:
            sys.exit(f"❌ 缺少绑定文件: {bind_path}")
    if not align_json.exists():
        sys.exit(f"❌ 缺少阶段四输出: {align_json}")

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir.mkdir(parents=True, exist_ok=True)
    log(f"🏋️  [阶段六] AvatarGaussian 训练  device={dev}")
    log(f"  📦 {ply_path.name} + {bind_path.name}")
    log(f"  🔁 epochs={epochs}, 增强触发 epoch={enh_epoch}")

    import smplx
    flame = smplx.create(model_path=os.environ["FLAME_MODEL"],
                         model_type="flame",
                         num_expression_coeffs=N_EXPR,
                         use_face_contour=False).to(dev)
    for p in flame.parameters():
        p.requires_grad_(False)

    align = json.loads(align_json.read_text())
    if pid not in align["persons"]:
        sys.exit(f"❌ 阶段四结果里没有 p{pid}，有: {list(align['persons'])}")
    ap = align["persons"][pid]

    model = AvatarModel(ply_path, bind_path, flame, ap, dev)
    n_frames = len(ap["local_q"])
    log(f"  🔢 {len(model._opacity):,} 高斯, {n_frames} 帧")

    params = [
        {"params": [model._bary_raw], "lr": lr * 0.1},
        {"params": [model._opacity], "lr": lr},
        {"params": [model._scaling], "lr": lr},
        {"params": [model._rotation], "lr": lr},
        {"params": [model._features_dc], "lr": lr},
        {"params": [model._features_rest], "lr": lr * 0.05},
        {"params": [model.id_coeff], "lr": lr * 0.1},
        {"params": [model.local_q, model.local_t], "lr": lr * 0.1},
        {"params": [model.global_q, model.global_t, model.log_s], "lr": lr * 0.1},
        {"params": [model.exp], "lr": lr_exp},
    ]
    if int(model.is_free.sum()) > 0:
        params.append({"params": [model._free_can], "lr": lr * 0.1})
    opt = torch.optim.Adam(params, lr=lr)

    sys.path.insert(0, os.environ.get("GS_DIR", ""))
    try:
        from gaussian_renderer import render
        from utils.loss_utils import l1_loss, ssim
        from scene.cameras import Camera as GSCamera
    except Exception as e:
        sys.exit(f"❌ 无法导入 gaussian-splatting（GS_DIR={os.environ.get('GS_DIR')}"
                 f"）: {type(e).__name__}: {e}\n"
                 f"   → 先跑 bash 00a_setup_env.sh 装子模块")
    # render() 只用到 pipe 的这几个字段，不用引 argparse
    from types import SimpleNamespace
    pipe = SimpleNamespace(compute_cov3D_python=False,
                           convert_SHs_python=False, debug=False)

    views = read_cameras(os.environ.get("SOURCE_DIR", ""))
    log(f"  📷 {len(views)} views")

    # 相机：只保留有该人观测的帧（阶段四的 frames 列表）
    stems = ap["frames"]
    view_by = {v["stem"]: v for v in views}
    cams = []
    for s in stems:
        v = view_by.get(s)
        if v is None:
            continue
        cams.append((s, v))
    log(f"  🎬 训练相机 {len(cams)} 帧")

    bg = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32, device=dev)
    best_psnr, best_path = -1.0, None
    t0 = time.time()
    for epoch in range(1, epochs + 1):
        model.active_sh = min(model.max_sh, (epoch - 1) // max(sh_every, 1))
        order = list(range(len(cams)))
        random.shuffle(order)
        running = 0.0
        for k, ci in enumerate(order):
            stem, v = cams[ci]
            model.set_frame(ci)
            img_path = Path(os.environ.get("IMAGES_DIR", "")) / f"{stem}.png"
            if not img_path.exists():
                img_path = Path(os.environ.get("IMAGES_DIR", "")) / f"{stem}.jpg"
            if not img_path.exists():
                continue
            from PIL import Image
            gt = torch.tensor(np.asarray(Image.open(img_path).convert("RGB")),
                              dtype=torch.float32, device=dev) / 255.0
            gt = gt.permute(2, 0, 1)
            cam_args = dict(
                colmap_id=0, R=v["R"].T, T=v["T"],
                FoVx=2 * np.arctan(v["W"] / (2 * v["fx"])),
                FoVy=2 * np.arctan(v["H"] / (2 * v["fy"])),
                image=gt, gt_alpha_mask=None, image_name=stem, uid=ci)
            try:
                cam = GSCamera(data_device=dev, **cam_args)  # 新版
            except TypeError:
                cam = GSCamera(**cam_args)                    # 旧版
            pkg = render(cam, model, pipe, bg, 1.0)
            loss = (1 - lam_dssim) * l1_loss(pkg["render"], gt) + \
                lam_dssim * (1.0 - ssim(pkg["render"], gt))
            loss.backward()
            opt.step()
            opt.zero_grad(set_to_none=True)
            running += float(loss.item())
            #  densify 用的梯度统计（官方同款：可见点的屏幕空间梯度范数）
            vp = pkg.get("viewspace_points", None)
            vf = pkg.get("visibility_filter", None)
            if (vp is not None and vp.grad is not None and vf is not None
                    and len(vf) == model.grad_accum.shape[0]):
                with torch.no_grad():
                    g = torch.norm(vp.grad[vf, :2], dim=-1)
                    model.grad_accum[vf] += g
                    model.denom[vf] += 1

        n = max(len(order), 1)
        log(f"  epoch {epoch}/{epochs} loss={running/n:.5f} "
            f"n={len(model._opacity):,} ({time.time()-t0:.0f}s)")

        if densify_from <= epoch <= densify_until and epoch % densify_every == 0:
            extent = float(torch.exp(model.log_s).item() * 0.5)
            kept, added = model.densify(grad_thresh, min_opacity, extent)
            log(f"     🌱 densify: 保留 {kept:,} 新增 {added:,} "
                f"→ {len(model._opacity):,}")

        # ── epoch 48：fix_exp + checkpoint + 增强 + 指标 + 可回滚 ──────
        if epoch == enh_epoch:
            ckpt = out_dir / f"ckpt_epoch{epoch}_pre_enhance.pth"
            model.save(ckpt)
            rep = model.fix_and_sync_exp()
            log(f"     🔒 fix_and_sync_exp: 统一到代表帧 #{rep}，SRT 未动")
            log(f"     💾 checkpoint: {ckpt.name}（指标变差可退回）")
            # 增强后的指标对比在 09_enhance_post 里做；这里只留钩子
            (out_dir / f"ENHANCE_TRIGGER_epoch{epoch}.json").write_text(
                json.dumps({"epoch": epoch, "rep_frame": rep,
                            "ckpt": str(ckpt)}, indent=1))

    final = out_dir / f"avatar_p{pid}_final.pth"
    model.save(final)
    log(f"💾 {final}")
    log("🎉 阶段六训练完成")


if __name__ == "__main__":
    main()

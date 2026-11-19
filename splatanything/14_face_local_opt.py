#!/usr/bin/env python3
"""14_face_local_opt.py — P0~P3 脸部局部高斯优化（诊断用测试时优化，非前馈成绩）。

实现 da3_human/EXPERIMENT_PLAN.md 的 P0~P3 最小消融：
  P0  恒等校验（重组渲染 == 直接渲染、固定块 hash、PLY 回读、K' 裁切 == 全图裁图）
  P1  仅 DC 颜色（f_dc）
  P2  f_dc + scale/quat/opacity，中心固定
  P3  P2 + 受限中心位移（≤5% 头宽，弱位移正则 + 同色 kNN 平滑）

关键纪律（对应计划原文）：
  - 固定块参数用原值重组（ buffers 不进 optimizer），不仅梯度乘 0；每 checkpoint 核对 hash。
  - 三视角同 batch 渲染、同一 renderer 共同合成（不是贴回背景）。
  - 损失按每视角有效脸像素数归一化；脸核/边界带/ROI 外分别计权；ROI 外锚定 P0 渲染。
  - 不增删点、固定相机、不优化曝光；初跑 100 步，链路正常再 300 步；只报预注册 0/100/300。
  - 停止条件：scale 爆炸 / opacity 塌缩 / ROI 外明显劣化 → 记失败例，不自动加步数。

用法：
  python 14_face_local_opt.py --result <02输出目录> --arm p1 --iters 100
"""
import argparse
import hashlib
import importlib.util
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as Fn
from PIL import Image

_RE03 = None


def re03():
    global _RE03
    if _RE03 is None:
        spec = importlib.util.spec_from_file_location(
            "render_eval03", str(Path(__file__).parent / "03_render_eval.py"))
        m = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(m)
        _RE03 = m
    return _RE03


# ------------------------------------------------------------------ PLY I/O
def load_ply_raw(path):
    """读原始参数空间：log-scale / logit-opacity / 原四元数 / f_dc。"""
    from plyfile import PlyData
    ply = PlyData.read(str(path))["vertex"]
    g = dict(
        means=np.stack([ply[n] for n in ("x", "y", "z")], 1).astype(np.float32),
        log_scales=np.stack([ply[f"scale_{i}"] for i in range(3)], 1).astype(np.float32),
        logit_opac=ply["opacity"].astype(np.float32),
        quats=np.stack([ply[f"rot_{i}"] for i in range(4)], 1).astype(np.float32),
        f_dc=np.stack([ply[f"f_dc_{i}"] for i in range(3)], 1).astype(np.float32),
    )
    return g


def write_ply(path, g):
    """写回标准 3DGS PLY（原始参数空间）。"""
    from plyfile import PlyData, PlyElement
    n = len(g["means"])
    cols = []
    names = (["x", "y", "z"] + [f"f_dc_{i}" for i in range(3)] +
             ["opacity"] + [f"scale_{i}" for i in range(3)] +
             [f"rot_{i}" for i in range(4)])
    data = np.concatenate(
        [g["means"], g["f_dc"], g["logit_opac"][:, None],
         g["log_scales"], g["quats"]], 1).astype(np.float32)
    for i, nm in enumerate(names):
        cols.append((nm, "f4"))
    arr = np.empty(n, dtype=cols)
    for i, nm in enumerate(names):
        arr[nm] = data[:, i]
    PlyData([PlyElement.describe(arr, "vertex")], text=False).write(str(path))


def sha256_bytes(b):
    return hashlib.sha256(np.ascontiguousarray(b).tobytes()).hexdigest()[:16]


# ------------------------------------------------------------------ 渲染
class Renderer:
    """固定相机 + 全高斯集合渲染；可训子集每步重组进完整参数。"""

    def __init__(self, g, c2w, K, w, h, sel, bg, eps2d=0.01, device="cuda"):
        self.dev = device
        self.w, self.h = w, h
        self.eps2d = eps2d
        self.sel = torch.from_numpy(sel).to(device)          # [S] 可训索引
        self.fixed_mask = torch.ones(len(g["means"]), dtype=torch.bool, device=device)
        self.fixed_mask[self.sel] = False
        # 原始值 buffer（固定块的"原值重组"来源；永不在 optimizer 里）
        self.base = {k: torch.from_numpy(v).to(device) for k, v in g.items()}
        self.viewmats = torch.from_numpy(
            np.linalg.inv(np.asarray(c2w, np.float64)).astype(np.float32)).to(device)
        self.Ks = torch.from_numpy(np.asarray(K, np.float32)).to(device)
        self.bg = torch.full((len(c2w), 3), float(bg), device=device)
        # 固定块 hash（训练前记录，之后每个 checkpoint 核对）
        self.fixed_hash = self._hash_fixed()

    def _hash_fixed(self):
        fm = self.fixed_mask
        h = hashlib.sha256()
        for k in ("means", "log_scales", "logit_opac", "quats", "f_dc"):
            h.update(self.base[k][fm].cpu().numpy().tobytes())
        return h.hexdigest()[:16]

    def assemble(self, name, train):
        """train=None → 直接返回 base（纯固定）；否则 原值 clone + 可训行覆盖。"""
        base = self.base[name]
        if train is None:
            return base
        full = base.clone()
        full[self.sel] = train
        return full

    def render(self, params, view_idx, Ks_override=None, wh=None):
        """params: dict(name -> tensor or None)。view_idx: list[int]。"""
        from gsplat import rasterization
        vi = list(view_idx)
        C = len(vi)
        means = self.assemble("means", params.get("means"))
        quats = Fn.normalize(self.assemble("quats", params.get("quats")), dim=-1)
        scales = torch.exp(self.assemble("log_scales", params.get("log_scales")))
        opac = torch.sigmoid(self.assemble("logit_opac", params.get("logit_opac")))
        fdc = self.assemble("f_dc", params.get("f_dc"))
        colors = fdc[None, :, None, :].expand(C, -1, -1, -1).contiguous()
        Ks = self.Ks[vi] if Ks_override is None else Ks_override
        W, H = (self.w, self.h) if wh is None else wh
        out, alpha, _ = rasterization(
            means=means, quats=quats, scales=scales, opacities=opac,
            colors=colors, viewmats=self.viewmats[vi], Ks=Ks,
            width=W, height=H, sh_degree=0, eps2d=self.eps2d,
            near_plane=0.01, far_plane=400.0,
            backgrounds=self.bg[vi] if Ks_override is None else
            self.bg[vi[:1]].expand(C, 3),
            render_mode="RGB")
        return out.clamp(0, 1), alpha


# ------------------------------------------------------------------ SSIM
def ssim_t(x, y, win=11, sigma=1.5):
    """x,y: [C,3,H,W] torch。高斯窗 SSIM，可微。"""
    c = torch.arange(win, dtype=torch.float32, device=x.device) - win // 2
    g = torch.exp(-(c ** 2) / (2 * sigma ** 2))
    g = (g / g.sum()).view(-1, 1)
    k = (g @ g.t()).expand(3, 1, win, win).contiguous()
    mu_x = Fn.conv2d(x, k, padding=win // 2, groups=3)
    mu_y = Fn.conv2d(y, k, padding=win // 2, groups=3)
    sx = Fn.conv2d(x * x, k, padding=win // 2, groups=3) - mu_x ** 2
    sy = Fn.conv2d(y * y, k, padding=win // 2, groups=3) - mu_y ** 2
    sxy = Fn.conv2d(x * y, k, padding=win // 2, groups=3) - mu_x * mu_y
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    s = ((2 * mu_x * mu_y + c1) * (2 * sxy + c2)) / \
        ((mu_x ** 2 + mu_y ** 2 + c1) * (sx + sy + c2))
    return s.mean()


# ------------------------------------------------------------------ ROI 工具
def roi_box(face_mask, pad=2.3, w=0, h=0):
    """脸 mask bbox → 方形 ROI（与 _tmp_faceroi 同口径，保证与基线数字可比）。"""
    ys, xs = np.where(face_mask)
    cx, cy = xs.mean(), ys.mean()
    side = max(xs.max() - xs.min(), ys.max() - ys.min()) * pad
    side = int(min(side, min(w, h)))
    x0 = int(np.clip(cx - side // 2, 0, w - side))
    y0 = int(np.clip(cy - side // 2, 0, h - side))
    return x0, y0, x0 + side, y0 + side


def crop_K(K, x0, y0):
    """K' = A_crop K：像素原点平移到 ROI 左上角。"""
    K2 = K.copy()
    K2[0, 2] -= x0
    K2[1, 2] -= y0
    return K2


def psnr_np(a, b):
    mse = float(np.mean((a.astype(np.float32) - b.astype(np.float32)) ** 2))
    return float("inf") if mse <= 1e-12 else 10.0 * np.log10(1.0 / mse)


def sharp_np(img, mask=None):
    return re03().sharpness(img, mask)


# ------------------------------------------------------------------ 主流程
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--result", required=True)
    ap.add_argument("--arm", required=True, choices=["p0", "p1", "p2", "p3"])
    ap.add_argument("--iters", type=int, default=100)
    ap.add_argument("--out", default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--lr-sh", type=float, default=2.5e-3)
    ap.add_argument("--lr-scale", type=float, default=5e-3)
    ap.add_argument("--lr-opacity", type=float, default=5e-2)
    ap.add_argument("--lr-quat", type=float, default=1e-3)
    ap.add_argument("--lr-mean", type=float, default=1.6e-4)
    ap.add_argument("--lr-final", type=float, default=0.05)
    ap.add_argument("--w-band", type=float, default=0.5)
    ap.add_argument("--lam-fid", type=float, default=0.5,
                    help="ROI 外相对 P0 渲染的保真项权重（预注册起点）")
    ap.add_argument("--lam-disp", type=float, default=10.0, help="P3 位移 L2 正则")
    ap.add_argument("--lam-smooth", type=float, default=10.0, help="P3 kNN 平滑")
    ap.add_argument("--max-disp-frac", type=float, default=0.05,
                    help="P3 位移上限 = 头宽 × 此值（预注册 5%）")
    ap.add_argument("--knn", type=int, default=8)
    ap.add_argument("--eval-at", default="0,100,300")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    dev = "cuda"
    res = Path(args.result)
    masks_dir = res / "face_opt" / "masks"
    sel_dir = res / "face_opt" / "select"
    out_dir = Path(args.out) if args.out else res / "face_opt" / args.arm
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---------- 加载 ----------
    g_raw = load_ply_raw(res / "gaussians.ply")
    re = re03()
    c2w, K, ids = re.load_cameras(res / "cameras.json")
    sel = np.load(sel_dir / "selected_ids.npy")
    sel_meta = json.loads((sel_dir / "select_meta.json").read_text())
    with Image.open(res / "rgb" / f"{ids[0]}.png") as im:
        w, h = im.size
    gts, face_ms, band_ms, out_ms, person_ms = [], [], [], [], []
    for cid in ids:
        gt = np.asarray(Image.open(res / "rgb" / f"{cid}.png").convert("RGB"),
                        np.float32) / 255.0
        face = np.asarray(Image.open(masks_dir / f"{cid}_face.png")) > 127
        band = np.asarray(Image.open(masks_dir / f"{cid}_band.png")) > 127
        person = np.asarray(Image.open(masks_dir / f"{cid}_person.png")) > 127
        gts.append(gt)
        face_ms.append(face)
        band_ms.append(band)
        person_ms.append(person)
        k16 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (33, 33))
        fdi = cv2.dilate(face.astype(np.uint8), k16).astype(bool)
        out_ms.append(~fdi)
    gt_t = torch.from_numpy(np.stack(gts)).permute(0, 3, 1, 2).to(dev)  # [C,3,H,W]
    # 权重图（face=1, band=w_band）与 ROI 外 mask
    w_face, m_out = [], []
    for f, b, o in zip(face_ms, band_ms, out_ms):
        wm = f.astype(np.float32) + b.astype(np.float32) * args.w_band
        w_face.append(torch.from_numpy(wm).to(dev))
        m_out.append(torch.from_numpy(o.astype(np.float32)).to(dev))
    roi_boxes = [roi_box(f, 2.3, w, h) for f in face_ms]

    # 背景灰度：与 03_render_eval 同口径（GT 四边中位数）
    edge = np.concatenate([gts[0][0], gts[0][-1], gts[0][:, 0], gts[0][:, -1]], 0)
    bg = float(np.median(edge))

    R = Renderer(g_raw, c2w, K, w, h, sel, bg, device=dev)
    n_sel = len(sel)
    print(f"🚀 [14] arm={args.arm} iters={args.iters} | 高斯 {len(g_raw['means']):,} "
          f"| 可训 {n_sel:,} | 视角 {len(ids)} | bg={bg:.3f}")
    print(f"   固定块 hash: {R.fixed_hash}")

    # ---------- P0 恒等校验 ----------
    checks = {}
    with torch.no_grad():
        direct, _ = R.render({}, [0])
        reasm = R.render({k: R.base[k][R.sel] for k in
                          ("means", "log_scales", "logit_opac", "quats", "f_dc")}, [0])
        mae_reasm = float((reasm[0] - direct[0]).abs().mean())
        checks["reassemble_mae"] = mae_reasm
        # ROI 裁切渲染 == 全图裁图（K' = A_crop K）
        x0, y0, x1, y1 = roi_boxes[0]
        K2 = torch.from_numpy(crop_K(K[0], x0, y0)[None].astype(np.float32)).to(dev)
        roi_rd, _ = R.render({}, [0], Ks_override=K2, wh=(x1 - x0, y1 - y0))
        crop = direct[0][y0:y1, x0:x1]
        checks["roi_crop_mae"] = float((roi_rd[0] - crop).abs().mean())
    # PLY 回读恒等
    tmp_ply = out_dir / "_p0_roundtrip.ply"
    write_ply(tmp_ply, g_raw)
    g2 = load_ply_raw(tmp_ply)
    R2 = Renderer(g2, c2w, K, w, h, sel, bg, device=dev)
    with torch.no_grad():
        rd2, _ = R2.render({}, [0])
    checks["ply_roundtrip_mae"] = float((rd2[0] - direct[0]).abs().mean())
    del R2
    tmp_ply.unlink()
    print(f"   P0 恒等: 重组 MAE={mae_reasm:.2e}  ROI裁切 MAE={checks['roi_crop_mae']:.2e}  "
          f"PLY回读 MAE={checks['ply_roundtrip_mae']:.2e}")
    if mae_reasm > 1e-4 or checks["ply_roundtrip_mae"] > 1e-4:
        print("❌ P0 恒等门未过（重组/回读 MAE > 1e-4）")
    if args.arm == "p0":
        # P0 只出基线指标 + 校验，不训练
        p0_render, _ = R.render({}, list(range(len(ids))))
        metrics = evaluate(R, {}, gt_t, gts, w_face, m_out, roi_boxes,
                           face_ms, ids, p0_render, 0, person_ms)
        save_outputs(out_dir, args, [metrics], checks, R, sel_meta, None, 0, 0.0)
        save_crops(out_dir, R, {}, gts, roi_boxes, ids)
        print(f"🎉 P0 done: {out_dir}")
        return

    # ---------- 可训参数组 ----------
    trainable = {"p1": ["f_dc"],
                 "p2": ["f_dc", "log_scales", "logit_opac", "quats"],
                 "p3": ["f_dc", "log_scales", "logit_opac", "quats", "means"]}[args.arm]
    params = {k: torch.nn.Parameter(R.base[k][R.sel].clone()) for k in trainable}
    lrs = {"f_dc": args.lr_sh, "log_scales": args.lr_scale,
           "logit_opac": args.lr_opacity, "quats": args.lr_quat,
           "means": args.lr_mean}
    opt = torch.optim.Adam(
        [{"params": [params[k]], "lr": lrs[k]} for k in trainable], eps=1e-15)
    base_lrs = [gp["lr"] for gp in opt.param_groups]
    print(f"   可优化组: {trainable}")

    # P3 约束预备：头宽、kNN 图、位移上限
    disp_ctx = None
    if args.arm == "p3":
        sel_meta_bbox = max(sel_meta["center_bbox_size"])
        head_w = float(sel_meta_bbox)          # 选中簇最大跨度 ≈ 头宽
        max_disp = head_w * args.max_disp_frac
        m0 = R.base["means"][R.sel]            # [S,3]
        d = torch.cdist(m0, m0)                # S~6k → 40M 元素，OK
        knn_idx = d.topk(args.knn + 1, largest=False).indices[:, 1:]
        sigma = float(d[d > 0].median())
        c0 = R.base["f_dc"][R.sel]
        cd = torch.cdist(c0, c0)
        sig_c = float(cd[cd > 0].median())
        # kNN 边权：空间近 × 颜色近（同色皮肤片内才平滑的近似门；
        # 不跨鼻翼/唇线/发际——这些区域 DC 颜色或深度不连续，权重自动压低）
        w_sp = torch.exp(-(torch.gather(d, 1, knn_idx) ** 2) / (2 * sigma ** 2))
        w_co = torch.exp(-(torch.gather(cd, 1, knn_idx) ** 2) / (2 * sig_c ** 2))
        knn_w = (w_sp * w_co).detach()         # [S,k]
        disp_ctx = dict(m0=m0, max_disp=max_disp, head_w=head_w,
                        knn_idx=knn_idx, knn_w=knn_w, sigma=sigma, sig_c=sig_c)
        print(f"   P3: 头宽≈{head_w*1000:.1f}mm → 位移上限 {max_disp*1000:.2f}mm, "
              f"kNN k={args.knn} σ={sigma*1000:.2f}mm")

    # ---------- P0 渲染快照（保真锚点 + 指标基准） ----------
    with torch.no_grad():
        p0_render, _ = R.render({}, list(range(len(ids))))
    p0_render = p0_render.detach()
    p0_perm = p0_render.permute(0, 3, 1, 2).contiguous()

    eval_at = sorted({int(x) for x in args.eval_at.split(",")} | {0, args.iters})
    history = []
    history.append(evaluate(R, params, gt_t, gts, w_face, m_out, roi_boxes,
                            face_ms, ids, p0_render, 0, person_ms))
    print_metrics(history[-1])

    # ---------- 训练 ----------
    t0 = time.time()
    stop_reason = None
    for it in range(1, args.iters + 1):
        frac = it / max(1, args.iters)
        for gp, lr0 in zip(opt.param_groups, base_lrs):
            gp["lr"] = lr0 * (args.lr_final ** frac)
        rd, _ = R.render(params, list(range(len(ids))))      # [C,H,W,3]
        rd_p = rd.permute(0, 3, 1, 2)                        # [C,3,H,W]
        loss_face = 0.0
        loss_fid = 0.0
        for i in range(len(ids)):
            wm = w_face[i][None]
            l1 = (rd_p[i:i + 1] - gt_t[i:i + 1]).abs() * wm
            loss_face = loss_face + l1.sum() / (wm.sum() * 3)
            x0, y0, x1, y1 = roi_boxes[i]
            ssim_v = ssim_t(rd_p[i:i + 1, :, y0:y1, x0:x1],
                            gt_t[i:i + 1, :, y0:y1, x0:x1])
            loss_face = loss_face + 0.2 * (1 - ssim_v)
            mo = m_out[i][None]
            loss_fid = loss_fid + ((rd_p[i:i + 1] - p0_perm[i:i + 1]).abs() * mo
                                   ).sum() / (mo.sum() * 3)
        loss = loss_face / len(ids) + args.lam_fid * loss_fid / len(ids)
        if args.arm == "p3":
            delta = params["means"] - disp_ctx["m0"]
            loss = loss + args.lam_disp * (delta ** 2).mean()
            kn = disp_ctx["knn_idx"]
            dd = delta[:, None, :] - delta[kn]               # [S,k,3]
            kw = disp_ctx["knn_w"]                           # [S,k]
            loss = loss + args.lam_smooth * (
                ((dd ** 2).sum(-1) * kw).sum() / kw.sum().clamp_min(1e-12))
        if not torch.isfinite(loss):
            stop_reason = f"loss 非有限值 @it{it}"
            break
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        with torch.no_grad():
            if "log_scales" in params:
                params["log_scales"].clamp_(-12.0, -2.0)
            if "logit_opac" in params:
                params["logit_opac"].clamp_(-12.0, 12.0)
            if "quats" in params:
                params["quats"].clamp_(-4.0, 4.0)
            if args.arm == "p3":
                delta = params["means"] - disp_ctx["m0"]
                nrm = delta.norm(dim=-1, keepdim=True)
                f = (disp_ctx["max_disp"] / nrm.clamp_min(1e-12)).clamp(max=1.0)
                params["means"].data = disp_ctx["m0"] + delta * f
        if it % 20 == 0 or it == 1:
            print(f"  it {it:4d}/{args.iters}  loss {loss.item():.5f}  "
                  f"face {(loss_face/len(ids)).item():.5f}  "
                  f"fid {(loss_fid/len(ids)).item():.5f}  "
                  f"{(time.time()-t0)/it:.3f}s/it")
        if it in eval_at:
            m = evaluate(R, params, gt_t, gts, w_face, m_out, roi_boxes,
                         face_ms, ids, p0_render, it, person_ms)
            history.append(m)
            print_metrics(m)
            # 停止条件：scale 爆炸 / opacity 塌缩 / ROI 外显著劣化
            with torch.no_grad():
                if "log_scales" in params:
                    smax = torch.exp(params["log_scales"]).max(dim=-1).values
                    if float(smax.quantile(0.99)) > 10 * disp_scale_ref(R):
                        stop_reason = "scale 爆炸（p99 超初值 10×）"
                if "logit_opac" in params:
                    if float(torch.sigmoid(params["logit_opac"]).mean()) < 0.05:
                        stop_reason = "opacity 塌缩（均值 <0.05）"
                if m["psnr_outside_p0"] is not None and m["psnr_outside_p0"] < 35.0:
                    stop_reason = (f"ROI 外保真劣化 "
                                   f"({m['psnr_outside_p0']:.1f}dB < 35dB)")
            if stop_reason:
                print(f"  ⚠️ 停止: {stop_reason}")
                break

    dt = time.time() - t0
    vram = torch.cuda.max_memory_allocated() / 1024 ** 3
    print(f"⏱️ {dt:.1f}s | 峰值显存 {vram:.2f} GB")

    # ---------- 导出 ----------
    final = {k: (params[k].detach() if k in params else None)
             for k in ("means", "log_scales", "logit_opac", "quats", "f_dc")}
    g_out = {k: R.assemble(k, final.get(k)).cpu().numpy()
             for k in ("means", "log_scales", "logit_opac", "quats", "f_dc")}
    g_out["quats"] = g_out["quats"] / np.linalg.norm(
        g_out["quats"], axis=-1, keepdims=True).clip(1e-12)
    ply_path = out_dir / f"gaussians_{args.arm}_{args.iters}.ply"
    write_ply(ply_path, g_out)
    save_outputs(out_dir, args, history, checks, R, sel_meta, stop_reason, dt, vram)
    save_crops(out_dir, R, final, gts, roi_boxes, ids)
    print(f"🎉 Done: {out_dir}")


def disp_scale_ref(R):
    return float(torch.exp(R.base["log_scales"][R.sel]).max(dim=-1).values.quantile(0.99))


def print_metrics(m):
    def f(v):
        return f"{v:.2f}" if v is not None else "  n/a"
    print(f"  📊 it{m['step']:4d}  full {f(m['psnr_full'])}  "
          f"face {f(m['psnr_face'])}  sharp {f(m['sharp_ratio'])}  "
          f"outP0 {f(m['psnr_outside_p0'])}  person {f(m['psnr_person'])}")


def evaluate(R, params, gt_t, gts, w_face, m_out, roi_boxes, face_ms, ids,
             p0_render, step, person_ms):
    """全指标：full / face ROI / 锐度比 / ROI 外 vs P0 / person 区(不含脸) vs GT。"""
    with torch.no_grad():
        rd, _ = R.render(params if params else {}, list(range(len(ids))))
        rd_np = rd.cpu().numpy()
    p0_np = p0_render.cpu().numpy()
    per = []
    for i, cid in enumerate(ids):
        gt, r9 = gts[i], rd_np[i]
        x0, y0, x1, y1 = roi_boxes[i]
        fm = face_ms[i]
        fdi = cv2.dilate(fm.astype(np.uint8),
                         cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (33, 33))
                         ).astype(bool)
        out_m = ~fdi                                   # ROI 外（相对 P0 保真）
        person_only = person_ms[i] & ~fdi              # 人物但非脸（验收口径）
        p_full = psnr_np(gt, r9)
        p_face = psnr_np(gt[y0:y1, x0:x1], r9[y0:y1, x0:x1])
        sh_gt = sharp_np(gt, fm)
        sh_rd = sharp_np(r9, fm)
        p_out = psnr_np(p0_np[i][out_m], r9[out_m]) if out_m.sum() > 256 else None
        p_person = (psnr_np(gt[person_only], r9[person_only])
                    if person_only.sum() > 256 else None)
        per.append(dict(id=cid, psnr_full=p_full, psnr_face=p_face,
                        sharp_ratio=(sh_rd / sh_gt) if sh_gt > 0 else None,
                        psnr_outside_p0=p_out, psnr_person=p_person))
    agg = dict(step=step, views=per)
    for k in ("psnr_full", "psnr_face", "sharp_ratio",
              "psnr_outside_p0", "psnr_person"):
        vals = [p[k] for p in per if p[k] is not None]
        agg[k] = float(np.mean(vals)) if vals else None
    return agg


def save_crops(out_dir, R, params, gts, roi_boxes, ids):
    """脸鼻裁图：每视角 GT | render | |diff|×4。"""
    with torch.no_grad():
        rd, _ = R.render(params if params else {}, list(range(len(ids))))
        rd_np = rd.cpu().numpy()
    tiles = []
    for i, cid in enumerate(ids):
        x0, y0, x1, y1 = roi_boxes[i]
        gt_c = (gts[i][y0:y1, x0:x1] * 255).astype(np.uint8)
        rd_c = (rd_np[i][y0:y1, x0:x1] * 255).astype(np.uint8)
        df = np.clip(np.abs(gt_c.astype(np.float32) -
                            rd_c.astype(np.float32)) * 4, 0, 255).astype(np.uint8)
        tiles.append(np.concatenate([gt_c, rd_c, df], 1))
    H = max(t.shape[0] for t in tiles)
    W = max(t.shape[1] for t in tiles)
    tiles = [np.pad(t, ((0, H - t.shape[0]), (0, W - t.shape[1]), (0, 0)))
             for t in tiles]
    Image.fromarray(np.concatenate(tiles, 0)).save(out_dir / "face_crops.png")


def save_outputs(out_dir, args, history, checks, R, sel_meta, stop_reason, dt, vram):
    src_ply = Path(args.result) / "gaussians.ply"
    src_hash = hashlib.sha256(src_ply.read_bytes()).hexdigest()[:16]
    summary = dict(
        arm=args.arm, iters=args.iters, args=vars(args),
        source_ply_sha256_16=src_hash,
        fixed_hash=R.fixed_hash,
        fixed_hash_ok=(R._hash_fixed() == R.fixed_hash),
        p0_checks=checks,
        n_selected=int(len(R.sel)),
        select_meta={k: v for k, v in sel_meta.items() if k != "args"},
        stop_reason=stop_reason,
        elapsed_s=round(dt, 1), vram_gb=round(vram, 2),
        history=history,
    )
    with open(out_dir / "metrics.json", "w", encoding="utf-8", newline="\n") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()

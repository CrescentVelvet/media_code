#!/usr/bin/env python3
"""16_face_constraint_audit.py — F0 约束审计（纯审计，无新增优化）。

对应 da3_human/EXPERIMENT_PLAN.md「F0：先修评测解释与约束审计，不重新训练」。
五个分节，全部读取已有 P0~P3 产物：

  A. target 评测口径补报：旧 2.3× 脸框（追溯）+ 紧脸框（1.0× bbox）+
     鼻部/眼周特写 LPIPS/PSNR/SSIM + 逐视角最差例；脸皮肤 mask 像素 PSNR。
  B. 原生 source 回渲：transforms.json 原生 K（1440×1920）+ cv2.undistort 原生 HEIC，
     c2w = Sim(3)·AR，P0/P3 回渲对比，核对 K/裁剪链与分辨率混杂。
  C. 鼻部遮挡分层审计：鼻尖点（FaceMesh 地标）沿视线的逐层高斯
     （选中/未选中 × 前/表面/后 深度分层 + opacity），外加 只选中/只未选中 两组渲染。
  D. P3 参数变化审计：ΔDC / Δopacity / Δscale / 位移分布、触顶率、正则终值复核、
     位移空间散布图（source view0 投影叠图）。
  E. P3 参数回退敏感性（非因果贡献）：P3 分别恢复 P0 的 DC / opacity / means，
     source 与 target 双侧渲染对拍。

输出：<result>/face_opt/audit/{audit_metrics.json, fig_*.png, strips ...}
用法：
  python 16_face_constraint_audit.py --result <02输出目录> \
      --target <t2场景> --masks <target mask目录>
"""
import argparse
import importlib.util
import json
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw

_M = {}


def load_mod(key, fname):
    if key not in _M:
        spec = importlib.util.spec_from_file_location(
            key, str(Path(__file__).parent / fname))
        m = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(m)
        _M[key] = m
    return _M[key]


def crop_K(K, x0, y0):
    K2 = K.copy()
    K2[0, 2] -= x0
    K2[1, 2] -= y0
    return K2


def sq_box(cx, cy, side, w, h):
    side = int(min(side, w, h))
    x0 = int(np.clip(round(cx - side / 2), 0, w - side))
    y0 = int(np.clip(round(cy - side / 2), 0, h - side))
    return x0, y0, x0 + side, y0 + side


def box_from_landmarks(pts, bbox, w, h, kind, approx=False):
    """kind: tight/nose/eyes。无地标时按 mask bbox 比例近似并标 approx。"""
    if pts is None:
        x0, y0, x1, y1 = bbox
        fw = x1 - x0
        fh = y1 - y0
        cx, cy = (x0 + x1) / 2, y0 + 0.70 * fh
        if kind == "eyes":
            cy = y0 + 0.38 * fh
        elif kind == "tight":
            side = max(fw, fh)
            return sq_box((x0 + x1) / 2, (y0 + y1) / 2, side, w, h), True
        return sq_box(cx, cy, fw * 0.4, w, h), True
    if kind == "tight":
        x0, y0, x1, y1 = bbox
        return sq_box((x0 + x1) / 2, (y0 + y1) / 2, max(x1 - x0, y1 - y0),
                      w, h), False
    fw = bbox[2] - bbox[0]
    if kind == "nose":
        p = pts["nose_tip"]
        return sq_box(p[0], p[1], fw * 0.45, w, h), False
    if kind == "eyes":
        a = np.array(pts["eye_l_out"]); b = np.array(pts["eye_r_out"])
        c = (a + b) / 2
        return sq_box(c[0], c[1] - fw * 0.06, fw * 0.72, w, h), False
    raise ValueError(kind)


def crop_metric(m14, gt, rd, box, lpips_fn, dev):
    x0, y0, x1, y1 = box
    g = gt[y0:y1, x0:x1]
    r = rd[y0:y1, x0:x1]
    out = dict(psnr=m14.psnr_np(g, r))
    out["ssim"] = float(m14.ssim_t(
        torch.from_numpy(g).permute(2, 0, 1)[None].to(dev),
        torch.from_numpy(r).permute(2, 0, 1)[None].to(dev)))
    if lpips_fn is not None and min(g.shape[:2]) >= 48:
        with torch.no_grad():
            out["lpips"] = float(lpips_fn(
                torch.from_numpy(g).permute(2, 0, 1)[None].to(dev) * 2 - 1,
                torch.from_numpy(r).permute(2, 0, 1)[None].to(dev) * 2 - 1))
    return out


def render_rgbd(R, params, vi_list):
    """render_mode=RGB+ED，第 4 通道 expected depth（gsplat 约定）。"""
    from gsplat import rasterization
    vi = list(vi_list)
    C = len(vi)
    means = R.assemble("means", params.get("means"))
    quats = torch.nn.functional.normalize(
        R.assemble("quats", params.get("quats")), dim=-1)
    scales = torch.exp(R.assemble("log_scales", params.get("log_scales")))
    opac = torch.sigmoid(R.assemble("logit_opac", params.get("logit_opac")))
    fdc = R.assemble("f_dc", params.get("f_dc"))
    colors = fdc[None, :, None, :].expand(C, -1, -1, -1).contiguous()
    out, alpha, _ = rasterization(
        means=means, quats=quats, scales=scales, opacities=opac,
        colors=colors, viewmats=R.viewmats[vi], Ks=R.Ks[vi],
        width=R.w, height=R.h, sh_degree=0, eps2d=R.eps2d,
        near_plane=0.01, far_plane=400.0,
        backgrounds=R.bg[vi], render_mode="RGB+ED")
    return out.clamp(0, 1), alpha


# ------------------------------------------------------------------ 画图
def hist_panel(draw, vals, title, xlabel, x_range=None, vline=None,
               vline_label="", size=(480, 300), color=(47, 107, 181)):
    """PIL 手画直方图（无 matplotlib）。返回 Image。"""
    W, H = size
    img = Image.new("RGB", (W, H), (255, 255, 255))
    d = ImageDraw.Draw(img)
    L, T, RR, B = 46, 28, W - 12, H - 34
    d.rectangle([L, T, RR, B], outline=(60, 60, 60))
    vals = np.asarray(vals, np.float64)
    lo, hi = (float(vals.min()), float(vals.max())) if x_range is None else x_range
    if hi <= lo:
        hi = lo + 1e-9
    nb = 40
    cnt, edges = np.histogram(vals, bins=nb, range=(lo, hi))
    cmax = max(int(cnt.max()), 1)
    bw = (RR - L) / nb
    for i in range(nb):
        if cnt[i] == 0:
            continue
        ch = min((B - T - 2) * cnt[i] / cmax, B - T - 2)
        ch = max(ch, 1.0)
        x = L + i * bw
        d.rectangle([x, B - ch, x + bw - 1, B - 1], fill=color)
    if vline is not None and lo <= vline <= hi:
        xv = L + (vline - lo) / (hi - lo) * (RR - L)
        d.line([xv, T, xv, B], fill=(200, 40, 40), width=2)
        d.text((min(xv + 4, W - 60), T + 4), vline_label, fill=(200, 40, 40))
    d.text((L, 6), f"{title}  n={len(vals)}", fill=(20, 20, 20))
    d.text((L, H - 22), f"{lo:.3g}", fill=(80, 80, 80))
    d.text((RR - 44, H - 22), f"{hi:.3g}  {xlabel}", fill=(80, 80, 80))
    d.text((8, T - 2), str(cmax), fill=(80, 80, 80))
    return img


def strip(tiles, labels, tile_h=180):
    """统一高度拼横条（上标签下图）。"""
    imgs = []
    for t in tiles:
        r = tile_h / t.shape[0]
        imgs.append(np.asarray(Image.fromarray(t).resize(
            (int(t.shape[1] * r), tile_h), Image.LANCZOS)))
    gap = 6
    W = sum(i.shape[1] for i in imgs) + gap * (len(imgs) - 1)
    cv = Image.new("RGB", (W, tile_h + 20), (255, 255, 255))
    dr = ImageDraw.Draw(cv)
    x = 0
    for lab, im in zip(labels, imgs):
        dr.text((x + 2, 2), lab, fill=(20, 20, 20))
        cv.paste(Image.fromarray(im), (x, 20))
        x += im.shape[1] + gap
    return cv


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--result", required=True)
    ap.add_argument("--source-scene", default=None)
    ap.add_argument("--target", required=True)
    ap.add_argument("--masks", required=True)
    ap.add_argument("--iters", type=int, default=300)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    dev = "cuda"
    m14 = load_mod("m14", "14_face_local_opt.py")
    m15 = load_mod("m15", "15_face_target_eval.py")
    res = Path(args.result)
    out_dir = Path(args.out) if args.out else res / "face_opt" / "audit"
    out_dir.mkdir(parents=True, exist_ok=True)
    lm = json.loads((out_dir / "landmarks.json").read_text())
    nat_meta = json.loads((out_dir / "native_meta.json").read_text())
    audit = {"args": vars(args), "sections": {}}

    import lpips as lpips_mod
    lpips_fn = lpips_mod.LPIPS(net="alex", verbose=False).to(dev)

    # ---------- 公共加载 ----------
    g0 = m14.load_ply_raw(res / "gaussians.ply")
    g3 = m14.load_ply_raw(res / "face_opt" / "p3" /
                          f"gaussians_p3_{args.iters}.ply")
    g2 = m14.load_ply_raw(res / "face_opt" / "p2" /
                          f"gaussians_p2_{args.iters}.ply")
    g1 = m14.load_ply_raw(res / "face_opt" / "p1" /
                          f"gaussians_p1_{args.iters}.ply")
    sel = np.load(res / "face_opt" / "select" / "selected_ids.npy")
    sel_meta = json.loads(
        (res / "face_opt" / "select" / "select_meta.json").read_text())
    p3m = json.loads((res / "face_opt" / "p3" / "metrics.json").read_text())
    c2w_s, K_s, src_ids = load_mod("m03", "03_render_eval.py").load_cameras(
        res / "cameras.json")
    gts_s, face_s = [], []
    for cid in src_ids:
        gts_s.append(np.asarray(Image.open(res / "rgb" / f"{cid}.png")
                                .convert("RGB"), np.float32) / 255.0)
        face_s.append(np.asarray(
            Image.open(res / "face_opt" / "masks" / f"{cid}_face.png")) > 127)
    ws, hs = gts_s[0].shape[1], gts_s[0].shape[0]
    edge = np.concatenate([gts_s[0][0], gts_s[0][-1],
                           gts_s[0][:, 0], gts_s[0][:, -1]], 0)
    bg_s = float(np.median(edge))

    source_scene = Path(args.source_scene) if args.source_scene else \
        Path("~/output/splatanything_results/_scenes").expanduser() / res.name
    s_sim, R_sim, T_sim, sim3_info = m15.estimate_sim3(res, source_scene)
    print(f"🔧 Sim(3): s={s_sim:.6f}")
    c2w_t, K_t, tgt_ids = m15.load_target_cams(Path(args.target))
    c2w_t = [m15.apply_sim3(c, s_sim, R_sim, T_sim) for c in c2w_t]
    gts_t, face_t = [], []
    mdir = Path(args.masks)
    for cid in tgt_ids:
        gts_t.append(np.asarray(
            Image.open(Path(args.target) / "images" / f"{cid}.png")
            .convert("RGB"), np.float32) / 255.0)
        face_t.append(np.asarray(Image.open(mdir / f"{cid}_face.png")) > 127)
    wt, ht = gts_t[0].shape[1], gts_t[0].shape[0]

    arms = [("p0", g0), ("p1", g1), ("p2", g2), ("p3", g3)]
    empty = np.array([], dtype=np.int64)

    # ============ A. target 评测口径补报 ============
    print("== A. target 紧框/鼻部/眼周 ==")
    secA = {"views": []}
    for a, g in arms:
        R = m14.Renderer(g, np.stack(c2w_t), np.stack(K_t), wt, ht, empty,
                         bg_s, device=dev)
        with torch.no_grad():
            rd, _ = R.render({}, list(range(len(tgt_ids))))
        rd = rd.cpu().numpy()
        for i, cid in enumerate(tgt_ids):
            lmv = lm["target"].get(cid)
            pts = lmv["pts"] if lmv else None
            bbox = lm["target"][cid]["bbox"] if lmv else None
            if bbox is None:
                ys, xs = np.where(face_t[i])
                bbox = [float(xs.min()), float(ys.min()),
                        float(xs.max()), float(ys.max())]
            view = dict(arm=a, target=cid)
            for kind in ("tight", "nose", "eyes"):
                box, approx = box_from_landmarks(pts, bbox, wt, ht, kind)
                mm = crop_metric(m14, gts_t[i], rd[i], box, lpips_fn, dev)
                view[kind] = dict(box=box, approx=approx, **mm)
            # 旧 2.3 框（追溯）
            box23 = m14.roi_box(face_t[i], 2.3, wt, ht)
            view["legacy_2p3"] = crop_metric(m14, gts_t[i], rd[i], box23,
                                             lpips_fn, dev)
            # 脸皮肤 mask 像素 PSNR
            fm = face_t[i]
            view["mask_psnr"] = m14.psnr_np(gts_t[i][fm], rd[i][fm])
            secA["views"].append(view)
        del R
        torch.cuda.empty_cache()
        print(f"  ✅ {a} target 口径补报完成")
    audit["sections"]["A_target_metrics"] = secA

    # ============ B. 原生 source 回渲 ============
    print("== B. 原生 source 回渲（1440×1920，去畸变）==")
    secB = {"views": []}
    cids_nat = list(nat_meta.keys())
    # transforms.json 是 AR 右手系；07/prior_camera 链路 = M @ diag(1,-1,-1,1)（实测，勿凭直觉改）
    FLIP = np.diag([1.0, -1.0, -1.0, 1.0])
    c2w_n = [m15.apply_sim3(np.asarray(nat_meta[c]["c2w_ar"], np.float64) @ FLIP,
                            s_sim, R_sim, T_sim) for c in cids_nat]
    K_n = [np.asarray(nat_meta[c]["K"], np.float64) for c in cids_nat]
    wn, hn = nat_meta[cids_nat[0]]["w"], nat_meta[cids_nat[0]]["h"]
    gt_n = {c: np.asarray(Image.open(out_dir / "native" / f"{c}.png")
                          .convert("RGB"), np.float32) / 255.0
            for c in cids_nat}
    renders_nat = {}
    for a in ("p0", "p3"):
        g = dict(arms)[a]
        R = m14.Renderer(g, np.stack(c2w_n), np.stack(K_n), wn, hn, empty,
                         bg_s, device=dev)
        with torch.no_grad():
            rd, _ = R.render({}, list(range(len(cids_nat))))
        renders_nat[a] = rd.cpu().numpy()
        del R
        torch.cuda.empty_cache()
    for j, cid in enumerate(cids_nat):
        pts = lm["native"][cid]["pts"]
        bbox = lm["native"][cid]["bbox"]
        for a in ("p0", "p3"):
            view = dict(arm=a, source=cid,
                        k_native=K_n[j][0, 0])
            for kind in ("tight", "nose"):
                box, approx = box_from_landmarks(pts, bbox, wn, hn, kind)
                view[kind] = dict(box=box, **crop_metric(
                    m14, gt_n[cid], renders_nat[a][j], box, lpips_fn, dev))
            view["psnr_full"] = m14.psnr_np(gt_n[cid], renders_nat[a][j])
            secB["views"].append(view)
    # K 链核对：756 工作 K 与原生 K 的尺度关系
    secB["k_check"] = dict(
        K756_fx=float(K_s[0][0, 0]),
        K_native_fx=float(K_n[0][0, 0]),
        fx_ratio=float(K_n[0][0, 0] / K_s[0][0, 0]),
        res_work=[ws, hs], res_native=[wn, hn],
        note="原生回渲直接用 transforms.json K（同 Sim(3) 模型系），"
             "与 756 工作分辨率的差异即预处理链缩放")
    audit["sections"]["B_native_rerender"] = secB
    # 裁图条
    for j, cid in enumerate(cids_nat):
        pts = lm["native"][cid]["pts"]
        bbox = lm["native"][cid]["bbox"]
        box, _ = box_from_landmarks(pts, bbox, wn, hn, "tight")
        x0, y0, x1, y1 = box
        tiles = [(gt_n[cid][y0:y1, x0:x1] * 255).astype(np.uint8)]
        for a in ("p0", "p3"):
            tiles.append((renders_nat[a][j][y0:y1, x0:x1] * 255).astype(
                np.uint8))
        strip(tiles, ["GT(原生去畸变)", "P0 原生回渲", "P3 原生回渲"],
              tile_h=200).save(out_dir / f"native_{cid}_tight.png")
    print(f"  ✅ 原生回渲 {len(cids_nat)} 视角 × 2 臂")

    # ============ C. 鼻部遮挡分层审计（source, P0） ============
    print("== C. 鼻部逐层审计 ==")
    R0 = m14.Renderer(g0, np.stack(c2w_s), np.stack(K_s), ws, hs, empty,
                      bg_s, device=dev)
    with torch.no_grad():
        rd_ed, _ = render_rgbd(R0, {}, list(range(len(src_ids))))
    rd_ed = rd_ed.cpu().numpy()
    means_t = torch.from_numpy(g0["means"]).to(dev)
    selset = set(int(x) for x in sel)
    secC = {"views": []}
    for i, cid in enumerate(src_ids):
        pts = lm["source"][cid]["pts"]
        nx, ny = pts["nose_tip"]
        w2c = np.linalg.inv(c2w_s[i]).astype(np.float32)
        Xc = (torch.from_numpy(w2c[:3, :3]).to(dev) @ means_t.T +
              torch.from_numpy(w2c[:3, 3]).to(dev)[:, None])
        z = Xc[2]
        px = Xc[:2] / z.clamp_min(1e-9)
        px = px * torch.tensor([[K_s[i][0, 0]], [K_s[i][1, 1]]]).to(dev) + \
            torch.tensor([[K_s[i][0, 2]], [K_s[i][1, 2]]]).to(dev)
        d2 = ((px[0] - nx) ** 2 + (px[1] - ny) ** 2)
        cand = (d2 <= 3.0 ** 2) & (z > 0.01)
        idx = cand.nonzero(as_tuple=True)[0]
        order = torch.argsort(z[idx])
        idx = idx[order]
        z_exp = float(rd_ed[i][int(round(ny)), int(round(nx)), 3])
        zexp = z_exp
        layers = []
        for k in idx.tolist():
            zg = float(z[k])
            layers.append(dict(
                id=int(k), z=round(zg, 5),
                rel=round(zg / zexp, 3),
                depth_bin=("front" if zg < 0.9 * zexp else
                           "at" if zg <= 1.1 * zexp else "behind"),
                selected=(int(k) in selset),
                opacity=round(float(torch.sigmoid(
                    torch.tensor(g0["logit_opac"][k])).cpu()), 3)))
        bins = {"front": [0, 0], "at": [0, 0], "behind": [0, 0]}
        for L in layers:
            bins[L["depth_bin"]][0 if L["selected"] else 1] += 1
        secC["views"].append(dict(
            source=cid, nose_px=[round(nx, 1), round(ny, 1)],
            z_expected=round(z_exp, 4), n_layers=len(layers),
            bins_selected_unselected=bins, layers=layers[:40]))
        # 只选中 / 只未选中 渲染鼻部裁图
        fw = lm["source"][cid]["bbox"][2] - lm["source"][cid]["bbox"][0]
        box, _ = box_from_landmarks(pts, lm["source"][cid]["bbox"],
                                    ws, hs, "nose")
        g_selonly = {k: v.copy() for k, v in g0.items()}
        m = np.zeros(len(g0["means"]), bool)
        m[sel] = True
        g_selonly["logit_opac"][~m] = -20.0
        g_unsel = {k: v.copy() for k, v in g0.items()}
        g_unsel["logit_opac"][m] = -20.0
        tiles = [(gts_s[i][box[1]:box[3], box[0]:box[2]] * 255).astype(np.uint8)]
        labs = ["GT"]
        for gg, lab in ((g0, "全部"), (g_selonly, "只选中"),
                        (g_unsel, "只未选中")):
            Rv = m14.Renderer(gg, np.stack(c2w_s[i:i + 1]),
                              np.stack(K_s[i:i + 1]), ws, hs, empty,
                              bg_s, device=dev)
            with torch.no_grad():
                rv, _ = Rv.render({}, [0])
            tiles.append(
                (rv.cpu().numpy()[0][box[1]:box[3], box[0]:box[2]] * 255
                 ).astype(np.uint8))
            labs.append(lab)
            del Rv
        strip(tiles, labs, tile_h=160).save(out_dir / f"nose_layers_{cid}.png")
        print(f"  ✅ {cid}: {len(layers)} 层 z_exp={z_exp:.4f} 分箱 {bins}")
    del R0
    torch.cuda.empty_cache()
    audit["sections"]["C_nose_layers"] = secC

    # ============ D. P3 参数变化审计 ============
    print("== D. P3 参数分布 ==")
    head_w = float(max(sel_meta["center_bbox_size"]))
    cap = head_w * 0.05
    d_dc = np.linalg.norm(g3["f_dc"][sel] - g0["f_dc"][sel], axis=1)
    d_op = (1 / (1 + np.exp(-g3["logit_opac"][sel])) -
            1 / (1 + np.exp(-g0["logit_opac"][sel])))
    d_sc = np.abs(g3["log_scales"][sel] - g0["log_scales"][sel]).max(axis=1)
    disp = np.linalg.norm(g3["means"][sel] - g0["means"][sel], axis=1)
    touch = disp > 0.999 * cap
    secD = dict(
        head_w_mm=round(head_w * 1000, 1), cap_mm=round(cap * 1000, 2),
        n_sel=len(sel),
        disp_mm=dict(mean=float(disp.mean() * 1000),
                     p50=float(np.quantile(disp, .5) * 1000),
                     p90=float(np.quantile(disp, .9) * 1000),
                     max=float(disp.max() * 1000)),
        touch_rate=float(touch.mean()),
        d_dc=dict(mean=float(d_dc.mean()), p90=float(np.quantile(d_dc, .9))),
        d_op=dict(mean=float(d_op.mean()), p10=float(np.quantile(d_op, .1)),
                  p90=float(np.quantile(d_op, .9)),
                  frac_decreased=float((d_op < 0).mean())),
        d_scale_log=dict(mean=float(d_sc.mean()),
                         p90=float(np.quantile(d_sc, .9))),
        lam_disp_term_check=float(10.0 * np.mean(disp ** 2)),
        note="正则终值复核 = 10*mean(delta²)，模型系尺度；"
             "回退链 gaussians_p3_300.ply ↔ gaussians.ply")
    audit["sections"]["D_p3_param_audit"] = secD
    print(f"  触顶率 {touch.mean():.1%} | disp p90 {np.quantile(disp,.9)*1000:.2f}mm"
          f" | 正则终值 {10.0*np.mean(disp**2):.3e}")
    # 直方图拼图
    panels = [
        hist_panel(None, disp * 1000, "中心位移 |Δmean|", "mm",
                   vline=cap * 1000, vline_label=f"cap {cap*1000:.1f}mm",
                   color=(200, 90, 40)),
        hist_panel(None, d_dc, "ΔDC（f_dc L2）", "f_dc"),
        hist_panel(None, d_op, "Δopacity（sigmoid 差）", "Δα",
                   color=(15, 110, 110)),
        hist_panel(None, d_sc, "Δscale（|log 比| 最大轴）", "log",
                   color=(120, 120, 120)),
    ]
    figD = Image.new("RGB", (960, 600), (255, 255, 255))
    for k, p in enumerate(panels):
        figD.paste(p, ((k % 2) * 480, (k // 2) * 300))
    figD.save(out_dir / "fig_p3_param_hist.png")
    # 位移散布图：view0 投影叠图
    R0v = m14.Renderer(g0, np.stack(c2w_s[0:1]), np.stack(K_s[0:1]),
                       ws, hs, empty, bg_s, device=dev)
    with torch.no_grad():
        base_img, _ = R0v.render({}, [0])
    canv = (base_img[0].cpu().numpy() * 255).astype(np.uint8)
    canv = cv2.resize(canv, (canv.shape[1] // 2, canv.shape[0] // 2))
    w2c0 = np.linalg.inv(c2w_s[0]).astype(np.float32)
    m_sel = g0["means"][sel].astype(np.float32)
    Xc = (w2c0[:3, :3] @ m_sel.T + w2c0[:3, 3:4])
    px = (Xc[:2] / np.clip(Xc[2], 1e-9, None) *
          np.array([[K_s[0][0, 0]], [K_s[0][1, 1]]], np.float32) +
          np.array([[K_s[0][0, 2]], [K_s[0][1, 2]]], np.float32)) * 0.5
    for k in range(len(sel)):
        x, y = px[0, k], px[1, k]
        if not (0 <= x < canv.shape[1] and 0 <= y < canv.shape[0]):
            continue
        t = min(disp[k] / cap, 1.0)
        col = (int(30 + 225 * t), int(60), int(230 - 200 * t))
        cv2.circle(canv, (int(x), int(y)), 2, colormap_rgb_to_bgr(col), -1)
    cv2.putText(canv, f"disp scatter (blue->red, red=cap)", (10, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    Image.fromarray(canv).save(out_dir / "fig_p3_disp_scatter.png")
    del R0v
    torch.cuda.empty_cache()

    # ============ E. P3 参数回退敏感性 ============
    print("== E. P3 回退敏感性 ==")
    variants = {}
    for field, lab in (("f_dc", "rb_dc"), ("logit_opac", "rb_op"),
                       ("means", "rb_mean")):
        gv = {k: v.copy() for k, v in g3.items()}
        gv[field][sel] = g0[field][sel]
        variants[lab] = gv
    secE = {"source": [], "target": []}
    # source 侧（紧脸框，756²）
    for lab, gv in variants.items():
        Rv = m14.Renderer(gv, np.stack(c2w_s), np.stack(K_s), ws, hs, empty,
                          bg_s, device=dev)
        with torch.no_grad():
            rd, _ = Rv.render({}, list(range(len(src_ids))))
        rd = rd.cpu().numpy()
        for i, cid in enumerate(src_ids):
            lmv = lm["source"][cid]
            box, approx = box_from_landmarks(lmv["pts"], lmv["bbox"],
                                             ws, hs, "tight")
            mm = crop_metric(m14, gts_s[i], rd[i], box, lpips_fn, dev)
            secE["source"].append(dict(variant=lab, source=cid, **mm))
        del Rv
        torch.cuda.empty_cache()
    # target 侧（紧脸框）
    for lab, gv in variants.items():
        Rv = m14.Renderer(gv, np.stack(c2w_t), np.stack(K_t), wt, ht, empty,
                          bg_s, device=dev)
        with torch.no_grad():
            rd, _ = Rv.render({}, list(range(len(tgt_ids))))
        rd = rd.cpu().numpy()
        for i, cid in enumerate(tgt_ids):
            lmv = lm["target"].get(cid)
            pts = lmv["pts"] if lmv else None
            bbox = lmv["bbox"] if lmv else None
            if bbox is None:
                ys, xs = np.where(face_t[i])
                bbox = [float(xs.min()), float(ys.min()),
                        float(xs.max()), float(ys.max())]
            box, approx = box_from_landmarks(pts, bbox, wt, ht, "tight")
            mm = crop_metric(m14, gts_t[i], rd[i], box, lpips_fn, dev)
            secE["target"].append(dict(variant=lab, target=cid, **mm))
        del Rv
        torch.cuda.empty_cache()
    audit["sections"]["E_rollback"] = secE
    # target 裁图条：GT | P0 | P3 | rb_dc | rb_op | rb_mean
    for i, cid in enumerate(tgt_ids):
        lmv = lm["target"].get(cid)
        pts = lmv["pts"] if lmv else None
        bbox = lmv["bbox"] if lmv else None
        if bbox is None:
            ys, xs = np.where(face_t[i])
            bbox = [float(xs.min()), float(ys.min()),
                    float(xs.max()), float(ys.max())]
        box, _ = box_from_landmarks(pts, bbox, wt, ht, "tight")
        x0, y0, x1, y1 = box
        tiles = [(gts_t[i][y0:y1, x0:x1] * 255).astype(np.uint8)]
        labs = ["GT"]
        for a, g in (("p0", g0), ("p3", g3)):
            Rv = m14.Renderer(g, np.stack(c2w_t[i:i + 1]),
                              np.stack(K_t[i:i + 1]), wt, ht, empty,
                              bg_s, device=dev)
            with torch.no_grad():
                rv, _ = Rv.render({}, [0])
            tiles.append((rv.cpu().numpy()[0][y0:y1, x0:x1] * 255).astype(
                np.uint8))
            labs.append(a.upper())
            del Rv
        for lab, gv in variants.items():
            Rv = m14.Renderer(gv, np.stack(c2w_t[i:i + 1]),
                              np.stack(K_t[i:i + 1]), wt, ht, empty,
                              bg_s, device=dev)
            with torch.no_grad():
                rv, _ = Rv.render({}, [0])
            tiles.append((rv.cpu().numpy()[0][y0:y1, x0:x1] * 255).astype(
                np.uint8))
            labs.append(lab)
            del Rv
        strip(tiles, labs, tile_h=200).save(
            out_dir / f"rollback_target_{cid}.png")
    print("  ✅ 回退渲染 source×3 + target×2 完成")

    with open(out_dir / "audit_metrics.json", "w", encoding="utf-8",
              newline="\n") as f:
        json.dump(audit, f, indent=1, ensure_ascii=False)
    print(f"\n🎉 Done: {out_dir}")

    # ---------- 汇总打印 ----------
    print("\n📊 A. target 紧脸框 LPIPS（越小越好）")
    for a in ("p0", "p1", "p2", "p3"):
        vals = [v["tight"].get("lpips") for v in secA["views"]
                if v["arm"] == a]
        vals = [x for x in vals if x is not None]
        if vals:
            print(f"  {a}: tight LPIPS {np.mean(vals):.4f} | "
                  + " ".join(f"{x:.4f}" for x in vals))
    print("📊 D. P3 参数审计:")
    print(json.dumps(secD, indent=1, ensure_ascii=False))


def colormap_rgb_to_bgr(c):
    return (c[2], c[1], c[0])


if __name__ == "__main__":
    main()

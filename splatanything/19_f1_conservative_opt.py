#!/usr/bin/env python3
"""19_f1_conservative_opt.py — F1 保守自由度对照（rosemary 静态包，诊断用测试时优化）。

实现 da3_human/EXPERIMENT_PLAN.md 的 F1（P0~P3 验收失败后的保守对照）：
  F1-A 中心/DC/opacity 固定；只微调 scale/quat，且**约束相对 P0 初值的变化**：
       scale 逐点钳制 [s0/r, s0*r]（r=4，预注册）；quat 测地角钳制 ≤45°（预注册）。
       目的：隔离 P2 的透明度补偿与覆盖变化；是诊断，不预设可修鼻子。
  F1-B DC/opacity/scale/quat 全固定；脸部平滑位移由**共享控制点**驱动：
       K=1024 控制点（种子固定子采样，预注册）；选中点位移 = 对 8-NN 控制点的
       空间×颜色门控凸组合（皮肤语义门控近似：鼻翼/唇线等颜色不连续处权重自动压低；
       可见性：控制点全部来自 select 阶段 support≥2 的选中簇）。
       位移上限 = 1% 头宽（预注册保守起点，非最优值），逐控制点投影并记录触顶率。
       正则：lam_disp·mean(u²) + lam_smooth·Σ_edge w_ij‖u_i−u_j‖²/Σw（边差平滑，
       图连接 = 控制点 kNN + 同色门控，不跨鼻翼/嘴唇）。
       凸组合 ⇒ 插值后逐点位移 ≤ 控制点位移上限，不需要逐点再钳。

关键纪律（对应计划原文）：
  - 各臂从**同一 P0**（原生 gaussians.ply + 同一 select）独立开始，固定预算 ≤300 步。
  - 只报预注册 checkpoint 0/100/300；不靠 target 挑最佳步。
  - 固定块参数用原值重组（不进 optimizer）；每 checkpoint 核对 fixed hash。
  - 禁止不透明度大幅减小来隐藏错层：F1-A/B 均不优化 opacity（物理冻结）。
  - 无新视角改善就停止该轮，不循环调学习率/正则。

预注册超参（写死为默认值，非调优结果）：
  f1a: lr_scale=5e-3, lr_quat=1e-3, rel_scale_range=4.0, max_quat_deg=45
  f1b: lr_u=1.6e-4（与 P3 mean lr 同量级）, n_ctrl=1024, knn=8,
       max_disp_frac=0.01, lam_disp=10, lam_smooth=10
  共用: lam_fid=0.5, eval_at=0,100,300, iters≤300

用法：
  python 19_f1_conservative_opt.py --result <02输出目录> --arm f1a --iters 300
  python 19_f1_conservative_opt.py --result <02输出目录> --arm f1b --iters 300
导出 PLY 命名 gaussians_<arm>_<iters>.ply，可被 15_face_target_eval.py --arms p0,f1a,f1b 验收。
"""
import argparse
import importlib.util
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as Fn
from PIL import Image

_M14 = None


def m14():
    """复用 14 的 Renderer / PLY IO / 损失与评估件。"""
    global _M14
    if _M14 is None:
        spec = importlib.util.spec_from_file_location(
            "face_local_opt", str(Path(__file__).parent / "14_face_local_opt.py"))
        m = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(m)
        _M14 = m
    return _M14


# ------------------------------------------------------------------ F1-B 图构造
def gated_knn_weights(d_kn, c_kn, sigma, sig_c):
    """kNN 边权 = 空间近 × 颜色近（与 14 P3 同门控；颜色不连续处权重自动压低）。"""
    w_sp = torch.exp(-(d_kn ** 2) / (2 * sigma ** 2))
    w_co = torch.exp(-(c_kn ** 2) / (2 * sig_c ** 2))
    return (w_sp * w_co).detach()


def build_f1b_graph(m0, c0, n_ctrl, knn, seed, device):
    """控制点子采样 + 插值权重 W[S,K] + 控制点边图。返回 dict。"""
    S = m0.shape[0]
    K = min(n_ctrl, S)
    gen = torch.Generator(device="cpu").manual_seed(seed)
    perm = torch.randperm(S, generator=gen)[:K].to(device)
    mc = m0[perm].contiguous()                       # [K,3] 控制点
    # 插值：选中点 → 控制点（空间×颜色门控，行归一化 = 凸组合）
    d = torch.cdist(m0, mc)                          # [S,K]
    idx = d.topk(knn, largest=False).indices         # [S,k]
    d_kn = torch.gather(d, 1, idx)
    cd = torch.cdist(c0, c0[perm])
    c_kn = torch.gather(cd, 1, idx)
    dpos = d[d > 0]
    cpos = cd[cd > 0]
    sigma = float(dpos.median()) if dpos.numel() else 1.0
    sig_c = float(cpos.median()) if cpos.numel() else 1.0
    w = gated_knn_weights(d_kn, c_kn, sigma, sig_c)  # [S,k]
    w = w / w.sum(dim=1, keepdim=True).clamp_min(1e-12)
    W = torch.zeros(S, K, device=device)
    W.scatter_(1, idx, w)                            # [S,K] 稀疏稠化（S≈15k,K=1k → 15M,OK）
    # 控制点边图（边差正则用）
    dcc = torch.cdist(mc, mc)
    eidx = dcc.topk(knn + 1, largest=False).indices[:, 1:]
    d_e = torch.gather(dcc, 1, eidx)
    c_e = torch.gather(torch.cdist(c0[perm], c0[perm]), 1, eidx)
    edge_w = gated_knn_weights(d_e, c_e, sigma, sig_c)
    return dict(K=K, perm=perm, W=W, eidx=eidx, edge_w=edge_w,
                sigma=sigma, sig_c=sig_c)


# ------------------------------------------------------------------ F1-A quat 工具
def quat_geodesic_deg(q, q0):
    """q,q0: [N,4]，返回逐点测地角（度）。"""
    q = Fn.normalize(q, dim=-1)
    q0 = Fn.normalize(q0, dim=-1)
    dot = (q * q0).sum(-1).abs().clamp(max=1.0)
    return 2.0 * torch.acos(dot) * 180.0 / np.pi


def clamp_quat_cone(q, q0, max_deg):
    """超出测地锥的点沿弧投影回锥面（slerp 到 q0→q 的 max_deg/angle 处）。"""
    q = Fn.normalize(q, dim=-1)
    q0n = Fn.normalize(q0, dim=-1)
    raw_dot = (q * q0n).sum(-1, keepdim=True)
    sign = torch.where(raw_dot < 0, -1.0, 1.0)
    q = q * sign                                     # 转到与 q0 同半球
    dot = (q * q0n).sum(-1).abs().clamp(max=1.0)
    ang = 2.0 * torch.acos(dot)                      # 弧度 [N]
    max_rad = max_deg * np.pi / 180.0
    over = ang > max_rad
    if not bool(over.any()):
        return q
    t = (max_rad / ang[over].clamp_min(1e-9)).clamp(max=1.0)
    qa, qb = q0n[over], q[over]
    sin_t = torch.sin(t)[:, None]
    out = qa * (torch.cos(t)[:, None]) + qb * sin_t  # 同半球 slerp ≈ 规范化插值
    q = torch.where(over[:, None], Fn.normalize(out, dim=-1), q)
    return q


# ------------------------------------------------------------------ 主流程
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--result", required=True)
    ap.add_argument("--arm", required=True, choices=["f1a", "f1b"])
    ap.add_argument("--iters", type=int, default=300)
    ap.add_argument("--out", default=None)
    ap.add_argument("--seed", type=int, default=0)
    # f1a
    ap.add_argument("--lr-scale", type=float, default=5e-3)
    ap.add_argument("--lr-quat", type=float, default=1e-3)
    ap.add_argument("--rel-scale-range", type=float, default=4.0,
                    help="scale 逐点钳制 [s0/r, s0*r]（预注册 r=4）")
    ap.add_argument("--max-quat-deg", type=float, default=45.0,
                    help="quat 测地角上限（预注册 45°）")
    # f1b
    ap.add_argument("--lr-u", type=float, default=1.6e-4)
    ap.add_argument("--n-ctrl", type=int, default=1024)
    ap.add_argument("--max-disp-frac", type=float, default=0.01,
                    help="控制点位移上限 = 头宽 × 此值（预注册 1% 头宽）")
    ap.add_argument("--lam-disp", type=float, default=10.0)
    ap.add_argument("--lam-smooth", type=float, default=10.0)
    # 共用
    ap.add_argument("--knn", type=int, default=8)
    ap.add_argument("--lam-fid", type=float, default=0.5)
    ap.add_argument("--lr-final", type=float, default=0.05)
    ap.add_argument("--eval-at", default="0,100,300")
    args = ap.parse_args()
    if args.iters > 300:
        raise SystemExit("❌ F1 固定预算 ≤300 步")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    dev = "cuda"
    m = m14()
    res = Path(args.result)
    masks_dir = res / "face_opt" / "masks"
    sel_dir = res / "face_opt" / "select"
    out_dir = Path(args.out) if args.out else res / "face_opt" / args.arm
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---------- 加载（与 14 同口径） ----------
    g_raw = m.load_ply_raw(res / "gaussians.ply")
    re = m.re03()
    c2w, K, ids = re.load_cameras(res / "cameras.json")
    sel = np.load(sel_dir / "selected_ids.npy")
    sel_meta = json.loads((sel_dir / "select_meta.json").read_text())
    with Image.open(res / "rgb" / f"{ids[0]}.png") as im:
        w, h = im.size
    gts, face_ms, band_ms, person_ms = [], [], [], []
    for cid in ids:
        gt = np.asarray(Image.open(res / "rgb" / f"{cid}.png").convert("RGB"),
                        np.float32) / 255.0
        face_ms.append(np.asarray(Image.open(masks_dir / f"{cid}_face.png")) > 127)
        band_ms.append(np.asarray(Image.open(masks_dir / f"{cid}_band.png")) > 127)
        person_ms.append(np.asarray(Image.open(masks_dir / f"{cid}_person.png")) > 127)
        gts.append(gt)
    gt_t = torch.from_numpy(np.stack(gts)).permute(0, 3, 1, 2).to(dev)
    w_face, m_out = [], []
    for f, b in zip(face_ms, band_ms):
        k16 = cv2_dilate(f)
        wm = f.astype(np.float32) + b.astype(np.float32) * 0.5
        w_face.append(torch.from_numpy(wm).to(dev))
        m_out.append(torch.from_numpy((~k16).astype(np.float32)).to(dev))
    roi_boxes = [m.roi_box(f, 2.3, w, h) for f in face_ms]
    edge = np.concatenate([gts[0][0], gts[0][-1], gts[0][:, 0], gts[0][:, -1]], 0)
    bg = float(np.median(edge))

    R = m.Renderer(g_raw, c2w, K, w, h, sel, bg, device=dev)
    n_sel = len(sel)
    print(f"🚀 [19] arm={args.arm} iters={args.iters} | 高斯 {len(g_raw['means']):,} "
          f"| 选中 {n_sel:,} | 视角 {len(ids)} | bg={bg:.3f}")
    print(f"   固定块 hash: {R.fixed_hash}")

    # ---------- P0 恒等校验（与 14 相同三项） ----------
    checks = {}
    with torch.no_grad():
        direct, _ = R.render({}, [0])
        reasm = R.render({k: R.base[k][R.sel] for k in
                          ("means", "log_scales", "logit_opac", "quats", "f_dc")}, [0])
        checks["reassemble_mae"] = float((reasm[0] - direct[0]).abs().mean())
        x0, y0, x1, y1 = roi_boxes[0]
        K2 = torch.from_numpy(m.crop_K(K[0], x0, y0)[None].astype(np.float32)).to(dev)
        roi_rd, _ = R.render({}, [0], Ks_override=K2, wh=(x1 - x0, y1 - y0))
        checks["roi_crop_mae"] = float((roi_rd[0] - direct[0][y0:y1, x0:x1]).abs().mean())
    print(f"   P0 恒等: 重组 MAE={checks['reassemble_mae']:.2e}  "
          f"ROI裁切 MAE={checks['roi_crop_mae']:.2e}")
    if checks["reassemble_mae"] > 1e-4:
        print("❌ P0 恒等门未过")

    # ---------- 可训参数组与约束上下文 ----------
    if args.arm == "f1a":
        trainable = ["log_scales", "quats"]
        params = {k: torch.nn.Parameter(R.base[k][R.sel].clone()) for k in trainable}
        ls0 = R.base["log_scales"][R.sel].detach()
        q0 = R.base["quats"][R.sel].detach()
        ls_lo = ls0 - float(np.log(args.rel_scale_range))
        ls_hi = ls0 + float(np.log(args.rel_scale_range))
        ctx = dict(ls0=ls0, q0=q0, ls_lo=ls_lo, ls_hi=ls_hi)
        lrs = {"log_scales": args.lr_scale, "quats": args.lr_quat}
        print(f"   F1-A: scale 钳 [{1/args.rel_scale_range:.2f}×, "
              f"{args.rel_scale_range}×]×s0, quat ≤{args.max_quat_deg}°")
    else:
        m0 = R.base["means"][R.sel].detach()
        c0 = R.base["f_dc"][R.sel].detach()
        head_w = float(max(sel_meta["center_bbox_size"]))
        max_disp = head_w * args.max_disp_frac
        graph = build_f1b_graph(m0, c0, args.n_ctrl, args.knn, args.seed, dev)
        u = torch.nn.Parameter(torch.zeros(graph["K"], 3, device=dev))
        W = graph["W"]
        ctx = dict(m0=m0, u=u, W=W, graph=graph, max_disp=max_disp, head_w=head_w)
        trainable = ["u"]
        lrs = {"u": args.lr_u}
        print(f"   F1-B: 控制点 {graph['K']} (kNN k={args.knn}), "
              f"头宽 {head_w:.4f} → 位移上限 {max_disp*1000:.3f} (模型单位×1e3), "
              f"σ_s={graph['sigma']:.4f} σ_c={graph['sig_c']:.3f}")

    opt = torch.optim.Adam(
        [{"params": [params[k] if args.arm == "f1a" else ctx["u"]],
          "lr": lrs[k]} for k in trainable], eps=1e-15)
    base_lrs = [gp["lr"] for gp in opt.param_groups]

    # ---------- P0 快照 + 初始评估 ----------
    with torch.no_grad():
        p0_render, _ = R.render({}, list(range(len(ids))))
    p0_render = p0_render.detach()
    p0_perm = p0_render.permute(0, 3, 1, 2).contiguous()

    eval_at = sorted({int(x) for x in args.eval_at.split(",")} | {0, args.iters})

    def cur_means():
        if args.arm == "f1b":
            return ctx["m0"] + ctx["W"] @ ctx["u"]
        return None

    def render_params():
        if args.arm == "f1a":
            return {k: params[k] for k in trainable}
        return {"means": cur_means()}

    def audit_extra():
        """F1 特有审计量（写进 history 对应 entry）。"""
        if args.arm == "f1a":
            with torch.no_grad():
                rel = (params["log_scales"] - ctx["ls0"]).abs().max().item()
                ang = float(quat_geodesic_deg(params["quats"], ctx["q0"]).max())
            return dict(max_log_scale_dev=round(rel, 4), max_quat_deg=round(ang, 2))
        with torch.no_grad():
            un = ctx["u"].norm(dim=-1)
            touch = (un >= 0.99 * ctx["max_disp"]).float().mean().item()
            interp = (ctx["W"] @ ctx["u"]).norm(dim=-1)
        return dict(disp_touch_rate=round(touch, 4),
                    u_mean_abs=round(float(un.mean()), 6),
                    interp_max=round(float(interp.max()), 6),
                    max_disp=round(ctx["max_disp"], 6))

    def run_eval(step):
        entry = m.evaluate(R, render_params(), gt_t, gts, w_face, m_out, roi_boxes,
                           face_ms, ids, p0_render, step, person_ms)
        entry.update(audit_extra())
        return entry

    history = [run_eval(0)]
    print_metrics_f1(history[-1])

    # ---------- 训练 ----------
    t0 = time.time()
    stop_reason = None
    for it in range(1, args.iters + 1):
        frac = it / max(1, args.iters)
        for gp, lr0 in zip(opt.param_groups, base_lrs):
            gp["lr"] = lr0 * (args.lr_final ** frac)
        rd, _ = R.render(render_params(), list(range(len(ids))))
        rd_p = rd.permute(0, 3, 1, 2)
        loss_face = 0.0
        loss_fid = 0.0
        for i in range(len(ids)):
            wm = w_face[i][None]
            l1 = (rd_p[i:i + 1] - gt_t[i:i + 1]).abs() * wm
            loss_face = loss_face + l1.sum() / (wm.sum() * 3)
            x0, y0, x1, y1 = roi_boxes[i]
            ssim_v = m.ssim_t(rd_p[i:i + 1, :, y0:y1, x0:x1],
                              gt_t[i:i + 1, :, y0:y1, x0:x1])
            loss_face = loss_face + 0.2 * (1 - ssim_v)
            mo = m_out[i][None]
            loss_fid = loss_fid + ((rd_p[i:i + 1] - p0_perm[i:i + 1]).abs() * mo
                                   ).sum() / (mo.sum() * 3)
        loss = loss_face / len(ids) + args.lam_fid * loss_fid / len(ids)
        if args.arm == "f1b":
            uu = ctx["u"]
            loss = loss + args.lam_disp * (uu ** 2).mean()
            eidx, ew = ctx["graph"]["eidx"], ctx["graph"]["edge_w"]
            dd = uu[:, None, :] - uu[eidx]
            loss = loss + args.lam_smooth * (
                ((dd ** 2).sum(-1) * ew).sum() / ew.sum().clamp_min(1e-12))
        if not torch.isfinite(loss):
            stop_reason = f"loss 非有限值 @it{it}"
            break
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        # ---------- 约束投影（预注册） ----------
        with torch.no_grad():
            if args.arm == "f1a":
                params["log_scales"].clamp_(ctx["ls_lo"], ctx["ls_hi"])
                params["quats"].data = clamp_quat_cone(
                    params["quats"].data, ctx["q0"], args.max_quat_deg)
            else:
                un = ctx["u"].norm(dim=-1, keepdim=True)
                f = (ctx["max_disp"] / un.clamp_min(1e-12)).clamp(max=1.0)
                ctx["u"].data = ctx["u"].data * f
        if it % 20 == 0 or it == 1:
            print(f"  it {it:4d}/{args.iters}  loss {loss.item():.5f}  "
                  f"face {(loss_face/len(ids)).item():.5f}  "
                  f"{(time.time()-t0)/it:.3f}s/it")
        if it in eval_at:
            entry = run_eval(it)
            history.append(entry)
            print_metrics_f1(entry)
            with torch.no_grad():
                if args.arm == "f1a":
                    smax = torch.exp(params["log_scales"]).max(dim=-1).values
                    if float(smax.quantile(0.99)) > 10 * m.disp_scale_ref(R):
                        stop_reason = "scale 爆炸（p99 超初值 10×）"
                if entry["psnr_outside_p0"] is not None and \
                        entry["psnr_outside_p0"] < 35.0:
                    stop_reason = (f"ROI 外保真劣化 "
                                   f"({entry['psnr_outside_p0']:.1f}dB < 35dB)")
            if stop_reason:
                print(f"  ⚠️ 停止: {stop_reason}")
                break

    dt = time.time() - t0
    vram = torch.cuda.max_memory_allocated() / 1024 ** 3
    print(f"⏱️ {dt:.1f}s | 峰值显存 {vram:.2f} GB | {audit_extra()}")

    # ---------- 导出 ----------
    if args.arm == "f1a":
        final = {"log_scales": params["log_scales"].detach(),
                 "quats": params["quats"].detach()}
    else:
        final = {"means": cur_means().detach()}
    g_out = {k: R.assemble(k, final.get(k)).cpu().numpy()
             for k in ("means", "log_scales", "logit_opac", "quats", "f_dc")}
    g_out["quats"] = g_out["quats"] / np.linalg.norm(
        g_out["quats"], axis=-1, keepdims=True).clip(1e-12)
    ply_path = out_dir / f"gaussians_{args.arm}_{args.iters}.ply"
    m.write_ply(ply_path, g_out)
    m.save_outputs(out_dir, args, history, checks, R, sel_meta, stop_reason, dt, vram)
    m.save_crops(out_dir, R, final, gts, roi_boxes, ids)
    print(f"🎉 Done: {out_dir}")


def cv2_dilate(face):
    import cv2
    k16 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (33, 33))
    return cv2.dilate(face.astype(np.uint8), k16).astype(bool)


def print_metrics_f1(e):
    def f(v):
        return f"{v:.2f}" if v is not None else "  n/a"
    extra = ""
    if "max_quat_deg" in e:
        extra = f" | Δscale {e['max_log_scale_dev']}  Δquat {e['max_quat_deg']}°"
    if "disp_touch_rate" in e:
        extra = (f" | 触顶 {e['disp_touch_rate']*100:.1f}%  "
                 f"|u| {e['u_mean_abs']}  interpMax {e['interp_max']}")
    print(f"  📊 it{e['step']:4d}  full {f(e['psnr_full'])}  "
          f"face {f(e['psnr_face'])}  sharp {f(e['sharp_ratio'])}  "
          f"outP0 {f(e['psnr_outside_p0'])}{extra}")


if __name__ == "__main__":
    main()

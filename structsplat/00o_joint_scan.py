# -*- coding: utf-8 -*-
"""
00o_joint_scan.py — 约定 × 对齐策略 联合扫描

为什么需要（2026-11-20）：
  00n 把相机系约定穷举了 16 种，全部落在 6.6~8.4 dB，而 pose-free 自洽基线是 20.5 dB。
  → 光换约定到不了目标，说明**对齐方式**也在丢精度。
  00m 诊断显示：对齐后中心残差 RMS 只有 0.033（拟合不错），但**逐视角旋转差恒为 90.00°**。
  恒定 90° 不可能来自估计误差，只能是 (a) 多一个轴交换，或 (b) 旋转根本没被正确传递。

  所以必须**联合**扫描：约定（决定 GT 相机长什么样）× 对齐策略（决定怎么搬进模型系）。
  任一单独固定都会把错误锁死。

对齐策略：
  A0 none          —— 不做对齐（对照）
  A1 center_kabsch —— 中心做 Kabsch（infer.py 现用）
  A2 rot_axes      —— 旋转用相机朝向 Kabsch（样本 3 倍，窄基线不退化）
  A3 procrustes    —— 中心 + 朝向联合

用法：
    python 00o_joint_scan.py <cache.pt> <transforms.json>
"""
import os, sys, math, itertools, json
import numpy as np
import torch
import torch.nn.functional as F

SD = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SD)
import infer as I
import importlib.util as _ilu

_spec = _ilu.spec_from_file_location("k00", os.path.join(SD, "00k_offline_align.py"))
_k = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_k)


def build_gt(tpath, image_paths, new_h, new_w, undist, gl2cv, transpose, kswap):
    with open(tpath, "r", encoding="utf-8") as f:
        meta = json.load(f)
    by = {os.path.basename(fr.get("file_path", "")): fr for fr in meta.get("frames", [])}
    _Kuse = undist[2] if undist is not None else None
    M = torch.tensor([[0., 1., 0.], [1., 0., 0.], [0., 0., 1.]], dtype=torch.float32)
    FLIP = torch.diag(torch.tensor([1., -1., -1.], dtype=torch.float32))
    c2ws, Ks = [], []
    for p in image_paths:
        fr = by[os.path.basename(p)]
        W0, H0 = fr["w"], fr["h"]
        if _Kuse is not None:
            fx, fy = float(_Kuse[0, 0]), float(_Kuse[1, 1])
            cx, cy = float(_Kuse[0, 2]), float(_Kuse[1, 2])
        else:
            fx, fy, cx, cy = fr["fl_x"], fr["fl_y"], fr["cx"], fr["cy"]
        tr = (new_h < new_w) and (H0 > W0)
        if tr:
            s_u, s_v = new_w / H0, new_h / W0
            fx2, fy2, cx2, cy2 = fy * s_u, fx * s_v, cy * s_u, cx * s_v
        else:
            sx, sy = new_w / W0, new_h / H0
            fx2, fy2, cx2, cy2 = fx * sx, fy * sy, cx * sx, cy * sy
        if kswap:
            fx2, fy2, cx2, cy2 = fy2, fx2, cy2, cx2
        K = torch.tensor([[fx2, 0., cx2], [0., fy2, cy2], [0., 0., 1.]], dtype=torch.float32)
        c2w = torch.tensor(fr["transform_matrix"], dtype=torch.float32).clone()
        if gl2cv:
            c2w[:3, :3] = c2w[:3, :3] @ FLIP
        if tr and transpose:
            c2w[:3, :3] = c2w[:3, :3] @ M
        c2ws.append(c2w); Ks.append(K)
    return torch.linalg.inv(torch.stack(c2ws)), torch.stack(Ks)


def render_psnr(d, w2cs, Ks_r, render_h, render_w, dev):
    from gsplat import rasterization
    means = d["means"].to(dev); quats = d["quats"].to(dev)
    scales = d["scales"].to(dev); opac = d["opac"].to(dev); colors = d["colors"].to(dev)
    src = d["src"].to(dev)[0]
    with torch.no_grad():
        out, _, _ = rasterization(
            means=means, quats=quats, scales=scales, opacities=opac, colors=colors,
            viewmats=w2cs.to(dev).float(), Ks=Ks_r.to(dev).float(),
            width=render_w, height=render_h, render_mode="RGB+D", packed=False,
            backgrounds=torch.zeros((w2cs.shape[0], 3), dtype=torch.float32, device=dev))
    rgb = out[..., :3]
    ps = []
    for i in range(rgb.shape[0]):
        gt = F.interpolate(src[i].unsqueeze(0), (render_h, render_w), mode="bilinear",
                           align_corners=False, antialias=True)[0].clamp(0, 1)
        pr = rgb[i].permute(2, 0, 1).clamp(0, 1)
        mse = float(((gt - pr) ** 2).mean())
        ps.append(10 * math.log10(1.0 / max(mse, 1e-12)))
    return float(np.mean(ps))


def main():
    cache, tpath = sys.argv[1], sys.argv[2]
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    d = torch.load(cache, map_location="cpu", weights_only=False)
    paths = d["paths"]; new_h, new_w = int(d["new_h"]), int(d["new_w"])
    render_h, render_w = int(d["render_h"]), int(d["render_w"])
    pred_w2c = d["pred_w2c"].to(dev); pred_K = d["pred_K"].to(dev)
    print(f"高斯 {d['means'].shape[0]}  输入 {new_h}×{new_w}  渲染 {render_h}×{render_w}")

    undist = I.probe_distortion(tpath)
    s = render_w / new_w
    Kr_pf = pred_K.clone(); Kr_pf[:, 0, :] *= s; Kr_pf[:, 1, :] *= s
    base = render_psnr(d, pred_w2c, Kr_pf, render_h, render_w, dev)
    print(f"★ pose-free 自洽基线: {base:.3f} dB\n")

    def _wrap(fn):
        """统一成 (pred, gt) -> aligned_w2c。fn 返回 (s,R,t,info)，且内部用 float64 计算，
        所以 apply_sim3 之前必须把 R/t 转回 gt 的 dtype，否则报
        "expected scalar type Float but found Double"（2026-11-20 实测）。"""
        def _f(P, G):
            s_, R_, t_, i_ = fn(P, G)
            R_ = R_.to(dtype=G.dtype)
            t_ = t_.to(dtype=G.dtype)
            return _k.apply_sim3(G, s_, R_, t_), i_
        return _f

    SOLVERS = {
        "A0_none": None,
        "A1_center": _wrap(_k.solve_center_kabsch),
        "A2_axes": _wrap(_k.solve_rot_from_axes),
        "A3_procrustes": _wrap(_k.solve_procrustes_full),
    }

    rows = []
    for gl2cv, transpose, kswap in itertools.product([0, 1], repeat=3):
        w2c_raw, K = build_gt(tpath, paths, new_h, new_w, undist,
                              bool(gl2cv), bool(transpose), bool(kswap))
        w2c_raw = w2c_raw.to(dev)
        Kr = K.clone(); Kr[:, 0, :] *= s; Kr[:, 1, :] *= s
        for name, fn in SOLVERS.items():
            err = ""
            try:
                if fn is None:
                    w2c_a = w2c_raw
                else:
                    w2c_a, info = fn(pred_w2c.float(), w2c_raw.float())
                p = render_psnr(d, w2c_a, Kr, render_h, render_w, dev)
            except Exception as e:
                p = float("nan")
                err = f"  ⚠️ {type(e).__name__}: {e}"
            rows.append((p, name, gl2cv, transpose, kswap))
            print(f"  gl2cv={gl2cv} transp={transpose} kswap={kswap} {name:14s} → {p:6.2f} dB{err}")

    rows.sort(key=lambda r: -(r[0] if not math.isnan(r[0]) else -99))
    print("\n" + "=" * 76)
    print(f"★ 基线 {base:.2f} dB — 按 PSNR 排序 Top 8：")
    for r in rows[:8]:
        print(f"  {r[0]:6.2f} dB  Δ{r[0]-base:+7.2f}  {r[1]:14s} gl2cv={r[2]} transp={r[3]} kswap={r[4]}")
    print("=" * 76)


if __name__ == "__main__":
    main()

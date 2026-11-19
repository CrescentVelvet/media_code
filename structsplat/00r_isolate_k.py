# -*- coding: utf-8 -*-
"""
00r_isolate_k.py — 分离「K 的贡献」与「外参的贡献」

为什么需要（2026-11-20）：
  00q 已测出：3° 相机旋转误差只吃掉 5.2dB（20.53→15.30），
  但 GT 分支最好只有 11.32dB，**还差 4dB 无解释**。

  观察：pred_K 的焦距在视角间**不一致**——
      v0 fx=385.5  v1 fx=397.7  v2 fx=387.9  v3 fx=399.6   （波动 3.6%）
  而 GT_K 恒定 fx=373.44。
  高斯是 output2gaussian 用 **各自视角的 pred_K** 反投影出来的。
  若各视角内参不一致，反投影出的点云本身就互相矛盾（同一物体在不同视角
  被放到略微不同的位置），**Sim(3) 是全局各向同性变换，吸收不了这种
  逐视角的径向尺度差**。

  本脚本做 2×2 分离：
      外参 pred / GT(对齐后)  ×  内参 pred_K / GT_K
  这样可以看清 4dB 该记在谁头上。

用法： python 00r_isolate_k.py <cache.pt> <transforms.json>
"""
import os, sys, math, itertools
import numpy as np
import torch
import torch.nn.functional as F

SD = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, SD)
import infer as I
import importlib.util as ilu


def _mod(name, f):
    sp = ilu.spec_from_file_location(name, os.path.join(SD, f))
    m = ilu.module_from_spec(sp); sp.loader.exec_module(m)
    return m


k00 = _mod("k00", "00k_offline_align.py")
o00 = _mod("o00", "00o_joint_scan.py")


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
    return float(np.mean(ps)), ps


def main():
    cache, tpath = sys.argv[1], sys.argv[2]
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    d = torch.load(cache, map_location="cpu", weights_only=False)
    paths = d["paths"]; new_h, new_w = int(d["new_h"]), int(d["new_w"])
    render_h, render_w = int(d["render_h"]), int(d["render_w"])
    P = d["pred_w2c"].to(dev).float()
    pred_K = d["pred_K"].to(dev).float()
    s = render_w / new_w

    def scale_K(K):
        K = K.clone(); K[:, 0, :] *= s; K[:, 1, :] *= s
        return K

    Kr_pred = scale_K(pred_K)
    base, _ = render_psnr(d, P, Kr_pred, render_h, render_w, dev)
    print(f"pose-free 自洽（pred 外参 + pred K）: {base:.3f} dB\n")

    undist = I.probe_distortion(tpath)
    G_raw, Kg = o00.build_gt(tpath, paths, new_h, new_w, undist,
                             gl2cv=True, transpose=True, kswap=False)
    G_raw = G_raw.to(dev).float(); Kg = Kg.to(dev).float()

    s_, R_, t_, info = k00.solve_rot_from_axes(P, G_raw)
    Ga = k00.apply_sim3(G_raw, s_, R_.to(torch.float32), t_.to(torch.float32))

    Kr_gt = scale_K(Kg)
    print("=" * 72)
    print(f"{'外参':<16s} {'内参':<16s} {'PSNR':>8s}   逐视角")
    print("-" * 72)
    for ext_name, ext in [("pred", P), ("GT(对齐后)", Ga)]:
        for k_name, Kx in [("pred_K", Kr_pred), ("GT_K", Kr_gt)]:
            p, ps = render_psnr(d, ext, Kx, render_h, render_w, dev)
            print(f"{ext_name:<16s} {k_name:<16s} {p:8.2f}   "
                  + " ".join(f"{x:5.1f}" for x in ps))
    print("=" * 72)

    # K 的一致性体检
    print("\npred_K 视角间一致性（相对 v0）：")
    for i in range(pred_K.shape[0]):
        fx, fy = pred_K[i, 0, 0].item(), pred_K[i, 1, 1].item()
        print(f"   v{i}: fx={fx:7.2f} ({(fx/pred_K[0,0,0].item()-1)*100:+6.2f}%)  "
              f"fy={fy:7.2f} ({(fy/pred_K[0,1,1].item()-1)*100:+6.2f}%)  "
              f"cx={pred_K[i,0,2].item():6.2f} cy={pred_K[i,1,2].item():6.2f}")
    print("GT_K 视角间一致性：全部相同（采集包共享一套内参）")


if __name__ == "__main__":
    main()

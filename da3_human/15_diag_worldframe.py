#!/usr/bin/env python3
"""15_diag_worldframe.py — 比较两次 DA3 推理的世界系（Sim(3) 是否只差均匀缩放）。

为什么必须有这个诊断：
  DA3 的世界系由「参考视角」定义。默认 `saddle_balanced` 在不同视角集上会选中不同帧，
  两次推理的世界系可差 83.6°；而 SH 与方向绑定，不转 SH 会白丢 1.2 dB。
  改用 `REF_STRATEGY=first` 后实测 R 与 I 夹角 0.10°，只剩均匀缩放，SH 原样可用。

  另一个用途：**跨 process_res 不能混用相机**。DA3 在不同分辨率下估计的 FOV 不同
  （504p fx/W=0.9780，672p 0.9566），拿 504p 相机渲染 672p 高斯实测掉 2.87 dB。

用法:
  python 15_diag_worldframe.py A/cameras.npz B/cameras.npz
  A 取 32 视角那次（含 view_indices），B 取 125 视角那次。
"""
import sys

import numpy as np


def centers_of(ext):
    r, t = ext[:, :3, :3], ext[:, :3, 3]
    return np.einsum("nij,nj->ni", -np.transpose(r, (0, 2, 1)), t)


def umeyama_sim3(src, dst):
    mu_s, mu_d = src.mean(0), dst.mean(0)
    a, b = src - mu_s, dst - mu_d
    u, sv, vt = np.linalg.svd((b.T @ a) / len(src))
    d = np.sign(np.linalg.det(u @ vt))
    diag = np.diag([1.0, 1.0, d])
    r = u @ diag @ vt
    s = float(np.trace(diag @ np.diag(sv)) / ((a ** 2).sum() / len(src)))
    return s, r, mu_d - s * (r @ mu_s)


def main():
    pa, pb = sys.argv[1], sys.argv[2]
    a, b = np.load(pa), np.load(pb)
    idx = a["view_indices"].astype(int) if "view_indices" in a.files else np.arange(len(a["extrinsics"]))

    ka, kb = a["intrinsics"][0], b["intrinsics"][0]
    ha, wa = a["depth"].shape[-2:]
    hb, wb = b["depth"].shape[-2:]
    print(f"A: {pa}\n   {len(a['extrinsics'])} 视角 @ {wa}x{ha} | sf={float(a['scale_factor']):.4f} "
          f"| fx/W={ka[0,0]/wa:.4f}")
    print(f"B: {pb}\n   {len(b['extrinsics'])} 视角 @ {wb}x{hb} | sf={float(b['scale_factor']):.4f} "
          f"| fx/W={kb[0,0]/wb:.4f}")
    print(f"   FOV 差 {abs(2*np.degrees(np.arctan(0.5*wa/ka[0,0])) - 2*np.degrees(np.arctan(0.5*wb/kb[0,0]))):.2f}° "
          f"(>1° 时不可混用相机)")

    ca = centers_of(a["extrinsics"].astype(np.float64))
    cb = centers_of(b["extrinsics"].astype(np.float64))[idx]
    s, r, t = umeyama_sim3(ca, cb)
    resid = np.linalg.norm(s * (ca @ r.T) + t - cb, axis=1)
    ang = np.degrees(np.arccos(np.clip((np.trace(r) - 1) / 2, -1, 1)))
    print(f"\ns={s:.4f} | R 与 I 夹角 {ang:.3f}° | 残差 mean {resid.mean()*1000:.2f}mm "
          f"max {resid.max()*1000:.2f}mm")
    print(f"不做任何变换：中心差 mean {np.linalg.norm(ca - cb, axis=1).mean()*1000:.2f}mm")
    print(f"\n判据：R 夹角 <1° 且 s≈1 → 可共用世界系，SH 无需旋转；否则必须转 SH 或用 ref=first 重跑。")


if __name__ == "__main__":
    main()

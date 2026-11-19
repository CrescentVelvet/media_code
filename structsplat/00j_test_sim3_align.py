#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
structsplat/00j_test_sim3_align.py — GT 相机 → 模型坐标系 Sim(3) 对齐的离线判据

为什么需要这个脚本（2026-11-19 实测）：
  GT 分支第一次跑出来 PSNR 只有 **6.41 dB**，而 pose-free 有 20.66 dB。
  根因不是「GT 相机不准」，而是**两套坐标系不相干**：
      gaussian_wrapper.output2gaussian 里
          coordinate = depth_to_world_coords_points(depth, extrinsic, intrinsic)
      这个 extrinsic 是**模型预测的**，所以高斯的绝对位置只跟预测世界系自洽。
  而 transforms.json 的 c2w 在采集包自己的世界系里，两者差一个 Sim(3)。
  直接换相机渲染 = 拿两套不相干的世界系硬拼 → 整体错位。

  官方 get_tar_camera 里本来就有这套数学（四元数均值定旋转 + 平移最小二乘定
  s,t），但它只用于「两次前向之间对齐 tar 相机」，不会把外部 GT 搬进来。
  infer.align_gt_to_model 补的就是这一环。

本脚本用**人造数据 + 已知真值**验证该实现，不碰 GPU、不读权重，秒级跑完。

五个判据（A~E）：
  A. 尺度可辨识：造数据时用一个已知 s0，估计出来的 s 必须等于 s0
  B. 投影等效（核心）：同一物理点，用「GT 相机 × GT 系点」与
     「aligned 相机 × 模型系点」投影，像素差必须 ≈ 0
  C. 正交性：相似变换后的 R 必须仍是正交阵（det=+1）
     —— 这是最容易写错的地方：用 `gt_w2c @ inv(Sim)` 会把 s 乘进旋转块，
        得到 R_g·Rᵀ/s，非正交，gsplat 渲出来是错乱画面
  D. identity 反向用例：gt 就是 pred 时，对齐结果必须等于原值（防过冲）
  E. 退化用例：纯平移 / 纯缩放两种极端，各自也要能正确还原

用法：
  python 00j_test_sim3_align.py            # 全部用例
  python 00j_test_sim3_align.py -v         # 打印每步中间量
"""

import argparse
import importlib.util
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))


def _load_infer():
    spec = importlib.util.spec_from_file_location("ss_infer", os.path.join(HERE, "infer.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


# ---------------------------------------------------------------
# 人造相机工具
# ---------------------------------------------------------------
def _rot(seed):
    """生成确定性的正交旋转矩阵（四元数 → 旋转阵，避免手写公式出错）。"""
    g = torch.Generator().manual_seed(seed)
    q = torch.randn(4, dtype=torch.float64, generator=g)
    q = q / q.norm()
    w, x, y, z = q.tolist()
    return torch.tensor([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ], dtype=torch.float64)


def _c2w_to_w2c(R_c2w, c):
    """c2w(旋转+光心) → w2c。光心 c 在**世界系**里。"""
    m = torch.eye(4, dtype=torch.float64)
    Rt = R_c2w.transpose(0, 1)
    m[:3, :3] = Rt
    m[:3, 3] = -Rt @ c
    return m


def make_pair(S=6, s0=2.37, seed=7, mode="sim3"):
    """
    造一对 (pred, gt) 相机，真值关系为  X_pred = s0·R0·X_gt + t0。

    ★ 注意 R 的处理（这是造数据最容易错的地方）：
      相似变换只缩放**位置**，不缩放**方向**。如果写成 R_pred = R_gt·R0ᵀ/s0，
      得到的是非正交矩阵 —— 现实里不存在这种相机，用它测出来的结论无效。
      正确做法：c2w 的旋转部分用 R0·R_gt（正交），光心用 s0·R0·c + t0。
    """
    gt_R = torch.stack([_rot(100 + i) for i in range(S)])     # c2w 旋转
    gt_c = torch.randn(S, 3, dtype=torch.float64) * 0.6       # 光心（gt 世界系）

    R0 = _rot(seed)
    t0 = torch.tensor([1.5, -2.0, 0.7], dtype=torch.float64)

    if mode == "sim3":
        s, R, t = s0, R0, t0
    elif mode == "translate":
        s, R, t = 1.0, torch.eye(3, dtype=torch.float64), t0
    elif mode == "scale":
        s, R, t = s0, torch.eye(3, dtype=torch.float64), torch.zeros(3, dtype=torch.float64)
    elif mode == "identity":
        s, R, t = 1.0, torch.eye(3, dtype=torch.float64), torch.zeros(3, dtype=torch.float64)
    else:
        raise ValueError(mode)

    pred_R = R @ gt_R                        # 正交（方向不缩放）
    pred_c = s * (gt_c @ R.transpose(0, 1)) + t

    gt = torch.stack([_c2w_to_w2c(gt_R[i], gt_c[i]) for i in range(S)])
    pred = torch.stack([_c2w_to_w2c(pred_R[i], pred_c[i]) for i in range(S)])
    return pred, gt, gt_R, gt_c, s, R, t


def _proj(w2c, pts, P):
    """w2c 相机 × 世界系点 → 像素坐标。"""
    cam = (w2c[:3, :3] @ pts.T).T + w2c[:3, 3]
    uv = (P @ (cam / cam[:, 2:3]).T).T
    return uv[:, :2]


def _centers(w2c):
    return -torch.einsum("sji,sj->si", w2c[:, :3, :3], w2c[:, :3, 3])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()

    inf = _load_infer()
    if not hasattr(inf, "align_gt_to_model"):
        print("❌ infer.py 里没有 align_gt_to_model，先更新脚本")
        return 2

    ok, fail = [], []
    def check(name, cond, detail=""):
        (ok if cond else fail).append(name)
        print(f"  {'✅' if cond else '❌'} {name}  {detail}")

    S = 6
    K = torch.eye(3).repeat(S, 1, 1)
    P = torch.tensor([[300.0, 0, 160.0], [0, 300.0, 120.0], [0, 0, 1.0]],
                     dtype=torch.float64)

    print("=" * 68)
    print(" A. 尺度可辨识（人造真值 s0）")
    print("=" * 68)
    s0 = 2.37
    pred, gt, gt_R, gt_c, s_true, R0, t0 = make_pair(S, s0=s0, mode="sim3")
    aligned, info = inf.align_gt_to_model(pred.float(), gt.float(), K, K)
    al = aligned.to(torch.float64)
    s_est = info["scale_gt_to_model"]
    check("A1 估计尺度 == 真值", abs(s_est - s_true) < 1e-4,
          f"s_est={s_est:.6f}  s_true={s_true:.6f}  相对误差={abs(s_est-s_true)/s_true:.2e}")
    dP = float((_centers(pred) - _centers(pred).mean(0)).norm(dim=1).mean())
    dG = float((_centers(gt) - _centers(gt).mean(0)).norm(dim=1).mean())
    check("A2 s 等于 dP/dG（口径自洽）", abs(s_est - dP / dG) < 1e-6,
          f"dP={dP:.4f} dG={dG:.4f} dP/dG={dP/dG:.6f}")

    print()
    print("=" * 68)
    print(" B. 投影等效（核心判据）")
    print("=" * 68)
    max_ab = 0.0
    max_ac = 0.0
    for i in range(S):
        d = torch.randn(300, 3, dtype=torch.float64) * 150.0
        d[:, 2] = d[:, 2].abs() + 700.0                 # 保证在相机前方
        pg = (d @ gt_R[i].transpose(0, 1)) + gt_c[i]    # gt 世界系点
        pm = s_true * (pg @ R0.transpose(0, 1)) + t0    # 同一物理点 → 模型系
        a = _proj(gt[i], pg, P)                         # 基准：GT 相机 × gt 点
        b = _proj(al[i], pm, P)                         # aligned 相机 × 模型点
        c = _proj(pred[i], pm, P)                       # 预测相机 × 模型点
        max_ab = max(max_ab, float((a - b).norm(dim=1).max()))
        max_ac = max(max_ac, float((a - c).norm(dim=1).max()))
    check("B1 基准 vs aligned ≈ 0", max_ab < 1e-2, f"max={max_ab:.3e} px")
    check("B2 基准 vs 预测相机 ≈ 0", max_ac < 1e-2, f"max={max_ac:.3e} px（自洽性）")

    print()
    print("=" * 68)
    print(" C. 正交性（踩过的坑）")
    print("=" * 68)
    Rn = al[:, :3, :3]
    orth = float((Rn.transpose(1, 2) @ Rn - torch.eye(3, dtype=torch.float64)).abs().max())
    det = torch.linalg.det(Rn)
    check("C1 R 正交", orth < 1e-6, f"max|RᵀR−I|={orth:.3e}")
    check("C2 det=+1（无反射）",
          float((det - 1).abs().max()) < 1e-6,
          f"det∈[{float(det.min()):.6f}, {float(det.max()):.6f}]")
    # 反向示范：旧写法 gt @ inv(Sim) 会破坏正交性 —— 用数值证明它确实非正交
    Sim = torch.eye(4, dtype=torch.float64)
    Sim[:3, :3] = (R0.transpose(0, 1) @ torch.linalg.inv(R0 @ R0.transpose(0, 1)))
    Sim[:3, :3] = torch.eye(3, dtype=torch.float64) * 0.0 + R0 / s_true
    Sim[:3, 3] = -R0 @ t0 / s_true
    bad = gt @ torch.linalg.inv(Sim)
    orth_bad = float((bad[:, :3, :3].transpose(1, 2) @ bad[:, :3, :3]
                      - torch.eye(3, dtype=torch.float64)).abs().max())
    check("C3 （反例）gt@inv(Sim) 必非正交", orth_bad > 1e-2,
          f"max|RᵀR−I|={orth_bad:.3e}  ← 证明不能那么写")

    print()
    print("=" * 68)
    print(" D. identity 反向用例（防过冲）")
    print("=" * 68)
    al2, i2 = inf.align_gt_to_model(pred.float(), pred.float(), K, K)
    diff2 = float((pred - al2).abs().max())
    check("D1 gt==pred 时对齐结果不变", diff2 < 1e-5,
          f"maxdiff={diff2:.3e}  s={i2['scale_gt_to_model']:.6f}")
    check("D2 identity 的 s==1", abs(i2["scale_gt_to_model"] - 1.0) < 1e-9,
          f"s={i2['scale_gt_to_model']:.9f}")

    print()
    print("=" * 68)
    print(" E. 退化用例（纯平移 / 纯缩放）")
    print("=" * 68)
    for mode, label, expect_s in (("translate", "纯平移", 1.0), ("scale", "纯缩放", s0)):
        p3, g3, _, _, s3t, R3, t3 = make_pair(S, s0=s0, mode=mode)
        a3, i3 = inf.align_gt_to_model(p3.float(), g3.float(), K, K)
        check(f"E1 {label}: 中心残差≈0", i3["center_max"] < 1e-4,
              f"max={i3['center_max']:.3e}  s={i3['scale_gt_to_model']:.6f}")
        check(f"E2 {label}: 尺度={expect_s}", abs(i3["scale_gt_to_model"] - expect_s) < 1e-6,
              f"s={i3['scale_gt_to_model']:.6f}")

    print()
    print("=" * 68)
    total = len(ok) + len(fail)
    if fail:
        print(f"❌ {len(fail)}/{total} 项未通过：")
        for f in fail:
            print("   -", f)
        print("=" * 68)
        return 1
    print(f"🎉 全部 {total} 项通过 —— Sim(3) 对齐数学正确")
    print("=" * 68)
    if args.verbose:
        print(f"\n参考量：s={s_est:.6f} 中心残差 RMS={info['center_rms']:.3e} "
              f"投影等效 max={max_ab:.3e}px")
    return 0


if __name__ == "__main__":
    sys.exit(main())

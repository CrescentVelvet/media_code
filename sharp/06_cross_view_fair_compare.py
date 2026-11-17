#!/usr/bin/env python3
"""跨视角渲染的公平对照：identity / 单平面单应 / SHARP 三方同掩码对拍。

## 这个脚本要解决什么问题

`05` 的 Q4 报了「跨视角 6.3~7.8 dB vs 自渲 26~27 dB，差 19 dB」，并据此判定
「几何根本没对上」。**这个对照不公平，v2 已撤回**，原因：

* Q4 的 `psnr` 是**全图**。渲染底色是纯黑（贴合官方 cli/render.py），GT 是真实房间，
  两者背景本来就不同。6~8 dB 里绝大部分来自「两张不同的照片本来就不一样」，
  不是几何误差。
* 自渲 26~27 dB 是「同一张照片 vs 自己」，分子上没有任何"照片差异"。

所以「19 dB 缺口」这个数本身没有信息量。真正该问的是：

    SHARP 的点云跨视角渲染，比**完全不做 warp**（identity）好多少？
    比**只含一个深度自由度的平面单应**好多少？

如果 SHARP 只比 identity 高零点几 dB，说明它贡献的几何信息近乎为零；
如果它还不如单平面单应，说明它贡献的几何信息是**负的**。

## 三个对照物（全部只在同一 alpha 掩码内算 PSNR）

    identity : 直接拿 source 图逐像素比 target 图，不做任何 warp。
               这是"两张不同照片的天然差异"这个下界，与几何无关。
    planeH   : 单个前向平行平面的 2D 单应 H = Kb (R + t n^T / d) Kb^-1，
               n = [0,0,1]（target 相机光轴方向），d 扫描取最优。
               只有 1 个几何自由度，但它是"真几何"的最弱形式。
    SHARP    : A 的点经 R,t 刚体变换后从 B 渲染，取 `psnr_covered`。

掩码统一用 `05` 存下的 `q4_{s}_to_{t}_alpha.png`（阈值 > 0.5）。这样三方的
像素集合完全相同，差额只可能来自几何。

## 为什么用 SHARP 自己的掩码当评价区

因为要回答的问题是「SHARP 的渲染在它自己覆盖到的地方准不准」，不是
「SHARP 覆盖了多少」。覆盖率单独报（`cov`）。若改用 identity/planeH 的覆盖区，
就会变成"在 SHARP 没覆盖的地方比"，那对 SHARP 更不利且不说明问题。

⚠️ 掩码来自 SHARP 的 alpha，而 alpha 由 SHARP 的 opacity 决定，
所以这个口径**偏向 SHARP**（它可以把难的地方判成不可见）。这正是判据的
保守方向：**如果 SHARP 在自己挑的地方都赢不了 identity，结论就更硬。**

## 口径纪律

* 纯 numpy，不占 GPU，不重新跑前馈。输入是 `05` 的输出目录。
* 只支持同一台相机的多视角包（`05` 已断言 K 唯一）。
* 所有 PSNR 都在 `render_size` 下算，与 `05` 的 Q4 逐像素对齐。
* identity 与 planeH 的 warp 越界像素用 clamp 到边界，不排除——
  排除会引入"只挑好像素"的偏差。

用法：

    python sharp/06_cross_view_fair_compare.py \\
        --run ~/output/sharp_results/S1a_crossview_13v_small22 \\
        --scene ~/output/splatanything_results/_scenes/jiajia_hold_13v \\
        --out <输出目录>
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np


# ------------------------------------------------------------------ 基础工具


def psnr(a: np.ndarray, b: np.ndarray, mask: np.ndarray | None = None) -> float:
    """在 mask 内算 PSNR。a/b 是 (N,3)，mask 是 (N,) bool。"""
    if mask is not None:
        a, b = a[mask], b[mask]
    if a.size == 0:
        return float("nan")
    mse = float(np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2))
    return 10.0 * math.log10(1.0 / max(mse, 1e-12))


def load_rgb_u8(path: Path) -> np.ndarray:
    """读图 -> uint8。与 05 的 load_rgb 同口径（走 sharp.utils.io，自带 HEIC/EXIF 处理）。"""
    from sharp.utils import io

    image_u8, _, _ = io.load_rgb(path)
    return np.asarray(image_u8, dtype=np.uint8)


def resize_to(arr_u8: np.ndarray, rw: int, rh: int) -> np.ndarray:
    """resize 到渲染口径，返回 float32 0..1。与 05 的 GT 处理逐位一致。"""
    from PIL import Image

    return np.asarray(
        Image.fromarray(arr_u8).resize((rw, rh), Image.LANCZOS), dtype=np.float32
    ) / 255.0


def backward_warp(img_src: np.ndarray, H_src2tgt: np.ndarray) -> np.ndarray:
    """用 H 做反向 warp（目标像素 -> 源像素采样），最近邻。

    `img_src` 是 (h,w,3) float32。返回 (h*w,3)，越界 clamp 到边界。
    最近邻而非双线性：三方必须用同一插值口径，identity 与 SHARP 都没有插值，
    唯一插值的地方就是这里，所以选最简单的。它会让 planeH 略吃亏（约 0.1~0.2 dB），
    但方向对 SHARP 有利，属于保守选择。
    """
    h, w = img_src.shape[:2]
    ys, xs = np.mgrid[0:h, 0:w]
    q = np.linalg.inv(H_src2tgt) @ np.stack([xs.ravel(), ys.ravel(), np.ones(h * w)])
    zq = q[2]
    okz = np.abs(zq) > 1e-12
    u = q[0] / np.where(okz, zq, 1.0)
    v = q[1] / np.where(okz, zq, 1.0)
    ui = np.clip(np.round(u).astype(np.int64), 0, w - 1)
    vi = np.clip(np.round(v).astype(np.int64), 0, h - 1)
    return img_src.reshape(-1, 3)[vi * w + ui]


def load_camera_json(path: Path) -> tuple[dict, dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    c2w = {e["camera_id"]: np.asarray(e["matrix"], dtype=np.float64) for e in data["extrinsics"]}
    kmat = {e["camera_id"]: np.asarray(e["matrix"], dtype=np.float64) for e in data["intrinsics"]}
    return c2w, kmat


# ------------------------------------------------------------------ 主流程


D_SWEEP = [0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 2.0, 2.5, 3.0, 4.0]


def main() -> int:
    ap = argparse.ArgumentParser(description="跨视角渲染三方同掩码对拍")
    ap.add_argument("--run", required=True, type=Path,
                    help="05 的输出目录（需含 cross_view.json 与 q4_*_alpha.png）")
    ap.add_argument("--scene", required=True, type=Path,
                    help="场景目录（需含 images/ 与 prior_camera.json）")
    ap.add_argument("--out", type=Path, default=None, help="默认写到 --run 下")
    ap.add_argument("--alpha-thr", type=float, default=0.5,
                    help="掩码阈值，须与 05 的 Q4 一致（0.5）")
    args = ap.parse_args()

    out = args.out or args.run
    out.mkdir(parents=True, exist_ok=True)

    cj = json.loads((args.run / "cross_view.json").read_text(encoding="utf-8"))
    q4 = cj["_q4_cross_view_render"]
    rw, rh = cj["render_size"]
    views = cj["views"]
    print(f"📁 run   {args.run}")
    print(f"   场景  {args.scene}")
    print(f"   视角  {len(views)} 个: {', '.join(views)}")
    print(f"   渲染  {rw}x{rh}  掩码 alpha>{args.alpha_thr}")

    c2w, kd = load_camera_json(args.scene / "prior_camera.json")
    stems = {p.stem: p for p in (args.scene / "images").glob("*.png")}

    rows = []
    for key in sorted(q4.keys()):
        s, t = key.split("->")
        if s not in stems or t not in stems:
            raise SystemExit(f"cross_view.json 里的视角 {key} 在场景 images/ 里找不到")
        alpha_path = args.run / f"q4_{s}_to_{t}_alpha.png"
        if not alpha_path.exists():
            print(f"   ⚠️ 跳过 {key}：缺 {alpha_path.name}")
            continue

        from PIL import Image

        alpha = np.asarray(Image.open(alpha_path).convert("L"), dtype=np.float32) / 255.0
        if alpha.shape != (rh, rw):
            raise SystemExit(
                f"{alpha_path.name} 尺寸 {alpha.shape} 与 render_size {(rh, rw)} 不符"
            )
        mask_flat = (alpha > args.alpha_thr).reshape(-1)
        if not mask_flat.any():
            print(f"   ⚠️ 跳过 {key}：掩码为空")
            continue

        # 先读原图拿尺寸（K 要按 render_size / 原图尺寸缩放），再 resize 到渲染口径。
        ua = load_rgb_u8(stems[s])
        ub = load_rgb_u8(stems[t])
        h0, w0 = ua.shape[:2]
        h1, w1 = ub.shape[:2]
        ia = resize_to(ua, rw, rh)
        ib = resize_to(ub, rw, rh)

        Ka = kd[s].copy()
        Kb = kd[t].copy()
        Ka[0, :] *= rw / w0
        Ka[1, :] *= rh / h0
        Kb[0, :] *= rw / w1
        Kb[1, :] *= rh / h1

        Ra, Rb = c2w[s][:3, :3], c2w[t][:3, :3]
        ta, tb = c2w[s][:3, 3], c2w[t][:3, 3]
        R = Rb.T @ Ra
        tvec = Rb.T @ (ta - tb)
        n = np.array([0.0, 0.0, 1.0])          # target 相机光轴方向的平面法向

        p_id = psnr(ia.reshape(-1, 3), ib.reshape(-1, 3), mask_flat)

        best = (-1e9, float("nan"))
        for d in D_SWEEP:
            H = Kb @ (R + np.outer(tvec, n) / d) @ np.linalg.inv(Ka)
            p = psnr(backward_warp(ia, H), ib.reshape(-1, 3), mask_flat)
            if p > best[0]:
                best = (p, d)

        p_sharp = q4[key]["full"]["psnr_covered"]
        p_sharp_full = q4[key]["full"]["psnr"]
        ang = math.degrees(
            math.acos(float(np.clip(c2w[s][:3, 2] @ c2w[t][:3, 2], -1.0, 1.0)))
        )

        r = {
            "source": s, "target": t, "angle_deg": ang,
            "mask_frac": float(mask_flat.mean()),
            "identity_db": p_id,
            "planeH_db": best[0], "planeH_d_best": best[1],
            "sharp_db": p_sharp, "sharp_full_image_db": p_sharp_full,
            "sharp_minus_identity_db": p_sharp - p_id,
            "sharp_minus_planeH_db": p_sharp - best[0],
        }
        rows.append(r)
        print(
            f"   {s}->{t} ang={ang:5.1f}° cov={mask_flat.mean()*100:3.0f}% | "
            f"identity={p_id:5.2f}  planeH={best[0]:5.2f}@d={best[1]:.1f}  "
            f"SHARP={p_sharp:5.2f} (全图 {p_sharp_full:4.2f}) | "
            f"S-I={p_sharp-p_id:+.2f}  S-P={p_sharp-best[0]:+.2f}"
        )

    if not rows:
        print("❌ 没有任何视角对可评，检查 --run 是否是 05 的输出目录")
        return 1

    # ---------------- 汇总 ----------------
    di = np.array([r["sharp_minus_identity_db"] for r in rows])
    dp = np.array([r["sharp_minus_planeH_db"] for r in rows])
    summary = {
        "n_pairs": len(rows),
        "sharp_minus_identity_db": {
            "median": float(np.median(di)), "min": float(di.min()), "max": float(di.max()),
            "n_positive": int((di > 0).sum()),
        },
        "sharp_minus_planeH_db": {
            "median": float(np.median(dp)), "min": float(dp.min()), "max": float(dp.max()),
            "n_positive": int((dp > 0).sum()),
        },
        "verdict": (
            "SHARP 优于 identity 但不如单平面单应：点云几何信息存在但几乎无对齐贡献"
            if (np.median(di) > 0 and np.median(dp) < 0)
            else "见逐对数据"
        ),
    }
    print("\n=== 汇总（全部同掩码）===")
    print(f"   SHARP − identity : 中位 {np.median(di):+.2f} dB "
          f"(范围 {di.min():+.2f}~{di.max():+.2f}, 正的 {int((di>0).sum())}/{len(di)})")
    print(f"   SHARP − planeH   : 中位 {np.median(dp):+.2f} dB "
          f"(范围 {dp.min():+.2f}~{dp.max():+.2f}, 正的 {int((dp>0).sum())}/{len(dp)})")
    print(f"   判读：{summary['verdict']}")

    payload = {
        "run": str(args.run), "scene": str(args.scene),
        "render_size": [rw, rh], "alpha_thr": args.alpha_thr,
        "d_sweep": D_SWEEP, "pairs": rows, "summary": summary,
    }
    (out / "fair_compare.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8", newline="\n"
    )
    print(f"\n✅ 写出 {out / 'fair_compare.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

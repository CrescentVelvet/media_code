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

> **v3 实测（2026-11-17，修 `05` 双重变换后）**：`S−I` 中位 **+3.73 dB**（16/16 为正）、
> `S−P` 中位 **+2.11 dB**（14/16 为正）。**两个"如果"都没发生**——几何贡献是真实正
> 且量级可观。但绝对水平只有 11~17 dB（SA 自渲 26~27 dB），仍不足以做融合基座。
> 详见 `EXPERIMENTS.md` 结果四 v3。

## 三个对照物（全部只在同一 alpha 掩码内算 PSNR）

    identity : 直接拿 source 图逐像素比 target 图，不做任何 warp。
               这是"两张不同照片的天然差异"这个下界，与几何无关。
    planeH   : 单个前向平行平面的 2D 单应 H = Kb (R + t n^T / d) Ka^-1，
               n = [0,0,1]（target 相机光轴方向），d 扫描取最优。
               只有 1 个几何自由度，但它是"真几何"的最弱形式。
    SHARP    : A 的点经 R,t 刚体变换后从 B 渲染，取 `psnr_covered`。

掩码统一用 `05` 存下的 `q4_{s}_to_{t}_alpha.png`（阈值 > 0.5）。这样三方的
像素集合完全相同，差额只可能来自几何。

⚠️ **`d` 是逐对 oracle 搜索**：每一对都在 `D_SWEEP` 上挑「使 PSNR 最高」的那个 `d`，
而 SHARP 侧**没有任何可调参数**。也就是说 planeH 的基线是**被喂了答案的最优值**，
对 planeH 单向有利。这个偏差的方向是「抬高基线、压低 SHARP 的胜率」——所以
SHARP 若仍能赢，这个赢是**保守的**、不是被高估的。

> ⚠️ **2026-11-17 修正（重要）**：本段原写「『SHARP 输给 planeH』这个结论在更严格的
> 设定下只会更明显，不会反转」。**那句话已被实测否证。** 它的前提是 `05` 的跨视角
> 渲染正确，而 `05` 当时带着**双重变换 bug**（`apply_w2c_to_gaussians` 之后又传了
> `extrinsics=w2c`，等于 `p → R(Rp+t)+t`），把 SHARP 的 PSNR 压低了 3.5~4.0 dB。
> 修正后 **SHARP 赢 planeH 14/16 对，中位 +2.11 dB**。所以这个 oracle 偏差现在
> 有了它该有的意义：**planeH 被喂了最优 `d` 仍然输，SHARP 的领先是干净的。**

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


def backward_warp(img_src: np.ndarray, H_src2tgt: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """用 H 做反向 warp（目标像素 -> 源像素采样），**双线性**。

    `img_src` 是 (h,w,3) float32。返回 `(pixels (h*w,3), inb_valid (h*w,) bool)`。

    两个 v4 复核指出的口径问题在此修正：

    1. **插值**：原用最近邻并把越界 clamp 到边界。planeH 是唯一需要插值的一方
       （identity 与 SHARP 都不插值），实测最近邻让它白吃 0.1~0.2 dB 的亏。
       改双线性，双方口径一致。
    2. **有效域**：原来越界像素 clamp 到边界后照样进 PSNR，等于拿"拉边"的假像素
       去比。现在返回 `inb_valid`，越界的像素**从三方的公共掩码里剔除**
       （见调用处），既不放水也不冤枉。

    返回 valid 而非直接置零，是因为掩码必须与 identity/SHARP 取交集，
    否则三方比的不是同一批像素。
    """
    h, w = img_src.shape[:2]
    ys, xs = np.mgrid[0:h, 0:w]
    q = np.linalg.inv(H_src2tgt) @ np.stack([xs.ravel(), ys.ravel(), np.ones(h * w)])
    zq = q[2]
    okz = np.abs(zq) > 1e-12
    u = q[0] / np.where(okz, zq, 1.0)
    v = q[1] / np.where(okz, zq, 1.0)

    # 双线性：采样点 (u,v) 落在 [0, w-1] x [0, h-1] 内才算有效
    eps = 1e-6
    inb = okz & (u >= -eps) & (u <= (w - 1) + eps) & (v >= -eps) & (v <= (h - 1) + eps)

    u = np.clip(u, 0, w - 1)
    v = np.clip(v, 0, h - 1)
    u0 = np.floor(u).astype(np.int64)
    v0 = np.floor(v).astype(np.int64)
    u1 = np.minimum(u0 + 1, w - 1)
    v1 = np.minimum(v0 + 1, h - 1)
    du = (u - u0)[:, None]
    dv = (v - v0)[:, None]

    src = img_src.reshape(-1, 3).astype(np.float64)
    out = (
        src[v0 * w + u0] * (1 - du) * (1 - dv)
        + src[v0 * w + u1] * du * (1 - dv)
        + src[v1 * w + u0] * (1 - du) * dv
        + src[v1 * w + u1] * du * dv
    )
    return out.astype(np.float32), inb


def load_camera_json(path: Path) -> tuple[dict, dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    c2w = {e["camera_id"]: np.asarray(e["matrix"], dtype=np.float64) for e in data["extrinsics"]}
    kmat = {e["camera_id"]: np.asarray(e["matrix"], dtype=np.float64) for e in data["intrinsics"]}
    return c2w, kmat


# ------------------------------------------------------------------ 主流程


D_SWEEP = [0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 2.0, 2.5, 3.0, 4.0]


def _verdict(
    md_i: float, md_p: float, n_i: int, n_p: int, n: int,
    n_eval_p: int | None = None,
) -> str:
    """按实测三方向给判读。**不预设结论**（v2 的硬编码判读被 bug 打了脸）。

    `n_eval_p` 是 S−P 实际可评的对数。为 0 时**不能**拿 md_p=0.0 去编「打平」——
    那是空集假造出来的结论（v5 首版就这么错过一次）。
    """
    if n_eval_p == 0:
        return (f"仅 S−I 可评：中位 {md_i:+.2f} dB（{n_i}/{n} 为正），跨视角对齐方向正确。"
                f"S−P 无可评对（planeH 在所有对上都 warp 出界，可用域 <20%），"
                f"**本轮不对「SHARP 是否强于单平面单应」作判读**。")
    if md_i <= 0.5:
        return (f"SHARP−identity 中位 {md_i:+.2f} dB（{n_i}/{n} 为正），几何净贡献接近 0"
                f"或为负：跨视角对齐没成立。")
    base = (f"SHARP−identity 中位 {md_i:+.2f} dB（{n_i}/{n} 为正），"
            f"跨视角对齐方向正确、量级真实；")
    if md_p > 0.5:
        return base + (f"且 SHARP−planeH 中位 {md_p:+.2f} dB（{n_p}/{n} 为正），"
                       f"赢过被喂了最优 d 的单平面单应。"
                       f"但绝对水平仍低，不代表可当融合基座。")
    if md_p > -0.5:
        return base + (f"SHARP−planeH 中位 {md_p:+.2f} dB（{n_p}/{n} 为正），与单平面单应打平："
                       f"几何信息真实但几乎不超出「一个平面」的假设。")
    return base + (f"但 SHARP−planeH 中位 {md_p:+.2f} dB（仅 {n_p}/{n} 为正），"
                   f"输给只含 1 个深度自由度的单平面单应：位置/朝向仍有系统性问题。")


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
        # 掩码：优先读 05 存的无损 float32（`.npy`），退回 8bit PNG。
        # 05 的 Q4 用 float alpha > thr 判定，若这里读 8bit 重建，量化会让边界像素
        # 归属反转，导致"引用 05 的分数但与 05 的像素集不同"（v4 复核指出的缺陷）。
        alpha_npy = args.run / f"q4_{s}_to_{t}_alpha.npy"
        alpha_png = args.run / f"q4_{s}_to_{t}_alpha.png"
        if alpha_npy.exists():
            alpha = np.load(alpha_npy).astype(np.float32)
            alpha_src = "npy(float32, 无损)"
        elif alpha_png.exists():
            from PIL import Image as _I

            alpha = np.asarray(_I.open(alpha_png).convert("L"), dtype=np.float32) / 255.0
            alpha_src = "png(8bit, 有量化误差)"
            print(f"   ⚠️ {key}: 只有 8bit alpha PNG，掩码与 05 的 float 判定不完全一致")
        else:
            print(f"   ⚠️ 跳过 {key}：缺 alpha（{alpha_npy.name} / {alpha_png.name}）")
            continue
        if alpha.shape != (rh, rw):
            raise SystemExit(
                f"{key} 的 alpha 尺寸 {alpha.shape} 与 render_size {(rh, rw)} 不符"
            )
        mask_sharp = (alpha > args.alpha_thr).reshape(-1)
        if not mask_sharp.any():
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

        # --- planeH：先按 SHARP 掩码扫 d 选出最优，再取其有效域交成公共掩码 ---
        best = (-1e9, float("nan"), None)
        for d in D_SWEEP:
            H = Kb @ (R + np.outer(tvec, n) / d) @ np.linalg.inv(Ka)
            warped, inb = backward_warp(ia, H)
            m_d = mask_sharp & inb
            if not m_d.any():
                continue
            p = psnr(warped, ib.reshape(-1, 3), m_d)
            if p > best[0]:
                best = (p, d, inb)
        if best[2] is None:
            print(f"   ⚠️ 跳过 {key}：面H 无有效 domain")
            continue

        p_planeH, d_best, inb_best = best
        # ── 掩码口径：**逐比较对定义，不再强求三方共用一个掩码** ──────────
        # 首版让 identity / planeH / SHARP 三方共用 `mask_sharp ∩ inb`，理由听起来对
        # （同一批像素谁都不占便宜），实测却错了：`inb` 是 planeH 独有的约束，
        # identity 与 SHARP 都**不需要 warp**，凭什么被单应的退化域拖下水？
        # 后果：large125 的 124.8° 对公共域只剩 0%，S−I 这种本来完全成立的比较被一并作废。
        #
        # 现在分成两个问题、两套掩码：
        #   (1) 「SHARP 比什么都不做强吗」→ 在 SHARP 自己的覆盖域 `mask_sharp` 上比 S−I。
        #       identity 用源图原样，无有效域问题，此比较在任何角度都成立。
        #   (2) 「SHARP 比单平面单应强吗」→ 必须限制在**两者都可定义**的 `mask_sp` 上。
        #       planeH 在大角度 warp 出界严重，`mask_sp` 会变小；此时 S−P 不可信，
        #       但这是 planeH 的局限，不该株连 S−I。
        mask_si = mask_sharp                       # (1) S−I 用
        mask_sp = mask_sharp & inb_best            # (2) S−P 用
        if not mask_sp.any() or not mask_si.any():
            print(f"   ⚠️ 跳过 {key}：掩码为空")
            continue

        # SHARP 的渲染图：优先 float32（.npy），退回 8bit PNG。
        sharp_npy = args.run / f"q4_{s}_to_{t}_full.npy"
        sharp_png = args.run / f"q4_{s}_to_{t}_full.png"
        if sharp_npy.exists():
            img_sharp = np.clip(np.load(sharp_npy).astype(np.float32), 0.0, 1.0).reshape(-1, 3)
            sharp_px_src = "npy(float32)"
        elif sharp_png.exists():
            from PIL import Image as _I

            img_sharp = np.asarray(
                _I.open(sharp_png).convert("RGB"), dtype=np.float32
            ).reshape(-1, 3) / 255.0
            sharp_px_src = "png(8bit)"
        else:
            print(f"   ⚠️ 跳过 {key}：缺 SHARP 渲染图")
            continue

        ib_px = ib.reshape(-1, 3)
        ia_px = ia.reshape(-1, 3)
        # (1) S−I：SHARP 自己的覆盖域
        p_id = psnr(ia_px, ib_px, mask_si)
        p_sharp = psnr(img_sharp, ib_px, mask_si)
        # (2) S−P：两者都能定义的域。注意 `p_planeH` 扫 d 时用的就是 mask_sharp ∩ inb，
        # 与 mask_sp 完全同一批像素，可直接复用，无需重算。
        p_sharp_sp = psnr(img_sharp, ib_px, mask_sp)
        p_sharp_full = q4[key]["full"]["psnr"]
        ang = math.degrees(
            math.acos(float(np.clip(c2w[s][:3, 2] @ c2w[t][:3, 2], -1.0, 1.0)))
        )

        # ── 有效度：两个比较各自判定 ────────────────────────────────
        # S−I 的掩码是 SHARP 自身覆盖域，不依赖任何 warp，只要覆盖不太小就成立。
        # S−P 受 planeH 的 warp 出界限制，大角度下残域会缩到几个百分点。
        # 门槛 20%（先写后看）：small 系列的 mask_sp 在 74~98%，large125 的在 0~61%，
        # 20% 能把两侧自然分开；低于它的 S−P 只记录不参与判读。
        f_si = float(mask_si.mean())
        f_sp = float(mask_sp.mean())
        sidx_powered = f_si >= 0.20
        sp_powered = f_sp >= 0.20

        r = {
            "source": s, "target": t,
            "angle_deg": ang,
            "mask_si_frac": f_si,
            "mask_sp_frac": f_sp,
            "mask_sharp_frac": float(mask_sharp.mean()),
            "mask_planeH_valid_frac": float(inb_best.mean()),
            "underpowered_si": not sidx_powered,
            "underpowered_sp": not sp_powered,
            "mask_src": alpha_src,
            "sharp_px_src": sharp_px_src,
            "identity_db": p_id,
            "planeH_db": p_planeH, "planeH_d_best": d_best,
            "sharp_db": p_sharp, "sharp_full_image_db": p_sharp_full,
            "sharp_db_sp_mask": p_sharp_sp,
            "sharp_db_05_reported": q4[key]["full"]["psnr_covered"],
            "sharp_minus_identity_db": p_sharp - p_id,
            "sharp_minus_planeH_db": p_sharp_sp - p_planeH,
        }
        rows.append(r)
        flag = "  " if (sidx_powered and sp_powered) else "⚠️"
        tags = []
        if not sidx_powered:
            tags.append(f"S-I域{f_si*100:.0f}%<20%")
        if not sp_powered:
            tags.append(f"S-P域{f_sp*100:.0f}%<20%")
        print(
            f" {flag}{s}->{t} ang={ang:5.1f}° 域(S-I {f_si*100:3.0f}% / S-P {f_sp*100:3.0f}%) | "
            f"identity={p_id:5.2f}  planeH={p_planeH:5.2f}@d={d_best:.1f}  "
            f"SHARP={p_sharp:5.2f} (全图 {p_sharp_full:4.2f}) | "
            f"S-I={p_sharp-p_id:+.2f}  S-P={p_sharp_sp-p_planeH:+.2f}"
            + (f"  ⚠️ {'、'.join(tags)}，该项不参与判读" if tags else "")
        )

    if not rows:
        print("❌ 没有任何视角对可评，检查 --run 是否是 05 的输出目录")
        return 1

    # ---------------- 汇总 ----------------
    # 两个比较各自筛样本，不再共用一个门：
    #   S−I 只需 SHARP 有覆盖（几乎所有对都成立）；
    #   S−P 还要 planeH 在大角度下没把画面 warp 出界。
    # 把它们绑在一起会把 S−I 一起废掉——首版就这么误伤了 large125 的全部 6 对。
    si_rows = [r for r in rows if not r["underpowered_si"]]
    sp_rows = [r for r in rows if not r["underpowered_sp"]]
    if not si_rows and not sp_rows:
        print(f"\n❌ {len(rows)} 对全部域不足（<20%），这批数据不能支撑任何比较。")
        payload = {
            "run": str(args.run), "scene": str(args.scene),
            "render_size": [rw, rh], "alpha_thr": args.alpha_thr,
            "d_sweep": D_SWEEP, "pairs": rows,
            "summary": {
                "n_pairs": len(rows), "n_evaluable_si": 0, "n_evaluable_sp": 0,
                "verdict": (
                    f"全部 {len(rows)} 对的可用域 <20%，本轮不给判读。"
                    "需降低视角差或不依赖 warp 的 baseline。"
                ),
            },
        }
        (out / "fair_compare.json").write_text(
            json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8", newline="\n"
        )
        print(f"✅ 写出 {out / 'fair_compare.json'}")
        return 3

    di = np.array([r["sharp_minus_identity_db"] for r in si_rows])
    dp = np.array([r["sharp_minus_planeH_db"] for r in sp_rows])

    def _blk(a):
        if a.size == 0:
            return None
        return {
            "n": int(a.size), "median": float(np.median(a)),
            "min": float(a.min()), "max": float(a.max()),
            "n_positive": int((a > 0).sum()),
        }

    summary = {
        "n_pairs": len(rows),
        "n_evaluable_si": len(si_rows),
        "n_evaluable_sp": len(sp_rows),
        "min_mask_frac_threshold": 0.20,
        "mask_definition": {
            "sharp_minus_identity": "mask_sharp（SHARP 自身 alpha>thr 覆盖域，无 warp 依赖）",
            "sharp_minus_planeH": "mask_sharp ∩ planeH 最优 d 的有效域（两者都可定义的像素）",
        },
        "sharp_minus_identity_db": _blk(di),
        "sharp_minus_planeH_db": _blk(dp),
        "verdict": _verdict(
            float(np.median(di)) if di.size else 0.0,
            float(np.median(dp)) if dp.size else 0.0,
            int((di > 0).sum()), int((dp > 0).sum()), max(len(si_rows), 1),
            n_eval_p=len(sp_rows),
        ),
    }
    print(f"\n=== 汇总（逐比较对筛选，门槛 20%）===")
    print(f"   S−I（{len(si_rows)}/{len(rows)} 对可用）: "
          + (f"中位 {np.median(di):+.2f} dB (范围 {di.min():+.2f}~{di.max():+.2f}, "
             f"正的 {int((di>0).sum())}/{len(di)})" if di.size else "无可评对"))
    print(f"   S−P（{len(sp_rows)}/{len(rows)} 对可用）: "
          + (f"中位 {np.median(dp):+.2f} dB (范围 {dp.min():+.2f}~{dp.max():+.2f}, "
             f"正的 {int((dp>0).sum())}/{len(dp)})" if dp.size else "无可评对"))
    for lab, rows_sub, key in (("S−I", si_rows, "underpowered_si"),
                               ("S−P", sp_rows, "underpowered_sp")):
        weak = [r for r in rows if r[key]]
        if weak:
            print(
                f"   ⚠️ {lab} 剔除 {len(weak)} 对（域<20%）："
                + "、".join(f"{r['source'][:3]}->{r['target'][:3]}" for r in weak)
            )
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

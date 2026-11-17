#!/usr/bin/env python3
"""SHARP 两层深度——受控消融（S0.5b）。

承接 `03_layer_ablation.py` 的观测：整层抽掉任一层，PSNR 从 30.6 掉到 23.9，
但两层的 z 中位差只有 0.19 mm（几乎共面），单层 alpha 空洞仅 0.4~1.3%，
所以「第二层补遮挡区」解释不了 6.7 dB。本脚本要把收益来源定性清楚。

## 待判定的假设（按可得性排序）

* **H1 覆盖密度 / 抗锯齿**：第二层的作用只是让同一像素有两次采样，等效于把点
  加密或把分辨率提高。若成立 → 任何"把点加密一倍"的手段都应获得同等收益，
  S2 里没必要保留"两层"这个语义。
* **H2 独立正确信息**：两层给出的是同表面两个**去相关**的估计，逐像素择优就能
  逼近 full。03 已观测到 oracle 逐像素择优 = 30.94 dB ≈ full 30.56 dB，
  且 corr(e0,e1)=+0.37（正相关，不是互补）——此假设需要更强的检验。
* **H3 深度排序修正**：full 的深度 99.68% 等于 min(d0,d1)，说明合成时按最近层
  取色，第二层只在第一层"更远"的像素上接管。若成立 → 收益集中在两层 z 有序分歧的
  像素上，与"加密"无关。

## 受控手段（都不改网络，只改渲染输入）

1. `opacity=1` 强制不透明：排除 opacity<1 的部分透明叠加。
2. `scale x k`（k=0.5/1/2）：改变点的覆盖半径，分离"点数"与"覆盖面积"。
3. 单层 + 副本抖动：把 layer0 复制一份并加极小随机位移，检验 H1（纯加密）。
4. 逐像素分层诊断：把像素按 |dz| 分档，看 full 相对单层的增益集中在哪一档。
5. 混合权重扫描 + 同像素替换（用 layer1 的颜色替换 layer0）：判别是否为颜色信息。

## 口径边界（必须随结果一起读）

* `color_space=linearRGB` 时渲染内部会做 `linearRGB2sRGB` 再返回，而 GT 是 sRGB
  原始图，两者同口径可比；但 `background_color="black"` 下 alpha<1 的像素会被
  压暗，所以**绝对 PSNR 不代表真实质量**，只用于同口径横向比较。
* 本脚本只喂**一张图**。任何结论都只对"单图前馈输出"成立，不能外推到多视角融合。
* SHARP 训练代码未开源，本节不训练、不微调，只做前向 + 渲染的因果拆解。
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch


# ---------------------------------------------------------------- 基础工具


def psnr(pred: np.ndarray, target: np.ndarray) -> float:
    mse = float(np.mean((pred.astype(np.float64) - target.astype(np.float64)) ** 2))
    return 10.0 * math.log10(1.0 / max(mse, 1e-12))


def load_gt(path: Path) -> np.ndarray:
    """读原图（sRGB，float32 0..1）。

    ⚠️ 不要直接用 PIL：HEIC 需要 pillow_heif 插件，裸 `Image.open` 会抛
    `UnidentifiedImageError`。官方 `io.load_rgb` 已注册该插件并处理 EXIF 旋转，
    直接复用可保证与推理输入是同一张、同一朝向。
    """
    from sharp.utils import io

    image_u8, _, _ = io.load_rgb(path)
    return np.asarray(image_u8, dtype=np.float32) / 255.0


def to_tensor_img(arr: np.ndarray, device) -> torch.Tensor:
    t = torch.from_numpy(arr).permute(2, 0, 1)[None].float().to(device)
    return t


def save_png(arr: np.ndarray, path: Path) -> None:
    from PIL import Image

    Image.fromarray(np.clip(arr * 255.0, 0, 255).astype(np.uint8)).save(path)


def save_gray(arr: np.ndarray, path: Path, vmin=None, vmax=None) -> None:
    from PIL import Image

    a = arr.astype(np.float64)
    lo = float(np.nanmin(a)) if vmin is None else vmin
    hi = float(np.nanmax(a)) if vmax is None else vmax
    if hi - lo < 1e-12:
        hi = lo + 1e-12
    Image.fromarray(((np.clip((a - lo) / (hi - lo), 0, 1)) * 255).astype(np.uint8)).save(path)


# ---------------------------------------------------------------- 渲染


def render(renderer, g, intrinsics: torch.Tensor, height: int, width: int):
    """渲染一张，返回 (color[H,W,3], alpha[H,W], depth[H,W])，全 numpy。"""
    with torch.no_grad():
        out = renderer(
            g,
            extrinsics=torch.eye(4, device=g.mean_vectors.device, dtype=torch.float32)[None],
            intrinsics=intrinsics[None],
            image_width=width,
            image_height=height,
        )
    color = np.clip(out.color[0].permute(1, 2, 0).float().cpu().numpy(), 0.0, 1.0)
    alpha = out.alpha[0, 0].float().cpu().numpy()
    depth = out.depth[0, 0].float().cpu().numpy()
    return color, alpha, depth


def clone_g(g, **overrides):
    """复制 Gaussians3D，可选替换某些字段。字段名同官方 NamedTuple。"""
    from sharp.utils.gaussians import Gaussians3D

    kw = dict(
        mean_vectors=g.mean_vectors.clone(),
        singular_values=g.singular_values.clone(),
        quaternions=g.quaternions.clone(),
        colors=g.colors.clone(),
        opacities=g.opacities.clone(),
    )
    kw.update(overrides)
    return Gaussians3D(**kw)


def subset_gaussians(g, start: int, end: int):
    from sharp.utils.gaussians import Gaussians3D

    return Gaussians3D(
        mean_vectors=g.mean_vectors[:, start:end].contiguous(),
        singular_values=g.singular_values[:, start:end].contiguous(),
        quaternions=g.quaternions[:, start:end].contiguous(),
        colors=g.colors[:, start:end].contiguous(),
        opacities=g.opacities[:, start:end].contiguous(),
    )


def concat_g(parts: list):
    from sharp.utils.gaussians import Gaussians3D

    return Gaussians3D(
        mean_vectors=torch.cat([p.mean_vectors for p in parts], dim=1),
        singular_values=torch.cat([p.singular_values for p in parts], dim=1),
        quaternions=torch.cat([p.quaternions for p in parts], dim=1),
        colors=torch.cat([p.colors for p in parts], dim=1),
        opacities=torch.cat([p.opacities for p in parts], dim=1),
    )


# ---------------------------------------------------------------- 推理


def predict(predictor, image_u8: np.ndarray, f_px: float, device):
    """复刻 cli/predict.py 的推理段，返回相机坐标高斯 + 内参 + 内部尺寸。

    ⚠️ 必须与 `03_layer_ablation.py::predict_and_split` **逐行一致**，否则基线
    PSNR（30.56）不能跨脚本比较。两点容易写错：
    * 内参是 **4x4**（`unproject_gaussians` 内部要做 `ndc @ K @ E` 的 4x4 连乘）。
    * `F.interpolate` 用 `align_corners=True`（与官方 cli 一致，不是 False）。
    """
    import torch.nn.functional as F

    from sharp.utils.gaussians import unproject_gaussians

    internal_shape = (1536, 1536)
    # .copy() 避开 "non-writable array" 警告（torch 2.x 对非可写 numpy 视图会告警）
    image_pt = torch.from_numpy(image_u8.copy()).float().to(device).permute(2, 0, 1) / 255.0
    _, height, width = image_pt.shape
    disparity_factor = torch.tensor([f_px / width]).float().to(device)

    image_resized_pt = F.interpolate(
        image_pt[None],
        size=(internal_shape[1], internal_shape[0]),
        mode="bilinear",
        align_corners=True,
    )

    intrinsics = torch.tensor(
        [
            [f_px, 0, width / 2, 0],
            [0, f_px, height / 2, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ],
        device=device,
        dtype=torch.float32,
    )
    intrinsics_resized = intrinsics.clone()
    intrinsics_resized[0] *= internal_shape[0] / width
    intrinsics_resized[1] *= internal_shape[1] / height

    with torch.no_grad():
        gaussians_ndc = predictor(image_resized_pt, disparity_factor)
        gaussians = unproject_gaussians(
            gaussians_ndc, torch.eye(4, device=device), intrinsics_resized, internal_shape
        )
    return {
        "gaussians": gaussians,
        "intrinsics": intrinsics,
        "size": (height, width),
        "image_resized_pt": image_resized_pt,
        "disparity_factor_tensor": disparity_factor,
    }


# ---------------------------------------------------------------- 主流程


def main() -> int:
    ap = argparse.ArgumentParser(description="SHARP 两层深度受控消融（S0.5b）")
    ap.add_argument("--input", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--ckpt", type=Path, default=None)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--render-scale", type=float, default=0.5,
                    help="渲染下采样（受控实验只需统计口径一致，0.5 省 4x 时间）")
    ap.add_argument("--jitter-px", type=float, default=0.5,
                    help="H1 复制抖动幅度（内部 768 网格的像素单位）")
    ap.add_argument("--bg", default="black,white",
                    help="渲染背景色，逗号分隔。底色会系统性偏袒覆盖率高的层，必须两种都报")
    args = ap.parse_args()

    device = torch.device(args.device)
    args.out.mkdir(parents=True, exist_ok=True)

    from sharp.models import PredictorParams, create_predictor
    from sharp.utils import io
    from sharp.utils.gsplat import GSplatRenderer

    print(f"🖼️  输入: {args.input}")
    image_u8, _, f_px = io.load_rgb(args.input)
    h, w = image_u8.shape[:2]
    print(f"   尺寸 {w}x{h}, f_px={f_px:.2f}")

    t0 = time.time()
    predictor = create_predictor(PredictorParams())
    if args.ckpt:
        state = torch.load(args.ckpt, weights_only=True)
    else:
        import sharp.cli.predict as _p

        print(f"⬇️  加载权重 {_p.DEFAULT_MODEL_URL}")
        state = torch.hub.load_state_dict_from_url(_p.DEFAULT_MODEL_URL, progress=True)
    predictor.load_state_dict(state)
    predictor.eval().to(device)
    print(f"🏋️  权重加载 {time.time() - t0:.1f}s")

    t0 = time.time()
    pred = predict(predictor, image_u8, f_px, device)
    g = pred["gaussians"]
    n_total = g.mean_vectors.shape[1]
    print(f"🔮 前向 {time.time() - t0:.1f}s, 高斯数 {n_total:,}")

    n_layers = 2
    hw = n_total // n_layers
    side = int(round(math.sqrt(hw)))
    if hw * n_layers != n_total or side * side != hw:
        raise RuntimeError(f"点数 {n_total} 与 2 x {side}^2 不符")
    slices = [(l * hw, (l + 1) * hw) for l in range(n_layers)]

    # 渲染尺寸
    rh, rw = h, w
    if args.render_scale != 1.0:
        rh = int(round(h * args.render_scale))
        rw = int(round(w * args.render_scale))
    ratio = rw / w
    # K 是 4x4；gsplat 内部取 Ks[:, :3, :3]，所以只缩放前两行即可
    K = pred["intrinsics"].clone()
    if ratio != 1.0:
        K[0] *= ratio
        K[1] *= ratio

    gt = load_gt(args.input)
    if ratio != 1.0:
        from PIL import Image

        gt = np.asarray(
            Image.fromarray((gt * 255).astype(np.uint8)).resize((rw, rh), Image.LANCZOS),
            dtype=np.float32,
        ) / 255.0
    print(f"🖥️  渲染 {rw}x{rh} (scale={args.render_scale}), 基线 GT 已同尺寸")

    # ⚠️ 底色必须两种都报：单层有 3.8% 的 alpha 空洞，black 会把它们压黑（惩罚
    # 覆盖不足的层），white 会把它们提亮。03 脚本用 black 报出 layer0=23.86，
    # 这里若只用 white 会报 17.96 —— 差 6 dB 全部来自那 3.8% 像素。这不是实现差异，
    # 而是「单层误差高度集中在它没盖住的那一小块」这一事实的直接体现。
    bgs = [b.strip() for b in args.bg.split(",") if b.strip()]
    renderers = {b: GSplatRenderer(color_space="linearRGB", background_color=b) for b in bgs}

    l0 = subset_gaussians(g, *slices[0])
    l1 = subset_gaussians(g, *slices[1])

    results: dict = {}
    images: dict = {}

    def record(name: str, gg, note: str = "", bg: str | None = None) -> dict:
        renderer = renderers[bg or bgs[0]]
        c, a, d = render(renderer, gg, K, rh, rw)
        p = psnr(c, gt)
        key = name if bg is None else f"{name}@{bg}"
        results[key] = {
            "n_gaussians": int(gg.mean_vectors.shape[1]),
            "psnr": p,
            "alpha_mean": float(a.mean()),
            "bg": bg or bgs[0],
            "depth_median": float(np.median(d[a > 0.5])) if np.any(a > 0.5) else None,
            "note": note,
        }
        images[key] = (c, a, d)
        print(
            f"   {key:32s} n={gg.mean_vectors.shape[1]:>9,}  PSNR={p:6.2f}  "
            f"alpha={a.mean():.4f}  {note}"
        )
        return results[key]

    print("\n=== 基线（两种底色各报一次，暴露单层对底色的敏感度）===")
    for bg in bgs:
        for nm, gg, note in (("full", g, "整份输出（两层）"),
                             ("layer0", l0, "仅第 0 层"),
                             ("layer1", l1, "仅第 1 层")):
            record(nm, gg, note, bg=bg)

    # 主口径 = 第一个底色（默认 black，贴合官方 cli/render.py）
    primary_bg = bgs[0]

    # ---------------- H1：覆盖密度 ----------------
    print("\n=== H1 覆盖密度：把单层复制一份（纯加密，无新信息）===")
    record("layer0_dup", concat_g([l0, clone_g(l0)]), "layer0 完全复制，点数翻倍")
    for s in (0.5, 0.707, 2.0):
        sv = l0.singular_values * s
        record(f"layer0_scale{s:g}", clone_g(l0, singular_values=sv), f"scale x{s:g}")
        record(f"layer1_scale{s:g}", clone_g(l1, singular_values=l1.singular_values * s), f"scale x{s:g}")

    # 抖动复制：在 NDC 平面加极小平移，模拟"第二个独立抖动样本"
    if args.jitter_px > 0.0:
        side_px = side
        pix_size_ndc = 2.0 / side_px

        def jitter_copy(layer_g, dx_px: float, dy_px: float):
            """在 normalized device 平面内平移。mean_vectors 已是相机坐标 z 不变，
            直接按 z 与内参把像素位移换成相机空间位移。"""
            m = layer_g.mean_vectors.clone()
            z = m[:, :, 2]
            fx = K[0, 0].item()
            fy = K[1, 1].item()
            # 像素位移 -> 相机空间位移（与内参、深度一致）
            sx = (dx_px * (rw / side_px)) / fx * z
            sy = (dy_px * (rh / side_px)) / fy * z
            m[:, :, 0] += sx
            m[:, :, 1] += sy
            return clone_g(layer_g, mean_vectors=m)

        for jp in (0.5, 1.0):
            jc = jitter_copy(l0, jp, jp)
            record(f"layer0_dup_jit{jp:g}px", concat_g([l0, jc]), f"复制 + {jp}px 抖动")
        _ = pix_size_ndc

    # ---------------- opacity 强制 ----------------
    print("\n=== 强制 opacity = 1（排除部分透明叠加）===")
    one0 = clone_g(l0, opacities=torch.ones_like(l0.opacities))
    one1 = clone_g(l1, opacities=torch.ones_like(l1.opacities))
    record("layer0_op1", one0, "仅 layer0, opacity=1")
    record("layer1_op1", one1, "仅 layer1, opacity=1")
    record("full_op1", concat_g([one0, one1]), "两层, opacity=1")

    # ---------------- D1：两层点位的真实空间关系 ----------------
    # 这决定「超采样」这个词能不能用。若两层投影到像素后完全重合，就不存在
    # 子像素互补，收益只能来自「深度排序选择」；若错位达亚像素级，才可能互补。
    print("\n=== D1 两层点位的空间关系（同索引 = 同像素格）===")
    m0 = l0.mean_vectors[0].cpu().numpy()   # [HW,3] 相机坐标
    m1 = l1.mean_vectors[0].cpu().numpy()
    _fx = K[0, 0].item() / ratio if ratio != 1.0 else K[0, 0].item()   # 回到内部尺度
    _fy = K[1, 1].item() / ratio if ratio != 1.0 else K[1, 1].item()
    _cx = K[0, 2].item() / ratio if ratio != 1.0 else K[0, 2].item()
    _cy = K[1, 2].item() / ratio if ratio != 1.0 else K[1, 2].item()
    u0 = _fx * m0[:, 0] / m0[:, 2] + _cx
    v0 = _fy * m0[:, 1] / m0[:, 2] + _cy
    u1 = _fx * m1[:, 0] / m1[:, 2] + _cx
    v1 = _fy * m1[:, 1] / m1[:, 2] + _cy
    du = u1 - u0
    dv = v1 - v0
    dpx_internal = np.sqrt(du**2 + dv**2)
    # 内部 1536 -> 768 网格，相邻格中心相距 stride=2 内部像素
    dpx_render = dpx_internal * (rw / 1536.0)
    dz_pt = m1[:, 2] - m0[:, 2]

    def _stat(name, a, unit=""):
        a = np.asarray(a, dtype=np.float64)
        print(
            f"   {name:26s} med={np.median(a):+9.5f} p10={np.percentile(a,10):+9.5f} "
            f"p90={np.percentile(a,90):+9.5f} |abs|med={np.median(np.abs(a)):8.5f}{unit}"
        )

    _stat("dz (相机 z 差)", dz_pt, " m")
    _stat("dxy (相机 xy 距离)", np.linalg.norm(m1[:, :2] - m0[:, :2], axis=1), " m")
    _stat("du (内部像素)", du)
    _stat("dv (内部像素)", dv)
    _stat("dpx (内部像素距离)", dpx_internal)
    _stat("dpx (渲染像素距离)", dpx_render, " px")
    geom = {
        "dz_abs_median_m": float(np.median(np.abs(dz_pt))),
        "dz_p90_abs_m": float(np.percentile(np.abs(dz_pt), 90)),
        "dxy_cam_median_m": float(np.median(np.linalg.norm(m1[:, :2] - m0[:, :2], axis=1))),
        "dpx_median_internal": float(np.median(dpx_internal)),
        "dpx_median_render": float(np.median(dpx_render)),
        "grid_spacing_internal": 2.0,
        "offset_over_grid_spacing": float(np.median(dpx_internal) / 2.0),
        "frac_dpx_lt_0p5px_render": float(np.mean(dpx_render < 0.5)),
        "frac_dpx_lt_1px_render": float(np.mean(dpx_render < 1.0)),
    }
    print(
        f"   → 渲染分辨率下错位中位 {np.median(dpx_render):.3f} px，"
        f"{np.mean(dpx_render < 1.0) * 100:.1f}% 的点错位 < 1 px，"
        f"错位/格距 = {np.median(dpx_internal) / 2.0:.4f}"
    )
    results["_d1_geometry"] = geom

    # ---------------- H3：按 dz 分档 ----------------
    print("\n=== H3 深度排序：按两层 |dz| 分档看增益来源 ===")
    c0, a0, d0 = images[f"layer0@{primary_bg}"]
    c1, a1, d1 = images[f"layer1@{primary_bg}"]
    cf, af, df = images[f"full@{primary_bg}"]
    valid = (a0 > 0.5) & (a1 > 0.5) & (a0 > 1e-6) & (a1 > 1e-6)
    dz = (d1 - d0).astype(np.float64)

    e0 = ((c0 - gt) ** 2).mean(2)
    e1 = ((c1 - gt) ** 2).mean(2)
    ef = ((cf - gt) ** 2).mean(2)

    print(f"   有效重叠像素 {valid.sum():,} / {valid.size:,}")
    bins = [(0.0, 0.001), (0.001, 0.005), (0.005, 0.02), (0.02, float("inf"))]
    bin_stats = []
    for lo, hi in bins:
        m = valid & (np.abs(dz) >= lo) & (np.abs(dz) < hi)
        if m.sum() < 100:
            continue
        st = {
            "abs_dz_range_m": [lo, hi],
            "n_px": int(m.sum()),
            "frac_px": float(m.sum() / max(valid.sum(), 1)),
            "mse_layer0": float(e0[m].mean()),
            "mse_layer1": float(e1[m].mean()),
            "mse_full": float(ef[m].mean()),
            "mse_best_single": float(np.minimum(e0[m], e1[m]).mean()),
        }
        st["gain_over_layer0_db"] = 10 * math.log10(max(st["mse_layer0"], 1e-12) / max(st["mse_full"], 1e-12))
        st["gain_over_best_db"] = 10 * math.log10(max(st["mse_best_single"], 1e-12) / max(st["mse_full"], 1e-12))
        bin_stats.append(st)
        print(
            f"   |dz| in [{lo*1000:7.1f}, {hi*1000 if hi<float('inf') else -1:7.1f})mm  "
            f"占 {st['frac_px']*100:5.2f}%  "
            f"l0={st['mse_layer0']:.6f} l1={st['mse_layer1']:.6f} best={st['mse_best_single']:.6f} "
            f"full={st['mse_full']:.6f}  gain_vs_l0={st['gain_over_layer0_db']:+6.2f}dB "
            f"gain_vs_best={st['gain_over_best_db']:+6.2f}dB"
        )
    results["_dz_bins"] = bin_stats

    # 谁更近，谁赢
    print("\n=== H3b 取最近层 vs full ===")
    closer0 = valid & (d0 <= d1)
    pick = np.where((d0 <= d1)[..., None], c0, c1)
    print(f"   full 深度 == min(d0,d1) 的比例: {np.mean(np.abs(df - np.minimum(d0, d1)) < 0.02):.4f}")
    print(f"   取最近层 PSNR = {psnr(pick, gt):.2f}   full = {psnr(cf, gt):.2f}")
    print(f"   rms(pick - full) = {np.sqrt(((pick - cf) ** 2).mean()):.4f}")
    print(f"   d0<=d1 的像素占比 {np.mean(d0 <= d1):.4f}")
    _ = closer0

    # ---------------- H2：信息独立性 ----------------
    print("\n=== H2 信息独立性 ===")
    for tag, m in (("全部", np.ones_like(valid, bool)), ("重叠区", valid)):
        if m.sum() < 100:
            continue
        g0 = (c0 - gt)[m].ravel()
        g1 = (c1 - gt)[m].ravel()
        corr = float(np.corrcoef(g0, g1)[0, 1])
        mix = 0.5 * (c0 + c1)
        print(
            f"   {tag}: corr(e0,e1)={corr:+.4f}  "
            f"PSNR l0={psnr(c0[m], gt[m]):.2f} l1={psnr(c1[m], gt[m]):.2f} "
            f"mix={psnr(mix[m], gt[m]):.2f} full={psnr(cf[m], gt[m]):.2f}"
        )
        results[f"_corr_{tag}"] = {
            "corr": corr,
            "psnr_l0": psnr(c0[m], gt[m]),
            "psnr_l1": psnr(c1[m], gt[m]),
            "psnr_mix": psnr(mix[m], gt[m]),
            "psnr_full": psnr(cf[m], gt[m]),
        }

    # 用 layer1 的颜色替换 layer0（几何不变，只换外观）——判别是否颜色信息增益
    print("\n=== H2b 只换颜色（几何用 layer0，颜色用 layer1）===")
    swapped = clone_g(l0, colors=l1.colors)
    record("layer0_geom_layer1_color", concat_g([swapped, l1]), "layer0 几何 + layer1 颜色")

    # ---------------- 汇总 ----------------
    print("\n=== 汇总 ===")
    base = results[f"full@{primary_bg}"]["psnr"]
    for k, v in results.items():
        if k.startswith("_"):
            continue
        print(f"   {k:34s} {v['psnr']:6.2f}  Δ vs full = {v['psnr'] - base:+.2f} dB")

    # 落盘（只存主口径 + 最有判别力的几张，避免目录膨胀）
    keep = {f"full@{primary_bg}", f"layer0@{primary_bg}", f"layer1@{primary_bg}",
            "layer0_dup", "layer0_dup_jit0.5px", "full_op1", "layer0_geom_layer1_color"}
    for name, (c, a, _d) in images.items():
        if name in keep:
            save_png(c, args.out / f"c_{name.replace('@', '_')}.png")
            save_gray(a, args.out / f"a_{name.replace('@', '_')}.png", 0.0, 1.0)
    if valid.any():
        save_gray(np.abs(dz), args.out / "dz_abs.png")

    out_json = {
        "input": str(args.input),
        "f_px": f_px,
        "image_hw": [h, w],
        "render_hw": [rh, rw],
        "render_scale": args.render_scale,
        "n_total": n_total,
        "n_layers": n_layers,
        "grid_side": side,
        "results": results,
        "primary_bg": primary_bg,
        "backgrounds": bgs,
        "caveat": [
            "PNG 无 EXIF 焦距时走 30mm 兜底，disparity_factor 本身错，绝对数值不代表真实质量",
            "底色会系统性偏袒覆盖率高的层：black 惩罚空洞、white 提亮空洞。"
            "同一 layer0 在 black 下 23.86、white 下 17.96（03/04 两脚本的 6 dB 差异全部来自此）",
            "只在 source 视角（extrinsic=I）渲染，未检查其它视角，不能代表跨视角一致性",
            "仅单图前馈，不外推到多视角融合",
            "未训练/未微调（SHARP 训练代码未开源）",
        ],
    }
    (args.out / "controlled_ablation.json").write_text(
        json.dumps(out_json, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"\n💾 已写 {args.out / 'controlled_ablation.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

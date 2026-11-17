#!/usr/bin/env python3
"""03_layer_ablation.py — SHARP 两层深度的「层数消融」（三图计划 S0.5）。

为什么需要这个脚本
------------------
SHARP 每个像素输出两个视差（前表面 / 被遮挡后层），各生成一套高斯，所以一张图
产出 2 x 768^2 = 1,179,648 个高斯。但**第二层到底贡献了什么**从来没有单独测过：
论文只报整体渲染指标，合成后的 PSNR 无法区分「第二层补了遮挡」和「第二层只是
把第一层推糊了一点」。

不改网络、不改权重也能测：composer 的展平顺序是 layer-major（见下），把输出按
H*W 切成两段就得到「只用第一层」「只用第二层」两个对照，与完整两层在同一相机、
同一 GT 下对拍。

展平顺序的证明（不靠猜）
------------------------
`composer.py:138` `mean_vectors.permute(0,2,3,4,1).flatten(1,3)`，入参形状
`[B, C, num_layers, H, W]` → permute 后 `[B, L, H, W, C]` → flatten(1,3) 后
索引 = `l*H*W + y*W + x`，即 **layer-major**。

更硬的不变量：base values 是同一张像素网格，两层共用 mean_x_ndc / mean_y_ndc，
只有 z（逆深度）不同（delta_factor.xy = 0.001 压得极小）。所以「正确切法」下
`gaussians[i]` 与 `gaussians[i+HW]` 的 x/y 应几乎重合，z 有真实落差；而「交错
切法」（index = 2i + l）下相邻点属于不同像素，x/y 必然差一个像素格。脚本两种
都算，用 xy 中位差直接判定，不依赖任何假设。

口径边界
--------
* source 视角渲染（extrinsic = I），不需要任何位姿估计，所以这个检验与「多图
  融合」「几何是否准确」无关，只回答「第二层对可见像素有没有贡献」。
* PSNR 是 source 自洽口径，不能跨视角数比较、不能当几何精度证据（ERRATA/E-08）。
* 第二层在 source 视角本来就大多被第一层遮住，所以「完整两层 ≈ 只留第一层」
  是**预期结果**，它证明的是第二层的价值只在遮挡区补面，不是 SHARP 变差了。

用法
----
    python sharp/03_layer_ablation.py --input <图> --out <输出目录>
    # 或用 wrapper：bash sharp/03_layer_ablation.sh INPUT=<图>
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


# ---------------------------------------------------------------- 基础工具


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def audit_focal(path: Path) -> dict:
    """审计焦距来源——决定度量尺度，也决定网络的条件输入（disparity_factor）。

    复刻 `utils/io.py::load_rgb` 的取数逻辑，但把「从哪来」记录下来。这一条是
    S0 的放行条件：走 30mm 兜底时 disparity_factor 本身就是错的，事后乘系数补不回来。
    """
    from sharp.utils.io import extract_exif

    if path.suffix.lower() == ".heic":
        import pillow_heif

        img_pil = pillow_heif.open_heif(path, convert_hdr_to_8bit=True).to_pillow()
    else:
        img_pil = Image.open(path)

    exif = extract_exif(img_pil)
    f_35mm = exif.get("FocalLengthIn35mmFilm", exif.get("FocalLenIn35mmFilm", None))
    source = "FocalLengthIn35mmFilm"
    if f_35mm is None or f_35mm < 1:
        f_35mm = exif.get("FocalLength", None)
        source = "FocalLength"
        if f_35mm is None:
            f_35mm = 30.0
            source = "FALLBACK_30mm"
        if f_35mm < 10.0:
            f_35mm *= 8.4
            source += "+(<10mm x8.4)"
    return {
        "f_35mm": float(f_35mm),
        "source": source,
        "orientation": exif.get("Orientation", 1),
        "is_fallback": source == "FALLBACK_30mm",
        "raw_exif_keys": sorted(k for k in exif if "ocal" in k or "rient" in k),
    }


def psnr(pred: np.ndarray, target: np.ndarray, mask: np.ndarray | None = None) -> float:
    """pred/target: HxWx3 float [0,1]；mask: HxW bool（None = 全图）。"""
    d = (pred - target).astype(np.float64) ** 2
    if mask is not None:
        if mask.sum() == 0:
            return float("nan")
        d = d[mask]
    mse = float(d.mean())
    return float(-10.0 * math.log10(mse + 1e-12))


def save_rgb(arr: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray((np.clip(arr, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)).save(path)


def gray_png(arr: np.ndarray, path: Path, vmax: float) -> None:
    g = np.clip(arr / max(vmax, 1e-9), 0.0, 1.0)
    save_rgb(np.repeat(g[..., None], 3, axis=2), path)


# ---------------------------------------------------------------- 推理


def predict_and_split(predictor, image: np.ndarray, f_px: float, device: torch.device):
    """复刻 `cli/predict.py::predict_image`，返回相机坐标高斯与内参。

    注意：本函数**不做层切分**。切分前必须先用 `verify_split_by_hook` 证明展平
    顺序，否则会拿到看起来合理但完全错误的层。
    """
    from sharp.utils.gaussians import unproject_gaussians

    internal_shape = (1536, 1536)
    image_pt = torch.from_numpy(image.copy()).float().to(device).permute(2, 0, 1) / 255.0
    _, height, width = image_pt.shape
    disparity_factor = torch.tensor([f_px / width]).float().to(device)

    image_resized_pt = F.interpolate(
        image_pt[None], size=(internal_shape[1], internal_shape[0]),
        mode="bilinear", align_corners=True,
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

    t0 = time.time()
    with torch.no_grad():
        gaussians_ndc = predictor(image_resized_pt, disparity_factor)
        gaussians = unproject_gaussians(
            gaussians_ndc, torch.eye(4, device=device), intrinsics_resized, internal_shape
        )
    fwd_s = time.time() - t0

    means = gaussians.mean_vectors[0]  # [N, 3]
    n_total = means.shape[0]
    return {
        "gaussians": gaussians,
        "gaussians_ndc": gaussians_ndc,
        "intrinsics": intrinsics,
        "size": (height, width),
        "n_total": n_total,
        "fwd_s": fwd_s,
        "disparity_factor": float(disparity_factor.item()),
        "image_resized_pt": image_resized_pt,
        "disparity_factor_tensor": disparity_factor,
    }


def split_layers(means: torch.Tensor, n_layers: int) -> tuple[list[tuple[int, int]], dict]:
    """按 layer-major 切层，并记录「为什么」——不靠几何猜。

    代码证据：`composer.py:138` 的 `mean_vectors.permute(0,2,3,4,1).flatten(1,3)`
    入参是 `[B, C, N, H, W]`（注释原文 `[B, C, N, H, W] -> [B, N, H, W, C]`），
    permute 后为 `[B, N, H, W, C]`，flatten(1,3) 的线性索引 = `l*H*W + y*W + x`。
    所以第 l 层是连续切片 `[l*HW, (l+1)*HW)`。

    ⚠️ 这里**不用世界坐标 xy 距离、alpha 覆盖率或像素格点计数来反推顺序**，三者
    实测都失效（详见 `verify_split_by_hook` 的 docstring）。真正的验证是 hook 住
    composer 拿未展平张量逐元素比对。
    """
    n_total = means.shape[0]
    hw = n_total // n_layers
    side = int(round(math.sqrt(hw)))
    n_total_ok = hw * n_layers == n_total

    info = {
        "n_total": n_total,
        "n_layers": n_layers,
        "gaussians_per_layer": hw,
        "grid_side": side,
        "grid_area_matches": side * side == hw,
        "assumed_order": "layer-major (composer.py:138 permute(0,2,3,4,1).flatten(1,3))",
        "evidence": "代码级：形状注释 [B,C,N,H,W] -> [B,N,H,W,C]，索引 = l*H*W + y*W + x",
    }
    if not n_total_ok or side * side != hw:
        raise RuntimeError(f"高斯数与 2 x 768^2 不符：n_total={n_total}, hw={hw}, side={side}")

    return [(l * hw, (l + 1) * hw) for l in range(n_layers)], info


def verify_split_by_hook(predictor, image_pt, disparity_factor, n_layers: int) -> dict:
    """决定性验证：hook 住 composer，拿到**未展平**的 `[B,3,N,H,W]`，按第 2 维切层，
    再与正常输出（已展平）的连续切片逐元素比对。

    为什么这个判据是决定性的（而几何/alpha 判据都不行）：
    * `initializer.py::_create_base_xy` 把 `mean_x_ndc` 造成 `[B,1,N,H,W]`，N 就是
      num_layers；`_create_surface_layer` 返回的 disparity 也是 `[B,1,N,H,W]`。
    * `composer.py::_forward_mean` 里 `base.mean_x_ndc.repeat((1,3,1,1,1))` → `[B,3,N,H,W]`，
      之后 `permute(0,2,3,4,1).flatten(1,3)` 的线性索引 = `l*H*W + y*W + x`。
    * 所以「展平结果的前 HW 个 == 未展平的第 0 层」必须**逐位相等**。相等就证明
      layer-major，不相等就说明我的切片用错了，直接报错。

    之前的两个尝试都失败且不该再用：
    - 世界坐标 xy 距离：两层 z 差大时同像素点的世界 xy 本来就分开，判据方向错。
    - alpha 覆盖率：768 格点在原分辨率上各覆盖多像素，抽掉一半覆盖率仍有 0.985，
      与正确切法的 0.999 无法区分（实测反而「错」的更高）。
    - 像素格点计数：delta 会移动点位（`xx = base_x + delta_xy`），点不再落规则网格，
      实测每格 3.5 点、填充 28%，两种切法数值几乎相同。
    """
    from sharp.models.composer import GaussianComposer

    captured: dict = {}
    orig_forward = GaussianComposer.forward

    def patched(self, delta, base_values, global_scale=None, flatten_output=True):
        out = orig_forward(self, delta, base_values, global_scale, flatten_output=False)
        captured["unflat"] = out.mean_vectors.detach().clone()   # [B,3,N,H,W]
        captured["global_scale"] = None if global_scale is None else global_scale.detach().clone()
        # 复原官方展平，供调用方使用（保持与原实现完全一致的输出）
        mv = out.mean_vectors.permute(0, 2, 3, 4, 1).flatten(1, 3)
        sv = out.singular_values.permute(0, 2, 3, 4, 1).flatten(1, 3)
        q = out.quaternions.permute(0, 2, 3, 4, 1).flatten(1, 3)
        c = out.colors.permute(0, 2, 3, 4, 1).flatten(1, 3)
        o = out.opacities.flatten(1, 3)
        from sharp.utils.gaussians import Gaussians3D as _G

        return _G(mean_vectors=mv, singular_values=sv, quaternions=q, colors=c, opacities=o)

    GaussianComposer.forward = patched
    try:
        with torch.no_grad():
            flat = predictor(image_pt, disparity_factor)
    finally:
        GaussianComposer.forward = orig_forward

    unflat = captured["unflat"]           # [B,3,N,H,W]
    if unflat.shape[2] != n_layers:
        raise RuntimeError(f"composer 第 2 维是 {unflat.shape[2]}，不是层数 {n_layers}")

    hw = unflat.shape[3] * unflat.shape[4]
    checks = {}
    for l in range(n_layers):
        per_layer = unflat[:, :, l].permute(0, 2, 3, 1).flatten(1, 2)   # [B,HW,3]
        sliced = flat.mean_vectors[:, l * hw:(l + 1) * hw]              # [B,HW,3]
        diff = (per_layer - sliced).abs().max().item()
        checks[f"layer{l}"] = {
            "max_abs_diff": diff,
            "matches_slice": bool(diff < 1e-5),
            "z_median_m": float(per_layer[0, :, 2].median().item()),
        }

    ok = all(v["matches_slice"] for v in checks.values())
    return {
        "criterion": "未展平 [B,3,N,H,W] 的第 2 维切片 == 已展平输出的连续 HW 切片（逐元素）",
        "unflat_shape": list(unflat.shape),
        "hw": hw,
        "layers": checks,
        "verdict": "layer-major" if ok else "MISMATCH (切片错误，禁止继续)",
    }


def subset_gaussians(g, start: int, end: int):
    from sharp.utils.gaussians import Gaussians3D

    return Gaussians3D(
        mean_vectors=g.mean_vectors[:, start:end].contiguous(),
        singular_values=g.singular_values[:, start:end].contiguous(),
        quaternions=g.quaternions[:, start:end].contiguous(),
        colors=g.colors[:, start:end].contiguous(),
        opacities=g.opacities[:, start:end].contiguous(),
    )


def render_source(renderer, g, intrinsics: torch.Tensor, height: int, width: int):
    with torch.no_grad():
        out = renderer(
            g,
            extrinsics=torch.eye(4, device=g.mean_vectors.device, dtype=torch.float32)[None],
            intrinsics=intrinsics[None],
            image_width=width,
            image_height=height,
        )
    color = out.color[0].permute(1, 2, 0).float().cpu().numpy()
    alpha = out.alpha[0, 0].float().cpu().numpy()   # [B,1,H,W] -> [H,W]
    depth = out.depth[0, 0].float().cpu().numpy()
    return np.clip(color, 0.0, 1.0), alpha, depth


# ---------------------------------------------------------------- 主流程


def main() -> int:
    ap = argparse.ArgumentParser(description="SHARP 两层深度层数消融（S0.5）")
    ap.add_argument("--input", required=True, type=Path, help="输入图（带 EXIF 焦距最佳）")
    ap.add_argument("--out", required=True, type=Path, help="输出目录")
    ap.add_argument("--ckpt", type=Path, default=None, help="权重（默认走 sharp 默认路径）")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--render-scale", type=float, default=1.0, help="渲染下采样比例（1.0=原尺寸）")
    args = ap.parse_args()

    device = torch.device(args.device)
    args.out.mkdir(parents=True, exist_ok=True)

    from sharp.models import PredictorParams, create_predictor
    from sharp.utils import io
    from sharp.utils.gsplat import GSplatRenderer

    print(f"🖼️  输入: {args.input}")
    focal_audit = audit_focal(args.input)
    print(f"🔍 焦距: {focal_audit['f_35mm']:.2f}mm (来源 {focal_audit['source']})")
    if focal_audit["is_fallback"]:
        print("⚠️  无 EXIF 焦距，走 30mm 兜底 → disparity_factor 本身错，结果只能算工程冒烟")

    image, _, f_px = io.load_rgb(args.input)
    print(f"   尺寸 {image.shape[1]}x{image.shape[0]}, f_px={f_px:.2f}")

    t0 = time.time()
    state = torch.load(args.ckpt, weights_only=True) if args.ckpt else None
    predictor = create_predictor(PredictorParams())
    if state is None:
        import sharp.cli.predict as _p

        print(f"⬇️  加载官方权重 {_p.DEFAULT_MODEL_URL}")
        state = torch.hub.load_state_dict_from_url(_p.DEFAULT_MODEL_URL, progress=True)
    predictor.load_state_dict(state)
    predictor.eval().to(device)
    load_s = time.time() - t0
    print(f"🏋️  权重加载 {load_s:.1f}s")

    pred = predict_and_split(predictor, image, f_px, device)
    print(f"🔮 前向 {pred['fwd_s']:.1f}s, 高斯数 {pred['n_total']:,}")

    n_layers = 2  # PredictorParams().num_monodepth_layers 默认值；多于 2 层时上游会改
    slices, order_info = split_layers(pred["gaussians"].mean_vectors[0], n_layers)
    print(f"🧭 切层依据: {order_info['assumed_order']}")
    print(f"   {order_info['evidence']}")

    height, width = pred["size"]
    if args.render_scale != 1.0:
        height = int(round(height * args.render_scale))
        width = int(round(width * args.render_scale))
    scale_ratio = width / pred["size"][1]

    intrinsics = pred["intrinsics"].clone()
    if scale_ratio != 1.0:
        intrinsics[0] *= scale_ratio
        intrinsics[1] *= scale_ratio

    renderer = GSplatRenderer(color_space="linearRGB", background_color="black")
    g = pred["gaussians"]

    # 决定性验证切分假设：hook composer 拿未展平 [B,3,N,H,W]，与展平输出的连续切片
    # 逐元素比对。通过则 layer-major 被证明；不通过立刻停，不产出可能看起来合理的错结果。
    split_check = verify_split_by_hook(
        predictor, pred["image_resized_pt"], pred["disparity_factor_tensor"], n_layers
    )
    lm = split_check["layers"]["layer0"]
    print(
        f"🧪 切分验证: 未展平 {split_check['unflat_shape']}, "
        f"layer0 逐元素最大差 {lm['max_abs_diff']:.3e} → {split_check['verdict']}"
    )
    if split_check["verdict"] != "layer-major":
        raise RuntimeError(
            "hook 判据否定 layer-major 切法，拒绝产出可能看起来合理的错误结果："
            f"{split_check}"
        )

    variants = {"full": g}
    for l, (start, end) in enumerate(slices):
        variants[f"layer{l}"] = subset_gaussians(g, start, end)

    gt = image
    if scale_ratio != 1.0:
        gt = np.asarray(Image.fromarray(gt).resize((width, height), Image.LANCZOS))
    gt_f = gt.astype(np.float32) / 255.0

    results = {}
    renders = {}
    for name, gv in variants.items():
        t = time.time()
        color, alpha, depth = render_source(renderer, gv, intrinsics, height, width)
        renders[name] = {"color": color, "alpha": alpha, "depth": depth}
        results[name] = {
            "n_gaussians": int(gv.mean_vectors.shape[1]),
            "render_s": round(time.time() - t, 2),
            "psnr_full": psnr(color, gt_f),
            "alpha_mean": float(alpha.mean()),
            "hole_ratio_lt_005": float((alpha < 0.05).mean()),
            "fg_ratio_gt_05": float((alpha > 0.5).mean()),
        }

    # 有人在的分区上比 PSNR（用完整两层的 alpha 定 mask，避免「少一层把背景也算进去」
    mask = renders["full"]["alpha"] > 0.5
    for name in results:
        results[name]["psnr_fg"] = psnr(renders[name]["color"], gt_f, mask)
        results[name]["psnr_bg"] = psnr(renders[name]["color"], gt_f, ~mask)

    # 层间深度差：第二层相对第一层远多少（metric，米）
    z0 = renders["layer0"]["depth"]
    z1 = renders["layer1"]["depth"]
    valid = mask & np.isfinite(z0) & np.isfinite(z1) & (z0 > 0) & (z1 > 0)
    gap = np.where(valid, z1 - z0, np.nan)
    rel_gap = np.where(valid, (z1 - z0) / np.maximum(z0, 1e-6), np.nan)
    gap_vals = gap[valid]
    rel_vals = rel_gap[valid]
    layer_gap = {
        "n_valid_px": int(valid.sum()),
        "z0_median_m": float(np.nanmedian(z0[valid])) if valid.any() else float("nan"),
        "z1_median_m": float(np.nanmedian(z1[valid])) if valid.any() else float("nan"),
        "gap_median_m": float(np.nanmedian(gap_vals)) if valid.any() else float("nan"),
        "gap_p90_m": float(np.nanpercentile(gap_vals, 90)) if valid.any() else float("nan"),
        "rel_gap_median": float(np.nanmedian(rel_vals)) if valid.any() else float("nan"),
        "frac_gap_lt_1mm": float((np.abs(gap_vals) < 1e-3).mean()) if valid.any() else float("nan"),
        "frac_gap_gt_5pct": float((rel_vals > 0.05).mean()) if valid.any() else float("nan"),
        "frac_gap_negative": float((gap_vals < 0).mean()) if valid.any() else float("nan"),
    }

    # ---- 层属性原始分布（不经过渲染，最干净） ----
    # 这一段是判「两层是混合还是遮挡」的关键：若两层的 opacity 都显著 <1，说明每个
    # 像素是靠两层叠加合成颜色的（混合）；若第一层 opacity≈1，才是「前层挡住后层」。
    raw_layers = {}
    for l, (start, end) in enumerate(slices):
        op = g.opacities[0, start:end]
        z = g.mean_vectors[0, start:end, 2]
        raw_layers[f"layer{l}"] = {
            "opacity_median": float(op.median()),
            "opacity_p10": float(op.quantile(0.10)),
            "opacity_p90": float(op.quantile(0.90)),
            "opacity_frac_gt_0.5": float((op > 0.5).float().mean()),
            "opacity_frac_gt_0.9": float((op > 0.9).float().mean()),
            "z_median_m": float(z.median()),
        }
    # 逐像素比较两层的原始 z（同一索引对应的就是同一像素格，layer-major 已由 hook 证明）
    z0_raw = g.mean_vectors[0, slices[0][0]:slices[0][1], 2]
    z1_raw = g.mean_vectors[0, slices[1][0]:slices[1][1], 2]
    dz_raw = (z1_raw - z0_raw).cpu().numpy()
    rel_raw = dz_raw / np.maximum(np.abs(z0_raw.cpu().numpy()), 1e-6)
    raw_layer_rel = {
        "dz_median_m": float(np.median(dz_raw)),
        "dz_p90_abs_m": float(np.percentile(np.abs(dz_raw), 90)),
        "dz_frac_abs_lt_1mm": float((np.abs(dz_raw) < 1e-3).mean()),
        "dz_frac_abs_lt_5mm": float((np.abs(dz_raw) < 5e-3).mean()),
        "dz_frac_negative": float((dz_raw < 0).mean()),
        "rel_dz_median": float(np.median(rel_raw)),
        "rel_dz_frac_abs_gt_5pct": float((np.abs(rel_raw) > 0.05).mean()),
    }

    # ---- 落盘 ----
    save_rgb(gt_f, args.out / "00_gt.png")
    for name, r in renders.items():
        save_rgb(r["color"], args.out / f"01_render_{name}.png")
        gray_png(r["alpha"], args.out / f"02_alpha_{name}.png", vmax=1.0)
        gray_png(np.nan_to_num(r["depth"]), args.out / f"03_depth_{name}.png", vmax=float(np.nanpercentile(z0[mask], 95)) if mask.any() else 5.0)
    gap_vis = np.nan_to_num(np.abs(gap), nan=0.0)
    gray_png(gap_vis, args.out / "04_layer_gap_abs.png",
             vmax=float(np.nanpercentile(np.abs(gap_vals), 95)) if valid.any() else 1.0)

    # 差图：完整两层 vs 只留第一层，看清第二层到底改了哪些像素
    d_full_l0 = np.abs(renders["full"]["color"] - renders["layer0"]["color"]).mean(axis=2)
    gray_png(d_full_l0, args.out / "05_diff_full_vs_layer0.png",
             vmax=max(float(np.percentile(d_full_l0, 99)), 1e-3))

    summary = {
        "input": str(args.input),
        "input_sha256": sha256(args.input),
        "focal_audit": focal_audit,
        "f_px": f_px,
        "disparity_factor": pred["disparity_factor"],
        "image_hw": list(pred["size"]),
        "render_hw": [height, width],
        "render_scale": args.render_scale,
        "load_s": round(load_s, 1),
        "forward_s": round(pred["fwd_s"], 1),
        "order_check": order_info,
        "split_verification": split_check,
        "variants": results,
        "layer_gap": layer_gap,
        "raw_layer_stats": raw_layers,
        "raw_layer_relation": raw_layer_rel,
        "color_space_note": "渲染 color_space=linearRGB（内部空间），与 GT sRGB 直接比会有系统色偏；"
                            "四组同口径比较仍有效，但绝对值不代表真实 PSNR",
    }
    with open(args.out / "layer_ablation.json", "w", encoding="utf-8", newline="\n") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    # ---- 控制台摘要 ----
    print()
    print("=" * 78)
    print(f"{'变体':<10}{'高斯数':>12}{'PSNR全图':>10}{'PSNR前景':>10}{'alpha均值':>11}{'空洞率':>9}")
    print("-" * 78)
    for name in ("full", "layer0", "layer1"):
        r = results[name]
        print(
            f"{name:<10}{r['n_gaussians']:>12,}{r['psnr_full']:>10.2f}"
            f"{r['psnr_fg']:>10.2f}{r['alpha_mean']:>11.3f}{r['hole_ratio_lt_005']:>9.3f}"
        )
    print("=" * 78)
    print(f"两层深度差（前景像素内）：中位 {layer_gap['gap_median_m']*1000:.2f} mm, "
          f"P90 {layer_gap['gap_p90_m']*1000:.2f} mm")
    print(f"  相对差 >5% 的像素占比 {layer_gap['frac_gap_gt_5pct']:.3f}, "
          f"层间差 <1mm 占比 {layer_gap['frac_gap_lt_1mm']:.3f}, "
          f"负值（第二层更近）占比 {layer_gap['frac_gap_negative']:.3f}")
    print()
    print("原始层属性（未渲染）：")
    for name, st in raw_layers.items():
        print(f"  {name}: opacity 中位 {st['opacity_median']:.3f} "
              f"[{st['opacity_p10']:.3f}, {st['opacity_p90']:.3f}], "
              f">0.5 占 {st['opacity_frac_gt_0.5']:.3f}, z 中位 {st['z_median_m']:.3f} m")
    print(f"  逐像素 dz: 中位 {raw_layer_rel['dz_median_m']*1000:.2f} mm, "
          f"P90|dz| {raw_layer_rel['dz_p90_abs_m']*1000:.2f} mm, "
          f"|dz|<1mm 占 {raw_layer_rel['dz_frac_abs_lt_1mm']:.3f}, "
          f"dz<0 占 {raw_layer_rel['dz_frac_negative']:.3f}")
    print(f"📁 {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
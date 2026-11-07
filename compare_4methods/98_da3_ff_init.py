#!/usr/bin/env python3
"""P1-0：一次 DA3 前向，同时产出「3DGS 初始化」所需的全部同源产物。

**为什么必须一次推理**：DA3 的世界坐标系由参考视角定义（参考相机基本落在原点），
换一次运行、换一个 `ref_view_strategy`，世界系就变了。P1 要把前馈高斯当作 3DGS
的初值，初值与训练相机必须处在同一世界系，否则几何整体错位 —— P0 已经踩过一次
（拿 504 的 npz 相机去渲染 1008 的 PLY，画面旋转错位、大片纯黑）。

一次前向出三样，全部同源：
  <OUT>/gs_ply/0000.ply          官方 GSDPT 高斯头（DC-only，已折算到**米制**）
  <OUT>/{cameras,images,points3D}.bin   COLMAP 稀疏场景（供 3DGS 读相机）
  <OUT>/cameras.npz              extrinsics/intrinsics/depth/conf + scale_factor

**坐标系口径（E-01，2026-11-06）**：DA3 的 `_apply_depth_alignment` 只把 depth 与
extrinsics 乘度量系数，`gaussians` 留在对齐前的坐标系。本脚本在导出前调用
`da3_frame.gaussians_to_metric()` 把 means/scales 乘回米制，否则 PLY 与 cameras.npz
差 1.8 倍尺度 → 参考视角因径向缩放"看起来正常"，其它视角错位 0.45×距离。
详见 `compare_4methods/ERRATA.md`。

用法（WSL，da3_human env）：
  DA3_MODEL_DIR=/mnt/d/wheel/da3_human_ms \
  IMAGES_DIR=~/output/da3_results/_captures/佳佳抱枕/images \
  OUT_DIR=~/output/da3_gs_results/佳佳抱枕_p1 \
  PROCESS_RES=504 REF_STRATEGY=first \
  ~/miniconda3/envs/da3_human/bin/python compare_4methods/98_da3_ff_init.py

Env:
  GS_SCALE_MODE  metric(默认，折到米制) | raw(复现 E-01 之前的旧结果)
  SAVE_FULL_SH   1 = 同一次前向额外导出 gs_ply_full/0000.ply（保留 SH rest，DA3 为 sh_degree=2）
"""
import json
import os
import time
from pathlib import Path

import numpy as np
import torch


def log(msg):
    print(msg, flush=True)


def camera_centers(w2c):
    """w2c (N,4,4) -> 相机中心 (N,3)：c = -R^T t（与 3DGS 的 camera_center 同约定）"""
    r = w2c[:, :3, :3]
    t = w2c[:, :3, 3]
    return np.einsum("nij,nj->ni", -r.transpose(0, 2, 1), t)


def main():
    model_dir = os.environ.get("DA3_MODEL_DIR", "/mnt/d/wheel/da3_human_ms")
    images_dir = os.path.expanduser(os.environ["IMAGES_DIR"])
    out_dir = Path(os.path.expanduser(os.environ["OUT_DIR"]))
    process_res = int(os.environ.get("PROCESS_RES", "504"))
    max_views = int(os.environ.get("MAX_VIEWS", "0"))
    # P0 的结论：跨运行对比必须钉死参考视角。P1 要与 P0 的 ref0 基线可比 → 默认 first。
    ref_strategy = os.environ.get("REF_STRATEGY", "first")
    out_dir.mkdir(parents=True, exist_ok=True)

    exts = ("*.png", "*.jpg", "*.jpeg")
    files = []
    for e in exts:
        files.extend(sorted(Path(images_dir).glob(e)))
    files = sorted(files)
    if max_views and len(files) > max_views:
        idx = np.linspace(0, len(files) - 1, max_views).round().astype(int)
        files = [files[i] for i in idx]
    if not files:
        raise SystemExit(f"❌ 没有图像: {images_dir}")
    log(f"🖼️ 输入 {len(files)} 视角 @ process_res={process_res}  ref_view_strategy={ref_strategy}")

    from depth_anything_3.api import DepthAnything3
    from depth_anything_3.utils import export as da3_export

    # api.inference 见到 "gs" 会强制追加 gs_video 渲染（需 gsplat）。本机没装，
    # 该步会抛 NameError 但此时 gs_ply 已写出 —— 直接短路，让导出干净结束。
    try:
        import importlib.util
        if importlib.util.find_spec("gsplat") is None:
            da3_export.export_to_gs_video = lambda *a, **k: None  # noqa: ARG005
            log("ℹ️ 未装 gsplat：跳过 gs_video 渲染")
    except Exception:
        pass

    t0 = time.time()
    model = DepthAnything3.from_pretrained(model_dir).to("cuda")
    t_load = time.time() - t0
    log(f"⏱️ 加载 {t_load:.1f}s")

    t0 = time.time()
    # ⚠️ export_dir=None：**先不导出**，拿到干净的 prediction 先把 npz 存下来。
    # 原因（2026-11-06 踩过）：export_to_colmap 会把 prediction.intrinsics 按**原图分辨率**
    # 放大（fx 372 → 1417，即 1439/378 倍）—— COLMAP 场景配原图是对的，但渲染用的是
    # 处理分辨率（378x504），拿放大后的内参去渲染 → FOV 收窄、画面只剩局部，看着像"几何错位"。
    pred = model.inference(
        [str(p) for p in files],
        process_res=process_res,
        infer_gs=True,
        ref_view_strategy=ref_strategy,
        export_dir=None,
        export_format="",
    )
    t_fwd = time.time() - t0
    log(f"⏱️ 前向 {t_fwd:.1f}s")

    g = pred.gaussians
    if g is None:
        raise SystemExit("❌ infer_gs=True 但 prediction.gaussians 为空")
    n_gauss = int(np.prod(np.asarray(g.means.shape)[:-1]))
    sh_shape = tuple(int(x) for x in np.asarray(g.harmonics.shape))
    log(f"📊 高斯 {n_gauss:,} ｜ harmonics {sh_shape}")

    # ── 坐标系口径（E-01）：gaussians 在度量对齐**之前**，depth/extrinsics 之后 ──
    # 先算系数（要写进 npz），再把高斯折到米制（必须在 export 之前）。
    from da3_frame import gaussians_to_metric, gs_scale_factor

    sf_raw, is_metric = gs_scale_factor(pred)
    gs_mode = os.environ.get("GS_SCALE_MODE", "metric")

    # ── 先落相机/深度（此时 intrinsics 还是处理分辨率口径） ──────────────────
    cam = out_dir / "cameras.npz"
    np.savez_compressed(
        cam,
        extrinsics=np.asarray(pred.extrinsics),
        intrinsics=np.asarray(pred.intrinsics),
        depth=np.asarray(pred.depth),
        conf=np.asarray(pred.conf),
        scale_factor=np.float32(sf_raw if sf_raw is not None else 0.0),
        is_metric=np.int32(is_metric),
        gs_scale_applied=np.int32(1 if gs_mode == "metric" and is_metric and sf_raw else 0),
    )

    sf = gaussians_to_metric(pred, mode=gs_mode)
    log(f"📷 相机/深度: {cam}  K[0] fx={float(np.asarray(pred.intrinsics)[0,0,0]):.2f}")
    # ── 再导出（这两步可能改写 prediction 内的量，必须在存 npz 之后） ─────────
    t0 = time.time()
    from depth_anything_3.utils.export import export_to_colmap, export_to_gs_ply
    export_to_gs_ply(pred, str(out_dir), 1)   # 第 3 位是 gs_views_interval，不是 kwargs

    # ── 可选：同一次前向再出一份「保留 SH rest」的 PLY（P0-2 复测用） ─────────
    # 与官方 export_to_gs_ply **逐参数对齐**，唯一差别是 save_sh_dc_only=False，
    # 保证两次导出的差别只有 SH rest（prune 规则 / opacity 反 sigmoid / 间隔全一致）。
    # 为什么必须同一次前向（清单 A5）：DA3 的世界系由参考视角定义，且 prune 依赖
    # ctx_depth，跨运行"应该一样"是赌，不值得赌。
    if os.environ.get("SAVE_FULL_SH", "0") == "1":
        from depth_anything_3.utils.gsply_helpers import save_gaussian_ply
        p_full = out_dir / "gs_ply_full" / "0000.ply"
        p_full.parent.mkdir(parents=True, exist_ok=True)
        t1 = time.time()
        save_gaussian_ply(
            gaussians=pred.gaussians,
            save_path=str(p_full),
            ctx_depth=torch.from_numpy(pred.depth).unsqueeze(-1).to(pred.gaussians.means),
            shift_and_scale=False,
            save_sh_dc_only=False,
            gs_views_interval=1,
            inv_opacity=True,
            prune_by_depth_percent=0.9,
            prune_border_gs=True,
            match_3dgs_mcmc_dev=False,
        )
        log(f"💾 full-SH {p_full}  {time.time()-t1:.1f}s  {p_full.stat().st_size/1024/1024:.0f} MB")

    # ── EXPORT_OFFICIAL=1：把「导出降级」的 2x2 组合一次全导出来（I-01/I-02） ──
    # 历史 PLY 同时开了两个降级选项：裁远点(0.9) + 去边界(True) → 610 万降到 484 万；
    # 且 save_sh_dc_only=True 丢掉全部 SH rest。合计相对官方口径 −3.14 dB。
    # 这里用同一个前向导出全部组合，保证「只差导出参数」这一单变量成立。
    #   i01_fullpts_dc : 不裁剪 + DC   （I-01 单独）
    #   i02_pruned_sh  : 裁剪   + 全SH （I-02 单独）
    #   i12_fullpts_sh : 不裁剪 + 全SH （官方等价，I-01+I-02）
    if os.environ.get("EXPORT_OFFICIAL", "0") == "1":
        from depth_anything_3.utils.gsply_helpers import save_gaussian_ply
        combos = (
            ("i01_fullpts_dc", 1.0, False, True),
            ("i02_pruned_sh", 0.9, True, False),
            ("i12_fullpts_sh", 1.0, False, False),
        )
        for tag, pct, border, dc_only in combos:
            p = out_dir / "official_export" / tag / "0000.ply"
            p.parent.mkdir(parents=True, exist_ok=True)
            t1 = time.time()
            save_gaussian_ply(
                gaussians=pred.gaussians,
                save_path=str(p),
                ctx_depth=torch.from_numpy(pred.depth).unsqueeze(-1).to(pred.gaussians.means),
                shift_and_scale=False,
                save_sh_dc_only=dc_only,
                gs_views_interval=1,
                inv_opacity=True,
                prune_by_depth_percent=pct,
                prune_border_gs=border,
                match_3dgs_mcmc_dev=False,
            )
            log(f"💾 {tag}: pct={pct} border={border} dc_only={dc_only}  "
                f"{time.time()-t1:.1f}s  {p.stat().st_size/1024/1024:.0f} MB  → {p}")

    export_to_colmap(pred, str(out_dir), image_paths=[str(p) for p in files])
    t_exp = time.time() - t0
    log(f"⏱️ 导出（gs_ply + colmap）{t_exp:.1f}s")

    # 与 3DGS getNerfppNorm 同口径的 cameras_extent：radius = 1.1 * max(相机中心到其均值距离)。
    # 短精调必须用它算 spatial_lr_scale，否则位置学习率尺度不对（3DGS 内部 lr × extent）。
    w2c = np.asarray(np.load(cam)["extrinsics"])
    C = camera_centers(w2c)
    extent = float(1.1 * np.linalg.norm(C - C.mean(axis=0), axis=1).max())
    log(f"📷 相机 {len(C)} 个 ｜ 轨迹半径 {np.linalg.norm(C - C.mean(0), axis=1).mean():.3f} m ｜ extent {extent:.3f} m")

    ply = out_dir / "gs_ply" / "0000.ply"
    ply_full = out_dir / "gs_ply_full" / "0000.ply"
    summary = {
        "images_dir": str(images_dir),
        "n_views": len(files),
        "process_res": process_res,
        "ref_view_strategy": ref_strategy,
        "model": model_dir,
        "is_metric": is_metric,
        "da3_scale_factor": sf_raw,
        "gs_scale_mode": gs_mode,
        "gs_scale_factor_applied": sf,
        "gpu": torch.cuda.get_device_name(0),
        "load_seconds": round(t_load, 1),
        "forward_seconds": round(t_fwd, 1),
        "export_seconds": round(t_exp, 1),
        "n_gaussians_raw": n_gauss,
        "harmonics_shape": list(sh_shape),
        "gs_ply": str(ply),
        "gs_ply_bytes": int(ply.stat().st_size) if ply.exists() else None,
        "gs_ply_vertices": None,
        "gs_ply_full": str(ply_full) if ply_full.exists() else None,
        "gs_ply_full_bytes": int(ply_full.stat().st_size) if ply_full.exists() else None,
        "cameras_npz": str(cam),
        "cameras_extent_m": round(extent, 4),
        "camera_radius_mean_m": round(float(np.linalg.norm(C - C.mean(0), axis=1).mean()), 4),
        "depth_median_m": round(float(np.median(pred.depth)), 4),
    }

    def _ply_vertices(p):
        """从 PLY 头读 vertex 数（不整读文件）。"""
        if not p.exists():
            return None
        with open(p, "rb") as f:
            head = f.read(4096).decode("ascii", "replace")
        import re
        m = re.search(r"element vertex (\d+)", head)
        return int(m.group(1)) if m else None

    n_dc = _ply_vertices(ply)
    if n_dc:
        summary["gs_ply_vertices"] = n_dc
        log(f"💾 PLY 实存 {n_dc:,} 点（导出时剪枝后）")
    n_full = _ply_vertices(ply_full)
    if n_full:
        summary["gs_ply_full_vertices"] = n_full
        log(f"💾 full-SH PLY 实存 {n_full:,} 点")
        # A/B 公平性闸门（清单 A2 的思路）：两组必须剪出同一批点，
        # 否则"SH rest 的收益"里混进了"点集不同"这个变量。
        if n_dc and n_full != n_dc:
            raise SystemExit(
                f"❌ dc 与 full-SH 的顶点数不一致（{n_dc:,} vs {n_full:,}）："
                "prune 规则应当产生同一批点，两份 PLY 不可直接 A/B。")

    sp = out_dir / "p1_init_summary.json"
    with open(sp, "w", encoding="utf-8", newline="\n") as f:
        json.dump(summary, f, indent=4, ensure_ascii=False)
    log(f"✅ {sp}")


if __name__ == "__main__":
    main()

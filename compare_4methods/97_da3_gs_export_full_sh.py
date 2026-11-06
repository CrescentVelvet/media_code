#!/usr/bin/env python3
"""P0-2：DA3 原生高斯头 —— 导出一份「保留 SH rest」的 PLY，与官方 DC-only 做同一次前向的 A/B。

背景：官方 `export_to_gs_ply` 里 `save_sh_dc_only=True` 是写死的（gs.py:52），
导出的 PLY 只带 SH DC（零阶，无视角相关外观）。本次不改 DA3 仓、不改模型权重，
只复用官方 `save_gaussian_ply`，把该 flag 翻成 False，其余参数逐项对齐官方，
保证两次导出的**唯一差别就是 SH rest**（抑制/prune/opacity 反 sigmoid 全部一致）。

输出目录结构：
  <OUT>/gs_ply_dc/0000.ply     与官方一致的 DC-only（对照组，应可复现基线体积）
  <OUT>/gs_ply_full/0000.ply   保留 SH rest（实验组）
  <OUT>/p0_export_summary.json 耗时 / 高斯数 / 体积 / harmonics 形状

用法（WSL，da3_human env）：
  DA3_MODEL_DIR=/mnt/d/wheel/da3_human_ms \
  IMAGES_DIR=~/output/da3_results/_captures/佳佳抱枕/images \
  OUT_DIR=~/output/da3_gs_results/佳佳抱枕_shfull \
  PROCESS_RES=504 \
  ~/miniconda3/envs/da3_human/bin/python compare_4methods/97_da3_gs_export_full_sh.py
"""
import json
import os
import time
from pathlib import Path

import numpy as np
import torch


def log(msg):
    print(msg, flush=True)


def main():
    model_dir = os.environ.get("DA3_MODEL_DIR", "/mnt/d/wheel/da3_human_ms")
    images_dir = os.path.expanduser(os.environ["IMAGES_DIR"])
    out_dir = Path(os.path.expanduser(os.environ["OUT_DIR"]))
    process_res = int(os.environ.get("PROCESS_RES", "504"))
    max_views = int(os.environ.get("MAX_VIEWS", "0"))
    # 参考视角策略：DA3 的世界坐标系由参考视角定义（ref 相机放在原点），
    # 换分辨率时 saddle_balanced 可能选出不同的 ref → 两次运行的世界系/质量分布都不可比。
    # 做分辨率 A/B 时必须钉死同一个策略（建议 first = 视角 0）。
    ref_strategy = os.environ.get("REF_STRATEGY", "saddle_balanced")
    out_dir.mkdir(parents=True, exist_ok=True)

    exts = ("*.png", "*.jpg", "*.jpeg")
    files = []
    for e in exts:
        files.extend(sorted(Path(images_dir).glob(e)))
    files = sorted(files)
    if max_views and len(files) > max_views:
        idx = np.linspace(0, len(files) - 1, max_views).round().astype(int)
        files = [files[i] for i in idx]
    log(f"🖼️ 输入 {len(files)} 视角 @ process_res={process_res}")

    from depth_anything_3.api import DepthAnything3
    from depth_anything_3.utils.gsply_helpers import save_gaussian_ply

    t0 = time.time()
    model = DepthAnything3.from_pretrained(model_dir).to("cuda")
    t_load = time.time() - t0

    t0 = time.time()
    # export_dir=None：完全跳过官方导出（空 export_format 会抛 ValueError），
    # 高斯留在 prediction.gaussians 里，由我们自己写盘。
    pred = model.inference(
        [str(p) for p in files],
        process_res=process_res,
        infer_gs=True,
        ref_view_strategy=ref_strategy,
        export_dir=None,
        export_format="",
    )
    t_fwd = time.time() - t0
    log(f"⏱️ 加载 {t_load:.1f}s ｜ 前向 {t_fwd:.1f}s")

    g = pred.gaussians
    if g is None:
        raise SystemExit("❌ infer_gs=True 但 prediction.gaussians 为空")
    n_gauss = int(np.prod(np.asarray(g.means.shape)[:-1]))
    sh_shape = tuple(int(x) for x in np.asarray(g.harmonics.shape))
    d_sh = sh_shape[-1]
    sh_degree = int(round(d_sh ** 0.5)) - 1
    log(f"📊 高斯 {n_gauss} ｜ harmonics {sh_shape} → d_sh={d_sh}, sh_degree={sh_degree}")

    # 坐标系口径（E-01）：官方导出把 raw 尺度的高斯直接写盘，与 cameras.npz（米制）差
    # 一个 scale_factor。这里先折到米制，否则只有参考视角因径向缩放"看起来正常"。
    from da3_frame import gaussians_to_metric, gs_scale_factor

    sf_raw, is_metric = gs_scale_factor(pred)
    gs_mode = os.environ.get("GS_SCALE_MODE", "metric")
    sf = gaussians_to_metric(pred, mode=gs_mode)

    ctx_depth = torch.from_numpy(pred.depth).unsqueeze(-1).to(g.means)   # (v,h,w,1)

    common = dict(
        gaussians=g,
        ctx_depth=ctx_depth,
        shift_and_scale=False,
        gs_views_interval=1,
        inv_opacity=True,
        prune_by_depth_percent=0.9,
        prune_border_gs=True,
        match_3dgs_mcmc_dev=False,
    )

    results = {}
    save_full = os.environ.get("SAVE_FULL", "1") == "1"
    for tag, dc_only in (("dc", True), ("full", False)):
        if not dc_only and not save_full:
            log("⏭️ SAVE_FULL=0：跳过 full-SH 导出（高斯数上千万时 plyfile 的 tuple 转换会吃 ~10GB 内存）")
            continue
        p = out_dir / f"gs_ply_{tag}" / "0000.ply"
        p.parent.mkdir(parents=True, exist_ok=True)
        t0 = time.time()
        if p.exists() and os.environ.get("FORCE", "0") != "1":
            log(f"⏭️ 已存在，跳过: {p}")
        else:
            save_gaussian_ply(save_path=str(p), save_sh_dc_only=dc_only, **common)
        dt = time.time() - t0
        size = p.stat().st_size
        results[tag] = {"path": str(p), "save_seconds": round(dt, 1),
                        "bytes": size, "save_sh_dc_only": dc_only}
        log(f"💾 {tag}: {dt:.1f}s  {size/1024/1024:.1f} MB  {p}")

    summary = {
        "images_dir": str(images_dir),
        "n_views": len(files),
        "process_res": process_res,
        "model": model_dir,
        "gpu": torch.cuda.get_device_name(0),
        "load_seconds": round(t_load, 1),
        "forward_seconds": round(t_fwd, 1),
        "n_gaussians": n_gauss,
        "harmonics_shape": list(sh_shape),
        "sh_degree": sh_degree,
        "ref_view_strategy": ref_strategy,
        "is_metric": is_metric,
        "da3_scale_factor": sf_raw,
        "gs_scale_mode": gs_mode,
        "gs_scale_factor_applied": sf,
        "depth_median_m": float(np.median(pred.depth)),
        "exports": results,
    }

    # 相机/深度单独落一份：DA3 的**世界坐标系由参考视角定义**，换分辨率时
    # saddle_balanced 可能选出不同的参考视角 → 世界系跟着变。渲染必须用同一次前向的相机，
    # 不能拿另一次运行的 npz 顶替（2026-11-06 踩过：1008 的 PLY 配 504 的相机 → 画面旋转错位）。
    cam = out_dir / "cameras.npz"
    np.savez_compressed(cam,
                        extrinsics=np.asarray(pred.extrinsics),
                        intrinsics=np.asarray(pred.intrinsics),
                        depth=np.asarray(pred.depth),
                        conf=np.asarray(pred.conf),
                        scale_factor=np.float32(sf_raw if sf_raw is not None else 0.0),
                        is_metric=np.int32(is_metric),
                        gs_scale_applied=np.int32(1 if gs_mode == "metric" and is_metric and sf_raw else 0))
    summary["cameras_npz"] = str(cam)
    log(f"📷 相机/深度: {cam}")
    sp = out_dir / "p0_export_summary.json"
    with open(sp, "w", encoding="utf-8", newline="\n") as f:
        json.dump(summary, f, indent=4, ensure_ascii=False)
    log(f"✅ {sp}")


if __name__ == "__main__":
    main()

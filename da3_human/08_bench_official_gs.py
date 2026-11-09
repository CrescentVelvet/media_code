#!/usr/bin/env python3
"""08_bench_official_gs.py — 官方 DA3 GSDPT 高斯头的「纯前向」耗时与导出开销拆解。

目的：回答两个问题
  1) 「秒级」在什么条件下成立？（加载 / 前向 / 导出 三段分别多少）
  2) 官方默认导出的两处降级（裁剪 + DC-only）各值多少 dB？

不改 DA3 仓、不改权重，只调 `save_gaussian_ply` 的官方入参：
    prune_border_gs        True/False   边缘 8/256 像素裁掉
    prune_by_depth_percent 0.9/1.0      按逐视角深度分位剪掉最远 10%
    save_sh_dc_only        True/False   SH 只留 DC
并把高斯折到米制（E-01，gaussians 不随 depth/extrinsics 缩放）。

Env:
  DA3_MODEL_DIR  权重目录（默认 /mnt/d/wheel/da3_human_ms）
  IMAGES_DIR     图像目录
  OUT_DIR        输出目录
  PROCESS_RES    504（默认）| 672 | 1008（须为 14 的倍数，且 W=0.75H 也须整除）
  N_VIEWS        0=全部；>0 均匀抽帧（测「秒级」边界用）
  REF_STRATEGY   saddle_balanced（默认）| first
  EXPORT         none = 只计时不落盘（最快）；all = 四档全导
  GS_SCALE_MODE  metric（默认）| raw
"""
import json
import os
import re
import sys
import time
from pathlib import Path

import numpy as np
import torch


def log(m):
    print(m, flush=True)


_OFFICIAL_EXPORT = {}


def _fast_export_ply(means, scales, rotations, harmonics, opacities, path,
                     shift_and_scale=False, save_sh_dc_only=True,
                     match_3dgs_mcmc_dev=False):
    """官方 export_ply 的等价实现，只换掉写盘路径。

    官方瓶颈是 `elements[:] = list(map(tuple, attributes))`——610 万行 × 62 列
    要构造 610 万个 Python tuple，实测 953 MB 写 28 s。这里直接拼 float32 矩阵
    一次 tofile，字段顺序与官方完全一致，下游 gsplat / 3DGS 读取无差异。
    """
    if shift_and_scale:
        raise NotImplementedError("fast_export 不支持 shift_and_scale")
    if match_3dgs_mcmc_dev:
        raise NotImplementedError("fast_export 不支持 match_3dgs_mcmc_dev")

    def np32(t):
        return t.detach().cpu().numpy().astype(np.float32)

    n = means.shape[0]
    k = 0 if save_sh_dc_only else harmonics.shape[-1] - 1
    names = ["x", "y", "z", "nx", "ny", "nz", "f_dc_0", "f_dc_1", "f_dc_2"]
    names += [f"f_rest_{i}" for i in range(3 * k)]
    names += ["opacity", "scale_0", "scale_1", "scale_2", "rot_0", "rot_1", "rot_2", "rot_3"]

    cols = [np32(means), np.zeros((n, 3), np.float32), np32(harmonics[..., 0])]
    if not save_sh_dc_only:
        cols.append(np32(harmonics[..., 1:].reshape(n, -1)))
    cols += [np32(opacities[..., None]), np32(scales.log()), np32(rotations)]
    arr = np.ascontiguousarray(np.concatenate(cols, axis=1), dtype="<f4")

    head = ["ply", "format binary_little_endian 1.0", f"element vertex {n}"]
    head += [f"property float {nm}" for nm in names]
    head += ["end_header"]
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        f.write(("\n".join(head) + "\n").encode("ascii"))
        arr.tofile(f)


def main():
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "compare_4methods"))
    model_dir = os.environ.get("DA3_MODEL_DIR", "/mnt/d/wheel/da3_human_ms")
    images_dir = os.path.expanduser(os.environ["IMAGES_DIR"])
    out_dir = Path(os.path.expanduser(os.environ["OUT_DIR"]))
    process_res = int(os.environ.get("PROCESS_RES", "504"))
    n_views = int(os.environ.get("N_VIEWS", "0"))
    ref_strategy = os.environ.get("REF_STRATEGY", "saddle_balanced")
    export_mode = os.environ.get("EXPORT", "none")
    gs_mode = os.environ.get("GS_SCALE_MODE", "metric")
    out_dir.mkdir(parents=True, exist_ok=True)

    files = []
    for e in ("*.jpg", "*.jpeg", "*.png"):
        files.extend(sorted(Path(images_dir).glob(e)))
    files = sorted(files)
    total_views = len(files)
    if n_views and len(files) > n_views:
        idx = np.linspace(0, len(files) - 1, n_views).round().astype(int)
        files = [files[i] for i in idx]
    log(f"🖼️ 输入 {len(files)}/{total_views} 视角 @ process_res={process_res} "
        f"| ref={ref_strategy} | export={export_mode}")

    from depth_anything_3.api import DepthAnything3

    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    model = DepthAnything3.from_pretrained(model_dir).to("cuda")
    torch.cuda.synchronize()
    t_load = time.time() - t0
    log(f"⏱️ 模型加载 {t_load:.1f}s")

    t0 = time.time()
    pred = model.inference(
        [str(p) for p in files],
        process_res=process_res,
        infer_gs=True,
        ref_view_strategy=ref_strategy,
        export_dir=None,
        export_format="",
    )
    torch.cuda.synchronize()
    t_fwd = time.time() - t0
    peak = torch.cuda.max_memory_allocated() / 1024 ** 3
    log(f"⏱️ 纯前向（含 GSDPT 头）{t_fwd:.1f}s | 显存峰值 {peak:.2f} GB")

    g = pred.gaussians
    if g is None:
        raise SystemExit("❌ infer_gs=True 但 prediction.gaussians 为空")
    n_gauss = int(np.prod(np.asarray(g.means.shape)[:-1]))
    sh_shape = tuple(int(x) for x in np.asarray(g.harmonics.shape))
    sh_degree = int(round(sh_shape[-1] ** 0.5)) - 1
    log(f"📊 高斯 {n_gauss:,} | harmonics {sh_shape} → degree {sh_degree}")

    from da3_frame import gaussians_to_metric, gs_scale_factor

    sf_raw, is_metric = gs_scale_factor(pred)
    sf = gaussians_to_metric(pred, mode=gs_mode)
    log(f"📐 scale_factor={sf_raw} is_metric={is_metric} → 应用 {sf}")
    if sf == 1.0 and sf_raw:
        log("⚠️ 尺度未折算，PLY 与 cameras.npz 不同尺度（E-01）")

    cam = out_dir / "cameras.npz"
    np.savez_compressed(cam,
                        extrinsics=np.asarray(pred.extrinsics),
                        intrinsics=np.asarray(pred.intrinsics),
                        depth=np.asarray(pred.depth),
                        conf=np.asarray(pred.conf),
                        scale_factor=np.float32(sf_raw if sf_raw is not None else 0.0),
                        is_metric=np.int32(is_metric),
                        # 抽帧时相机不再与 sorted(images) 一一对应，必须把原始索引带出去，
                        # 否则下游拿 sorted[i] 当第 i 个视角 → GT 与渲染错位（实测掉到 10 dB）
                        view_indices=np.asarray(idx if (n_views and total_views > n_views)
                                               else np.arange(total_views), np.int64))
    log(f"📷 相机/深度 → {cam}")

    exports = {}
    if export_mode != "none":
        from depth_anything_3.utils import gsply_helpers as GH

        if os.environ.get("FAST_EXPORT", "1") == "1":
            _OFFICIAL_EXPORT["fn"] = GH.export_ply   # 留一份官方实现，供 VERIFY 比对
            GH.export_ply = _fast_export_ply
            log("⚡ 启用快速导出（numpy 单次 tofile，替换官方 list(map(tuple))）")
        save_gaussian_ply = GH.save_gaussian_ply

        ctx_depth = torch.from_numpy(pred.depth).unsqueeze(-1).to(g.means)
        combos = [
            ("official",  True,  0.9),   # 官方默认：裁剪 + DC-only
            ("full_sh",   True,  0.9),   # 只开全 SH
            ("noclip",    False, 1.0),   # 不裁剪 + DC-only
            ("max",       False, 1.0),   # 不裁剪 + 全 SH（质量上限）
        ]
        if export_mode != "all":
            combos = [c for c in combos if c[0] == export_mode]
            if not combos:
                raise SystemExit(f"❌ EXPORT={export_mode} 不在 {['all','none','official','full_sh','noclip','max']}")
        for tag, clip, depth_pct in combos:
            dc_only = tag in ("official", "noclip")
            p = out_dir / f"gs_{tag}" / "0000.ply"
            p.parent.mkdir(parents=True, exist_ok=True)
            if os.environ.get("VERIFY") == "1" and tag == combos[0][0]:
                # 同一次推理、同一份 gaussians，分别用官方/快速导出，做逐字段比对
                import importlib
                gh = importlib.import_module("depth_anything_3.utils.gsply_helpers")
                po = p.with_name("official.ply")
                gh.export_ply = _OFFICIAL_EXPORT["fn"]
                t0 = time.time()
                save_gaussian_ply(
                    gaussians=g, save_path=str(po), ctx_depth=ctx_depth,
                    shift_and_scale=False, save_sh_dc_only=dc_only, gs_views_interval=1,
                    inv_opacity=True, prune_by_depth_percent=depth_pct,
                    prune_border_gs=clip, match_3dgs_mcmc_dev=False)
                log(f"🔍 [verify] 官方导出 {time.time()-t0:.1f}s → {po}")
                gh.export_ply = _fast_export_ply
            t0 = time.time()
            save_gaussian_ply(
                gaussians=g, save_path=str(p), ctx_depth=ctx_depth,
                shift_and_scale=False, save_sh_dc_only=dc_only, gs_views_interval=1,
                inv_opacity=True, prune_by_depth_percent=depth_pct,
                prune_border_gs=clip, match_3dgs_mcmc_dev=False)
            dt = time.time() - t0
            size = p.stat().st_size
            with open(p, "rb") as fh:
                head = fh.read(1024).decode("ascii", "replace")
            n_pts = int(re.search(r"element vertex (\d+)", head).group(1))
            exports[tag] = {"path": str(p), "seconds": round(dt, 1),
                            "MB": round(size / 1024 ** 2, 1), "points": n_pts,
                            "sh_dc_only": dc_only, "clip": clip, "depth_pct": depth_pct}
            log(f"💾 {tag:9s} {dt:6.1f}s  {size/1024**2:8.1f} MB  {n_pts:,} 点  "
                f"(clip={clip}, depth={depth_pct}, dc_only={dc_only})")
    else:
        log("⏭️ EXPORT=none：跳过落盘（只测前向耗时）")

    summary = {
        "images_dir": str(images_dir), "n_views": len(files), "total_views": total_views,
        "process_res": process_res, "model": model_dir, "ref_view_strategy": ref_strategy,
        "gpu": torch.cuda.get_device_name(0),
        "load_seconds": round(t_load, 1), "forward_seconds": round(t_fwd, 1),
        "peak_vram_gb": round(peak, 2),
        "n_gaussians": n_gauss, "harmonics_shape": list(sh_shape), "sh_degree": sh_degree,
        "is_metric": is_metric, "scale_factor": sf_raw, "scale_applied": sf,
        "gs_scale_mode": gs_mode, "exports": exports,
    }
    sp = out_dir / f"bench_{process_res}_{len(files)}v.json"
    with open(sp, "w", encoding="utf-8", newline="\n") as f:
        json.dump(summary, f, indent=4, ensure_ascii=False)
    log(f"✅ {sp}")
    log(f"🎉 汇总：加载 {t_load:.1f}s + 前向 {t_fwd:.1f}s = {t_load + t_fwd:.1f}s（不含导出）")


if __name__ == "__main__":
    main()

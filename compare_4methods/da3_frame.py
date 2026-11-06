#!/usr/bin/env python3
"""DA3 的坐标系口径（E-01，见 compare_4methods/ERRATA.md）

## 问题

`da3.py:_apply_depth_alignment()`（约 :385-415）在度量对齐时：

    scale_factor = least_squares_scale_scalar(valid_metric_depth, valid_depth)
    output.depth      *= scale_factor
    output.extrinsics[:, :, :3, 3] *= scale_factor
    # gaussians.means / gaussians.scales 未动

于是 `prediction.gaussians` 留在**对齐前**的坐标系，而 `prediction.depth` /
`prediction.extrinsics` 已经是米制。两者相差一个 `scale_factor`（本例 1.81）。

任何把「官方导出的 gs_ply」与「cameras.npz」混用的下游（本仓的 94 / 97 / 98 / 98b），
都必须先乘这个因子。DA3 自己的 `export_to_gs_video` 走的是反方向：
`tgt_extrs[:, :, :3, 3] /= scale_factor`（utils/export/gs.py:91-93）。

## 用法

```python
from da3_frame import gaussians_to_metric
sf = gaussians_to_metric(pred, mode=os.environ.get("GS_SCALE_MODE", "metric"))
```

`scales` 是长度量，必须与 `means` 一起乘；`rotations` / `harmonics` / `opacities`
是无量纲的，不动。
"""
from __future__ import annotations


def _log(msg):
    print(msg, flush=True)


def gs_scale_factor(pred):
    """返回 (scale_factor 或 None, is_metric)。"""
    sf = getattr(pred, "scale_factor", None)
    is_metric = int(getattr(pred, "is_metric", 0) or 0)
    try:
        sf = float(sf) if sf is not None else None
    except (TypeError, ValueError):
        sf = None
    return sf, is_metric


def gaussians_to_metric(pred, mode: str = "metric", strict: bool = True) -> float:
    """把 prediction.gaussians 折算到米制（**就地修改**）。返回实际使用的系数。

    mode="metric"：乘 scale_factor，使 PLY 与 cameras.npz/depth 同尺度（推荐，默认）
    mode="raw"   ：不动，返回 1.0（仅用于复现 E-01 之前的旧结果）
    """
    sf, is_metric = gs_scale_factor(pred)

    if mode == "raw":
        _log("⚠️ GS_SCALE_MODE=raw：高斯保留 DA3 内部尺度，与 cameras.npz **不同尺度**"
             "（仅供复现 E-01 之前的旧结果）")
        return 1.0

    if not is_metric or sf is None:
        msg = ("❌ 模型没有度量对齐（is_metric=0 / scale_factor=None），高斯与相机本应同尺度；"
               "若确实想按 raw 导出请显式设 GS_SCALE_MODE=raw")
        if strict:
            raise SystemExit(msg)
        _log("⚠️ " + msg)
        return 1.0

    g = pred.gaussians
    if g is None:
        raise SystemExit("❌ prediction.gaussians 为空")
    g.means = g.means * sf
    g.scales = g.scales * sf          # 长度量，同步缩放
    _log(f"📏 高斯 → 米制：means/scales × scale_factor={sf:.4f}"
         f"（DA3 is_metric={is_metric}；E-01）")
    return sf

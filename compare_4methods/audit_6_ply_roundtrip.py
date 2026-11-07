#!/usr/bin/env python3
"""AUDIT-6：前馈高斯对象 ↔ 导出 PLY 的逐位对账（最后一环）

## 为什么必须做

AUDIT-1/2/3/4 都以 PLY 为"真值"。如果 `export_to_gs_ply` 在组装属性时错位
（通道顺序、prune 索引错行、少乘/多乘一次激活），下游所有检查都会**自洽地错**，
查不出来。所以要在同一个环境里再跑一次前向，把内存里的 `pred.gaussians`
与 PLY 逐点比：

  means     PLY.xyz          == gaussians.means × sf            （E-01 的米制折算）
  scales    exp(PLY.scale)   == gaussians.scales × sf
  opacity   sigmoid(PLY.op)  == gaussians.opacities
  rot       PLY.rot          == gaussians.rotations（wxyz，**不做任何重排**）
  f_dc      PLY.f_dc         == gaussians.harmonics[...,0]

判据：float32 量化级（相对误差 < 1e-5）。任一属性错位都会在这一步暴露。

用法（da3_human env，参数必须与 98 完全一致）：
  DA3_MODEL_DIR=/mnt/d/wheel/da3_human_ms \
  IMAGES_DIR=~/output/da3_results/_captures/佳佳抱枕/images \
  OUT_DIR=~/output/da3_gs_results/佳佳抱枕_p1m PROCESS_RES=504 REF_STRATEGY=first \
  ~/miniconda3/envs/da3_human/bin/python compare_4methods/audit_6_ply_roundtrip.py
"""
import os
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from da3_frame import gaussians_to_metric, gs_scale_factor  # noqa: E402


def log(s):
    print(s, flush=True)


def read_ply(path):
    with open(path, "rb") as f:
        raw = f.read(65536)
        end = raw.index(b"end_header") + len(b"end_header") + 1
        head = raw[:end].decode("ascii", "replace")
        n = int(re.search(r"element vertex (\d+)", head).group(1))
        props = [l.split()[2] for l in head.splitlines() if l.startswith("property")]
        f.seek(end)
        arr = np.fromfile(f, dtype=np.float32).reshape(n, len(props))
    return {k: arr[:, i] for i, k in enumerate(props)}, n


def main():
    model_dir = os.environ.get("DA3_MODEL_DIR", "/mnt/d/wheel/da3_human_ms")
    images_dir = os.path.expanduser(os.environ["IMAGES_DIR"])
    out_dir = Path(os.path.expanduser(os.environ["OUT_DIR"]))
    process_res = int(os.environ.get("PROCESS_RES", "504"))
    ref_strategy = os.environ.get("REF_STRATEGY", "first")

    files = []
    for e in ("*.png", "*.jpg", "*.jpeg"):
        files.extend(sorted(Path(images_dir).glob(e)))
    files = sorted(files)
    log(f"🖼️ {len(files)} 视角 @ {process_res}  ref={ref_strategy}")

    import torch
    from depth_anything_3.api import DepthAnything3
    from depth_anything_3.utils import export as da3_export

    try:
        import importlib.util
        if importlib.util.find_spec("gsplat") is None:
            da3_export.export_to_gs_video = lambda *a, **k: None
    except Exception:
        pass

    model = DepthAnything3.from_pretrained(model_dir).to("cuda")
    pred = model.inference([str(p) for p in files], process_res=process_res,
                           infer_gs=True, ref_view_strategy=ref_strategy,
                           export_dir=None, export_format="")

    g = pred.gaussians
    sf, is_metric = gs_scale_factor(pred)
    log(f"📏 scale_factor={sf:.6f} is_metric={is_metric}")
    # ⚠️ 与 98 同序：先用 npz 的相机口径，再折算米制
    sf_applied = gaussians_to_metric(pred, mode=os.environ.get("GS_SCALE_MODE", "metric"))
    assert abs(sf_applied - float(sf)) < 1e-9

    V, H, W = pred.depth.shape
    means = g.means.detach().float().cpu().numpy().reshape(V, H, W, 3)
    scales = g.scales.detach().float().cpu().numpy().reshape(V, H, W, 3)
    opac = g.opacities.detach().float().cpu().numpy().reshape(V, H, W)
    rots = g.rotations.detach().float().cpu().numpy().reshape(V, H, W, 4)
    harm = g.harmonics.detach().float().cpu().numpy()          # (1, N, 3, d_sh)
    d_sh = harm.shape[-1]
    harm = harm.reshape(V, H, W, 3, d_sh)
    log(f"🧠 内存高斯 {V*H*W:,} 个  d_sh={d_sh}（sh_degree={int(round(d_sh**0.5))-1}）")

    ply, n = read_ply(out_dir / "gs_ply" / "0000.ply")
    log(f"💾 PLY {n:,} 顶点")

    # 复刻屏蔽（prune_border + depth q90），与 save_gaussian_ply 逐行对应
    dep = np.asarray(pred.depth)
    th, tw = int(8 / 256 * H), int(8 / 256 * W)
    m = np.zeros((V, H, W), bool)
    m[:, th:H - th, tw:W - tw] = True
    m &= dep <= np.quantile(dep.reshape(V, -1), 0.90, axis=1).reshape(V, 1, 1)
    if int(m.sum()) != n:
        raise SystemExit(f"❌ 屏蔽 {int(m.sum()):,} ≠ PLY {n:,}")
    log(f"✅ 屏蔽复刻一致（{n:,}）")

    def cmp(name, a, b):
        a = np.asarray(a, np.float64).ravel()
        b = np.asarray(b, np.float64).ravel()
        den = np.maximum(np.abs(b), 1e-6)
        rel = np.abs(a - b) / den
        log(f"  {name:8s} max|Δ|={np.abs(a-b).max():.3e}  max_rel={rel.max():.3e}  "
            f"p99.9_rel={np.percentile(rel, 99.9):.3e}")
        return rel.max()

    log("\n逐属性对账（PLY vs 内存）")
    mk = m.reshape(-1)     # (V*H*W,) 只作用在内存侧；PLY 已是被压缩后的顺序
    r = {}
    r["means"] = cmp("means", np.stack([ply["x"], ply["y"], ply["z"]], -1),
                     means.reshape(-1, 3)[mk])
    r["scales"] = cmp("scales",
                      np.exp(np.stack([ply[f"scale_{i}"] for i in range(3)], -1)),
                      scales.reshape(-1, 3)[mk])
    r["opacity"] = cmp("opacity",
                       1.0 / (1.0 + np.exp(-ply["opacity"].astype(np.float64))),
                       opac.reshape(-1)[mk])
    r["rot"] = cmp("rot", np.stack([ply[f"rot_{i}"] for i in range(4)], -1),
                   rots.reshape(-1, 4)[mk])
    r["f_dc"] = cmp("f_dc", np.stack([ply[f"f_dc_{i}"] for i in range(3)], -1),
                    harm[..., 0].reshape(-1, 3)[mk])

    ok = all(v < 1e-5 for v in r.values())
    log(f"\n⇒ {'✅ 导出逐位一致，PLY 可作下游真值' if ok else '❌ 导出与内存不一致，下游全部结论需重审'}")
    if not ok:
        bad = [k for k, v in r.items() if v >= 1e-5]
        log(f"   异常属性：{bad}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()

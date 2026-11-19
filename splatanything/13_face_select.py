#!/usr/bin/env python3
"""13_face_select.py — 把 2D face mask 转成 3D 高斯 ID 集合（P0-2）。

判据（逐视角）：
  1. 中心投影像素落在 dilate(face) 内 → 候选；
  2. 深度层：用 gsplat 的 expected depth（render_mode RGB+ED）在该像素的值 z_exp，
     只保留 z_g <= z_exp * depth_tol 的高斯 —— 排除脸后背景高斯（计划原话：
     "仅中心投影落在脸框会选到脸后背景"）；
  3. footprint：2D 投影半径 >= min_fp_px（太小的是远处碎点，对脸无贡献）。
跨视角汇总：support 计数 >= min_views（默认 2，三视角中至少两张支持，保留合法侧脸）。

产物：
  <out>/selected_ids.npy   int64 高斯索引
  <out>/select_meta.json   统计（候选数/支持数/尺度分布/每视角计数）
  <out>/audit_<view>.png   三视角覆盖审核图（投影点按支持数着色叠加在 rgb 上）

运行环境：splatanything env（gsplat）。用法：
  python 13_face_select.py --result <02输出目录> [--min-support 2]
"""
import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

# 复用 03_render_eval 的 load_ply / load_cameras（同目录，文件名带数字前缀不能直接 import）
_spec_mod = None


def _load_03():
    global _spec_mod
    if _spec_mod is None:
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "render_eval03", str(Path(__file__).parent / "03_render_eval.py"))
        m = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(m)
        _spec_mod = m
    return _spec_mod


def render_expected_depth(g, c2w, K, w, h, device="cuda", eps2d=0.01,
                          near=0.01, far=400.0, background=0.0):
    """用 gsplat 渲染期望深度（RGB+ED 模式取第 4 通道）。"""
    from gsplat import rasterization
    c2w64 = np.asarray(c2w, np.float64).reshape(-1, 4, 4)
    viewmats = torch.from_numpy(np.linalg.inv(c2w64).astype(np.float32)).to(device)
    Ks = torch.from_numpy(np.asarray(K, np.float32).reshape(-1, 3, 3)).to(device)
    C = viewmats.shape[0]
    colors = torch.from_numpy(g["sh"]).to(device)[
        None, :, None, :].expand(C, -1, -1, -1).contiguous()
    bg = torch.full((C, 3), float(background), device=device)
    out, alpha, _ = rasterization(
        means=torch.from_numpy(g["means"]).to(device),
        quats=torch.from_numpy(g["quats"]).to(device),
        scales=torch.from_numpy(g["scales"]).to(device),
        opacities=torch.from_numpy(g["opacities"]).to(device),
        colors=colors, viewmats=viewmats, Ks=Ks, width=w, height=h,
        sh_degree=0, eps2d=eps2d, near_plane=near, far_plane=far,
        backgrounds=bg, render_mode="RGB+ED")
    # out: [C,H,W,4]，第 4 通道 expected depth
    return out[..., 3].cpu().numpy(), alpha.cpu().numpy()


def project_points(means, c2w, K):
    """返回 (u, v, z) 像素坐标与深度；z<=0 为相机后方。"""
    w2c = np.linalg.inv(np.asarray(c2w, np.float64))
    pc = means @ w2c[:3, :3].T + w2c[:3, 3]      # [N,3] 相机系
    z = pc[:, 2]
    with np.errstate(divide="ignore", invalid="ignore"):
        u = K[0, 0] * pc[:, 0] / z + K[0, 2]
        v = K[1, 1] * pc[:, 1] / z + K[1, 2]
    return u, v, z


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--result", required=True)
    ap.add_argument("--masks", default=None, help="默认 <result>/face_opt/masks")
    ap.add_argument("--out", default=None, help="默认 <result>/face_opt/select")
    ap.add_argument("--min-support", type=int, default=2)
    ap.add_argument("--dilate-px", type=int, default=8,
                    help="face mask 外扩，吃进压线/发际边缘高斯")
    ap.add_argument("--depth-tol", type=float, default=1.10,
                    help="深度层容差倍率：z_g <= z_exp * depth_tol")
    ap.add_argument("--min-fp-px", type=float, default=0.3,
                    help="最小 2D footprint 半径（像素）")
    ap.add_argument("--eps2d", type=float, default=0.01)
    args = ap.parse_args()

    re03 = _load_03()
    res = Path(args.result)
    masks_dir = Path(args.masks) if args.masks else res / "face_opt" / "masks"
    out = Path(args.out) if args.out else res / "face_opt" / "select"
    out.mkdir(parents=True, exist_ok=True)

    g = re03.load_ply(res / "gaussians.ply")
    c2w, K, ids = load_cams(res / "cameras.json")
    N = len(g["means"])
    print(f"🚀 [13] 选择脸部高斯 | 高斯 {N:,} | 视角 {len(ids)}")

    with Image.open(res / "rgb" / f"{ids[0]}.png") as im:
        w, h = im.size

    # 逐视角：face mask + 期望深度 + 中心投影 → 支持表
    support = np.zeros((N,), np.int32)
    per_view = []
    audit_tiles = []
    for vi, cid in enumerate(ids):
        fm = np.asarray(Image.open(masks_dir / f"{cid}_face.png")) > 127
        if int(fm.sum()) == 0:
            print(f"  ⚠️ {cid}: 无 face mask，跳过")
            per_view.append(dict(id=cid, cand=0, passed=0))
            continue
        import cv2
        k = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (2 * args.dilate_px + 1,) * 2)
        fm_d = cv2.dilate(fm.astype(np.uint8), k).astype(bool)

        ed, _alpha = render_expected_depth(
            g, c2w[vi:vi + 1], K[vi:vi + 1], w, h, eps2d=args.eps2d)
        ed = ed[0]  # [H,W]

        u, v, z = project_points(g["means"], c2w[vi], K[vi])
        ui, vi_ = np.round(u).astype(np.int64), np.round(v).astype(np.int64)
        in_img = (z > 0.01) & (ui >= 0) & (ui < w) & (vi_ >= 0) & (vi_ < h)
        idx_img = np.where(in_img)[0]
        in_face = np.zeros(N, bool)
        in_face[idx_img] = fm_d[vi_[idx_img], ui[idx_img]]
        # 深度层：候选点的 z 不超过该像素期望深度 * tol（排除脸后背景）
        z_exp = np.full(N, np.inf, np.float64)
        z_exp[idx_img] = ed[vi_[idx_img], ui[idx_img]]
        front = z <= z_exp * args.depth_tol
        # footprint 粗估：最大 scale 投影为 fx*s/z 半径
        fpx = 0.5 * (K[vi][0, 0] + K[vi][1, 1])
        smax = g["scales"].max(axis=1)
        fp = np.where(z > 0.01, fpx * smax / np.maximum(z, 1e-6), 0.0)
        big = fp >= args.min_fp_px

        ok = in_face & front & big
        support += ok.astype(np.int32)
        per_view.append(dict(id=cid, cand=int(in_face.sum()),
                             depth_pass=int((in_face & front).sum()),
                             fp_pass=int(ok.sum())))
        print(f"  {cid}: 候选 {int(in_face.sum()):,} → "
              f"深度 {int((in_face & front).sum()):,} → footprint {int(ok.sum()):,}")

        # 审核图：rgb 底 + face mask 描边 + 通过点（红）
        base = np.asarray(Image.open(res / "rgb" / f"{cid}.png").convert("RGB")).copy()
        cnt, _ = cv2.findContours(fm_d.astype(np.uint8), cv2.RETR_EXTERNAL,
                                  cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(base, cnt, -1, (0, 255, 0), 1)
        pts = np.where(ok)[0]
        if len(pts) > 20000:  # 审核图抽稀，免得全红看不清
            pts = pts[np.linspace(0, len(pts) - 1, 20000).astype(np.int64)]
        pu, pv = ui[pts], vi_[pts]
        valid = (pu >= 0) & (pu < w) & (pv >= 0) & (pv < h)
        base[pv[valid], pu[valid]] = (255, 40, 40)
        audit_tiles.append(base)

    sel = np.where(support >= args.min_support)[0].astype(np.int64)
    np.save(out / "selected_ids.npy", sel)
    print(f"🎯 选中 {len(sel):,} / {N:,}（{100*len(sel)/N:.2f}%），"
          f"min_support={args.min_support}")

    # 选中高斯的尺度/深度统计（P3 位移上限要用头宽，这里顺便记录包围盒）
    if len(sel):
        smax = g["scales"][sel].max(axis=1)
        ctr = g["means"][sel]
        bbox = np.ptp(ctr, axis=0)
        stats = dict(
            n_selected=int(len(sel)), n_total=int(N),
            scale_max_pct=[float(x) for x in np.percentile(smax, [50, 90, 99])],
            center_bbox_size=[float(x) for x in bbox],
            center_mean=[float(x) for x in ctr.mean(axis=0)],
        )
    else:
        stats = dict(n_selected=0, n_total=int(N))
    meta = dict(args=vars(args), per_view=per_view, **stats)
    with open(out / "select_meta.json", "w", encoding="utf-8", newline="\n") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    # 审核拼图
    if audit_tiles:
        tiles = [Image.fromarray(t).resize((w // 2, h // 2)) for t in audit_tiles]
        sheet = Image.new("RGB", (w // 2 * len(tiles), h // 2))
        for i, t in enumerate(tiles):
            sheet.paste(t, (i * w // 2, 0))
        sheet.save(out / "audit_sheet.png")
    print(f"🎉 Done: {out}")


def load_cams(path):
    re03 = _load_03()
    return re03.load_cameras(path)


if __name__ == "__main__":
    main()

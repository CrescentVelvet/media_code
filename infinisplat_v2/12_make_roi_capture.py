#!/usr/bin/env python3
"""从手机采集包裁 ROI（人物/手部区域）→ 生成 InfiniSplat v2 输入，实现"把像素预算给到手"。

与 07_capture_to_npz.py 的唯一区别：**在处理前裁一块物理更小的视场**，再 resize 到
同样的处理长边。效果等价于数字变焦 —— 手部像素密度变成 1/roi_frac 倍。

为什么必须从原图（1440x1920）裁而不是从 392x518 的处理图裁：
  07 已经把原图下采样到处理尺寸，手部只剩约 30 px。从那张图裁出来再放大是插值，
  不产生新信息。ROI 的价值来自"原图里本来就有、只是被整图下采样丢掉的像素"。

ROI 定位（零检测器方案）：
  transforms.json 的 anchor_point 实测在人物躯干上（32/32 帧投影落画面内，深度 1.1~1.7 m），
  逐帧投影后以它为中心裁框，框随人物漂移。实测锚点系统性偏人物右侧约 0.05~0.1
  （归一化），用 --offset-u 补。

内参三处必须同步改（漏一处就渲染错位）：
  1. undistort 得到 newK + roi 裁剪 → cx/cy 减 roi 偏移（同 07）
  2. 裁 ROI [x0,y0] → cx/cy 再减 x0/y0
  3. resize 到 target_long → fx/fy/cx/cy 同乘 scale（底行保持 [0,0,1]）

Env / args:
  --raw-dir     采集包路径（必填）
  --out-dir     输出目录（必填）
  --n-views     抽帧数（默认 32）
  --roi-frac    ROI 边长占原图的比例（默认 0.5 → 视场边长减半，像素密度 2x）
  --target-long 输出图长边（默认 518；推理时 LONG_SIDE 要填同一个值）
  --offset-u    锚点水平偏移（归一化，默认 -0.06，实测锚点偏人物右侧）
  --offset-v    锚点垂直偏移（默认 0.02）
  --preview     只画框不写图（输出带框的 32 帧网格，用来看框是否覆盖手）
  --preview-out 预览图路径
"""
import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pillow_heif
from PIL import Image, ImageDraw, ImageOps

pillow_heif.register_heif_opener()
FLIP = np.diag([1.0, -1.0, -1.0, 1.0])  # OpenGL c2w → OpenCV c2w


def pick_frames(n_frames, n_views):
    if n_views >= n_frames:
        return list(range(n_frames))
    idx = np.linspace(0, n_frames - 1, n_views).round().astype(int)
    return sorted(set(idx.tolist()))


def anchor_uv(fr, anchor, w, h):
    """anchor_point → 原始（带畸变）像素坐标。返回 None 表示在相机背后。"""
    c2w = np.array(fr["transform_matrix"], dtype=np.float64) @ FLIP
    R, t = c2w[:3, :3], c2w[:3, 3]
    pc = R.T @ (anchor - t)
    if pc[2] <= 1e-6:
        return None
    u = fr["fl_x"] * pc[0] / pc[2] + fr["cx"]
    v = fr["fl_y"] * pc[1] / pc[2] + fr["cy"]
    return np.array([u, v], dtype=np.float64)


def undistort_map(K, dist, w, h):
    """复现 07 的去畸变：返回 (newK, x_off, y_off, out_w, out_h)。"""
    newK, roi = cv2.getOptimalNewCameraMatrix(K, dist, (w, h), 0, (w, h))
    x, y, rw, rh = roi
    if rw <= 0 or rh <= 0:
        return newK, 0, 0, w, h
    return newK, int(x), int(y), int(rw), int(rh)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--n-views", type=int, default=32)
    ap.add_argument("--roi-source", choices=["anchor", "mask"], default="anchor",
                    help="anchor=用 anchor_point 投影（零依赖，落在躯干）；mask=SAM3 手部 bbox（精确）")
    ap.add_argument("--mask-manifest", default=None,
                    help="SAM3 masks_manifest.json（--roi-source mask 时必填）")
    ap.add_argument("--roi-frac", type=float, default=0.5)
    ap.add_argument("--target-long", type=int, default=518)
    ap.add_argument("--offset-u", type=float, default=-0.06)
    ap.add_argument("--offset-v", type=float, default=0.02)
    ap.add_argument("--preview", action="store_true", help="缩略图上画 ROI 框")
    ap.add_argument("--preview-out", default=None)
    ap.add_argument("--fixed-center", action="store_true",
                    help="所有帧用同一个中心（各帧检测中心的中位数），消除框随帧移动 —— "
                         "用于区分「裁切本身」与「框逐帧漂移导致 DA3 跨帧一致性变差」")
    ap.add_argument("--preview-roi", action="store_true",
                    help="把裁好的 ROI 本身拼网格（判读框是否真的框住了手，比画框可靠）")
    ap.add_argument("--preview-roi-out", default=None)
    ap.add_argument("--no-undistort", action="store_true")
    args = ap.parse_args()

    raw = Path(args.raw_dir)
    out = Path(args.out_dir)
    tj = raw / "transforms.json"
    if not tj.is_file():
        sys.exit(f"❌ 缺 transforms.json: {tj}")
    d = json.loads(tj.read_text(encoding="utf-8"))
    frames = d["frames"]
    anchor = np.array(d["anchor_point"], dtype=np.float64)
    picked = pick_frames(len(frames), args.n_views)

    # SAM3 手部 bbox（归一化 xyxy）。注意 mask 是在 07 的输出图（已去畸变+roi裁切，
    # 只是尺寸更小）上算的，与本文的 arr **同一坐标系**，归一化坐标可直接用。
    mask_boxes = {}
    if args.roi_source == "mask":
        if not args.mask_manifest:
            sys.exit("❌ --roi-source mask 需要 --mask-manifest")
        mf = json.loads(Path(args.mask_manifest).read_text(encoding="utf-8"))
        for stem, insts in mf.get("frames", {}).items():
            if not insts:
                continue
            # manifest key 是 SAM3 worker 的输出名 "000_<原图stem>"，去掉序号前缀才能对上 src.stem。
            # 早期版本直接拿 src.stem 查 .get() → 全部 None → 静默回退到 anchor（框偏但仍出图，
            # 极难发现）。所以下面必须做命中率硬校验。
            key = stem.split("_", 1)[1] if "_" in stem else stem
            x0s = [it["box_xywh"][0] for it in insts]
            y0s = [it["box_xywh"][1] for it in insts]
            x1s = [it["box_xywh"][0] + it["box_xywh"][2] for it in insts]
            y1s = [it["box_xywh"][1] + it["box_xywh"][3] for it in insts]
            mask_boxes[key] = (min(x0s), min(y0s), max(x1s), max(y1s))
        hit = sum(1 for i in picked
                  if (raw / frames[i]["file_path"].replace("./", "")).stem in mask_boxes)
        print(f"🔍 手部 bbox: manifest {len(mask_boxes)} 帧，命中 {hit}/{len(picked)}")
        if hit < 0.8 * len(picked):
            sys.exit(f"❌ mask 命中率过低 {hit}/{len(picked)}，manifest key 与图像名对不上")

    fixed_cu = fixed_cv = None
    if args.fixed_center:
        if args.roi_source != "mask":
            sys.exit("❌ --fixed-center 目前只支持 --roi-source mask")
        uu = np.array([0.5 * (b[0] + b[2]) for b in mask_boxes.values()])
        vv = np.array([0.5 * (b[1] + b[3]) for b in mask_boxes.values()])
        fixed_cu = float(np.median(uu)) + args.offset_u
        fixed_cv = float(np.median(vv)) + args.offset_v
        print(f"📌 固定中心 u={fixed_cu:.3f} v={fixed_cv:.3f}"
              f"（手部中心漂移 u ±{float((uu.max() - uu.min()) / 2):.3f}）")

    if not (args.preview or args.preview_roi):
        (out / "images").mkdir(parents=True, exist_ok=True)

    # 预览图布局
    need_preview = args.preview or args.preview_roi
    if need_preview:
        TW, TH, COLS = 180, 240, 8
        ROWS = (len(picked) + COLS - 1) // COLS
    if args.preview:
        canvas = Image.new("RGB", (COLS * TW, ROWS * TH), (15, 15, 15))
        dr = ImageDraw.Draw(canvas)
    if args.preview_roi:
        RTW, RTH = 160, 213
        canvas_roi = Image.new("RGB", (COLS * RTW, ROWS * RTH), (15, 15, 15))
        dr_roi = ImageDraw.Draw(canvas_roi)

    c2ws, Ks, names, records = [], [], [], []
    box_stats = []
    for n, i in enumerate(picked):
        fr = frames[i]
        src = raw / fr["file_path"].replace("./", "")
        if not src.is_file():
            sys.exit(f"❌ 图片缺失: {src}")
        with Image.open(src) as im:
            arr = np.asarray(ImageOps.exif_transpose(im).convert("RGB"))
        h, w = arr.shape[:2]

        K = np.array([[fr["fl_x"], 0.0, fr["cx"]],
                      [0.0, fr["fl_y"], fr["cy"]],
                      [0.0, 0.0, 1.0]], dtype=np.float64)
        if (w, h) != (fr["w"], fr["h"]):
            K[0] *= w / fr["w"]
            K[1] *= h / fr["h"]

        uv = anchor_uv(fr, anchor, w, h)
        if uv is None and args.roi_source == "anchor":
            sys.exit(f"❌ 帧 {i} 锚点在相机背后，无法定位 ROI")

        # --- 去畸变（同 07 口径），并把锚点映射到去畸变后的坐标 ---
        if not args.no_undistort and "k1" in fr:
            dist = np.array([fr["k1"], fr["k2"], fr["p1"], fr["p2"], fr["k3"]], dtype=np.float64)
            newK, xo, yo, ow, oh = undistort_map(K, dist, w, h)
            arr = cv2.undistort(arr, K, dist, None, newK)
            if (xo, yo, ow, oh) != (0, 0, w, h):
                arr = arr[yo:yo + oh, xo:xo + ow]
                newK = newK.copy()
                newK[0, 2] -= xo
                newK[1, 2] -= yo
            H, W = arr.shape[:2]
            # 锚点：原图畸变像素 →（原始 K）→ 去畸变归一化 →（P=newK）投射。
            # ⚠️ 这里 newK 已在上面的裁切里减过 xo/yo，所以**不能再减一次**；
            # 早期版本多减一次会把框整体往左上推（实测偏移可达 0.03~0.05 归一化）。
            if uv is not None:
                und = cv2.undistortPoints(uv.reshape(1, 1, 2), K, dist, P=newK).reshape(2)
            else:
                und = np.array([newK[0, 2] + 0.5 * W, newK[1, 2] + 0.5 * H])  # 兜底：主点
            K = newK
        else:
            und = uv if uv is not None else np.array([w / 2.0, h / 2.0])
            H, W = h, w

        # --- 定 ROI 中心：固定物理视场（边长 = roi_frac × 图边长），中心跟随手/锚点 ---
        # 固定视场而非自适应 bbox 大小，是为了让每帧的物理视场一致、内参只差 cx/cy 平移。
        if fixed_cu is not None:
            cu, cv_ = fixed_cu, fixed_cv
        elif args.roi_source == "mask" and src.stem in mask_boxes:
            mb = mask_boxes[src.stem]
            cu = 0.5 * (mb[0] + mb[2]) + args.offset_u
            cv_ = 0.5 * (mb[1] + mb[3]) + args.offset_v
        else:
            cu = float(und[0] / W) + args.offset_u
            cv_ = float(und[1] / H) + args.offset_v
        bw, bh = args.roi_frac * W, args.roi_frac * H
        x0 = cu * W - bw / 2.0
        y0 = cv_ * H - bh / 2.0
        x0 = float(np.clip(x0, 0, W - bw))
        y0 = float(np.clip(y0, 0, H - bh))
        xi0, yi0 = int(round(x0)), int(round(y0))
        xi1, yi1 = int(round(x0 + bw)), int(round(y0 + bh))
        xi1, yi1 = min(xi1, W), min(yi1, H)
        roi = arr[yi0:yi1, xi0:xi1]
        rh_, rw_ = roi.shape[:2]

        # --- 内参：先减裁切偏移，再按 resize 缩放 ---
        Kr = K.copy()
        Kr[0, 2] -= xi0
        Kr[1, 2] -= yi0
        scale = args.target_long / max(rw_, rh_)
        out_w, out_h = int(round(rw_ * scale)), int(round(rh_ * scale))
        Kr[0] *= scale
        Kr[1] *= scale

        if need_preview:
            r, c = n // COLS, n % COLS
        box_stats.append((xi0 / W, yi0 / H, xi1 / W, yi1 / H))

        if args.preview:
            th = Image.fromarray(arr).resize((TW, TH), Image.Resampling.BILINEAR)
            canvas.paste(th, (c * TW, r * TH))
            sx, sy = TW / W, TH / H
            dr.rectangle([c * TW + xi0 * sx, r * TH + yi0 * sy,
                          c * TW + xi1 * sx, r * TH + yi1 * sy],
                         outline=(0, 255, 0), width=2)
            dr.text((c * TW + 3, r * TH + 2), f"{n}:f{i}", fill=(255, 255, 0))
        if args.preview_roi:
            th = Image.fromarray(roi).resize((RTW, RTH), Image.Resampling.BILINEAR)
            canvas_roi.paste(th, (c * RTW, r * RTH))
            dr_roi.text((c * RTW + 3, r * RTH + 2), f"{n}:f{i}", fill=(255, 255, 0))
        if not need_preview:
            roi_r = cv2.resize(roi, (out_w, out_h), interpolation=cv2.INTER_AREA)
            name = f"{n:03d}_{src.stem}.png"
            Image.fromarray(roi_r).save(out / "images" / name, optimize=False)
            names.append(name)
            c2ws.append(np.array(fr["transform_matrix"], dtype=np.float64) @ FLIP)
            Ks.append(Kr.astype(np.float32))
            records.append(dict(idx=int(i), src=src.name, out=name,
                                roi_px=[xi0, yi0, xi1, yi1], src_wh=[int(W), int(H)],
                                out_wh=[out_w, out_h],
                                upscale_vs_full=round(scale / (args.target_long / max(w, h)), 3)))

    bs = np.array(box_stats)
    print(f"📦 ROI 框（归一化，占去畸变图）x:[{bs[:,0].min():.3f},{bs[:,2].max():.3f}] "
          f"y:[{bs[:,1].min():.3f},{bs[:,3].max():.3f}]  边长 {args.roi_frac:.2f}")

    if args.preview:
        p = args.preview_out or str(out / "roi_preview.png")
        canvas.save(p)
        print(f"🖼️ 框预览写出 {p}")
    if args.preview_roi:
        p = args.preview_roi_out or str(out / "roi_content_preview.png")
        canvas_roi.save(p)
        print(f"🖼️ ROI 内容预览写出 {p}")
    if need_preview:
        return

    c2w = np.stack(c2ws).astype(np.float32)
    K = np.stack(Ks).astype(np.float32)
    np.savez(out / "cameras.npz", c2w=c2w, intrinsics=K, names=np.array(names))

    C = c2w[:, :3, 3]
    rad = np.linalg.norm(C, axis=1)
    meta = dict(source=str(raw), mode="roi", n_frames_total=len(frames),
                n_views=len(picked), picked_idx=picked,
                roi_frac=args.roi_frac, target_long=args.target_long,
                offset_uv=[args.offset_u, args.offset_v],
                anchor_point=[float(x) for x in anchor],
                undistort=not args.no_undistort,
                out_wh=[records[0]["out_wh"]], frames=records,
                cam_radius=dict(min=float(rad.min()), max=float(rad.max()), mean=float(rad.mean())))
    (out / "meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"✅ 输出 {out}")
    print(f"   🖼️  {len(picked)} 帧 → {records[0]['out_wh'][0]}x{records[0]['out_wh'][1]}")
    print(f"   📐 相对整图放大 {records[0]['upscale_vs_full']:.2f}x（等于 1/roi_frac × 长边比）")
    print(f"   📷 相机半径 {rad.min():.3f}~{rad.max():.3f} m")
    print(f"   🔭 K[0] fx={K[0,0,0]:.1f} cx={K[0,0,2]:.1f} cy={K[0,1,2]:.1f}")


if __name__ == "__main__":
    main()

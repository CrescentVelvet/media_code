#!/usr/bin/env python3
"""17_fbxgt_to_scene.py — human_fbx_gt（静态合成人物，Blender 渲染 + GT 相机/深度）
转成 02_run_inference.sh 可直接消费的 07 式场景目录。

对应 F2「先选一个可信相机的静态高质量人物模型渲染包：三张 source + 独立 target + GT 表面」。
按 yaw 均分选 3 张 source + 2 张 target（间隔其中），生成：
  _scenes/<out_src>/{images/*.png, prior_camera.json, meta.json}
  _scenes/<out_tgt>/{images/*.png, prior_camera.json, meta.json}

相机约定：包内 cameras.json 标称 OpenCV，但 2026-11-19 实测发现生成器
（flash_human/01_render_gt.py v1）把 Blender→OpenCV 转换误写为 C_CONV @ R_b
（应为 R_b @ C_CONV，左乘=世界系翻转），导致跨视角几何漂移。
本适配器对存量包一律做修正 R = C @ R_stored @ C、t 不变
（鼻尖三视角反投影残差 ~2mm 实测定案）。新渲的包（生成器已修）无需修正，
用 --no-fix 关闭。

用法：
  python 17_fbxgt_to_scene.py --pkg /mnt/d/dataset/human_fbx_gt/rosemary_face_v1 \
      --out-root ~/output/splatanything_results/_scenes --name rosemary_face
"""
import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image


def yaw_of(c2w, target):
    """相机绕目标点的方位角（世界系 XY 平面；该包 bbox 显示 Z 是竖直轴）。"""
    d = c2w[:3, 3] - target
    return float(np.degrees(np.arctan2(d[1], d[0])))


def write_scene(out, picks, K, ids_w, cams_by_id):
    out.mkdir(parents=True, exist_ok=True)
    (out / "images").mkdir(exist_ok=True)
    extr, intr = [], []
    for new_id, (cid, yaw) in enumerate(picks):
        stem = f"{new_id:03d}_{cid:06d}"
        img = Image.open(ids_path[cid]).convert("RGB")
        img.save(out / "images" / f"{stem}.png")
        extr.append({"camera_id": stem,
                     "matrix": cams_by_id[cid]["c2w"].tolist()})
        intr.append({"camera_id": stem, "matrix": K.tolist()})
    (out / "prior_camera.json").write_text(json.dumps(
        {"num_cameras": len(extr), "extrinsics": extr, "intrinsics": intr}))
    (out / "meta.json").write_text(json.dumps({
        "source_package": str(pkg_dir), "view_ids": [c for c, _ in picks],
        "yaws": [round(y, 1) for _, y in picks],
        "gt_depth": "depth/*.exr (相机系线性 z 米)",
        "gt_mesh": meta.get("fbx")}, ensure_ascii=False, indent=1))
    print(f"  ✅ {out}  ids={[c for c,_ in picks]} yaw={[round(y,1) for _,y in picks]}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--pkg", required=True)
    ap.add_argument("--out-root", required=True)
    ap.add_argument("--name", default="rosemary_face")
    ap.add_argument("--src", default=None,
                    help="显式 source 视角 id 列表，如 20,30,38（省去自动 yaw 选择）")
    ap.add_argument("--tgt", default=None, help="显式 target 视角 id 列表，如 24,34")
    ap.add_argument("--no-fix", action="store_true",
                    help="生成器已修复的新包，关闭 C@R@C 修正")
    args = ap.parse_args()

    pkg_dir = Path(args.pkg)
    cj = json.loads((pkg_dir / "cameras.json").read_text())
    meta = json.loads((pkg_dir / "meta.json").read_text())
    K = np.asarray(cj["intrinsics"]["K"], np.float64)
    target = np.asarray(meta["target"], np.float64)
    C3 = np.diag([1.0, -1.0, -1.0])
    cams_by_id = {}
    for c in cj["cameras"]:
        cid = int(c["id"])
        m = np.asarray(c["transform_cam2world"], np.float64)
        if not args.no_fix:
            m2 = m.copy()
            m2[:3, :3] = C3 @ m[:3, :3] @ C3   # 只修旋转；平移不变
            m = m2
        cams_by_id[cid] = dict(c2w=m)
    ids_path = {c["id"]: pkg_dir / c["file_rgb"] for c in cj["cameras"]}
    yaws = {cid: yaw_of(v["c2w"], target) for cid, v in cams_by_id.items()}
    order = sorted(yaws.items(), key=lambda kv: kv[1])
    print(f"视角数 {len(order)}  yaw 范围 "
          f"{order[0][1]:.1f}° ~ {order[-1][1]:.1f}°")

    # 3 source：中位视角 + 两端约 ±50°（nearest 去重）；2 target：夹在 source 之间
    if args.src and args.tgt:
        src_picks = [(int(x), yaws[int(x)]) for x in args.src.split(",")]
        tgt_picks = [(int(x), yaws[int(x)]) for x in args.tgt.split(",")]
    else:
        n = len(order)

        def nearest(want):
            return min(order, key=lambda kv: abs(kv[1] - want))
        mid = order[n // 2][1]
        span = 50.0
        src_picks = []
        for want in (mid - span, mid, mid + span):
            p = nearest(want)
            if p[0] not in {u[0] for u in src_picks}:
                src_picks.append(p)
        if len(src_picks) < 3:
            raise SystemExit("❌ 视角不够分散，无法选出 3 张 source")
        y0, y1, y2 = (p[1] for p in src_picks)
        tgt_picks = []
        for want in ((y0 + y1) / 2, (y1 + y2) / 2):
            p = nearest(want)
            if p[0] not in {u[0] for u in src_picks}:
                tgt_picks.append(p)

    root = Path(args.out_root)
    write_scene(root / f"{args.name}3v", src_picks, K, None, cams_by_id)
    write_scene(root / f"{args.name}_t2", tgt_picks, K, None, cams_by_id)

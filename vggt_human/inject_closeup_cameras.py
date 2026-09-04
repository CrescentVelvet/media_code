#!/usr/bin/env python3
"""inject_closeup_cameras.py — 把增强后的 512 近景图作为额外相机注入 COLMAP 场景。

背景：06c 产出的近景渲染（closeup_XXXX.png）经 HYPIR 整图增强后，
希望作为新增训练视角喂回 3DGS（06d 续训）。这些近景相机的位姿/内参
已由 render_closeup.py 写入 06c_closeup_poses_p{pid}.json：
  R (3x3, 行 = [right; down; forward]), T, C, W, H, fx, fy, cx, cy
与 COLMAP cam_from_world 同约定：x_cam = R @ x_world + T。

pycolmap 4.2 注入路径（C builtin，无 inspect.signature，经 probe 确认）：
  rec.add_camera_with_trivial_rig(cam)                    # 建 trivial rig
  rec.add_image_with_trivial_frame(img, cam_from_world)   # 注入带位姿的图

用法:
  python inject_closeup_cameras.py \
      --source_dir  /mnt/d/output/vggt_human_ms/03b_source_ba \
      --poses_dir   /mnt/d/output/vggt_human_ms/06c_512_enhanced \
      --enhanced_root /mnt/d/output/vggt_human_ms/06c_512_enhanced \
      --out_dir     /mnt/d/output/vggt_human_ms/06e_source_closeup \
      [--persons p00,p01,p02] [--dry_run]

输出:
  out_dir/images/       原图拷贝 + 增强近景（重命名 closeup_p{pid}_{n}.png）
  out_dir/sparse/0/     注入后的重建（write_binary）
  out_dir/inject_report.json  每人/每视角 sanity（脸部中心投影偏差、深度符号）
"""
import argparse
import glob
import json
import os
import shutil
import sys

import numpy as np
import pycolmap


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source_dir", required=True,
                   help="COLMAP 场景（含 images/ 与 sparse/0）")
    p.add_argument("--poses_dir", required=True,
                   help="含 06c_closeup_poses_p*.json 的目录")
    p.add_argument("--enhanced_root", required=True,
                   help="含 enhanced_p{pid}/images/closeup_*.png 的目录")
    p.add_argument("--out_dir", required=True, help="输出场景目录")
    p.add_argument("--images_folder", default="images",
                   help="源场景图像目录名（默认 images）")
    p.add_argument("--persons", default="",
                   help="逗号分隔的 pid 过滤，如 p00,p01,p02；空=全部")
    p.add_argument("--dry_run", action="store_true",
                   help="只做投影 sanity，不写输出")
    return p.parse_args()


def load_poses(poses_dir, persons):
    """返回 {pid: [view_dict, ...]}，按 json 文件顺序。"""
    out = {}
    pattern = os.path.join(poses_dir, "06c_closeup_poses_p*.json")
    files = sorted(glob.glob(pattern))
    if not files:
        sys.exit(f"❌ no pose json found: {pattern}")
    for f in files:
        pid = os.path.basename(f).replace("06c_closeup_poses_", "").replace(".json", "")
        if persons and pid not in persons:
            continue
        with open(f) as fh:
            views = json.load(fh)
        if views:
            out[pid] = views
    return out


def sanity_project(views, center):
    """把 3D 人脸中心投到每个近景相机，检查深度>0 与落点离心程度。"""
    reps = []
    for cv in views:
        R = np.asarray(cv["R"], dtype=np.float64)
        T = np.asarray(cv["T"], dtype=np.float64)
        Pc = R @ np.asarray(center, dtype=np.float64) + T
        depth = float(Pc[2])
        u = cv["fx"] * Pc[0] / depth + cv["cx"]
        v = cv["fy"] * Pc[1] / depth + cv["cy"]
        reps.append({
            "name": cv["name"],
            "depth": round(depth, 4),
            "u": round(float(u), 1), "v": round(float(v), 1),
            "dist_px": round(float(np.hypot(u - cv["cx"], v - cv["cy"])), 1),
            "positive_depth": depth > 0,
        })
    return reps


def main():
    args = parse_args()
    persons = set(x for x in args.persons.split(",") if x) if args.persons else None

    sparse_dir = os.path.join(args.source_dir, "sparse", "0")
    rec = pycolmap.Reconstruction(sparse_dir)
    n_cam0, n_img0 = len(rec.cameras), len(rec.images)
    print(f"📂 source: {n_cam0} cameras / {n_img0} images / {len(rec.points3D)} pts")

    poses = load_poses(args.poses_dir, persons or set())
    if not poses:
        sys.exit("❌ no poses after person filter")
    print(f"👥 persons: {', '.join(f'{k}({len(v)}v)' for k, v in sorted(poses.items()))}")

    # 逐人 sanity：脸部 3D 中心应投影在近景图中心附近（这就是近景的设计目标）
    center_file = os.path.join(os.path.dirname(args.poses_dir), "06c_face_center.json")
    report = {"source": args.source_dir, "persons": {}}
    if os.path.exists(center_file):
        with open(center_file) as fh:
            fc = json.load(fh)
        for pid, views in sorted(poses.items()):
            # face_center.json 的 persons 键是 "0"/"1"/...（int pid），poses 文件是 p00
            pkey = str(int(pid[1:])) if pid.startswith("p") else pid
            pinfo = fc.get("persons", {}).get(pkey)
            if not pinfo:
                print(f"  ⚠️ {pid}: no face_center entry, skip projection sanity")
                continue
            reps = sanity_project(views, pinfo["center"])
            ok = sum(r["positive_depth"] for r in reps)
            mean_d = float(np.mean([r["dist_px"] for r in reps]))
            print(f"  🔎 {pid}: depth>0 {ok}/{len(reps)}, mean center offset {mean_d:.1f}px")
            report["persons"][pid] = {"projection": reps,
                                      "mean_offset_px": round(mean_d, 1)}
    else:
        print(f"  ⚠️ face center file not found: {center_file} (skip sanity)")

    if args.dry_run:
        print("🏃 dry_run, exiting")
        return

    # 准备输出场景：拷 sparse 之外的图像目录 + 复制 sparse
    os.makedirs(os.path.join(args.out_dir, args.images_folder), exist_ok=True)
    out_sparse = os.path.join(args.out_dir, "sparse", "0")
    os.makedirs(out_sparse, exist_ok=True)

    src_images = os.path.join(args.source_dir, args.images_folder)
    dst_images = os.path.join(args.out_dir, args.images_folder)
    n_orig = 0
    for f in sorted(os.listdir(src_images)):
        s, d = os.path.join(src_images, f), os.path.join(dst_images, f)
        if not os.path.exists(d):
            shutil.copy2(s, d)
        n_orig += 1
    print(f"🖼  copied/linked {n_orig} original images")

    # 注入：每视角独立相机（fx 按视角距离自适应，可能不同）
    cam_id = max(rec.cameras) + 1
    img_id = max(rec.images) + 1
    n_inj = 0
    for pid, views in sorted(poses.items()):
        enh_dir = os.path.join(args.enhanced_root, f"enhanced_{pid}", "images")
        if not os.path.isdir(enh_dir):
            print(f"  ⚠️ {pid}: enhanced dir missing: {enh_dir}, skip")
            continue
        for cv in views:
            src_png = os.path.join(enh_dir, cv["name"])
            if not os.path.exists(src_png):
                print(f"  ⚠️ missing enhanced image: {src_png}, skip")
                continue
            flat_name = f"closeup_{pid}_{cv['name']}"  # closeup_p00_0001.png
            dst_png = os.path.join(dst_images, flat_name)
            shutil.copy2(src_png, dst_png)

            cam = pycolmap.Camera(
                model="PINHOLE", width=cv["W"], height=cv["H"],
                params=[cv["fx"], cv["fy"], cv["cx"], cv["cy"]],
                camera_id=cam_id)
            rec.add_camera_with_trivial_rig(cam)

            img = pycolmap.Image(name=flat_name, camera_id=cam_id, image_id=img_id)
            cam_from_world = pycolmap.Rigid3d(
                pycolmap.Rotation3d(np.asarray(cv["R"], dtype=np.float64)),
                np.asarray(cv["T"], dtype=np.float64))
            rec.add_image_with_trivial_frame(img, cam_from_world)

            cam_id += 1
            img_id += 1
            n_inj += 1
        print(f"  ✅ {pid}: injected views, total now {n_inj}")

    rec.write_binary(out_sparse)
    print(f"💾 wrote {out_sparse}: {len(rec.cameras)} cams / {len(rec.images)} imgs "
          f"(+{len(rec.cameras)-n_cam0} cams, +{len(rec.images)-n_img0} imgs)")

    report["out_dir"] = args.out_dir
    report["injected"] = n_inj
    report["total_images"] = len(rec.images)
    with open(os.path.join(args.out_dir, "inject_report.json"), "w") as fh:
        json.dump(report, fh, indent=2, ensure_ascii=False)
    print(f"📄 report: {os.path.join(args.out_dir, 'inject_report.json')}")


if __name__ == "__main__":
    main()

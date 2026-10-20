"""按帧序裁剪场景：只保留前 KEEP 帧（垂手段），重建不含抬手段的子场景。

输入: colmap_refined/{sparse_bin/0, persons/pXX}
输出: colmap_trim{KEEP}/{images/, sparse/0/, persons/pXX/masks/, persons/pXX/person.json}
"""
import os
import shutil
import struct
import glob
import numpy as np

ROOT = "/home/velvet/output/som_human_results/hand10_shuangren_zhaoshou"
SRC_SPARSE = os.path.join(ROOT, "colmap_refined", "sparse_bin", "0")
SRC_IMAGES = os.path.join(ROOT, "colmap_scene", "images")
KEEP = 105
PIDS = ["p07"]  # 只裁 p07 验证


def read_images_bin(path):
    out = []
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        for _ in range(n):
            iid = struct.unpack("<i", f.read(4))[0]
            q = struct.unpack("<4d", f.read(32))
            t = struct.unpack("<3d", f.read(24))
            cid = struct.unpack("<i", f.read(4))[0]
            name = b""
            while True:
                c = f.read(1)
                if c == b"\x00":
                    break
                name += c
            npts = struct.unpack("<Q", f.read(8))[0]
            pts = f.read(24 * npts)
            out.append((iid, q, t, cid, name.decode(), npts, pts))
    return out


def write_images_bin(path, recs):
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(recs)))
        for iid, q, t, cid, name, npts, pts in recs:
            f.write(struct.pack("<i", iid))
            f.write(struct.pack("<4d", *q))
            f.write(struct.pack("<3d", *t))
            f.write(struct.pack("<i", cid))
            f.write(name.encode() + b"\x00")
            f.write(struct.pack("<Q", npts))
            f.write(pts)


def main():
    dst_root = os.path.join(ROOT, f"colmap_trim{KEEP}")
    os.makedirs(dst_root, exist_ok=True)
    recs = read_images_bin(os.path.join(SRC_SPARSE, "images.bin"))
    recs_sorted = sorted(recs, key=lambda r: r[4])
    keep = recs_sorted[:KEEP]
    keep_names = [r[4] for r in keep]
    print(f"总相机 {len(recs)}，保留 {len(keep)} 帧，首/末: {keep_names[0]} / {keep_names[-1]}")

    # sparse
    dsparse = os.path.join(dst_root, "sparse", "0")
    os.makedirs(dsparse, exist_ok=True)
    write_images_bin(os.path.join(dsparse, "images.bin"), keep)
    for fn in ("cameras.bin", "points3D.bin"):
        shutil.copy2(os.path.join(SRC_SPARSE, fn), os.path.join(dsparse, fn))
    print("sparse 已写")

    # images 软链
    dimg = os.path.join(dst_root, "images")
    os.makedirs(dimg, exist_ok=True)
    n = 0
    for nm in keep_names:
        src = os.path.join(SRC_IMAGES, nm)
        if not os.path.exists(src):
            # 找任意扩展
            cand = glob.glob(os.path.join(SRC_IMAGES, nm.rsplit(".", 1)[0] + ".*"))
            if cand:
                src = cand[0]
                nm = os.path.basename(src)
            else:
                continue
        dst = os.path.join(dimg, nm)
        if not os.path.lexists(dst):
            os.symlink(src, dst)
        n += 1
    print(f"images 软链 {n}")

    # persons
    for pid in PIDS:
        psr = os.path.join(ROOT, "colmap_refined", "persons", pid)
        pds = os.path.join(dst_root, "persons", pid)
        os.makedirs(pds, exist_ok=True)
        # masks 软链前 KEEP 帧
        dmk = os.path.join(pds, "masks")
        os.makedirs(dmk, exist_ok=True)
        m = 0
        for nm in keep_names:
            stem = nm.rsplit(".", 1)[0]
            src = os.path.join(psr, "masks", stem + ".png")
            if not os.path.exists(src):
                continue
            dst = os.path.join(dmk, stem + ".png")
            if not os.path.lexists(dst):
                os.symlink(src, dst)
            m += 1
        print(f"{pid} masks 软链 {m}")
        pj = os.path.join(psr, "person.json")
        if os.path.exists(pj):
            shutil.copy2(pj, os.path.join(pds, "person.json"))

    print(f"DONE -> {dst_root}")


if __name__ == "__main__":
    main()

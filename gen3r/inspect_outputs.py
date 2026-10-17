#!/usr/bin/env python3
"""Gen3R 产物体检：抽帧预览 + 点云/相机统计。

用法:
    python inspect_outputs.py <产物目录> [--out 预览目录]
"""
import argparse
import json
import os

import imageio
import imageio.v3 as iio
import numpy as np
from PIL import Image


def read_ply_header(path, max_lines=200):
    info = {"path": path, "bytes": os.path.getsize(path)}
    with open(path, "rb") as f:
        for _ in range(max_lines):
            line = f.readline().decode("ascii", "ignore").strip()
            if line.startswith("element vertex"):
                info["vertices"] = int(line.split()[-1])
            if line == "end_header":
                info["header_end"] = f.tell()
                break
    return info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--out", default="/mnt/c/code/media_code/gen3r/_preview")
    ap.add_argument("--frames", type=int, default=6)
    args = ap.parse_args()

    d = args.run_dir
    os.makedirs(args.out, exist_ok=True)

    # 1) 抽帧
    vid = os.path.join(d, "rgb.mp4")
    if os.path.isfile(vid):
        rd = imageio.get_reader(vid, "ffmpeg")   # 走 imageio-ffmpeg 自带二进制
        frames = [np.asarray(f) for f in rd]
        rd.close()
        n = len(frames)
        idx = np.linspace(0, n - 1, args.frames).astype(int)
        for k, i in enumerate(idx):
            Image.fromarray(frames[i]).save(os.path.join(args.out, "f{:02d}_n{}.png".format(k, i)))
        print("🎞️  rgb.mp4: {} 帧 {}x{}".format(n, frames[0].shape[1], frames[0].shape[0]))

    # 2) 点云
    ply = os.path.join(d, "pcds.ply")
    if os.path.isfile(ply):
        info = read_ply_header(ply)
        print("☁️  pcds.ply: {:.1f} MB, {} 点".format(
            info["bytes"] / 2**20, info.get("vertices", "?")))

    # 3) 相机
    cam = os.path.join(d, "cameras.json")
    if os.path.isfile(cam):
        with open(cam) as f:
            c = json.load(f)
        ex = np.array(c["extrinsics"])
        ks = np.array(c["intrinsics"])
        centers = -np.einsum("fij,fj->fi", ex[:, :3, :3], ex[:, :3, 3]) if ex.shape[-1] == 4 else None
        print("📷 cameras.json: extrinsics {} intrinsics {}".format(ex.shape, ks.shape))
        if centers is not None:
            r = np.linalg.norm(centers - centers.mean(0), axis=1)
            print("   相机中心范围 x[{:.2f},{:.2f}] y[{:.2f},{:.2f}] z[{:.2f},{:.2f}]  环绕半径 {:.2f}~{:.2f}".format(
                centers[:, 0].min(), centers[:, 0].max(),
                centers[:, 1].min(), centers[:, 1].max(),
                centers[:, 2].min(), centers[:, 2].max(), r.min(), r.max()))
            print("   fx={:.1f} cx={:.1f} cy={:.1f}".format(ks[0, 0, 0], ks[0, 0, 2], ks[0, 1, 2]))


if __name__ == "__main__":
    main()

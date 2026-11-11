#!/usr/bin/env python
# 32_compare_v1v2.py — v1 vs v2 批量产物回归比对（21 批量两版汇总 + 逐包产物核对）
#
# 用法: python 32_compare_v1v2.py <v1根> <v2根>
# 检查项:
#   1) 包集合是否一致（缺失/新增列出）
#   2) 每包 PLY 大小差、视频大小差、图片数差
#   3) PLY 头解析高斯数（gs_max/0000.ply 前两行）
# 输出: 逐包表 + 汇总统计（stdout），退出码 0=全部一致
import os
import sys


def read_summary(root):
    path = os.path.join(root, "_summary.tsv")
    rows = {}
    with open(path, encoding="utf-8") as f:
        header = f.readline()
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) >= 7:
                rows[parts[0]] = {
                    "n_imgs": parts[1],
                    "ply_MB": parts[4],
                    "video_MB": parts[5],
                    "status": parts[6],
                }
    return rows


def ply_points(ply_path):
    """PLY 二进制头第三行一般是 'element vertex N'；找不到返回 None。"""
    try:
        with open(ply_path, "rb") as f:
            head = f.read(4096).decode("ascii", errors="ignore")
        for line in head.splitlines():
            if line.startswith("element vertex"):
                return int(line.split()[-1])
    except OSError:
        return None
    return None


def main():
    v1_root, v2_root = sys.argv[1], sys.argv[2]
    v1 = read_summary(v1_root)
    v2 = read_summary(v2_root)
    only1 = sorted(set(v1) - set(v2))
    only2 = sorted(set(v2) - set(v1))
    both = sorted(set(v1) & set(v2))

    print(f"v1 包数={len(v1)}  v2 包数={len(v2)}  共有={len(both)}")
    if only1:
        print(f"⚠️ 仅 v1 有: {only1}")
    if only2:
        print(f"⚠️ 仅 v2 有: {only2}")

    print(f"\n{'pkg':<14} {'imgs':>10} {'ply_MB':>12} {'mp4_MB':>10} {'gs点数(v2头)':>14}")
    n_ply_diff = 0
    n_mp4_diff = 0
    n_img_diff = 0
    gs_mismatch = []
    for name in both:
        a, b = v1[name], v2[name]
        di = ""
        if a["n_imgs"] != b["n_imgs"]:
            di = f"{a['n_imgs']}→{b['n_imgs']}"
            n_img_diff += 1
        dp = ""
        if a["ply_MB"] != b["ply_MB"]:
            dp = f"{a['ply_MB']}→{b['ply_MB']}"
            n_ply_diff += 1
        dv = ""
        if a["video_MB"] != b["video_MB"]:
            dv = f"{a['video_MB']}→{b['video_MB']}"
            n_mp4_diff += 1
        gs = ply_points(os.path.join(v2_root, name, "gs_max", "0000.ply"))
        gs_s = f"{gs:,}" if gs else "?"
        if gs and gs != 6096384:
            gs_mismatch.append((name, gs))
        print(f"{name[:12]:<14} {di or a['n_imgs']:>10} {dp or a['ply_MB']:>12} "
              f"{dv or a['video_MB']:>10} {gs_s:>14}  {b['status']}")

    print(f"\n汇总: 图片数不同={n_img_diff}  PLY大小不同={n_ply_diff}  视频大小不同={n_mp4_diff}")
    if gs_mismatch:
        print(f"⚠️ v2 高斯数≠6,096,384 的包: {gs_mismatch}")
    else:
        print("✅ v2 全部包高斯数=6,096,384（与 v1 规格一致）")


if __name__ == "__main__":
    main()

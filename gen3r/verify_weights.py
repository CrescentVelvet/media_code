#!/usr/bin/env python3
"""校验 Gen3R 权重是否齐全（大小精确匹配 HF tree API 的记录值）。

用法：
    python verify_weights.py --root ~/model/gen3r            # 只校验，可选 --link-into
    python verify_weights.py --root /mnt/d/wheel/gen3r_ms --link-into ~/model/gen3r

目录结构容错：优先按相对路径找；找不到就按**文件名**全盘搜（迅雷批量下载常把
text_encoder/xxx 摊平成 xxx 放在根目录），命中即视为已下好。
--link-into 会把找到的文件统一整理成上游要的相对路径结构（软链接，不占空间）。

退出码 0 = 齐全；1 = 有缺失或大小不符（逐条打印）。
"""
import argparse
import fnmatch
import os
import sys

# 与 download_models.py 共用同一份清单定义
from download_models import FILES


def _walk_index(root):
    """root 下所有文件的 basename -> [绝对路径]（一次遍历，避免重复 os.walk）。"""
    index = {}
    for dirpath, _dirs, files in os.walk(root):
        for f in files:
            index.setdefault(f, []).append(os.path.join(dirpath, f))
    return index


def resolve(root, path, index=None):
    """返回该清单项在 root 下的实际文件路径；找不到返回 None。"""
    p = os.path.join(root, path)
    if os.path.isfile(p):
        return p
    if index is None:
        index = _walk_index(root)
    base = os.path.basename(path)
    if base in index:
        return index[base][0]
    # 迅雷重名后缀："xxx (1).safetensors" / "xxx(1).safetensors"
    stem, ext = os.path.splitext(base)
    for pat in (stem + " (*)" + ext, stem + "(*)" + ext):
        for name, paths in index.items():
            if fnmatch.fnmatch(name, pat):
                return paths[0]
    return None


def link_into(root, dest, index=None):
    """把 root 下找到的文件按相对路径挂进 dest（软链接），已存在且大小对的跳过。"""
    if index is None:
        index = _walk_index(root)
    made, skipped = 0, 0
    for path, size in FILES:
        src = resolve(root, path, index)
        if src is None:
            continue
        dst = os.path.join(dest, path)
        if os.path.lexists(dst):
            if os.path.exists(dst) and os.path.getsize(dst) == size:
                skipped += 1
                continue
            os.unlink(dst)  # 只摘链接/坏文件，不动 src
        os.makedirs(os.path.dirname(dst) or dest, exist_ok=True)
        os.symlink(os.path.abspath(src), dst)
        made += 1
    return made, skipped


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=os.path.expanduser("~/model/gen3r"))
    ap.add_argument("--allow-extra", action="store_true", help="允许额外文件（默认就允许，占位）")
    ap.add_argument("--link-into", default="", help="把找到的文件按相对路径软链到该目录")
    args = ap.parse_args()

    if not os.path.isdir(args.root):
        print("❌ {} 不存在".format(args.root))
        sys.exit(1)

    index = _walk_index(args.root)
    bad = []
    total = 0
    flat = []
    for path, size in FILES:
        src = resolve(args.root, path, index)
        if src is None:
            bad.append((path, "missing", size))
            continue
        got = os.path.getsize(src)
        if got != size:
            bad.append((path, "size {} != {}".format(got, size), size))
        else:
            total += size
            if os.path.normpath(src) != os.path.normpath(os.path.join(args.root, path)):
                flat.append((path, src))

    print("✅ {}/{} 文件，{:.2f} GB".format(len(FILES) - len(bad), len(FILES), total / 2**30))
    if flat:
        print("ℹ️  {} 个文件不在标准相对路径下（已按文件名匹配到）：".format(len(flat)))
        for path, src in flat[:5]:
            print("     {:<55s} <- {}".format(path, src))
        if len(flat) > 5:
            print("     ... 其余 {} 个略".format(len(flat) - 5))

    if bad:
        print("❌ 有问题：")
        for path, why, size in bad:
            print("   {:<70s} {}".format(path, why))
        sys.exit(1)

    if args.link_into:
        made, skipped = link_into(args.root, args.link_into, index)
        print("🔗 已整理到 {}：新建 {} 个软链，跳过 {} 个".format(args.link_into, made, skipped))


if __name__ == "__main__":
    main()

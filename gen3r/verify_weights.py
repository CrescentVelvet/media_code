#!/usr/bin/env python3
"""校验 Gen3R 权重是否齐全（大小精确匹配 HF tree API 的记录值）。

用法：
    python verify_weights.py --root ~/model/gen3r
    python verify_weights.py --root /mnt/d/wheel/gen3r_ms --link-into ~/model/gen3r

目录结构容错：优先按相对路径找；找不到就按**文件名**全盘搜（迅雷批量下载常把
text_encoder/xxx 摊平成 xxx 放在根目录），命中即视为已下好。
⚠️ 多个子目录有同名文件（如 geo_adapter/transformer/wan_vae/clip_image_encoder 的
   diffusion_pytorch_model.safetensors），所以扁平匹配必须**同时比对大小**并去重认领，
   否则会张冠李戴（曾把 wan_vae 的 507MB 当成 clip 的 4.44GB 报 size 不符）。

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
    """root 下所有文件的 basename -> [绝对路径]（跳过 *.parts 分块目录）。"""
    index = {}
    for dirpath, dirs, files in os.walk(root):
        dirs[:] = [d for d in dirs if not d.endswith(".parts")]
        for f in files:
            index.setdefault(f, []).append(os.path.join(dirpath, f))
    return index


def resolve(root, path, size=None, index=None, claimed=None):
    """返回该清单项在 root 下的实际文件路径；找不到返回 None。

    size 给了就只认大小完全一致的候选（扁平匹配时防同名误配）。
    claimed 是已认领文件集合，避免一个文件被两个清单项重复使用。
    """
    p = os.path.join(root, path)
    if os.path.isfile(p):
        return p
    if index is None:
        index = _walk_index(root)
    base = os.path.basename(path)

    def ok(c):
        if claimed is not None and c in claimed:
            return False
        return True if size is None else os.path.getsize(c) == size

    cands = [c for c in index.get(base, []) if ok(c)]
    if cands:
        return cands[0]
    # 迅雷重名后缀："xxx (1).safetensors" / "xxx(1).safetensors"
    stem, ext = os.path.splitext(base)
    for pat in (stem + " (*)" + ext, stem + "(*)" + ext):
        for name, paths in index.items():
            if fnmatch.fnmatch(name, pat):
                cands = [c for c in paths if ok(c)]
                if cands:
                    return cands[0]
    return None


def link_into(root, dest, index=None):
    """把 root 下找到的文件按相对路径挂进 dest（软链接），已存在且大小对的跳过。"""
    if index is None:
        index = _walk_index(root)
    claimed = set()
    made, skipped = 0, 0
    for path, size in FILES:
        src = resolve(root, path, size, index, claimed)
        if src is None:
            continue
        claimed.add(src)
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
    claimed = set()
    bad = []
    total = 0
    flat = []
    for path, size in FILES:
        src = resolve(args.root, path, size, index, claimed)
        if src is None:
            # 精确路径存在但大小不对 → 报 size 问题，更利于定位半截文件
            exact = os.path.join(args.root, path)
            if os.path.isfile(exact):
                bad.append((path, "size {} != {}".format(os.path.getsize(exact), size), size))
            else:
                bad.append((path, "missing", size))
            continue
        claimed.add(src)
        got = os.path.getsize(src)
        if got != size:
            bad.append((path, "size {} != {}".format(got, size), size))
        else:
            total += size
            if os.path.normpath(src) != os.path.normpath(os.path.join(args.root, path)):
                flat.append((path, src))

    print("✅ {}/{} 文件，{:.2f} GB".format(len(FILES) - len(bad), len(FILES), total / 2**30))
    if flat:
        print("ℹ️  {} 个文件不在标准相对路径下（已按文件名+大小匹配到）：".format(len(flat)))
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

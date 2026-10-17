#!/usr/bin/env python3
"""校验 Gen3R 权重是否齐全（大小精确匹配 HF tree API 的记录值）。

用法：python verify_weights.py --root ~/model/gen3r [--allow-extra]

退出码 0 = 齐全；1 = 有缺失或大小不符（逐条打印）。
"""
import argparse
import os
import sys

# 与 download_models.py 共用同一份清单定义
from download_models import FILES


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=os.path.expanduser("~/model/gen3r"))
    ap.add_argument("--allow-extra", action="store_true", help="允许额外文件（默认就允许，占位）")
    args = ap.parse_args()

    bad = []
    total = 0
    for path, size in FILES:
        p = os.path.join(args.root, path)
        if not os.path.exists(p):
            bad.append((path, "missing", size))
            continue
        got = os.path.getsize(p)
        if got != size:
            bad.append((path, "size {} != {}".format(got, size), size))
        else:
            total += size

    print("✅ {}/{} 文件，{:.2f} GB".format(len(FILES) - len(bad), len(FILES), total / 2**30))
    if bad:
        print("❌ 有问题：")
        for path, why, size in bad:
            print("   {:<70s} {}".format(path, why))
        sys.exit(1)


if __name__ == "__main__":
    main()

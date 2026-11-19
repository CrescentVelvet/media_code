#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
并行分块下载器（断点续传 + 并发 range）。

用途：HF 单流经代理仅 ~90 KB/s，实测 8 并发 696 KB/s、16 并发 1063 KB/s。
本脚本把大文件切成固定大小的块并发下载，已完成的块落盘为 .part.N 分块文件，
全部完成后按序拼接成目标文件。中断后重跑自动跳过已完成块。

用法:
  python 00c_par_download.py <url> <out_path> [--chunk-mb 8] [--workers 16] [--expect-size N]

设计要点:
  - 分块文件命名 <out>.part/000123，用「文件大小 == 块大小」判定完成，避免半块误判。
  - 最后一块按 expect_size 精确计算，短了视为未完成 → 重下。
  - 完全不用 requests，只用 curl（复用 proxy.env 的 http_proxy/https_proxy 环境变量）。
  - 拼接用 shutil.copyfileobj，最后 os.replace 原子落盘。
"""

import argparse
import os
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

MB = 1024 * 1024


def human(n):
    for u in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024 or u == "TB":
            return f"{n:.1f}{u}" if u != "B" else f"{n}{u}"
        n /= 1024.0
    return f"{n:.1f}TB"


def head_size(url, timeout=60):
    """取 Content-Length（跟随重定向）。"""
    p = subprocess.run(
        ["curl", "-sIL", "-m", str(timeout), url],
        capture_output=True, text=True, errors="replace",
    )
    size = None
    for line in p.stdout.splitlines():
        k, _, v = line.partition(":")
        if k.strip().lower() == "content-length":
            try:
                n = int(v.strip())
            except ValueError:
                continue
            # 302 的 content-length 很小，只认最大的那个（= 真实文件）
            if size is None or n > size:
                size = n
    return size


def chunk_ok(path, want):
    try:
        return os.path.getsize(path) == want
    except OSError:
        return False


def fetch_chunk(url, path, start, end, retries=8):
    """下载 [start, end] 闭区间到 path。成功返回 (idx, bytes, None)。"""
    want = end - start + 1
    if chunk_ok(path, want):
        return (path, want, None)
    tmp = path + ".tmp"
    last_err = ""
    for attempt in range(retries):
        if os.path.exists(tmp):
            os.remove(tmp)
        cmd = [
            "curl", "-sL", "-f", "--max-time", "900",
            "-r", f"{start}-{end}",
            "--retry", "3", "--retry-delay", "2", "--retry-all-errors",
            "-o", tmp, url,
        ]
        p = subprocess.run(cmd, capture_output=True, text=True, errors="replace")
        if p.returncode == 0 and os.path.exists(tmp) and os.path.getsize(tmp) == want:
            os.replace(tmp, path)
            return (path, want, None)
        got = os.path.getsize(tmp) if os.path.exists(tmp) else 0
        last_err = f"rc={p.returncode} got={got}/{want} {p.stderr.strip()[:120]}"
        time.sleep(min(2 ** attempt, 20))
    return (path, 0, last_err)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("url")
    ap.add_argument("out")
    ap.add_argument("--chunk-mb", type=int, default=8)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--expect-size", type=int, default=0)
    ap.add_argument("--resume-only", action="store_true",
                    help="只下载缺失块，不拼接（用于分批推进）")
    args = ap.parse_args()

    url, out = args.url, os.path.abspath(args.out)
    total = args.expect_size or head_size(url)
    if not total:
        print("❌ 无法取得文件总大小（--expect-size 可手工指定）", flush=True)
        return 2
    print(f"目标: {out}", flush=True)
    print(f"总大小: {total:,} bytes ({human(total)})", flush=True)

    if os.path.exists(out) and os.path.getsize(out) == total:
        print("✅ 目标文件已完整，跳过", flush=True)
        return 0

    csize = args.chunk_mb * MB
    nchunk = (total + csize - 1) // csize
    parts_dir = out + ".part"
    os.makedirs(parts_dir, exist_ok=True)

    jobs = []
    for i in range(nchunk):
        s = i * csize
        e = min(s + csize - 1, total - 1)
        jobs.append((i, os.path.join(parts_dir, f"{i:06d}"), s, e))

    done_bytes = sum(os.path.getsize(p) for _, p, _, _ in jobs if os.path.exists(p))
    print(f"分块: {nchunk} 块 × {args.chunk_mb}MB, 并发 {args.workers}", flush=True)
    print(f"已有: {done_bytes:,} / {total:,} ({done_bytes*100.0/total:.1f}%)", flush=True)

    t0 = time.time()
    base = done_bytes
    ctr = {"n": 0}
    fails = []

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(fetch_chunk, url, p, s, e): i for i, p, s, e in jobs}
        for fu in as_completed(futs):
            path, got, err = fu.result()
            ctr["n"] += 1
            if err:
                fails.append((path, err))
            if ctr["n"] % 12 == 0 or ctr["n"] == nchunk:
                cur = sum(os.path.getsize(p) for _, p, _, _ in jobs if os.path.exists(p))
                el = max(time.time() - t0, 1e-6)
                spd = (cur - base) / el
                eta = (total - cur) / spd if spd > 1 else 0
                print(f"  [{ctr['n']}/{nchunk}] {cur*100.0/total:5.1f}%  "
                      f"{human(cur)}/{human(total)}  {human(spd)}/s  "
                      f"ETA {eta/60:.1f}min  fail={len(fails)}", flush=True)

    if fails:
        print(f"⚠️ {len(fails)} 块失败，重跑本命令会自动续传。前 3 条:", flush=True)
        for p, e in fails[:3]:
            print(f"  {os.path.basename(p)}: {e}", flush=True)
        if args.resume_only:
            return 1

    if args.resume_only:
        print("✅ resume-only 结束", flush=True)
        return 0

    # 拼接校验
    missing = [p for _, p, s, e in jobs if not chunk_ok(p, e - s + 1)]
    if missing:
        print(f"❌ 仍有 {len(missing)} 块不完整，不拼接。重跑续传。", flush=True)
        return 1

    print("拼接中...", flush=True)
    tmp_out = out + ".assembling"
    with open(tmp_out, "wb") as fo:
        for i, p, s, e in jobs:
            with open(p, "rb") as fi:
                shutil.copyfileobj(fi, fo, 4 * MB)
    got = os.path.getsize(tmp_out)
    if got != total:
        print(f"❌ 拼接后大小不符: {got:,} != {total:,}", flush=True)
        os.remove(tmp_out)
        return 1
    os.replace(tmp_out, out)
    shutil.rmtree(parts_dir, ignore_errors=True)
    print(f"🎉 完成: {out}  {got:,} bytes  耗时 {(time.time()-t0)/60:.1f}min", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""dl_parallel.py — 多线程 Range 分块下载（针对单连接被限速 ~20KB/s 的海外源）。

用法:
    python dl_parallel.py <URL> <输出路径> [--jobs N] [--chunk-mb N]

断点续传：已完成的块记 <输出路径>.chunks，重跑跳过已完成块。
2026-11-09 实测：naverlabs 单连接 ~22KB/s，32 连接 ~700KB/s。
"""
import argparse
import os
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor

p = argparse.ArgumentParser()
p.add_argument("url")
p.add_argument("out")
p.add_argument("--jobs", type=int, default=32)
p.add_argument("--chunk-mb", type=int, default=8)
a = p.parse_args()

CHUNK = a.chunk_mb * 1024 * 1024
req = urllib.request.Request(a.url, method="HEAD")
with urllib.request.urlopen(req, timeout=30) as r:
    total = int(r.headers["Content-Length"])
n_chunks = (total + CHUNK - 1) // CHUNK
done_file = a.out + ".chunks"
done = set()
if os.path.exists(done_file):
    with open(done_file, encoding="utf-8") as f:
        done = {int(x) for x in f.read().split() if x}
print(f"total={total} ({total/1e9:.2f} GB) chunks={n_chunks} done={len(done)}", flush=True)


def fetch(i):
    if i in done:
        return i, True
    lo, hi = i * CHUNK, min((i + 1) * CHUNK, total) - 1
    for attempt in range(8):
        try:
            rq = urllib.request.Request(a.url, headers={"Range": f"bytes={lo}-{hi}"})
            with urllib.request.urlopen(rq, timeout=120) as r:
                buf = r.read()
            if len(buf) != hi - lo + 1:
                raise IOError(f"short read {len(buf)} != {hi-lo+1}")
            with open(a.out, "r+b") as f:
                f.seek(lo)
                f.write(buf)
            return i, True
        except Exception as e:
            if attempt == 7:
                print(f"chunk {i} FAIL: {e}", flush=True)
                return i, False
            time.sleep(2 * (attempt + 1))


if not os.path.exists(a.out):
    with open(a.out, "wb") as f:
        f.truncate(total)

t0 = time.time()
ok = 0
with ThreadPoolExecutor(max_workers=a.jobs) as ex:
    for i, success in ex.map(fetch, range(n_chunks)):
        if success:
            ok += 1
            done.add(i)
            if ok % 50 == 0:
                with open(done_file, "w", encoding="utf-8") as f:
                    f.write(" ".join(map(str, done)))
                el = time.time() - t0
                sp = ok * CHUNK / el / 1024
                print(f"{ok}/{n_chunks} {sp:.0f} KB/s eta={((n_chunks-ok)*CHUNK/sp/1024/60):.0f}min", flush=True)

with open(done_file, "w", encoding="utf-8") as f:
    f.write(" ".join(map(str, done)))
if ok == n_chunks:
    os.remove(done_file)
    print(f"OK all {n_chunks} chunks in {(time.time()-t0)/60:.1f} min", flush=True)
    sys.exit(0)
print(f"INCOMPLETE {ok}/{n_chunks}，重跑续传", flush=True)
sys.exit(1)

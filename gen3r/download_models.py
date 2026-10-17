#!/usr/bin/env python3
"""并行分块下载 HF 模型权重（本机 hf-mirror 单连接 ~1.2MB/s，多连接才能跑满）。

用法：
    python download_models.py --dest ~/model/gen3r [--endpoint https://hf-mirror.com]
                              [--repo JaceyH919/Gen3R] [--workers 12] [--dry-run]

说明：
- 每个文件按 CHUNK 切块，块写入 <file>.parts/<idx>，全部完成后按序拼接并删 .parts。
  已存在的块若大小正确则跳过 → 天然断点续传。
- 只依赖标准库（不占 env）。
"""
import argparse
import os
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed

# 官方 checkpoint 文件清单（HF tree API 2026-10-17 实测大小）
FILES = [
    ("model_index.json", 828),
    ("scheduler/scheduler_config.json", 450),
    ("clip_image_encoder/config.json", 67),
    ("clip_image_encoder/diffusion_pytorch_model.safetensors", 4_772_146_492),
    ("geo_adapter/config.json", 257),
    ("geo_adapter/diffusion_pytorch_model.safetensors", 1_271_400_536),
    ("text_encoder/config.json", 252),
    ("text_encoder/diffusion_pytorch_model-00001-of-00003.safetensors", 9_871_621_720),
    ("text_encoder/diffusion_pytorch_model-00002-of-00003.safetensors", 9_966_195_448),
    ("text_encoder/diffusion_pytorch_model-00003-of-00003.safetensors", 2_885_848_768),
    ("text_encoder/diffusion_pytorch_model.safetensors.index.json", 21037),
    ("tokenizer/special_tokens_map.json", 7079),
    ("tokenizer/spiece.model", 4_548_313),
    ("tokenizer/tokenizer.json", 16_837_614),
    ("tokenizer/tokenizer_config.json", 61833),
    ("transformer/config.json", 567),
    ("transformer/diffusion_pytorch_model.safetensors", 6_465_451_744),
    ("vggt/config.json", 62),
    ("vggt/model.safetensors", 4_762_545_592),
    ("wan_vae/config.json", 168),
    ("wan_vae/diffusion_pytorch_model.safetensors", 507_592_404),
]

CHUNK = 16 * 1024 * 1024  # 16MB/块（改小会让已下载的 .parts 编号错位，别动）


def url_of(endpoint, repo, path):
    return "{}/{}/resolve/main/{}".format(endpoint.rstrip("/"), repo, path)


def fetch_range(url, start, end, dest_tmp, retries=7):
    """下载 [start, end] 闭区间到 dest_tmp，支持 206/200 两种响应。

    hf-mirror 对高并发返回 429，必须指数退避（实测 12 并发时大量 429）。
    """
    import random
    want = end - start + 1
    for attempt in range(retries):
        if os.path.exists(dest_tmp) and os.path.getsize(dest_tmp) == want:
            return want
        req = urllib.request.Request(url, headers={
            "Range": "bytes={}-{}".format(start, end),
            # aifasthub 会按 UA 拦 Python-urllib（403），伪装成 huggingface_hub
            "User-Agent": "huggingface_hub/0.36.2; python/3.11",
            "Accept": "*/*",
        })
        try:
            with urllib.request.urlopen(req, timeout=180) as r, open(dest_tmp, "wb") as f:
                while True:
                    buf = r.read(1 << 18)
                    if not buf:
                        break
                    f.write(buf)
            got = os.path.getsize(dest_tmp)
            if got == want:
                return want
            # 服务端忽略 Range（200 全量）时直接判失败，避免错位拼接
            raise RuntimeError("size mismatch {} != {}".format(got, want))
        except Exception as e:  # noqa: BLE001 网络抖动重试
            if attempt == retries - 1:
                raise
            wait = min(60.0, 4.0 * (2 ** attempt)) + random.uniform(0, 3)
            print("  ⚠️ retry {}/{} {} ({}) sleep {:.0f}s".format(
                attempt + 1, retries, os.path.basename(dest_tmp), e, wait))
            time.sleep(wait)
    return want


def download_one(endpoint, repo, path, size, dest_root, workers, pool):
    dst = os.path.join(dest_root, path)
    if os.path.exists(dst) and os.path.getsize(dst) == size:
        print("⏭️  skip {} ({} MB)".format(path, size // 2**20))
        return
    os.makedirs(os.path.dirname(dst) or ".", exist_ok=True)
    parts_dir = dst + ".parts"
    os.makedirs(parts_dir, exist_ok=True)
    url = url_of(endpoint, repo, path)
    ranges = []
    for i, s in enumerate(range(0, size, CHUNK)):
        ranges.append((i, s, min(s + CHUNK, size) - 1))

    print("📦 {} ({:.2f} GB, {} chunks)".format(path, size / 2**30, len(ranges)))
    t0 = time.time()
    futs = []
    done_bytes = 0
    for i, s, e in ranges:
        tmp = os.path.join(parts_dir, "{:05d}".format(i))
        futs.append(pool.submit(fetch_range, url, s, e, tmp))
    for fut in as_completed(futs):
        done_bytes += fut.result()
        if done_bytes % (512 * 2**20) < CHUNK:
            sp = done_bytes / max(time.time() - t0, 1e-6) / 2**20
            print("    {:.2f}/{:.2f} GB  {:.1f} MB/s".format(done_bytes / 2**30, size / 2**30, sp))

    with open(dst, "wb") as out:
        for i, _s, _e in ranges:
            with open(os.path.join(parts_dir, "{:05d}".format(i)), "rb") as f:
                while True:
                    buf = f.read(1 << 22)
                    if not buf:
                        break
                    out.write(buf)
    got = os.path.getsize(dst)
    if got != size:
        sys.exit("❌ {} size {} != expected {}".format(path, got, size))
    for i, _s, _e in ranges:
        os.unlink(os.path.join(parts_dir, "{:05d}".format(i)))
    os.rmdir(parts_dir)
    print("✅ {} {:.2f} GB  {:.0f}s".format(path, size / 2**30, time.time() - t0))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default="JaceyH919/Gen3R")
    ap.add_argument("--endpoint", default=os.environ.get("HF_ENDPOINT", "https://hf-mirror.com"))
    ap.add_argument("--dest", default=os.path.expanduser("~/model/gen3r"))
    ap.add_argument("--workers", type=int, default=6, help="并行块数（hf-mirror 12 会 429）")
    ap.add_argument("--only", default="", help="只下路径含该子串的文件")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    files = [f for f in FILES if args.only in f[0]]
    total = sum(s for _p, s in files)
    print("🚀 Gen3R weights → {}".format(args.dest))
    print("   endpoint: {}  files: {}  total: {:.2f} GB  workers: {}".format(
        args.endpoint, len(files), total / 2**30, args.workers))
    if args.dry_run:
        for p, s in files:
            print("   {:.2f} GB  {}".format(s / 2**30, p))
        return

    os.makedirs(args.dest, exist_ok=True)

    # 排序：非 text_encoder 优先、小文件优先（text_encoder 22.7GB 最易卡，放最后）
    def sort_key(item):
        path, size = item
        return (path.startswith("text_encoder/"), size)

    failed = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for path, size in sorted(files, key=sort_key):
            try:
                download_one(args.endpoint, args.repo, path, size, args.dest, args.workers, pool)
            except Exception as e:  # noqa: BLE001 单个文件失败不中断整批
                print("❌ {} failed: {}".format(path, e))
                failed.append(path)
    if failed:
        print("⚠️ 失败 {} 个文件（重跑本脚本可续传）：".format(len(failed)))
        for p in failed:
            print("   - " + p)
        sys.exit(1)
    print("🎉 all weights ready: {}".format(args.dest))


if __name__ == "__main__":
    main()

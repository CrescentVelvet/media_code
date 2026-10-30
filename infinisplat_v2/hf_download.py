#!/usr/bin/env python3
"""并发分块下载 HF 文件（支持镜像站 + 断点续传 + SHA256 校验）。

官方 huggingface.co 与 hf-mirror.com 在本机均不可达，实测可用的是国内镜像
aifasthub.com（HF 兼容 API，支持 Range）。HF 大文件单连接约 1.3 MB/s，
8 路 Range 可到 ~10 MB/s。

Env vars:
  HF_BASE_URL   镜像站根地址，默认 https://aifasthub.com
  DOWNLOAD_JOBS 分块并发数，默认 8

用法:
  python hf_download.py <repo_id> <out_dir> <file1> [file2 ...] [--sha256 HEX]
"""
import hashlib
import json
import os
import subprocess
import sys
import threading
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor

BASE = os.environ.get("HF_BASE_URL", "https://aifasthub.com").rstrip("/")
JOBS = int(os.environ.get("DOWNLOAD_JOBS", "8"))
TIMEOUT = 60
# 镜像站的 WAF 会拦 Python-urllib 默认 UA（实测 403），必须伪装。
UA = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) InfiniSplat-v2-fetch"}


def url_of(repo, name):
    return f"{BASE}/{repo}/resolve/main/{name}"


def head_size(url):
    """返回 (总字节数, 是否支持 Range)。

    镜像站的 HEAD 可能返回 403（实测 aifasthub），改用 1 字节 Range GET，
    从 Content-Range 里解析总长度，同时验证 Range 可用性。
    """
    req = urllib.request.Request(url, headers={**UA, "Range": "bytes=0-0"})
    with urllib.request.urlopen(req, timeout=TIMEOUT) as r:
        code = r.getcode()
        cr = r.headers.get("Content-Range") or ""
        if code == 206 and "/" in cr:
            return int(cr.rsplit("/", 1)[1]), True
        # 服务端忽略 Range，整包返回：长度来自 Content-Length，但不要读完
        total = int(r.headers.get("Content-Length") or 0)
    return total, False


def fetch_range(url, start, end, dest, progress):
    """把 [start, end] 闭区间下到 dest，支持中途断点续传。

    踩过的坑：早期版本重试时用 open(dest,"wb") 从头写，只要连接抖一次就把整块
    已下字节丢掉，6.6 GB 反复重启永远走不到 100%。这里改成两点：
      1) 以追加模式写，每次重试只请求「还缺的那一段」；
      2) 用 curl 子进程（自带 --retry 与 stdout 追加），比 urllib 的半开连接稳。
    """
    want = end - start + 1
    if os.path.exists(dest) and os.path.getsize(dest) == want:
        progress(want)
        return
    for attempt in range(40):
        have = os.path.getsize(dest) if os.path.exists(dest) else 0
        if have >= want:
            if have == want:
                return
            # 多写了（并发写同一文件导致），截断到正确长度
            with open(dest, "r+b") as f:
                f.truncate(want)
            return
        rng = f"{start + have}-{end}"
        with open(dest, "ab") as f:
            proc = subprocess.run(
                ["curl", "-sSL", "-A", UA["User-Agent"], "--retry", "3", "--retry-delay", "2",
                 "--max-time", "900", "-r", rng, url],
                stdout=f, stderr=subprocess.PIPE,
            )
        now = os.path.getsize(dest) if os.path.exists(dest) else 0
        progress(max(0, now - start - have))
        if now == want:
            return
        if proc.returncode != 0:
            err = (proc.stderr or b"").decode("utf-8", "ignore").strip()[:160]
            print(f"  ⚠️ chunk {start}-{end} 第 {attempt+1} 次中断（{now-start}/{want}）: {err}", flush=True)
            time.sleep(1 + attempt)
        else:
            # curl 正常退出但没下全：服务端提前关了连接，继续续
            time.sleep(0.5)
    raise RuntimeError(f"chunk {start}-{end} 下载失败（40 次重试）")


def sha256_of(path, chunk=8 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for buf in iter(lambda: f.read(chunk), b""):
            h.update(buf)
    return h.hexdigest()


def download(repo, name, out_dir, expect_sha=None):
    dest = os.path.join(out_dir, name)
    os.makedirs(os.path.dirname(dest) or ".", exist_ok=True)
    url = url_of(repo, name)

    if os.path.exists(dest):
        if expect_sha is None or sha256_of(dest) == expect_sha:
            print(f"⏭️  已存在，跳过: {dest}")
            return dest
        print(f"⚠️  SHA256 不匹配，重新下载: {dest}")
        os.remove(dest)

    total, ranged = head_size(url)
    if total == 0:
        raise RuntimeError(f"无法获取文件大小: {url}")
    print(f"📦 {name}  {total/1e9:.2f} GB  range={ranged}", flush=True)

    if not ranged or JOBS <= 1 or total < 64 << 20:
        tmp = dest + ".part"
        done = [0]
        lock = threading.Lock()

        def tick(n):
            with lock:
                done[0] += n
                pct = 100.0 * done[0] / total
                if int(pct) % 5 == 0:
                    print(f"\r  {pct:5.1f}%  {done[0]/1e6:.0f}/{total/1e6:.0f} MB", end="", flush=True)

        fetch_range(url, 0, total - 1, tmp, tick)
        print()
        os.replace(tmp, dest)
    else:
        # 固定 96 MB 分块：与并发数解耦，换并发重启时已下的分块仍能命中续传
        chunk = int(os.environ.get("CHUNK_MB", "96")) << 20
        parts = [(s, min(s + chunk - 1, total - 1)) for s in range(0, total, chunk)]
        done = [0]
        lock = threading.Lock()
        state = {"last": 0.0}

        def tick(n):
            with lock:
                done[0] += n
                now = time.time()
                if now - state["last"] > 3:
                    state["last"] = now
                    pct = 100.0 * done[0] / total
                    print(f"\r  {pct:5.1f}%  {done[0]/1e9:.2f}/{total/1e9:.2f} GB", end="", flush=True)

        with ThreadPoolExecutor(max_workers=JOBS) as pool:
            futures = [
                pool.submit(fetch_range, url, s, e, f"{dest}.part{i:04d}", tick)
                for i, (s, e) in enumerate(parts)
            ]
            for f in futures:
                f.result()
        print()

        with open(dest + ".part", "wb") as out:
            for i in range(len(parts)):
                piece = f"{dest}.part{i:04d}"
                with open(piece, "rb") as f:
                    while True:
                        buf = f.read(1 << 20)
                        if not buf:
                            break
                        out.write(buf)
                os.remove(piece)
        os.replace(dest + ".part", dest)

    if expect_sha:
        got = sha256_of(dest)
        if got != expect_sha:
            raise RuntimeError(f"SHA256 校验失败\n  期望 {expect_sha}\n  实际 {got}")
        print(f"✅ SHA256 校验通过: {got[:16]}…")
    print(f"✅ {dest}")
    return dest


def main():
    argv = sys.argv[1:]
    expect_sha = None
    if "--sha256" in argv:
        i = argv.index("--sha256")
        expect_sha = argv[i + 1]
        del argv[i:i + 2]
    if len(argv) < 3:
        sys.exit(__doc__)
    repo, out_dir, files = argv[0], argv[1], argv[2:]
    for name in files:
        download(repo, name, out_dir, expect_sha if len(files) == 1 else None)


if __name__ == "__main__":
    main()

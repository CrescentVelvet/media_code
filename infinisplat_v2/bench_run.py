#!/usr/bin/env python3
"""跑一条命令，同时后台线程按固定间隔采样 GPU 显存，输出峰值/基线与墙钟耗时。

shell 里用后台子进程采样容易踩「命令替换等 stdout 关闭」「wait 挂住」这类竞态，
放到单进程 + 线程里最省心。

用法:
  python bench_run.py --gpu 0 --interval 0.15 -- out.json -- bash 02_run_inference.sh

输出 JSON: {"sec": 20.3, "peak_mb": 9750, "base_mb": 1582, "samples": 134, "rc": 0}
"""
import argparse
import json
import subprocess
import sys
import threading
import time


def sample_mem(gpu, stop_evt, sink):
    while not stop_evt.is_set():
        try:
            out = subprocess.run(
                ["nvidia-smi", "-i", str(gpu), "--query-gpu=memory.used",
                 "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=5,
            )
            v = out.stdout.strip().splitlines()
            if v:
                sink.append(int(float(v[0].strip())))
        except Exception:
            pass
        stop_evt.wait(0.10)   # 采样本身的耗时也算进间隔，避免空转拉满 CPU


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", default="0")
    ap.add_argument("--out", required=True, help="结果 JSON 路径")
    ap.add_argument("cmd", nargs=argparse.REMAINDER, help="-- 之后的命令")
    args = ap.parse_args()

    cmd = args.cmd
    if cmd and cmd[0] == "--":
        cmd = cmd[1:]
    if not cmd:
        sys.exit("❌ 没有要执行的命令（用 -- 分隔）")

    sink = []
    stop = threading.Event()
    th = threading.Thread(target=sample_mem, args=(args.gpu, stop, sink), daemon=True)
    th.start()

    t0 = time.time()
    rc = subprocess.run(cmd).returncode
    sec = time.time() - t0

    stop.set()
    th.join(timeout=2)

    res = {
        "sec": round(sec, 2),
        "peak_mb": max(sink) if sink else None,
        "base_mb": min(sink) if sink else None,
        "samples": len(sink),
        "rc": rc,
    }
    with open(args.out, "w", encoding="utf-8", newline="\n") as f:
        json.dump(res, f, ensure_ascii=False)
    print(f"⏱️  {res['sec']}s | 🎮 峰值 {res['peak_mb']} MB（基线 {res['base_mb']}，{res['samples']} 次采样）")
    sys.exit(rc)


if __name__ == "__main__":
    main()

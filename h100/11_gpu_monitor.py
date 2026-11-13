#!/usr/bin/env python3
"""H100 显卡状态实时监控（在登录节点跑）。

存在的理由：登录节点 nvidia-smi 报 "Failed to initialize NVML"，
所以真正的 GPU 读数必须进计算节点拿。两条路子：

  A. 本任务有正在跑的作业 → `srun --jobid=<ID> --overlap -n1 nvidia-smi ...`
     （最准，而且不需要新排队；但要求作业真的在 RUNNING）
  B. 全节点概览、没有自己的作业 → `srun -p <分区> --gres=gpu:N --overlap ...`
     会新排一个 2~4 秒的小作业，仅用于取一次读数（能拿到就别频繁刷）

本脚本优先 A，落不到 A 时提示用 B（--probe）。每个采样点都标注数据来源，
避免把"取不到"误读成"0 占用"。
"""
import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time

Q = "index,name,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw,power.limit"


def sh(cmd, timeout=30):
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return r.returncode, r.stdout, r.stderr
    except subprocess.TimeoutExpired:
        return 124, "", "timeout"
    except FileNotFoundError:
        return 127, "", "not found"


def my_jobs(name_filter=""):
    """返回 [(jobid, partition, name, state, runtime, nodelist)]"""
    rc, out, _ = sh(["squeue", "-a", "-u", os.environ.get("USER", ""), "-h",
                     "-o", "%i|%P|%j|%T|%M|%R"])
    if rc != 0:
        return []
    rows = []
    for ln in out.strip().splitlines():
        p = ln.split("|")
        if len(p) < 6:
            continue
        if name_filter and name_filter not in p[2]:
            continue
        rows.append(p)
    return rows


def parse_nvsmi(csv_text):
    """解析 nvidia-smi --format=csv,noheader,nounits 的输出为 dict 列表。"""
    gpus = []
    for ln in csv_text.strip().splitlines():
        v = [x.strip() for x in ln.split(",")]
        if len(v) < 8:
            continue
        def num(s, d=0.0):
            try:
                return float(s)
            except ValueError:
                return d
        gpus.append({
            "idx": int(num(v[0], -1)),
            "name": v[1],
            "util": int(num(v[2], -1)),
            "mem_used": int(num(v[3])),
            "mem_total": int(num(v[4])),
            "temp": int(num(v[5], -1)),
            "power": num(v[6], -1.0),
            "power_cap": num(v[7], -1.0),
        })
    return gpus


def read_via_job(jobid):
    """路子 A：借已有作业的执行上下文进节点读 nvidia-smi。"""
    cmd = ["srun", "--jobid=" + str(jobid), "--overlap", "-n1",
           "nvidia-smi", "--query-gpu=" + Q, "--format=csv,noheader,nounits"]
    rc, out, err = sh(cmd, timeout=25)
    if rc != 0 or not out.strip():
        return None, (err.strip() or "srun 无输出")
    gpus = parse_nvsmi(out)
    return (gpus or None), ""


def read_via_probe(partition, ngpu):
    """路子 B：临时排一个小作业进节点读一次（2~4s）。"""
    cmd = ["srun", "-p", partition, "--gres=gpu:%d" % ngpu,
           "-c", "1", "--mem=2G", "-t", "2", "--overlap", "-n1",
           "nvidia-smi", "--query-gpu=" + Q, "--format=csv,noheader,nounits"]
    rc, out, err = sh(cmd, timeout=60)
    if rc != 0 or not out.strip():
        return None, (err.strip() or "probe 无输出")
    return (parse_nvsmi(out) or None), ""


def read_via_local():
    """路子 C：万一就在计算节点上（或登录节点开放了 NVML）。"""
    rc, out, _ = sh(["nvidia-smi", "--query-gpu=" + Q,
                     "--format=csv,noheader,nounits"], timeout=10)
    if rc != 0 or not out.strip():
        return None
    return parse_nvsmi(out) or None


def bar(pct, width=20):
    pct = max(0, min(100, pct))
    fill = int(round(pct / 100.0 * width))
    return "█" * fill + "░" * (width - fill)


def paint(text, code, enable):
    return ("\033[%sm%s\033[0m" % (code, text)) if enable else text


def color_util(u, enable):
    if u < 0:
        return paint("  --", "90", enable)
    if u >= 85:
        return paint("%4d%%" % u, "92", enable)   # 绿：吃满
    if u >= 30:
        return paint("%4d%%" % u, "93", enable)   # 黄：中等
    return paint("%4d%%" % u, "91", enable)       # 红：闲


def color_mem(pct, enable):
    if pct >= 80:
        return paint("%4.0f%%" % pct, "92", enable)
    if pct >= 30:
        return paint("%4.0f%%" % pct, "93", enable)
    return paint("%4.0f%%" % pct, "91", enable)


def pad(text, width, align="left"):
    """按显示宽度补空格（中文占 2 格，直接用 %-Ns 会错位）。"""
    w = 0
    for ch in text:
        w += 2 if ord(ch) > 0x2E7F else 1
    if w >= width:
        return text
    fill = " " * (width - w)
    return (fill + text) if align == "right" else (text + fill)


def render(gpus, header, src, enable_color, show_bar=True):
    lines = []
    lines.append(header)
    if src:
        lines.append(paint("数据来源: " + src, "90", enable_color))
    lines.append("")
    if show_bar:
        hdr = (pad("GPU", 4) + pad("型号", 22) + pad("算力", 7) + pad("算力占用", 15)
               + pad("显存", 7) + pad("显存占用", 15) + pad("显存GiB", 12)
               + pad("温度", 6) + pad("功耗", 11))
        lines.append("  " + paint(hdr, "1", enable_color))
        lines.append("  " + "-" * 100)
    for g in gpus:
        mpct = (100.0 * g["mem_used"] / g["mem_total"]) if g["mem_total"] else 0.0
        pw = ("%.0f/%.0fW" % (g["power"], g["power_cap"])
              if g["power"] >= 0 else "   --")
        if show_bar:
            # 颜色码本身是零宽，pad 必须在着色之后按可见文本单独算 → 先把
            # 可见宽度算好再拼，避免 \033 被计入宽度。
            cu_plain = ("%d%%" % g["util"]) if g["util"] >= 0 else "--"
            cm_plain = "%d%%" % int(round(mpct))
            row = (pad(str(g["idx"]), 4)
                   + pad(g["name"][:20], 22)
                   + pad(cu_plain, 7)
                   + pad(bar(g["util"], 12), 15)
                   + pad(cm_plain, 7)
                   + pad(bar(mpct, 12), 15)
                   + pad("%.1f/%.0f" % (g["mem_used"] / 1024.0,
                                        g["mem_total"] / 1024.0), 12)
                   + pad(("%dC" % g["temp"]) if g["temp"] >= 0 else "--", 6)
                   + pad(pw, 11))
            if enable_color:
                row = row.replace(cu_plain, color_util(g["util"], True), 1)
                row = row.replace(cm_plain, color_mem(mpct, True), 1)
            lines.append("  " + row)
        else:
            lines.append("  gpu%d util=%d%% mem=%.1f/%.1fGiB temp=%d"
                         % (g["idx"], g["util"], g["mem_used"] / 1024.0,
                            g["mem_total"] / 1024.0, g["temp"]))
    return lines


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--interval", type=float, default=5.0, help="刷新间隔秒（默认 5）")
    ap.add_argument("--jobid", default="", help="指定作业 ID；默认自动找自己的 hold_gpu")
    ap.add_argument("--partition", default="h100a011", help="probe 用的分区")
    ap.add_argument("--probe", action="store_true",
                    help="没有自己的作业时，临时排个小作业进节点读一次（2~4s）")
    ap.add_argument("--probe-gpus", type=int, default=1, help="probe 申请的卡数（默认 1）")
    ap.add_argument("--once", action="store_true", help="只打印一次")
    ap.add_argument("--no-bar", action="store_true", help="纯文本模式（便于重定向到文件）")
    ap.add_argument("--no-color", action="store_true")
    ap.add_argument("--log", default="", help="把每次采样追加到该 jsonl 文件")
    args = ap.parse_args()

    enable_color = (not args.no_color) and sys.stdout.isatty()
    tty = sys.stdout.isatty()

    def sample():
        """返回 (gpus, header, src) 或 (None, header, err)"""
        job = args.jobid
        state = ""
        if not job:
            for r in my_jobs("hold_gpu") or my_jobs():
                job, state = r[0], r[3]
                break
        now = time.strftime("%Y-%m-%d %H:%M:%S")
        node = sh(["hostname"], 5)[1].strip() or "?"
        header = "[%s] 登录节点 %s" % (now, node)

        gpus = read_via_local()
        if gpus:
            return gpus, header, "本机 nvidia-smi（在计算节点上）"

        if job:
            header += "  job=%s(%s)" % (job, state or "?")
            gpus, err = read_via_job(job)
            if gpus:
                return gpus, header, "srun --jobid=%s --overlap" % job
            header += "  读取失败: %s" % err.replace("\n", " ")[:80]
        else:
            header += "  无自己的作业"

        if args.probe:
            gpus, err = read_via_probe(args.partition, args.probe_gpus)
            if gpus:
                return gpus, header, "srun -p %s --gres=gpu:%d（临时探针）" % (
                    args.partition, args.probe_gpus)
            header += "  probe 失败: %s" % err.replace("\n", " ")[:80]
        return None, header, ""

    while True:
        gpus, header, src = sample()
        out_lines = []
        if gpus is None:
            out_lines.append(header)
            out_lines.append("")
            out_lines.append("  取不到 GPU 读数。" + (src or ""))
            out_lines.append("  登录节点没有 NVML，必须进计算节点。可以：")
            out_lines.append("    · 有作业在跑 → 本脚本会自动 srun --jobid 进去读")
            out_lines.append("    · 只想看现状 → 加 --probe（临时排 2~4s 的小作业）")
        else:
            out_lines = render(gpus, header, src, enable_color,
                               show_bar=not args.no_bar)
            if args.log:
                rec = {"t": time.strftime("%Y-%m-%dT%H:%M:%S"), "src": src,
                       "job": args.jobid or "", "gpus": gpus}
                try:
                    with open(args.log, "a", encoding="utf-8", newline="\n") as f:
                        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                except Exception:
                    pass

        if tty and not args.once:
            sys.stdout.write("\033[H\033[2J")   # 清屏不闪
        print("\n".join(out_lines), flush=True)
        if args.once:
            return 0 if gpus else 1
        if args.probe and gpus is not None:
            print("\n(--probe 每轮都会新排作业，建议调大 --interval；Ctrl-C 退出)",
                  flush=True)
        elif gpus is not None:
            print("\n(Ctrl-C 退出；不会影响正在跑的任务)", flush=True)
        time.sleep(max(1.0, args.interval))


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n已退出监控。")
        sys.exit(0)

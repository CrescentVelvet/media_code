#!/usr/bin/env python3
"""H100 显卡实时监控（在登录节点跑，nvitop 风格界面）。

## 为什么不能直接用 nvitop

两台机器上都试过：
1. **登录节点没有 NVML**（`pynvml.nvmlInit()` 直接报 `NVMLError_Unknown`），
   nvitop 起不来，必须在登录节点套 `srun` 进计算节点，它的交互界面就废了。
2. **nvitop 的 TUI 是 GPL-3.0**（`nvitop/tui/` + `cli.py`），抄进来会传染整个仓库；
   `nvitop/api` 是 Apache-2.0，但它依赖 NVML 现场直连，同问题 1。

所以本脚本不抄代码，只学它的**原理**：不用 `nvidia-smi` 起进程（0.6~1.6 s/次），
改成 `pynvml` 直连 NVML（实测 0.094 ms/次，快 4 个数量级）。

## 架构：远端采样器 + 本地渲染

登录节点 → `srun --jobid=<ID> --overlap` 起一个常驻 python → 它用 pynvml 每
`--interval` 采一次，把「一帧」以单行 JSON 打到 stdout → 本地读取、渲染 TUI。

这样做的收益：
- 刷新率能到 1s 甚至 0.5s（原来每次采样都要新起一个 srun + nvidia-smi，至少 2s）
- **进程表**只需要一次 `/proc` 读取，不再每帧拉 `nvidia-smi`
- srun 只在启动时握手一次，之后走管道，不会每次排队

## 三块信息（对齐 nvitop）

  MEM      每卡 已用/总量 + 占用率条
  UTIL     每卡 算力占用率条
  PROCS    进程表：GPU / PID / USER / 进程名 / 显存

用法：
  ./11_gpu_monitor.py                    # 自动找自己的 hold_gpu，1s 刷新
  ./11_gpu_monitor.py -i 0.5             # 0.5s 刷新
  ./11_gpu_monitor.py --jobid 201362442  # 指定作业
  ./11_gpu_monitor.py --once             # 打一帧就退出（可重定向到文件）
  ./11_gpu_monitor.py --no-color         # 无颜色（日志/重定向）
"""
import argparse
import json
import os
import queue
import shutil
import subprocess
import sys
import threading
import time

# 远端采样器：把这段源码通过 stdin 喂给计算节点的 python，不落盘、不依赖仓库同步。
# 只依赖 pynvml（nvidia-ml-py），实测计算节点上装了；缺失时退化成 nvidia-smi 慢速模式。
SAMPLER_SRC = r'''
import json, os, sys, time

def emit(obj):
    sys.stdout.write(json.dumps(obj, ensure_ascii=False) + "\n")
    sys.stdout.flush()

interval = float(sys.argv[1]) if len(sys.argv) > 1 else 1.0
with_procs = (len(sys.argv) > 2 and sys.argv[2] == "1")

try:
    import pynvml
    pynvml.nvmlInit()
    NV = pynvml
except Exception as e:
    emit({"err": "pynvml 不可用: %r" % (e,), "fatal": True})
    sys.exit(2)

n = NV.nvmlDeviceGetCount()
handles = [NV.nvmlDeviceGetHandleByIndex(i) for i in range(n)]

def proc_user(pid):
    try:
        st = os.stat("/proc/%d" % pid)
        import pwd
        return pwd.getpwuid(st.st_uid).pw_name
    except Exception:
        return "?"

def proc_cmd(pid):
    """取一个短命令名。nvmlSystemGetProcessName 给的是解释器全路径，
    真正的脚本名在 /proc/<pid>/cmdline 里，优先后者。"""
    try:
        raw = open("/proc/%d/cmdline" % pid, "rb").read().split(b"\0")
        parts = [p.decode("utf-8", "replace") for p in raw if p]
        # 挑第一个像 .py 的参数，否则用 argv[0] 的 basename
        for p in parts[1:]:
            if p.endswith(".py") or "/" in p and not p.startswith("-"):
                if p.endswith(".py"):
                    return os.path.basename(p)
        if parts:
            return os.path.basename(parts[0])
    except Exception:
        pass
    try:
        return os.path.basename(NV.nvmlSystemGetProcessName(pid))
    except Exception:
        return "?"

while True:
    t0 = time.time()
    gpus = []
    for i, h in enumerate(handles):
        try:
            u = NV.nvmlDeviceGetUtilizationRates(h)
            m = NV.nvmlDeviceGetMemoryInfo(h)
            try:
                temp = NV.nvmlDeviceGetTemperature(h, NV.NVML_TEMPERATURE_GPU)
            except Exception:
                temp = -1
            try:
                pw = NV.nvmlDeviceGetPowerUsage(h) / 1000.0
            except Exception:
                pw = -1.0
            try:
                cap = NV.nvmlDeviceGetEnforcedPowerLimit(h) / 1000.0
            except Exception:
                cap = -1.0
            g = {"i": i,
                 "util": int(u.gpu), "mem_util": int(u.memory),
                 "used": int(m.used / 2**20), "total": int(m.total / 2**20),
                 "temp": int(temp), "power": round(pw, 1), "cap": round(cap, 1),
                 "procs": []}
            if with_procs:
                try:
                    ps = NV.nvmlDeviceGetComputeRunningProcesses(h)
                except Exception:
                    ps = []
                for p in ps:
                    g["procs"].append({
                        "pid": p.pid,
                        "mem": int((p.usedGpuMemory or 0) / 2**20),
                        "user": proc_user(p.pid),
                        "cmd": proc_cmd(p.pid),
                    })
            gpus.append(g)
        except Exception as e:
            gpus.append({"i": i, "err": str(e)[:60]})
    emit({"t": time.strftime("%Y-%m-%dT%H:%M:%S"), "gpus": gpus})
    dt = time.time() - t0
    time.sleep(max(0.05, interval - dt))
'''

Q_GPU = ("index,utilization.gpu,memory.used,memory.total,"
         "temperature.gpu,power.draw,power.limit")


def sh(cmd, timeout=30):
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return r.returncode, r.stdout, r.stderr
    except subprocess.TimeoutExpired:
        return 124, "", "timeout"
    except FileNotFoundError:
        return 127, "", "not found"


def my_jobs(name_filter=""):
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


# ---------------------------------------------------------------- 渲染

C = {
    "reset": "\033[0m", "bold": "\033[1m", "dim": "\033[90m",
    "red": "\033[91m", "green": "\033[92m", "yellow": "\033[93m",
    "blue": "\033[94m", "cyan": "\033[96m", "mag": "\033[95m",
}


def paint(s, code, on):
    return ("%s%s%s" % (code, s, C["reset"])) if on else s


def dwidth(s):
    """显示宽度：CJK 占 2 格。\033 转义序列按 0 计。"""
    w = 0
    i = 0
    while i < len(s):
        if s[i] == "\033":
            j = s.find("m", i)
            i = (j + 1) if j != -1 else i + 1
            continue
        w += 2 if ord(s[i]) > 0x2E7F else 1
        i += 1
    return w


def pad(s, width, align="left"):
    fill = " " * max(0, width - dwidth(s))
    return (fill + s) if align == "right" else (s + fill)


def bar(pct, width=14, color=None, on=True):
    """两段式进度条：实心 + 空心。颜色按占用高低分档。"""
    pct = max(0.0, min(100.0, float(pct)))
    fill = int(round(pct / 100.0 * width))
    body = "█" * fill + "░" * (width - fill)
    if not on:
        return body
    if color is None:
        color = C["green"] if pct >= 85 else (C["yellow"] if pct >= 30 else C["red"])
    return color + body + C["reset"]


def util_color(u, on):
    if u < 0:
        return paint("  --", C["dim"], on)
    if u >= 85:
        return paint("%3d%%" % u, C["green"], on)
    if u >= 30:
        return paint("%3d%%" % u, C["yellow"], on)
    return paint("%3d%%" % u, C["red"], on)


def human_gib(mib):
    return "%.1fG" % (mib / 1024.0)


def render(frame, header_lines, enable_color, show_procs=True, width=100):
    """把一帧 JSON 渲染成文本行。"""
    gpus = frame.get("gpus", [])
    out = list(header_lines)
    out.append("")

    tot_used = sum(g.get("used", 0) for g in gpus)
    tot_mem = sum(g.get("total", 0) for g in gpus)
    tot_pw = sum(g.get("power", 0) for g in gpus if g.get("power", 0) > 0)
    avg_util = (sum(g.get("util", 0) for g in gpus) / len(gpus)) if gpus else 0
    summary = ("  合计  %d 卡 · 显存 %s/%s (%.0f%%) · 平均 UTIL %.0f%% · 功耗 %.0fW"
               % (len(gpus), human_gib(tot_used), human_gib(tot_mem),
                  100.0 * tot_used / tot_mem if tot_mem else 0, avg_util, tot_pw))
    out.append(paint(summary, C["bold"], enable_color))

    out.append("  " + paint("─" * width, C["dim"], enable_color))
    hdr = (pad("GPU", 5) + pad("型号/状态", 14) + pad(" UTIL", 7) + pad("算力占用", 20)
           + pad(" MEM%", 7) + pad("显存占用", 20) + pad("显存GiB", 13)
           + pad("温度", 6) + pad("功耗", 12))
    out.append("  " + paint(hdr, C["bold"], enable_color))
    out.append("  " + paint("─" * width, C["dim"], enable_color))

    for g in gpus:
        idx = str(g.get("i", "?"))
        if "err" in g:
            out.append("  " + pad(idx, 5) + paint("读取失败: " + g["err"], C["red"], enable_color))
            continue
        used, total = g["used"], g["total"]
        mpct = 100.0 * used / total if total else 0.0
        pw = ("%.0f/%.0fW" % (g["power"], g["cap"])) if g["power"] >= 0 else "  --"
        row = (pad(idx, 5)
               + pad("H100 80G", 14)
               + pad(("  " + util_color(g["util"], False)), 7)
               + pad(bar(g["util"], 14, on=False), 20)
               # 这列是「显存控制器利用率」(nvml memory util)，跟下面的真实占用率
               # 是两回事，并排很容易看混，所以只留占用率。原值仍在 --log 的 jsonl 里。
               + pad("%4.0f%%" % mpct, 7)
               + pad(bar(mpct, 14, on=False), 20)
               + pad("%.1f/%.1f" % (used / 1024.0, total / 1024.0), 13)
               + pad(("%dC" % g["temp"]) if g["temp"] >= 0 else "--", 6)
               + pad(pw, 12))
        if enable_color:
            # 着色要在「宽度已对齐」之后做：颜色码零宽，先拼再着色不会错位
            row = (pad(idx, 5)
                   + pad("H100 80G", 14)
                   + pad("  " + util_color(g["util"], True), 7)
                   + pad(bar(g["util"], 14, on=True), 20)
                   + pad("%4.0f%%" % mpct, 7)
                   + pad(bar(mpct, 14, on=True), 20)
                   + pad("%.1f/%.1f" % (used / 1024.0, total / 1024.0), 13)
                   + pad(("%dC" % g["temp"]) if g["temp"] >= 0 else "--", 6)
                   + pad(pw, 12))
        out.append("  " + row)

    if show_procs:
        rows = []
        for g in gpus:
            for p in g.get("procs", []):
                rows.append((g["i"], p))
        out.append("")
        title = "  PROCS  %d 个进程" % len(rows)
        if not rows:
            title += paint("（无）", C["dim"], enable_color)
        out.append(paint(title, C["bold"], enable_color))
        out.append("  " + paint("─" * width, C["dim"], enable_color))
        ph = (pad("GPU", 5) + pad("PID", 10) + pad("USER", 12)
              + pad("进程名", 34) + pad("显存", 10))
        out.append("  " + paint(ph, C["bold"], enable_color))
        for gi, p in rows:
            row = (pad(str(gi), 5) + pad(str(p["pid"]), 10)
                   + pad(p["user"][:10], 12) + pad(p["cmd"][:32], 34)
                   + pad(human_gib(p["mem"]), 10))
            out.append("  " + row)
    return out


# ---------------------------------------------------------------- 采样

def build_remote_cmd(jobid, interval, with_procs, native, env_script):
    """构造进计算节点跑采样器的命令。

    ⚠️ 计算节点的非交互 shell 里 **没有 python**（实测 `which python` 报 command
    not found，exit 127），必须显式 source conda 环境脚本才能拿到解释器。
    """
    if native:
        # 兜底：没有 pynvml 时用 nvidia-smi 自带的循环模式（-lms），
        # 但拿不到进程表之外的细节，且每个采样点都要 fork 一次
        inner = ("nvidia-smi --query-gpu=%s --format=csv,noheader,nounits -lms %d"
                 % (Q_GPU, max(200, int(interval * 1000))))
    else:
        # python 从 stdin 读源码执行，避免把采样器落盘到计算节点
        inner = ("source %s >/dev/null 2>&1; "
                 "python -c \"import sys; exec(sys.stdin.read())\" %s %d"
                 % (env_script, interval, 1 if with_procs else 0))
    cmd = ["srun", "--jobid=" + str(jobid), "--overlap", "-n1", "bash", "-c", inner]
    return cmd


def start_stream(jobid, interval, with_procs, native, env_script):
    """起远端采样器，返回 (proc, err)。stdin 写完就关。"""
    cmd = build_remote_cmd(jobid, interval, with_procs, native, env_script)
    try:
        # 需要把源码喂到 stdin，所以不能用 capture_output 的便捷形式
        p = subprocess.Popen(cmd, stdin=subprocess.PIPE,
                             stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                             text=True, bufsize=1)
        if not native:
            p.stdin.write(SAMPLER_SRC)
            p.stdin.flush()
            p.stdin.close()
        return p, ""
    except Exception as e:  # noqa: BLE001
        return None, str(e)


def parse_nvsmi_loop(line):
    """把 nvidia-smi -lms 的一行 CSV 转成帧。"""
    v = [x.strip() for x in line.split(",")]
    if len(v) < 7:
        return None

    def num(s, d=0.0):
        try:
            return float(s)
        except ValueError:
            return d

    return {"t": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "gpus": [{"i": int(num(v[0], -1)), "util": int(num(v[1])),
                      "mem_util": -1, "used": int(num(v[2])),
                      "total": int(num(v[3])), "temp": int(num(v[4])),
                      "power": num(v[5], -1), "cap": num(v[6], -1),
                      "procs": []}]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--interval", type=float, default=1.0,
                    help="刷新间隔秒（默认 1.0，最小 0.2）")
    ap.add_argument("--jobid", default="", help="指定作业 ID；默认自动找自己的 hold_gpu")
    ap.add_argument("--once", action="store_true", help="只打一帧就退出")
    ap.add_argument("--no-bar", action="store_true", help="纯文本模式")
    ap.add_argument("--no-color", action="store_true", help="禁用颜色")
    ap.add_argument("--no-procs", action="store_true", help="不显示进程表")
    ap.add_argument("--native", action="store_true",
                    help="不用 pynvml，改用 nvidia-smi -lms（兼容兜底，慢很多）")
    ap.add_argument("--log", default="", help="把每帧追加到该 jsonl")
    ap.add_argument("--env-script",
                    default="/data/00950754/envs/activate_splatanything.sh",
                    help="计算节点上激活 conda 环境的脚本（默认 splatanything；"
                         "计算节点非交互 shell 里没有 python，必须显式 source）")
    args = ap.parse_args()

    interval = max(0.2, args.interval)
    enable_color = (not args.no_color) and sys.stdout.isatty()
    tty = sys.stdout.isatty()
    show_procs = (not args.no_procs) and (not args.no_bar)

    # 定位作业
    jobid, jstate = args.jobid, ""
    if not jobid:
        for r in my_jobs("hold_gpu") or my_jobs():
            jobid, jstate = r[0], r[3]
            break
    if not jobid:
        print("没有找到自己的作业，无法进计算节点读数。")
        print("先启动保活： ./10_hold.sh start        然后： ./11_gpu_monitor.py")
        return 2

    node = "?"
    rc, out, _ = sh(["squeue", "-j", jobid, "-h", "-o", "%N|%T|%M"], 10)
    if rc == 0 and out.strip():
        parts = out.strip().split("|")
        node = parts[0] if parts else "?"
        jstate = parts[1] if len(parts) > 1 else jstate

    proc, err = start_stream(jobid, interval, show_procs, args.native,
                             args.env_script)
    if proc is None:
        print("启动远端采样器失败: %s" % err)
        return 1

    frames = queue.Queue(maxsize=8)
    stopped = {"v": False}
    err_lines = []

    def read_stderr():
        """独立线程读 stderr，一是避免 PIPE 写满阻塞，二是出错时能给出真实原因。"""
        try:
            for line in proc.stderr:
                line = line.rstrip()
                if line:
                    err_lines.append(line)
                    del err_lines[:-6]      # 只留最后 6 行
        except Exception:
            pass

    threading.Thread(target=read_stderr, daemon=True).start()

    def reader():
        for line in proc.stdout:
            if stopped["v"]:
                break
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except ValueError:
                continue
            if obj.get("fatal"):
                frames.put(obj)
                continue
            if args.native:
                obj = parse_nvsmi_loop(line) or {}
            if obj.get("gpus"):
                frames.put(obj)
        frames.put({"eof": True})

    th = threading.Thread(target=reader, daemon=True)
    th.start()

    nframes = 0
    last_err = ""
    try:
        while True:
            try:
                frame = frames.get(timeout=max(5.0, interval * 6))
            except queue.Empty:
                last_err = "采样超时（远端 %.0fs 无输出；作业可能在排队/刚启动）" % (
                    max(5.0, interval * 6))
                frame = None
            if frame and frame.get("eof"):
                last_err = last_err or "远端采样器已退出"
                if err_lines:
                    last_err += "\n  远端 stderr: " + " | ".join(err_lines[-3:])
                frame = None
            if frame and frame.get("fatal"):
                print(frame["err"])
                return 2

            now = time.strftime("%Y-%m-%d %H:%M:%S")
            header = ["[%s] 登录节点 %s · job=%s(%s) · 刷新 %.1fs"
                      % (now, sh(["hostname"], 5)[1].strip() or "?", jobid,
                         jstate or "?", interval)]
            if frame is None:
                header.append("")
                header.append("  " + (last_err or "等待第一帧…"))
                lines = header
            else:
                nframes += 1
                src = ("nvidia-smi -lms" if args.native
                       else "pynvml @ 计算节点 %s" % node)
                header.append("数据来源: %s" % src)
                lines = render(frame, header, enable_color,
                               show_procs=show_procs)
                if args.log:
                    try:
                        with open(args.log, "a", encoding="utf-8", newline="\n") as f:
                            f.write(json.dumps(frame, ensure_ascii=False) + "\n")
                    except Exception:
                        pass

            if tty and not args.once:
                sys.stdout.write("\033[H\033[2J")
            print("\n".join(lines), flush=True)

            if args.once:
                return 0
            if not tty:
                print("", flush=True)   # 非 tty 时留空行分隔
            if not args.once and nframes == 1:
                print("\n(Ctrl-C 退出；不会影响正在跑的任务)", flush=True)
    except KeyboardInterrupt:
        print("\n已退出监控。")
        return 0
    finally:
        stopped["v"] = True
        try:
            proc.terminate()
        except Exception:
            pass


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n已退出监控。")
        sys.exit(0)

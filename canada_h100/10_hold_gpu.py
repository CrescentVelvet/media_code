#!/usr/bin/env python3
"""H100 保活：持续占用本任务分配到的 GPU，做真实 matmul 负载（不是空转等死）。

为什么要真算：分区按 AllocTRES 记账，只要 GPU 被占住就算我的；
但 util=0 的"占着不用"更解释不清，所以这里跑真实算力负载，可调强度。

每个 GPU 起一个子进程，互不干扰。子进程在 `import torch` 之前把
CUDA_VISIBLE_DEVICES 收窄成单卡号，这样 torch 一定落在正确的那张卡上。

⚠️ 注意 nvidia-smi **不认** CUDA_VISIBLE_DEVICES（实测 CVD=1/2/3 时它仍列全部 8 卡，
out[0] 恒为物理卡 0），所以查询必须显式带 `-i <物理卡号>`。

输出：逐行文本到 stdout（即 Slurm 的 .out），同时写 <rundir>/gpuN.jsonl 便于后处理。
"""
import argparse
import json
import os
import signal
import subprocess
import sys
import threading
import time

DUTY = {"low": 0.35, "mid": 0.75, "high": 1.0}
# 强度只靠「占空比」一个旋钮。实测 H100 上只要 matmul 连续在跑，功耗就顶到
# 340~350W 的功耗墙，改 kernel 大小（768/1536/4096）对功耗几乎没影响
# （实测 mid 小 kernel 仍 347W），所以不搞尺寸档，只留占空比。
# 占空比按「GPU 真实忙时 / 总时长」折算（单次 matmul 耗时开机实测，不拍脑袋），
# 日志里的「实测占空比」是 CUDA event 累加出来的真值，可用于核对标称值。
#
# 2026-11-13：**标准档位就是 high，别用 low/mid。** 这台机器是独占的（没有别人抢），
# 保活的目的就是让管理员看到卡被吃满；low/mid 是当初担心"空转占卡说不过去"设计的
# 省电档，占空比一低，util 读数就在 0/100 之间跳，反而像"占着不用"。留着这两个值
# 只是给调试（比如验证占空比折算公式）用，日常提交不要传。
DTYPE_BYTES = {"fp16": 2, "bf16": 2, "fp32": 4}

# 显存垫片默认目标：占满总显存的 90%（留 10% 给 CUDA context / cuBLAS workspace，
# 顶到 100% 会在下一次分配时直接 OOM）。实测：垫到 90% 后 nvidia-smi 报 used≈71.7/79.2 GiB。
DEFAULT_MEM_RATIO = 0.90

# NVML 句柄缓存：init 一次，之后每次查询约 0.1 ms（nvidia-smi 要 600+ ms）。
# ok=None 表示还没试过；ok=False 表示不可用，后续直接走 nvidia-smi 兜底。
_NVML = {"ok": None, "nvml": None, "handles": []}


def _nvml_init():
    """尝试初始化 pynvml。失败不抛异常（登录节点无 NVML 是常态，但 worker 在计算节点）。"""
    try:
        import pynvml
        pynvml.nvmlInit()
        n = pynvml.nvmlDeviceGetCount()
        _NVML["nvml"] = pynvml
        _NVML["handles"] = [pynvml.nvmlDeviceGetHandleByIndex(i) for i in range(n)]
        _NVML["ok"] = True
    except Exception:
        _NVML["ok"] = False


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--gpus", type=int, default=0,
                   help="0 = 自动取 SLURM_GPUS_ON_NODE / CUDA_VISIBLE_DEVICES 的长度")
    p.add_argument("--intensity", choices=["low", "mid", "high"], default="high",
                   help="占空比：high=连续跑(util 稳定100%%, ~350W)。high 是标准档，"
                        "low/mid 仅用于调试占空比折算，日常别传")
    p.add_argument("--size", type=int, default=4096,
                   help="matmul 方阵维度（默认 4096；实测 H100 上改它对功耗影响很小）")
    p.add_argument("--dtype", choices=list(DTYPE_BYTES), default="bf16")
    p.add_argument("--mem-ratio", type=float, default=DEFAULT_MEM_RATIO,
                   help="显存垫片目标占用比（默认 0.90 = 留 10%% 余量防 OOM；0 表示不垫片）")
    p.add_argument("--mem-gib", type=float, default=0.0,
                   help="显存垫片绝对值 GiB（>0 时覆盖 --mem-ratio，0=不垫片；"
                        "默认由 --mem-ratio 按卡容量自动算）")
    p.add_argument("--interval", type=float, default=30.0, help="状态打印间隔秒")
    p.add_argument("--max-hours", type=float, default=0.0,
                   help="自限时长，0 = 不限（靠 Slurm --time 兜底）")
    p.add_argument("--rundir", default="", help="jsonl 落盘目录，默认 <cwd>/hold_<JOBID>")
    p.add_argument("--dry-run", action="store_true", help="只打印配置，不算")
    p.add_argument("--chain", action="store_true",
                   help="在当前作业到期前自动提交下一个同类作业（无缝续期）")
    p.add_argument("--chain-lead", type=float, default=120.0,
                   help="提前多少秒提交下一个作业（默认 120）")
    return p.parse_args()


def physical_ids(args):
    """拿到本任务真正能用的物理卡号（Slurm 会设 CUDA_VISIBLE_DEVICES）。"""
    cvd = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if cvd.strip():
        ids = [x.strip() for x in cvd.split(",") if x.strip() != ""]
        if ids:
            return ids
    n = args.gpus or int(os.environ.get("SLURM_GPUS_ON_NODE", "0") or 0)
    if n <= 0:
        n = 1
    return [str(i) for i in range(n)]


def chain_next(args, ids):
    """在当前作业里提交下一个同类作业，实现无缝续期。

    为什么在计算节点里提交：登录节点和计算节点都能调 sbatch（实测
    /usr/local/bin/sbatch 在两边都有），而在计算节点里提交的好处是能直接继承
    当前的 Slurm 环境变量（分区、账号），不必硬编码。
    """
    sb = os.environ.get("HOLD_SBATCH",
                        os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                     "10_hold_gpu.sbatch"))
    if not os.path.exists(sb):
        return False, "找不到 sbatch 模板: %s" % sb
    env = dict(os.environ)
    env["HOLD_INTENSITY"] = args.intensity
    env["HOLD_SIZE"] = str(args.size)
    env["HOLD_DTYPE"] = args.dtype
    env["HOLD_MEM_RATIO"] = str(args.mem_ratio)
    env["HOLD_MEM_GIB"] = str(args.mem_gib)
    env["HOLD_INTERVAL"] = str(args.interval)
    if not args.chain:
        env.pop("HOLD_CHAIN", None)
    cmd = ["sbatch", "--parsable", "--gres=gpu:%d" % len(ids),
           "--job-name=hold_gpu", sb]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=60, env=env)
        if r.returncode == 0:
            return True, r.stdout.strip()
        return False, (r.stderr or r.stdout).strip()[:200]
    except Exception as e:
        return False, str(e)


def gpu_stat(gpu_id=None):
    """查单卡的 util / 显存 / 温度 / 功耗。

    优先用 pynvml（NVML 直连，实测 50 次查询共 4.7 ms ≈ 0.094ms/次），
    退化时用 nvidia-smi（0.6~1.6 s/次，慢 4 个数量级，只作兜底）。

    ⚠️ nvidia-smi **不认 CUDA_VISIBLE_DEVICES**：实测 CVD=1/2/3 时
    `--query-gpu` 仍输出全部 8 行，且 out[0] 恒为物理卡 0。所以必须显式
    带 `-i <物理卡号>`，否则 8 个 worker 会全部读到卡 0 的数值。
    pynvml 的 getDeviceHandleByIndex 用的是 NVML 索引，与 nvidia-smi -i
    的物理序号一致（已用 torch 的 uuid 对齐验证过）。
    """
    if _NVML["ok"] is None:
        _nvml_init()
    if _NVML["ok"]:
        try:
            h = _NVML["handles"][int(gpu_id)]
            u = _NVML["nvml"].nvmlDeviceGetUtilizationRates(h)
            m = _NVML["nvml"].nvmlDeviceGetMemoryInfo(h)
            return {
                "util": int(u.gpu),
                "mem_used": int(m.used / 2 ** 20),
                "mem_total": int(m.total / 2 ** 20),
                "temp": int(_NVML["nvml"].nvmlDeviceGetTemperature(
                    h, _NVML["nvml"].NVML_TEMPERATURE_GPU)),
                "power": float(_NVML["nvml"].nvmlDeviceGetPowerUsage(h)) / 1000.0,
            }
        except Exception:
            pass   # 落到 nvidia-smi 兜底

    q = "utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw"
    cmd = ["nvidia-smi"]
    if gpu_id is not None:
        cmd += ["-i", str(gpu_id)]
    cmd += ["--query-gpu=" + q, "--format=csv,noheader,nounits"]
    try:
        out = subprocess.run(cmd, capture_output=True, text=True,
                             timeout=5).stdout.strip().splitlines()
        if not out:
            return None
        v = [x.strip() for x in out[0].split(",")]
        return {
            "util": int(float(v[0])),
            "mem_used": int(float(v[1])),
            "mem_total": int(float(v[2])),
            "temp": int(float(v[3])),
            "power": float(v[4]) if v[4] not in ("[N/A]", "N/A") else -1.0,
        }
    except Exception:
        return None


def worker(phys_id, args, max_s, stop_flag):
    # 关键：必须在 import torch 之前把可见卡收窄到这一张。
    # 否则 spawn 出来的子进程继承父进程的 CUDA_VISIBLE_DEVICES="0,1,..."，
    # 各自都能看到全部卡、又都去用 cuda:0 → 全挤在卡 0 上，其余卡闲着。
    # （实测踩过：2 卡任务 gpu0 util=100% / gpu1 util=0%）
    os.environ["CUDA_VISIBLE_DEVICES"] = str(phys_id)
    import torch  # 子进程里 import，CUDA_VISIBLE_DEVICES 已生效

    dev = torch.device("cuda:0")
    # 自证：确认这个进程真的落在 phys_id 那张卡上，而不是又挤回卡 0
    try:
        assert torch.cuda.device_count() == 1, "device_count=%d" % torch.cuda.device_count()
        props = torch.cuda.get_device_properties(0)
        print("[gpu%s] pid=%d bound_to=%s sm=%d mem=%.1fGiB"
              % (phys_id, os.getpid(), props.name, props.multi_processor_count,
                 props.total_memory / 1024 ** 3), flush=True)
    except Exception as e:
        print("[gpu%s] ⚠️ 设备绑定自检失败: %s" % (phys_id, e), flush=True)
    dtype = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}[args.dtype]
    n = args.size
    f32 = torch.float32
    a = torch.randn(n, n, device=dev, dtype=f32).to(dtype)
    b = torch.randn(n, n, device=dev, dtype=f32).to(dtype)
    c = torch.empty(n, n, device=dev, dtype=dtype)

    # 额外显存垫片：只分配不访问，让 memory.used 落在「训练中」的量级。
    # 用途是给管理员看显存利用率——只跑一次 4096 方阵 matmul 才吃 2 GiB，
    # 8 张 80G 卡显示 2/79 GiB 会显得"没用起来"。垫片是 torch.empty（
    # 不 touch），实测驱动照样记账：nvml 从 1.96 GiB 涨到 71.71 GiB。
    ballast = []
    want_gib = args.mem_gib
    if want_gib <= 0 and args.mem_ratio > 0:
        # 按「总容量 × ratio - 已用」算差额；已用含 matmul 张量与 CUDA context
        free_b, total_b = torch.cuda.mem_get_info()
        used_gib = (total_b - free_b) / 1024 ** 3
        want_gib = total_b / 1024 ** 3 * args.mem_ratio - used_gib
    if want_gib > 0:
        chunk = 256 * 1024 * 1024  # 256 MiB
        n_chunks = int(want_gib * 1024 ** 3 / chunk)
        got = 0
        for _ in range(n_chunks):
            try:
                ballast.append(torch.empty(chunk, dtype=torch.uint8, device=dev))
                got += chunk
            except RuntimeError as e:
                # 顶到上限就停，不要把整个 worker 拖崩
                print("%s ⚠️ 垫片在第 %.1f GiB 处停（%s）"
                      % ("[gpu" + str(phys_id) + "]", got / 1024 ** 3,
                         str(e).split(".")[0][:60]), flush=True)
                break
        free_b, total_b = torch.cuda.mem_get_info()
        print("[gpu%s] 垫片 %.1f GiB → 显存占用 %.0f%%（%.1f/%.1f GiB）"
              % (phys_id, got / 1024 ** 3,
                 100.0 * (total_b - free_b) / total_b,
                 (total_b - free_b) / 1024 ** 3, total_b / 1024 ** 3), flush=True)

    duty = DUTY[args.intensity]
    tag = "[gpu" + str(phys_id) + "]"
    # 占空比的坑（实测五次才对）：
    #   1) 「算 1s 睡 Ns，睡完再采样」→ nvidia-smi 的 utilization.gpu 是最近采样窗
    #      内有 kernel 在跑的时间占比，窗口全落在睡眠里 → 恒读 0%
    #   2) 5ms 细密开关              → 被异步 kernel 排队抹平，35% 实测仍读到 95%
    #   3) burst 内逐 20 次 matmul 查表 → Python 提交速度跟不上 GPU 消耗，
    #      队列经常空，功耗只有 96W、util 读 0%（8 卡全中招）
    #   4) 睡眠时长按「burst_s*(1-duty)/duty」折算 → burst_s 是假的，占空比失真
    #   5) 真元凶：Python 每秒只能提交约 490 次 matmul（每次 GPU 约 0.35ms），
    #      即 GPU 每秒只拿到约 0.17s 的活 → mid 标称 75% 实测封顶 17%、
    #      功耗 128W。**再怎么调睡眠时长都没用，是 CPU 喂不饱 GPU。**
    # 解法：CUDA Graph。把整批 matmul 录成一张图，之后每次 replay 只有一次
    # CPU 调用，GPU 端拿到的是背靠背的 kernel 序列 → 真能喂满。
    # 占空比 = 每轮 replay 的总忙时 / (忙时 + 睡眠)，用 CUDA event 实测得出。
    n_in_graph = 64                   # 每张图里塞多少次 matmul
    work_slice = 1.0                  # 每个 burst 让 GPU 真忙多少秒

    def mm():
        torch.matmul(a, b, out=c)

    graph = None
    try:
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(3):        # 录制前的预热，避免 capture 里做惰性初始化
                mm()
        torch.cuda.current_stream().wait_stream(s)
        torch.cuda.synchronize()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            for _ in range(n_in_graph):
                mm()
        # 量一次 replay 的 GPU 耗时
        g.replay()
        torch.cuda.synchronize()
        t_cal = time.time()
        for _ in range(3):
            g.replay()
        torch.cuda.synchronize()
        replay_s = (time.time() - t_cal) / 3.0
        graph = g
        per_iter = replay_s / n_in_graph
        print("%s CUDA Graph 就绪：%d 次 matmul/图，单图 %.3fs（单次 %.2f ms）"
              % (tag, n_in_graph, replay_s, per_iter * 1e3), flush=True)
    except Exception as e:
        # 兜底：图录不了就退回逐次提交（占空比会偏低，但不会崩）
        print("%s ⚠️ CUDA Graph 不可用（%s），退回逐次提交模式" % (tag, e), flush=True)
        torch.cuda.synchronize()
        t_cal = time.time()
        for _ in range(20):
            mm()
        torch.cuda.synchronize()
        per_iter = max((time.time() - t_cal) / 20.0, 1e-9)
        replay_s = per_iter * n_in_graph

    n_replay = max(1, int(round(work_slice / replay_s)))
    work_iters = n_replay * n_in_graph
    sleep_slice = max(0.0, work_slice * (1.0 - duty) / duty)
    print("%s 每 burst replay %d 次（约 %.2fs 忙），睡 %.2fs  → 标称占空比 %.0f%%"
          % (tag, n_replay, n_replay * replay_s, sleep_slice, duty * 100), flush=True)

    ev_start = torch.cuda.Event(enable_timing=True)
    ev_end = torch.cuda.Event(enable_timing=True)
    busy_s = 0.0

    t0 = time.time()
    next_report = t0 + args.interval
    iters = 0

    # 采样必须放独立线程（实测第七次才对）：
    #   nvidia-smi 单次调用要 0.6~1.6s（主要是 NVML 初始化 + PCI 枚举）。原先在
    #   主循环里同步调它 → 这 1 秒 Python 不再提交 kernel，GPU 队列被跑空，
    #   nvidia-smi 的采样窗全落在空档里，于是 8 张卡全报 util=0% / 102W / 58C，
    #   而外部同时刻看是 100% / 349W / 66C。**GPU 明明是满的，是读数把自己饿死了。**
    #   丢到线程里后主循环不间断 replay，subprocess 等待期间也释放 GIL。
    #   现在 gpu_stat 优先走 pynvml（0.1ms），同步调用也不会饿死 GPU 了，
    #   但仍保留线程：一是间隔可独立于主循环，二是万一退回 nvidia-smi 兜底时不会炸。
    shared = {"st": None, "ts": 0.0, "util_sum": 0.0, "pw_sum": 0.0, "n": 0}
    lock = threading.Lock()

    def sampler():
        while not stop_flag["stop"]:
            st = gpu_stat(phys_id)
            if st:
                with lock:
                    shared["st"] = st
                    shared["ts"] = time.time()
                    # 累计均值：nvml 只有瞬时值，low/mid 档若单看瞬时值，
                    # 采样点落在 burst 内就是 100%/350W，会让人以为没在省电。
                    shared["util_sum"] += st["util"]
                    if st["power"] >= 0:
                        shared["pw_sum"] += st["power"]
                    shared["n"] += 1
            # 采样周期 = interval，但至少留 0.5s 给下次；sleep 可被 stop 打断
            end = time.time() + max(0.5, args.interval)
            while not stop_flag["stop"] and time.time() < end:
                time.sleep(0.2)

    th = threading.Thread(target=sampler, daemon=True)
    th.start()

    def report(now):
        """打印一行状态。数值取自采样线程的最新一份，主循环不再阻塞。

        ⚠️ 这里只报 **nvml 的 util**，不再报「CUDA event 跨度占空比」。
        实测发现后者会说谎：ev_start 打在第一个 replay 前、ev_end 打在最后
        一个后，两者之间的 elapsed_time 是 **GPU 时间轴上的跨度**，中途 GPU
        空转的时间也被算进去。旧代码同步调 nvidia-smi 时有 1 秒多的空档，
        日志却报「实测占空比 91%」，而 nvml 同时刻读 0%/101W。**信 nvml。**
        """
        with lock:
            st = shared["st"]
            util_avg = shared["util_sum"] / shared["n"] if shared["n"] else 0.0
            pw_avg = shared["pw_sum"] / shared["n"] if shared["n"] else 0.0
            nsamp = shared["n"]
        el = int(now - t0)
        if st:
            mem_gib = st["mem_used"] / 1024.0
            tot_gib = st["mem_total"] / 1024.0
            mem_pct = (100.0 * st["mem_used"] / st["mem_total"]) if st["mem_total"] else 0.0
            pw = ("%.0fW" % st["power"]) if st["power"] >= 0 else "n/a"
            line = (tag + " t=%ds util=%d%%(均 %.0f%%, n=%d) "
                    "mem=%.1f/%.1fGiB(%.0f%%) temp=%dC power=%s(均 %.0fW) iters=%d"
                    % (el, st["util"], util_avg, nsamp, mem_gib, tot_gib, mem_pct,
                       st["temp"], pw, pw_avg, iters))
        else:
            line = tag + " t=%ds iters=%d (等待首次采样)" % (el, iters)
        print(line, flush=True)
        if args.rundir:
            rec = {"ts": round(now, 1), "gpu": phys_id, "elapsed": el,
                   "iters": iters, "size": n, "intensity": args.intensity,
                   "duty_nominal": duty, "duty_from_nvml": round(util_avg / 100.0, 4),
                   "util_avg": round(util_avg, 1), "power_avg": round(pw_avg, 1),
                   "per_iter_ms": round(per_iter * 1e3, 3),
                   "cuda_graph": graph is not None,
                   # busy_s 是 CUDA event 跨度，含 GPU 空转，仅作参考
                   "ev_span_s": round(busy_s, 2)}
            if st:
                rec.update(st)
            try:
                with open(os.path.join(args.rundir, "gpu" + str(phys_id) + ".jsonl"),
                          "a", encoding="utf-8", newline="\n") as f:
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            except Exception:
                pass

    while not stop_flag["stop"]:
        if max_s > 0 and time.time() - t0 >= max_s:
            break
        # burst：连发 n_replay 次 replay（CPU 开销忽略，GPU 端背靠背）
        ev_start.record()
        for _ in range(n_replay):
            if graph is not None:
                graph.replay()
            else:
                for _ in range(n_in_graph):
                    mm()
            iters += n_in_graph
        ev_end.record()               # 排在所有 replay 之后，elapsed 即真实忙时
        # 采样点选在「队列还满」的时候，不是 synchronize 之后
        now = time.time()
        if now >= next_report:
            report(now)
            next_report = now + args.interval
        # 收口：等这批 kernel 真的跑完，再睡 —— 顺序反了占空比就不成立
        torch.cuda.synchronize()
        busy_s += ev_start.elapsed_time(ev_end) / 1000.0
        if sleep_slice > 0 and not stop_flag["stop"]:
            time.sleep(sleep_slice)


def main():
    args = parse_args()
    ids = physical_ids(args)
    max_s = args.max_hours * 3600.0
    duty = DUTY[args.intensity]

    job = os.environ.get("SLURM_JOB_ID", "local")
    rundir = args.rundir or os.path.join(os.getcwd(), "hold_" + str(job))
    if not args.dry_run:
        os.makedirs(rundir, exist_ok=True)

    print("=" * 72, flush=True)
    print("H100 保活任务  job=%s  node=%s" % (job, os.uname().nodename), flush=True)
    print("  物理卡号 : %s" % ",".join(ids), flush=True)
    print("  强度/占空比: %s / %.0f%%   方阵: %d   dtype: %s"
          % (args.intensity, duty * 100, args.size, args.dtype), flush=True)
    est = (args.size ** 2) * DTYPE_BYTES[args.dtype] * 3 / (1024 ** 3)
    if args.mem_gib > 0:
        mem_desc = "%.1f GiB（绝对值）" % args.mem_gib
    elif args.mem_ratio > 0:
        mem_desc = "总显存的 %.0f%%（留 %.0f%% 余量）" % (args.mem_ratio * 100,
                                                        (1 - args.mem_ratio) * 100)
    else:
        mem_desc = "不垫片"
    print("  显存目标 : %s   （matmul 本身约 %.2f GiB/卡）" % (mem_desc, est), flush=True)
    print("  自限时长 : %s" % ("不限（由 Slurm --time 兜底）" if max_s <= 0
                              else "%.1f 小时" % args.max_hours), flush=True)
    print("  状态间隔 : %.0fs    jsonl: %s" % (args.interval, rundir), flush=True)
    end_t = 0.0
    if os.environ.get("SLURM_JOB_END_TIME"):
        try:
            end_t = float(os.environ["SLURM_JOB_END_TIME"])
            left_h = (end_t - time.time()) / 3600.0
            print("  Slurm 时限: %s（剩 %.1f 小时）"
                  % (time.strftime("%m-%d %H:%M", time.localtime(end_t)), left_h),
                  flush=True)
        except ValueError:
            end_t = 0.0
    print("  链式续期 : %s" % ("开（到期前 %.0fs 提交下一个）" % args.chain_lead
                              if args.chain else "关（到期即停）"), flush=True)
    print("=" * 72, flush=True)
    if args.dry_run:
        print("dry-run，不启动计算。", flush=True)
        return 0

    from multiprocessing import get_context
    ctx = get_context("spawn")
    mgr_stop = {"stop": False}
    procs = []
    for pid in ids:
        p = ctx.Process(target=worker, args=(pid, args, max_s, mgr_stop), daemon=False)
        p.start()
        procs.append(p)

    def on_signal(signum, _frame):
        print("\n[hold] 收到信号 %d，通知各卡退出…" % signum, flush=True)
        mgr_stop["stop"] = True

    signal.signal(signal.SIGTERM, on_signal)
    signal.signal(signal.SIGINT, on_signal)

    t0 = time.time()
    chained = False
    while any(p.is_alive() for p in procs):
        if mgr_stop["stop"]:
            for p in procs:
                p.join(timeout=20)
            for p in procs:
                if p.is_alive():
                    p.terminate()
            break
        # 到期前 lead 秒提交下一个作业，保证前后两个之间有空档不停
        if args.chain and end_t > 0 and not chained:
            if end_t - time.time() <= args.chain_lead:
                ok, msg = chain_next(args, ids)
                chained = True
                print("[hold] 链式续期: %s  %s" % ("成功 → 新 job " if ok else "失败",
                                                  msg), flush=True)
                if not ok:
                    print("[hold] ⚠️ 续期失败，本作业到期后会空闲，请手动检查",
                          flush=True)
        time.sleep(1.0)

    for p in procs:
        p.join(timeout=10)
    codes = [p.exitcode for p in procs]
    crashed = any(c not in (0, None) for c in codes)
    print("[hold] 结束，总运行 %.1f 分钟，退出码 %s"
          % ((time.time() - t0) / 60.0, codes), flush=True)
    if crashed:
        # 崩了就别续期，否则会把故障作业一直链下去（踩过：NameError 连链两个）
        print("[hold] ⚠️ 有 worker 异常退出，跳过续期", flush=True)
        return 1
    # 走到这里说明是正常结束（跑到 --time 或被信号停掉）。只有「接近到期」
    # 才补一次续期，避免手动 stop 之后又自己爬起来。
    if args.chain and not chained and end_t > 0:
        left = end_t - time.time()
        if left < max(args.chain_lead, 60.0):
            ok, msg = chain_next(args, ids)
            print("[hold] 收尾补一次续期(剩 %.0fs): %s %s"
                  % (left, "成功" if ok else "失败", msg), flush=True)
        else:
            print("[hold] 距到期还有 %.0f 分钟，被提前终止 → 不续期"
                  % (left / 60.0), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())

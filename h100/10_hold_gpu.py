#!/usr/bin/env python3
"""H100 保活：持续占用本任务分配到的 GPU，做真实 matmul 负载（不是空转等死）。

为什么要真算：分区按 AllocTRES 记账，只要 GPU 被占住就算我的；
但 util=0 的"占着不用"更解释不清，所以这里跑真实算力负载，可调强度。

每个 GPU 起一个子进程，互不干扰；CUDA_VISIBLE_DEVICES 已经由 Slurm 约束，
这里只做「按物理卡号各起一个进程」这一件事。

输出：逐行文本到 stdout（即 Slurm 的 .out），同时写 <rundir>/gpuN.jsonl 便于后处理。
"""
import argparse
import json
import os
import signal
import subprocess
import sys
import time

DUTY = {"low": 0.35, "mid": 0.75, "high": 1.0}
# 强度只靠「占空比」一个旋钮。实测 H100 上只要 matmul 连续在跑，功耗就顶到
# 340~350W 的功耗墙，改 kernel 大小（768/1536/4096）对功耗几乎没影响
# （实测 mid 小 kernel 仍 347W），所以不搞尺寸档，只留占空比。
# 占空比按「GPU 真实忙时 / 总时长」折算（单次 matmul 耗时开机实测，不拍脑袋），
# 日志里的「实测占空比」是 CUDA event 累加出来的真值，可用于核对标称值。
DTYPE_BYTES = {"fp16": 2, "bf16": 2, "fp32": 4}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--gpus", type=int, default=0,
                   help="0 = 自动取 SLURM_GPUS_ON_NODE / CUDA_VISIBLE_DEVICES 的长度")
    p.add_argument("--intensity", choices=["low", "mid", "high"], default="high",
                   help="占空比：high=连续跑(util 稳定100%%, ~350W) mid=75%% low=35%%"
                        "（低档省电但 util 读数会在 0/100 间跳）")
    p.add_argument("--size", type=int, default=4096,
                   help="matmul 方阵维度（默认 4096；实测 H100 上改它对功耗影响很小）")
    p.add_argument("--dtype", choices=list(DTYPE_BYTES), default="bf16")
    p.add_argument("--mem-gib", type=float, default=0.0,
                   help="额外占用的显存 GiB（只占不算，用来让显存占用率看起来正常；0=不占）")
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

    gpu_id=None 表示「本进程可见的卡」——worker 里 CUDA_VISIBLE_DEVICES 已被收窄
    成单卡，nvidia-smi 不带 -i 就只会列出那一张，比按序号/UUID 指定更可靠
    （踩过两次：物理序号在 CVD 收窄后语义漂移；torch 的 uuid 不带 GPU- 前缀，
    而 nvidia-smi -i 还必须是本作业可见的卡）。
    """
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

    # 额外显存垫片：只分配不访问，让 memory.used 落在正常训练的量级
    ballast = []
    if args.mem_gib > 0:
        chunk = 256 * 1024 * 1024  # 256 MiB
        n_chunks = int(args.mem_gib * 1024 ** 3 / chunk)
        for _ in range(n_chunks):
            ballast.append(torch.empty(chunk, dtype=torch.uint8, device=dev))

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
    util_hist = []

    def report(now):
        """采样并打印。必须在 burst 提交之后、synchronize 之前调用：nvidia-smi 的
        utilization.gpu 是上次查询到现在的活跃占比，若在睡眠段之后查会读 0%
        （low 强度全打印 0% 就是这么来的）。"""
        st = gpu_stat()
        el = int(now - t0)
        real = (busy_s / (now - t0) * 100.0) if now > t0 else 0.0
        if st:
            mem_gib = st["mem_used"] / 1024.0
            tot_gib = st["mem_total"] / 1024.0
            pw = ("%.0fW" % st["power"]) if st["power"] >= 0 else "n/a"
            util_hist.append(st["util"])
            if len(util_hist) > 10:
                util_hist.pop(0)
            avg = sum(util_hist) / float(len(util_hist))
            line = (tag + " t=%ds util=%d%%(近%d次均 %.0f%%) 实测占空比=%.0f%% "
                    "mem=%.1f/%.1fGiB temp=%dC power=%s iters=%d"
                    % (el, st["util"], len(util_hist), avg, real,
                       mem_gib, tot_gib, st["temp"], pw, iters))
        else:
            line = tag + " t=%ds iters=%d (nvidia-smi 查询失败)" % (el, iters)
        print(line, flush=True)
        if args.rundir:
            rec = {"ts": round(now, 1), "gpu": phys_id, "elapsed": el,
                   "iters": iters, "size": n, "intensity": args.intensity,
                   "duty": duty, "duty_measured": round(real / 100.0, 4),
                   "per_iter_ms": round(per_iter * 1e3, 3),
                   "cuda_graph": graph is not None}
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
    print("  预估常驻显存: 每卡约 %.2f GiB matmul + %.1f GiB 垫片"
          % (est, args.mem_gib), flush=True)
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

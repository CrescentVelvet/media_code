#!/bin/bash
# 12_sa_train.sh —— SplatAnything 训练的提交 / 查看 / 停止（登录节点跑）
#
#   ./12_sa_train.sh start                 8 卡 / 168h / 全参 / 25 万步
#   ./12_sa_train.sh start 4 168h          4 卡 / 168h
#   ./12_sa_train.sh start 8 168h gshead   8 卡 / 只训解码头（冒烟）
#   ./12_sa_train.sh scene <场景目录>      指定单场景（默认轮转 examples/ 下 6 个）
#   ./12_sa_train.sh ls                    我的训练任务
#   ./12_sa_train.sh tail [JOBID]          跟日志
#   ./12_sa_train.sh metrics [RUNDIR]      汇总各 rank 指标
#   ./12_sa_train.sh stop [JOBID]          停止（不给 ID 就停全部）
#
# 作业形态：单节点 8 卡，torchrun 起 8 个进程，NCCL 数据并行。各 rank 读不同的
# 真实场景（examples/ 下 6 个按 rank 轮转），梯度 all-reduce 平均，rank0 落盘。
set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOGD=/data/00950754/logs
RUNS=/data/00950754/runs/sa_train

need_login_node() {
    if [ -n "${SLURM_JOB_ID:-}" ]; then
        echo "⚠️ 当前在作业里（SLURM_JOB_ID=$SLURM_JOB_ID），请回登录节点执行"
        exit 1
    fi
}

my_jobs() {
    squeue -a -u "$(whoami)" -h -o '%i|%P|%j|%T|%M|%D|%R|%b' 2>/dev/null \
        | awk -F'|' '$3=="sa_train"'
}

cmd_start() {
    local gpus="${1:-8}" hours="${2:-168h}" scope="${3:-all}"
    local t
    case "$hours" in
        *h) t=$(printf "%02d:00:00" "${hours%h}") ;;
        *:*) t="$hours" ;;
        *)  t=$(printf "%02d:00:00" "$hours") ;;
    esac
    need_login_node

    if [ -n "$(my_jobs)" ]; then
        echo "已有训练任务，先停掉再提交："
        my_jobs | awk -F'|' '{printf "  job %s  %s  %s\n", $1, $4, $5}'
        echo "  → $0 stop"
        exit 1
    fi

    echo "提交: gpus=$gpus time=$t scope=$scope iters=${SA_ITERS:-250000} scene=${SA_SCENE:-<examples 轮转>}"
    SA_SCRIPT_DIR="$SCRIPT_DIR" \
    SA_SCENE="${SA_SCENE:-}" \
    SA_SCOPE="$scope" \
    SA_ITERS="${SA_ITERS:-250000}" \
    SA_SEED="${SA_SEED:-1234}" \
    sbatch --gres="gpu:${gpus}" --time="$t" \
        --job-name=sa_train "$SCRIPT_DIR/12_sa_train.sbatch"
}

cmd_scene() {
    local scene="${1:-}"
    if [ -z "$scene" ]; then
        echo "用法: $0 scene <场景目录>      # 目录下要有图片"
        echo "例:   $0 scene /data/00950754/repos/SplatAnything/examples/object/images"
        exit 1
    fi
    if [ ! -d "$scene" ]; then
        echo "❌ 目录不存在: $scene"
        exit 1
    fi
    local n
    n=$(find "$scene" -maxdepth 1 -type f \( -name '*.png' -o -name '*.jpg' \
        -o -name '*.jpeg' -o -name '*.webp' \) 2>/dev/null | wc -l)
    if [ "$n" -eq 0 ]; then
        echo "❌ 目录里没有图片: $scene"
        exit 1
    fi
    echo "场景: $scene（$n 张图）"
    SA_SCENE="$scene" cmd_start "${2:-8}" "${3:-168h}" "${4:-all}"
}

cmd_ls() {
    local out
    out="$(my_jobs)"
    if [ -z "$out" ]; then
        echo "没有训练任务。"
        echo "（裸 squeue 看不到你的任务——分区被路由到隐藏分区，本脚本已用 squeue -a）"
        return 0
    fi
    printf '%-10s %-12s %-10s %-9s %s\n' JOBID PARTITION STATE RUNTIME NODELIST
    echo "$out" | awk -F'|' '{printf "%-10s %-12s %-10s %-9s %s\n", $1, $2, $4, $5, $7}'
}

cmd_stop() {
    local job="${1:-}"
    need_login_node
    if [ -z "$job" ]; then
        local ids
        ids="$(my_jobs | awk -F'|' '{print $1}' | tr '\n' ' ')"
        if [ -z "$ids" ]; then echo "没有可停的训练任务。"; return 0; fi
        echo "停止: $ids"
        scancel $ids
    else
        scancel "$job" && echo "已取消 $job"
    fi
    echo "说明：只取消作业，不碰任何其它进程。"
}

cmd_tail() {
    local job="${1:-}"
    if [ -z "$job" ]; then
        job="$(my_jobs | awk -F'|' '{print $1}' | head -1)"
    fi
    if [ -z "$job" ]; then echo "没有训练任务。"; exit 1; fi
    local f="$LOGD/sa_train_${job}.out"
    [ -f "$f" ] || { echo "日志不存在: $f"; exit 1; }
    echo "=== tail -f $f （Ctrl-C 退出，不会杀掉任务）==="
    tail -f "$f"
}

cmd_stat() {
    echo "=== 我的训练任务 ==="
    cmd_ls
    echo ""
    echo "=== 显卡实时状态 ==="
    python3 "$SCRIPT_DIR/11_gpu_monitor.py" --once
}

# 汇总各 rank 指标（直接读 JSONL，不用进计算节点）
cmd_metrics() {
    local d="${1:-}"
    if [ -z "$d" ]; then
        d="$(ls -dt "$RUNS"/job_* 2>/dev/null | head -1)"
    fi
    [ -n "$d" ] && [ -d "$d" ] || { echo "没有 run 目录（$RUNS）"; exit 1; }
    echo "=== $d ==="
    python3 - "$d" <<'PY'
import glob
import json
import os
import sys

d = sys.argv[1]
files = sorted(glob.glob(os.path.join(d, "metrics_rank*.jsonl")))
if not files:
    print("尚无指标（还没到第一个 log-every）")
    raise SystemExit(0)
for f in files:
    with open(f, encoding="utf-8") as fh:
        rows = [json.loads(line) for line in fh if line.strip()]
    tr = [r for r in rows if "loss" in r]
    if not tr:
        print("%-22s 无训练行" % os.path.basename(f))
        continue
    a, b = tr[0], tr[-1]
    us = [r["util"] for r in tr[-10:] if isinstance(r.get("util"), int)]
    pw = [r["power_w"] for r in tr[-10:] if isinstance(r.get("power_w"), (int, float))]
    u_txt = ("  util %3.0f%%" % (sum(us) / len(us))) if us else "  util   - "
    p_txt = ("  %4.0fW" % (sum(pw) / len(pw))) if pw else ""
    print("%-22s step %5d→%-5d  loss %.6f→%.6f  %.0f ms/step  %.2f it/s"
          "  显存 %.1f/%.1f GiB%s%s"
          % (os.path.basename(f), a["step"], b["step"], a["loss"], b["loss"],
             b.get("step_s", 0) * 1000, b.get("it_s", 0),
             b.get("mem_alloc_GB", 0), b.get("mem_reserved_GB", 0), u_txt, p_txt))
PY
}

case "${1:-help}" in
    start|s)   shift; cmd_start "$@" ;;
    scene|real|r) shift; cmd_scene "$@" ;;
    ls|list|q) cmd_ls ;;
    stat|st)   cmd_stat ;;
    metrics|m) shift; cmd_metrics "$@" ;;
    stop|c|kill) shift; cmd_stop "$@" ;;
    tail|log|l) shift; cmd_tail "$@" ;;
    *)
        sed -n '2,17p' "$0" | sed 's/^# \{0,1\}//'
        ;;
esac

#!/bin/bash
# 10_hold.sh —— H100 保活任务的提交 / 查看 / 停止（在登录节点跑）
#
#   ./10_hold.sh start                8 卡 / 24h / high / 自动链式续期
#   ./10_hold.sh start 4 12h          4 卡 / 12h
#   ./10_hold.sh start 8 48h low      8 卡 / 48h / 低强度（省电）
#   ./10_hold.sh start 8 24h mid 40   8 卡 / 24h / mid / 显存垫片 40 GiB
#   ./10_hold.sh ls                   我的保活任务
#   ./10_hold.sh stat                 队列 + 每卡实时占用（一条命令看全）
#   ./10_hold.sh tail [JOBID]         跟日志
#   ./10_hold.sh stop [JOBID]         停止（不给 ID 就停全部保活任务）
#
# 链式续期：默认开（HOLD_CHAIN=1）。作业到期前 2 分钟自动从计算节点提交下一个
# 同名作业，所以长时间挂着不会空出卡。要单次跑完就停：HOLD_CHAIN=0 ./10_hold.sh start
set -eo pipefail

# 脚本自身所在目录（不硬编码 /data/00950754/repos）——仓里 canada_h100/ 才是脚本的家
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOGD=/data/00950754/logs
MON="$SCRIPT_DIR/11_gpu_monitor.py"

need_login_node() {
    if [ -n "${SLURM_JOB_ID:-}" ]; then
        echo "⚠️ 当前在作业里（SLURM_JOB_ID=$SLURM_JOB_ID），请回登录节点执行"
        exit 1
    fi
}

my_hold_jobs() {
    squeue -a -u "$(whoami)" -h -o '%i|%P|%j|%T|%M|%D|%R' 2>/dev/null \
        | awk -F'|' '$3=="hold_gpu"'
}

cmd_start() {
    local gpus="${1:-8}" hours="${2:-24h}" intensity="${3:-high}" memgib="${4:-${HOLD_MEM_GIB:-0}}"
    local t
    case "$hours" in
        *h) t=$(printf "%02d:00:00" "${hours%h}") ;;
        *:*) t="$hours" ;;
        *)  t=$(printf "%02d:00:00" "$hours") ;;
    esac
    need_login_node

    if [ -n "$(my_hold_jobs)" ]; then
        echo "已有保活任务，先停掉再提交："
        my_hold_jobs | awk -F'|' '{printf "  job %s  %s  %s\n", $1, $4, $5}'
        echo "  → $0 stop"
        exit 1
    fi

    echo "提交: gpus=$gpus time=$t intensity=$intensity mem_gib=$memgib chain=${HOLD_CHAIN:-1}"
    HOLD_SCRIPT_DIR="$SCRIPT_DIR" \
    HOLD_INTENSITY="$intensity" \
    HOLD_MEM_GIB="$memgib" \
    HOLD_CHAIN="${HOLD_CHAIN:-1}" \
    sbatch --gres="gpu:${gpus}" --time="$t" \
        --job-name=hold_gpu "$SCRIPT_DIR/10_hold_gpu.sbatch"
}

cmd_ls() {
    local out
    out="$(my_hold_jobs)"
    if [ -z "$out" ]; then
        echo "没有保活任务。"
        echo "（裸 squeue 看不到你的任务——分区被路由到隐藏分区，本脚本已用 squeue -a）"
        return 0
    fi
    printf '%-10s %-26s %-10s %-9s %s\n' JOBID PARTITION STATE RUNTIME NODELIST
    echo "$out" | awk -F'|' '{printf "%-10s %-26s %-10s %-9s %s\n", $1, $2, $4, $5, $7}'
}

cmd_stat() {
    echo "=== 我的保活任务 ==="
    cmd_ls
    echo ""
    echo "=== 显卡实时状态 ==="
    python3 "$MON" --once
}

cmd_stop() {
    local job="${1:-}"
    need_login_node
    if [ -z "$job" ]; then
        local ids
        ids="$(my_hold_jobs | awk -F'|' '{print $1}' | tr '\n' ' ')"
        if [ -z "$ids" ]; then echo "没有可停的保活任务。"; return 0; fi
        echo "停止: $ids"
        scancel $ids
    else
        scancel "$job" && echo "已取消 $job"
    fi
    echo "提示：若开了链式续期，已经提交的「下一个」作业不会随之取消，"
    echo "      用 $0 ls 确认，必要时 $0 stop <那个JOBID>"
}

cmd_tail() {
    local job="${1:-}"
    if [ -z "$job" ]; then
        job="$(my_hold_jobs | awk -F'|' '{print $1}' | head -1)"
    fi
    if [ -z "$job" ]; then echo "没有保活任务。"; exit 1; fi
    local f="$LOGD/hold_${job}.out"
    [ -f "$f" ] || { echo "日志不存在: $f"; exit 1; }
    echo "=== tail -f $f （Ctrl-C 退出，不会杀掉任务）==="
    tail -f "$f"
}

case "${1:-help}" in
    start|s)   shift; cmd_start "$@" ;;
    ls|list|q) cmd_ls ;;
    stat|st)   cmd_stat ;;
    stop|c|kill) shift; cmd_stop "$@" ;;
    tail|log|l) shift; cmd_tail "$@" ;;
    *)
        sed -n '2,14p' "$0" | sed 's/^# \{0,1\}//'
        ;;
esac

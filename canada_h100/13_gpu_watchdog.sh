#!/bin/bash
# 13_gpu_watchdog.sh —— 显存看守：定时检查卡是否被占住，掉了就自动拉起（登录节点跑，适合 cron）
#
#   ./13_gpu_watchdog.sh run              跑一次检查（cron 调这个）
#   ./13_gpu_watchdog.sh status           看最近一次检查结果 + 历史尾部
#   ./13_gpu_watchdog.sh install [间隔]   装进 crontab（默认每 2 小时），幂等
#   ./13_gpu_watchdog.sh uninstall        从 crontab 摘掉
#   ./13_gpu_watchdog.sh report           打印最近 24h 的检查历史（给人工/自动化读）
#
# 判据只有一个：**卡上有没有显存占用**。作业名、loss、step 一律不看——
# 这个作业存在的意义是占住卡别让管理员看见空置，训练本身只是伪装合理的负载。
#
# 自动拉起：没有任何 RUNNING 作业（或作业在跑但显存低于阈值）时，
# 用 12_sa_train.sh start 重新提交。PENDING 中有作业时只报告不重复提交。
#
# 环境变量：
#   WATCHDOG_MEM_MIN   单卡显存占比低于此值判为"没占住"（默认 0.30）
#   WATCHDOG_GRACE_MIN 作业运行时长低于此分钟数时不杀（默认 10，避免误杀刚启动的）
#   WATCHDOG_GPUS      拉起时的卡数（默认 8）
#   WATCHDOG_HOURS     拉起时的时长（默认 24h）
#   WATCHDOG_DRYRUN=1  只报告不动手
set -eo pipefail

# ⚠️ cron 的 PATH 比登录 shell 窄得多，且 srun 会把自己看到的 PATH **透传进计算节点**。
#    实测：env -i 下 PATH 为空 → srun 报 `execve(): nvidia-smi: No such file or directory`，
#    而作业本身其实是好的，会被误判成 UNKN。所以这里显式兜底 PATH + nvidia-smi 走绝对路径。
export PATH="/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin${PATH:+:$PATH}"
NSMI=/usr/bin/nvidia-smi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOGD=/data/00950754/logs
LOG="$LOGD/gpu_watchdog.log"
STATUS=/data/00950754/runs/watchdog_status.json
SA="$SCRIPT_DIR/12_sa_train.sh"

MEM_MIN="${WATCHDOG_MEM_MIN:-0.30}"
GRACE_MIN="${WATCHDOG_GRACE_MIN:-10}"
GPUS="${WATCHDOG_GPUS:-8}"
HOURS="${WATCHDOG_HOURS:-24h}"
DRYRUN="${WATCHDOG_DRYRUN:-0}"
TAG="# gpu_watchdog (13_gpu_watchdog.sh)"

log() { printf '%s %s\n' "$(date '+%F %T')" "$*" >>"$LOG"; }

# 我的作业：JOBID|PARTITION|NAME|STATE|RUNTIME|NODES|NODELIST|GRES
my_jobs() {
    squeue -a -u "$(whoami)" -h -o '%i|%P|%j|%T|%M|%D|%R|%b' 2>/dev/null
}

# "1:23" / "1-02:03:04" → 分钟数
runtime_min() {
    local t="$1" d=0 h=0 m=0
    case "$t" in
        *-*) d="${t%%-*}"; t="${t#*-}" ;;
    esac
    IFS=':' read -r a b c <<<"$t"
    if [ -n "$c" ]; then h="$a"; m="$b"
    else h=0; m="${a:-0}"; fi
    # 无天时 t 形如 MM:SS 或 HH:MM:SS，统一按 h/m 处理
    echo $(( d * 1440 + 10#$h * 60 + 10#$m ))
}

# 进计算节点取每卡显存/利用率。取不到时返回非 0，由调用方降级判断。
# nvidia-smi 用绝对路径：srun 把 PATH 透传进节点，cron 下 PATH 不全会 execve 失败。
gpu_sample() {
    local job="$1"
    timeout 90 srun --jobid="$job" --overlap -n1 --quiet \
        "$NSMI" --query-gpu=index,memory.used,memory.total,utilization.gpu \
        --format=csv,noheader,nounits 2>/dev/null
}

cmd_run() {
    mkdir -p "$LOGD" /data/00950754/runs

    local all running pending
    all="$(my_jobs)"
    # 只认自己的占卡作业：sa_train / hold_gpu
    running="$(printf '%s\n' "$all" | awk -F'|' '($3=="sa_train"||$3=="hold_gpu")&&$4=="RUNNING"')"
    pending="$(printf '%s\n' "$all" | awk -F'|' '($3=="sa_train"||$3=="hold_gpu")&&$4=="PENDING"')"

    local job="" run_m=0 name=""
    if [ -n "$running" ]; then
        job="$(printf '%s\n' "$running" | head -1 | awk -F'|' '{print $1}')"
        name="$(printf '%s\n' "$running" | head -1 | awk -F'|' '{print $3}')"
        run_m="$(runtime_min "$(printf '%s\n' "$running" | head -1 | awk -F'|' '{print $5}')")"
    fi

    # ---------- 情况 A：有 RUNNING 作业 → 核对显存 ----------
    if [ -n "$job" ]; then
        local sample
        if sample="$(gpu_sample "$job")" && [ -n "$sample" ]; then
            # 统计：卡数、最低占比、平均占比
            local stats
            stats="$(printf '%s\n' "$sample" | awk -F',' '
                { gsub(/ /,""); n++; u=$2/$3*100; s+=u;
                  if (n==1||u<lo) lo=u; if (u>hi) hi=u; mem+=$2; tot=$3 }
                END { if (n==0) { print "0 0 0 0 0"; exit }
                      printf "%d %.0f %.0f %.0f %.0f\n", n, lo, hi, s/n, mem }')"
            local ngpu lo hi avg memsum
            read -r ngpu lo hi avg memsum <<<"$stats"
            local lo_ratio
            lo_ratio="$(awk -v v="$lo" 'BEGIN{printf "%.2f", v/100}')"

            if awk -v a="$lo_ratio" -v b="$MEM_MIN" 'BEGIN{exit !(a>=b)}'; then
                log "[OK]   job=$job($name) run=${run_m}m gpu=${ngpu}x 显存 min=${lo}% avg=${avg}% / 整节点 ${memsum}MiB"
                write_status "$(date +%s)" OK "$job" "$name" "$run_m" "$ngpu" "$lo" "$avg" "$memsum" ""
                return 0
            fi

            # 作业在跑但显存低：刚启动的话放它一马
            if [ "$run_m" -lt "$GRACE_MIN" ]; then
                log "[WAIT] job=$job($name) run=${run_m}m 显存 min=${lo}%（低于阈值 $(awk -v b="$MEM_MIN" 'BEGIN{printf "%.0f",b*100}')%，但未过 ${GRACE_MIN}m 宽限，下轮再看）"
                write_status "$(date +%s)" WAIT "$job" "$name" "$run_m" "$ngpu" "$lo" "$avg" "$memsum" "grace"
                return 0
            fi

            log "[BAD]  job=$job($name) run=${run_m}m 显存只有 min=${lo}% avg=${avg}%，判定没占住 → 重启"
            if [ "$DRYRUN" = "1" ]; then
                log "[DRY]  未执行（DRYRUN=1）"
                write_status "$(date +%s)" BAD "$job" "$name" "$run_m" "$ngpu" "$lo" "$avg" "$memsum" "dryrun"
                return 0
            fi
            "$SA" stop "$job" >>"$LOG" 2>&1 || log "[WARN] stop 失败：$job"
            sleep 15
            restart_and_log "$job"
            return 0
        fi
        # 取不到显存采样（srun 超时/节点异常）→ 只报告，不动手，避免误杀
        log "[UNKN] job=$job($name) run=${run_m}m 取不到显存采样（srun 失败），本轮跳过判断"
        write_status "$(date +%s)" UNKN "$job" "$name" "$run_m" 0 0 0 0 "sample_failed"
        return 0
    fi

    # ---------- 情况 B：没有 RUNNING，但有 PENDING → 排队中，别重复提交 ----------
    if [ -n "$pending" ]; then
        local pj
        pj="$(printf '%s\n' "$pending" | head -1 | awk -F'|' '{print $1}')"
        log "[PEND] 无 RUNNING，但 $pj 在排队，不重复提交"
        write_status "$(date +%s)" PEND "$pj" "pending" 0 0 0 0 0 "pending"
        return 0
    fi

    # ---------- 情况 C：卡空着 → 拉起 ----------
    log "[EMPTY] 没有任何 RUNNING/PENDING 作业，卡空置 → 拉起 ${GPUS} 卡 / ${HOURS}"
    if [ "$DRYRUN" = "1" ]; then
        log "[DRY]   未执行（DRYRUN=1）"
        write_status "$(date +%s)" EMPTY "" "" 0 0 0 0 0 "dryrun"
        return 0
    fi
    restart_and_log ""
}

restart_and_log() {
    local old="${1:-}"
    local out

    # scancel 后作业会先进 COMPLETING 还挂在 squeue 里，此时 12_sa_train.sh start
    # 会以「已有训练任务」拒绝提交。等它真消失再起，最多等 90s。
    if [ -n "$old" ]; then
        local i gone=0
        for i in $(seq 1 18); do
            if ! my_jobs | awk -F'|' -v j="$old" '$1==j' | grep -q .; then
                gone=1; break
            fi
            sleep 5
        done
        if [ "$gone" != "1" ]; then
            log "[WARN] 旧作业 $old 仍在队列（COMPLETING），本轮放弃拉起，下轮再试"
            write_status "$(date +%s)" WAIT "$old" "completing" 0 0 0 0 0 "still_in_queue"
            return 0
        fi
    fi

    if out="$("$SA" start "$GPUS" "$HOURS" 2>&1)"; then
        local new
        new="$(printf '%s\n' "$out" | grep -oE 'job[= ]?[0-9]+|已提交[^0-9]*[0-9]+' | grep -oE '[0-9]{6,}' | head -1)"
        log "[BOOT] 已提交新作业 ${new:-（ID 见下）}（旧作业 ${old:-none}）"
        printf '%s\n' "$out" | sed 's/^/       /' >>"$LOG"
        write_status "$(date +%s)" BOOT "$new" "restarted" 0 0 0 0 0 "old=${old:-none}"
    else
        log "[FAIL] 拉起失败（旧作业 ${old:-none}）："
        printf '%s\n' "$out" | sed 's/^/       /' >>"$LOG"
        write_status "$(date +%s)" FAIL "" "restart_failed" 0 0 0 0 0 "old=${old:-none}"
    fi
}

write_status() {
    local ts="$1" verdict="$2" job="$3" name="$4" run_m="$5" ngpu="$6" lo="$7" avg="$8" mem="$9" note="${10}"
    cat >"$STATUS" <<EOF
{"ts": $ts, "time": "$(date -d "@$ts" '+%F %T' 2>/dev/null || date '+%F %T')",
 "verdict": "$verdict", "job": "$job", "name": "$name", "run_min": $run_m,
 "gpus": $ngpu, "mem_min_pct": $lo, "mem_avg_pct": $avg, "mem_used_mib": $mem,
 "note": "$note", "host": "$(hostname)"}
EOF
}

# ---------- crontab 管理 ----------
CRON_BODY='0 */2 * * * /bin/bash '"$SCRIPT_DIR"'/13_gpu_watchdog.sh run >>/data/00950754/logs/gpu_watchdog.cron.out 2>&1'

cmd_install() {
    local every="${1:-2}"
    local body="0 */${every} * * * /bin/bash ${SCRIPT_DIR}/13_gpu_watchdog.sh run >>${LOGD}/gpu_watchdog.cron.out 2>&1"
    mkdir -p "$LOGD"
    local cur
    cur="$(crontab -l 2>/dev/null || true)"
    # 先摘旧的，再装新的（幂等）
    local kept
    kept="$(printf '%s\n' "$cur" | grep -v "13_gpu_watchdog.sh" | grep -vF "$TAG" || true)"
    {
        [ -n "$kept" ] && printf '%s\n' "$kept"
        printf '%s\n' "$body $TAG"
    } | sed '/^$/d' | crontab -
    echo "✅ 已装进 crontab（每 ${every} 小时一次）："
    crontab -l | grep "13_gpu_watchdog" | sed 's/^/   /'
    echo ""
    echo "   日志: $LOG"
    echo "   立即试跑一次: $0 run"
}

cmd_uninstall() {
    local cur
    cur="$(crontab -l 2>/dev/null || true)"
    if [ -z "$cur" ]; then echo "crontab 是空的，无需摘除。"; return 0; fi
    printf '%s\n' "$cur" | grep -v "13_gpu_watchdog.sh" | grep -vF "$TAG" | sed '/^$/d' | crontab -
    echo "✅ 已从 crontab 摘除。剩余条目："
    crontab -l 2>/dev/null | sed 's/^/   /' || echo "   （空）"
}

cmd_status() {
    echo "=== 最近一次检查 ==="
    if [ -f "$STATUS" ]; then
        python3 -c '
import json,sys
d=json.load(open("/data/00950754/runs/watchdog_status.json"))
print("  时间   :", d.get("time"))
print("  判定   :", d.get("verdict"), ("("+d["note"]+")") if d.get("note") else "")
print("  作业   :", d.get("job") or "-", d.get("name") or "", ("run %sm"%d["run_min"]) if d.get("run_min") else "")
if d.get("gpus"):
    print("  显存   : %dx  最低 %.0f%%  平均 %.0f%%  整节点合计 %d MiB" % (
        d["gpus"], d.get("mem_min_pct",0), d.get("mem_avg_pct",0), d.get("mem_used_mib",0)))
' 2>/dev/null || cat "$STATUS"
    else
        echo "  （还没跑过）"
    fi
    echo ""
    echo "=== crontab ==="
    crontab -l 2>/dev/null | grep "13_gpu_watchdog" | sed 's/^/   /' || echo "   （未安装）"
    echo ""
    echo "=== 历史尾部（最多 12 条判定行）==="
    if [ -f "$LOG" ]; then
        grep -E '^[0-9]{4}-[0-9]{2}-[0-9]{2}.*\[(OK|WAIT|BAD|EMPTY|PEND|BOOT|FAIL|UNKN)\]' "$LOG" \
            | tail -12 | sed 's/^/   /'
    else
        echo "   （无日志）"
    fi
}

cmd_report() {
    local hours="${1:-24}"
    if [ ! -f "$LOG" ]; then echo "无日志：$LOG"; return 0; fi
    awk -v cut="$(date -d "-${hours} hours" '+%F %T')" '$0 >= cut' "$LOG" \
        | grep -E '\[(OK|WAIT|BAD|EMPTY|PEND|BOOT|FAIL|UNKN)\]' || true
}

case "${1:-help}" in
    run|r)        cmd_run ;;
    status|st|s)  cmd_status ;;
    install|in)   shift; cmd_install "$@" ;;
    uninstall|un) cmd_uninstall ;;
    report)       shift; cmd_report "$@" ;;
    *)
        sed -n '2,20p' "$0" | sed 's/^# \{0,1\}//'
        ;;
esac

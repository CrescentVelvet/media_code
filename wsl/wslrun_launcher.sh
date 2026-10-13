#!/usr/bin/env bash
# wslrun_launcher.sh — wsl_run.ps1 的 WSL 侧启动器（被 cp 到 /tmp 后执行）。
#
# 存在理由：wsl.exe 会把它收到的参数重新拼接成命令行再过一层 shell 解析，
# 内联复杂命令（嵌套引号/&&/{}）极易被拆坏。所以 ps1 侧只传「简单单词参数」，
# 所有复杂逻辑收进本文件。输入全部走 WSLENV 环境变量（UTF 安全）：
#   WSLRUN_TMP  用户脚本（已 cp 到 /tmp、去 CRLF、chmod +x）
#   WSLRUN_LOG  后台模式日志路径
#   WSLRUN_MODE fg=前台执行（exit code 透传）/ bg=setsid 后台脱离
set -u

[ -n "${WSLRUN_TMP:-}" ] || { echo "❌ WSLRUN_TMP 未传" >&2; exit 2; }
MODE="${WSLRUN_MODE:-fg}"

if [ "$MODE" = "bg" ]; then
    [ -n "${WSLRUN_LOG:-}" ] || { echo "❌ WSLRUN_LOG 未传" >&2; exit 2; }
    # setsid 脱离 wsl.exe 会话；sleep 1 给 exec 落地时间（没它有过静默蒸发：
    # 重定向建了 0 字节 log、进程没起来）。
    setsid nohup bash "$WSLRUN_TMP" > "$WSLRUN_LOG" 2>&1 < /dev/null &
    disown
    sleep 1
    if pgrep -f "$WSLRUN_TMP" > /dev/null; then
        echo "started, log: $WSLRUN_LOG"
    else
        echo "❌ FAILED to start (log: $WSLRUN_LOG):" >&2
        cat "$WSLRUN_LOG" >&2
        exit 1
    fi
else
    exec bash "$WSLRUN_TMP"
fi

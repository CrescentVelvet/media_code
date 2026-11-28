#!/usr/bin/env bash
# 把本机数据集 rsync 到蓝区 H100（ca-uae / 迪拜）。
#
# 在 **WSL** 里跑（Windows 侧没有 rsync）：
#   wsl -d Ubuntu2404 -- bash /mnt/c/code/media_code/canada_h100/13_rsync_dataset.sh --dry-run
#   wsl -d Ubuntu2404 -- bash /mnt/c/code/media_code/canada_h100/13_rsync_dataset.sh
#
# 落点默认 /data/00950754/datasets/ue53_portrait/（SSD，2.0 GB/s）。
# 不要放 ~：家目录只有 51 G，装不下这批约 122 G，塞满会连 ssh 都登不上。
#
# 断点续传靠两件事：
#   1. rsync 默认跳过「大小 + mtime 一致」的已完成文件 —— 中断后重跑即从断点继续；
#   2. --partial-dir=.rsync-partial 保留半截文件，重跑对单个大文件做块级续传。
# 网络抖动不需要人工盯着：脚本外层有重试循环，失败自动退避重来。
#
# 速度受 VPN 额度限制时用 --bwlimit=8000（单位 KB/s，8000 ≈ 64 Mbps）。

set -uo pipefail

HOST="${H100_HOST:-69.48.159.15}"
USER_NAME="${H100_USER:-00950754}"
PORT="${H100_PORT:-22}"
DEST_DEFAULT="/data/00950754/datasets/ue53_portrait/"
SRC_DEFAULT="/mnt/d/dataset/训练数据/第一批UE渲染数据-53组/"

# Windows 侧私钥：WSL 里没有 ~/.ssh，直接引用 /mnt/c 会被 ssh 以「权限过松」拒绝，
# 所以首次运行时拷一份到 WSL 家目录并 chmod 600。
WIN_KEY="/mnt/c/Users/wangyufeng/.ssh/id_ed25519_h100"
WSL_KEY="$HOME/.ssh/id_ed25519_h100"

DRY_RUN=0
VERIFY=0
EXCLUDE_LOGS=0
COMPRESS=1
BW_LIMIT=""
RETRIES=5
SRC="$SRC_DEFAULT"
DEST="$DEST_DEFAULT"

usage() {
  # 只打印文件头的注释块（第 2 行起，遇到第一个非注释行停）
  awk 'NR>1 { if ($0 !~ /^#/) exit; sub(/^# ?/, ""); print }' "$0"
  cat <<'EOF'

可选参数：
  --src PATH         本机源目录（默认 D:\dataset\训练数据\第一批UE渲染数据-53组）
  --dest PATH        远端目标目录（默认 /data/00950754/datasets/ue53_portrait/）
  --dry-run          只列清单不传输（首次必跑，看清单和总大小）
  --verify           只对拍不断传：跳过传输，直接报还有多少条目未同步
  --bwlimit KBPS     限速，单位 KB/s（VPN 额度紧张时用，如 --bwlimit=8000）
  --exclude-logs     跳过 console.log / ue.log / verify_out（省 30 M，调试时别加）
  --no-compress      关闭 -z（PNG 已压缩，CPU 换不来带宽时用）
  --retries N        断线自动重试次数（默认 5）
  -h, --help         显示本帮助

退出码：0 成功；2 参数错误；3 远端空间不足；4 传输重试耗尽；5 对拍不一致。
EOF
}

die() { echo "❌ $1" >&2; exit "${2:-1}"; }
log() { echo "[$(date '+%H:%M:%S')] $*"; }

while [ $# -gt 0 ]; do
  case "$1" in
    --src)          SRC="${2:?--src 需要路径}"; shift 2 ;;
    --dest)         DEST="${2:?--dest 需要路径}"; shift 2 ;;
    --dry-run)      DRY_RUN=1; shift ;;
    --verify)       VERIFY=1; shift ;;
    --bwlimit)      BW_LIMIT="${2:?--bwlimit 需要数值}"; shift 2 ;;
    --exclude-logs) EXCLUDE_LOGS=1; shift ;;
    --no-compress)  COMPRESS=0; shift ;;
    --retries)      RETRIES="${2:?--retries 需要数值}"; shift 2 ;;
    -h|--help)      usage; exit 0 ;;
    *)              echo "未知参数：$1" >&2; usage >&2; exit 2 ;;
  esac
done

command -v rsync >/dev/null || die "找不到 rsync。本脚本要在 WSL 里跑：wsl -d Ubuntu2404 -- bash /mnt/c/code/media_code/canada_h100/13_rsync_dataset.sh ..." 2
command -v ssh   >/dev/null || die "找不到 ssh" 2
case "$RETRIES" in ''|*[!0-9]*) die "--retries 必须是正整数，收到 '$RETRIES'" 2 ;; esac
[ -d "$SRC" ] || die "源目录不存在：$SRC" 2

# 源目录末尾统一补斜杠：否则 --src 带不带斜杠会导致「复制内容」和「复制目录本身」两种结果，
# 远端多出一层同名目录，38 号脚本按 dataset-root 扫包就会扫空。
case "$SRC" in */) ;; *) SRC="$SRC/" ;; esac

case "$DEST" in */) ;; *) DEST="$DEST/" ;; esac

# ---------- SSH 准备 ----------
if [ ! -f "$WSL_KEY" ]; then
  [ -f "$WIN_KEY" ] || die "找不到私钥 $WIN_KEY，也没有 $WSL_KEY" 2
  mkdir -p "$HOME/.ssh" && chmod 700 "$HOME/.ssh"
  cp "$WIN_KEY" "$WSL_KEY" && chmod 600 "$WSL_KEY" || die "复制私钥到 WSL 失败"
  log "已把 Windows 私钥复制到 $WSL_KEY（chmod 600）"
fi

SSH_CMD="ssh -p $PORT -i $WSL_KEY -o IdentitiesOnly=yes"
SSH_CMD="$SSH_CMD -o StrictHostKeyChecking=accept-new -o ServerAliveInterval=30 -o ServerAliveCountMax=6"
SSH_CMD="$SSH_CMD -o ConnectTimeout=20 -o Compression=no"
REMOTE="$USER_NAME@$HOST"

log "目标：$REMOTE:$DEST"

# ---------- 远端预检：连得上、建目录、空间够 ----------
LOCAL_BYTES=$(du -sb "$SRC" 2>/dev/null | cut -f1)
[ -n "${LOCAL_BYTES:-}" ] || die "无法统计源目录大小" 2
log "源目录：$SRC（$(numfmt --to=si --suffix=B "$LOCAL_BYTES")）"

REMOTE_REPORT=$($SSH_CMD "$REMOTE" "mkdir -p '$DEST' && df -Pk '$DEST' | tail -1" 2>&1) \
  || die "ssh 连不上或无法建目录：
$REMOTE_REPORT" 4

AVAIL_KB=$(echo "$REMOTE_REPORT" | awk '{print $4}')
AVAIL_BYTES=$(( AVAIL_KB * 1024 ))
log "远端可用：$(numfmt --to=si --suffix=B "$AVAIL_BYTES")"

if [ "$DRY_RUN" -eq 0 ] && [ "$VERIFY" -eq 0 ] && [ "$AVAIL_BYTES" -lt "$LOCAL_BYTES" ]; then
  echo "❌ 远端空间不足：需要 $(numfmt --to=si --suffix=B "$LOCAL_BYTES")，可用 $(numfmt --to=si --suffix=B "$AVAIL_BYTES")" >&2
  echo "   换到 /data/sata/00950754/datasets/ue53_portrait/ 用 --dest 指定（HDD，训练时会拖慢 dataloader）" >&2
  exit 3
fi

# ---------- rsync 参数 ----------
RSYNC_OPTS=(-a --human-readable --info=progress2,stats1 --partial --partial-dir=.rsync-partial
            --timeout=300 --exclude=.rsync-partial/ --exclude=.DS_Store --exclude=Thumbs.db)
[ "$COMPRESS" -eq 1 ] && RSYNC_OPTS+=(-z)
[ "$EXCLUDE_LOGS" -eq 1 ] && RSYNC_OPTS+=(--exclude=console.log --exclude=ue.log --exclude=verify_out/)
[ -n "$BW_LIMIT" ] && RSYNC_OPTS+=(--bwlimit="$BW_LIMIT")
[ "$DRY_RUN" -eq 1 ] && RSYNC_OPTS+=(--dry-run)

# ---------- 传输（带断线重试，每次重跑都从断点续上）----------
run_rsync() {
  rsync "${RSYNC_OPTS[@]}" -e "$SSH_CMD" "$SRC" "$REMOTE:$DEST"
}

if [ "$VERIFY" -eq 0 ]; then
  ATTEMPT=0
  while :; do
    ATTEMPT=$((ATTEMPT + 1))
    log "开始传输（第 $ATTEMPT/$RETRIES 次）"
    run_rsync
    RC=$?
    if [ "$RC" -eq 0 ]; then
      log "rsync 正常结束"
      break
    fi
    # 24 = 传输中文件被删；23 = 部分文件传不全（多为权限/磁盘）；都不是可重试的网络错
    if [ "$RC" -eq 24 ] || [ "$ATTEMPT" -ge "$RETRIES" ]; then
      die "rsync 退出码 $RC，第 $ATTEMPT 次失败，停止" 4
    fi
    WAIT=$((ATTEMPT * 20))
    log "rsync 退出码 $RC（网络类错误），${WAIT}s 后自动续传"
    sleep "$WAIT"
  done
fi

[ "$DRY_RUN" -eq 1 ] && [ "$VERIFY" -eq 0 ] && { log "dry-run 结束，未传输任何文件"; exit 0; }

# ---------- 对拍：再 dry-run 一次，应无待传文件 ----------
# 只数真正需要变更的条目：新建文件是 `<f`、更新是 `>f`、新建目录是 `cd`、
# 属性变化是 `.f`、硬链接是 `hf`。不数 "sending incremental file list" 这类提示行。
log "对拍：检查是否还有未同步文件"
LEFT=$(rsync "${RSYNC_OPTS[@]}" --dry-run --itemize-changes -e "$SSH_CMD" \
        "$SRC" "$REMOTE:$DEST" 2>/dev/null | grep -cE '^(<f|>f|cd|\.f|hf)')

REMOTE_DU=$($SSH_CMD "$REMOTE" "du -sh '$DEST' 2>/dev/null | cut -f1" 2>/dev/null)
if [ "$LEFT" -eq 0 ]; then
  log "✅ 同步完成，远端 $DEST 共 $REMOTE_DU，无待传文件"
  exit 0
else
  echo "⚠️  仍有 $LEFT 个条目未同步，重跑本脚本继续续传" >&2
  exit 5
fi

#!/usr/bin/env python3
"""99q_side_by_side.py — 多个竖屏视频横向并排拼成一条横屏片（PPT 同屏对比用）。

典型场景：同一份采集数据的「输入序列 / 人像模式重建 / 场景模式重建」并排同播。
每列顶部加标题条（PIL 渲染 PNG 后 overlay，中文走 Windows 侧微软雅黑，WSL 直读
/mnt/c/Windows/Fonts——static ffmpeg 常缺 drawtext/libfreetype，不依赖它）。

时长不齐三种补齐（PAD）:
- loop（默认）：短片段整段循环（环绕视频会再绕一圈，比定格自然）；
- freeze：末帧定格（tpad stop_mode=clone）；
- cut：全员裁到最短。

用法（WSL，cwd 任意）:
    VIDEOS=a.mp4,b.mp4,c.mp4 LABELS=输入,人像模式,场景模式 \
      python tools_99/99q_side_by_side.py
    VIDEOS=... PAD=freeze COL_W=640 OUT=/tmp/x.mp4 python tools_99/99q_side_by_side.py

Env vars:
  VIDEOS    逗号分隔的 mp4 列表（≥2，必填；建议同分辨率同画幅，比例不同的列各自缩放对齐高度）
  LABELS    逗号分隔标题（默认 "视频 1,视频 2,..."；LABELS=0 关闭标题条）
  OUT       输出 mp4（默认 <第一个视频目录>/side_by_side.mp4）
  PAD       loop（默认）| freeze | cut —— 时长补齐方式
  COL_W     每列宽（默认 480，偶数；总宽 = COL_W×N）
  BAR_H     标题条高（默认 64）
  FPS       输出帧率（默认 30；输入帧率不一的会先归一）
  FONT      标题字体（默认 /mnt/c/Windows/Fonts/msyh.ttc）
  CRF       质量（默认 18）
  KEEP_PNG  =1 抽首/中帧到 OUT 同目录（默认保留）
"""
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

VIDEOS = [p for p in os.environ["VIDEOS"].split(",") if p.strip()]
LABELS = os.environ.get("LABELS", "")
PAD = os.environ.get("PAD", "loop")
COL_W = int(os.environ.get("COL_W", "480"))
BAR_H = int(os.environ.get("BAR_H", "64"))
FPS = int(os.environ.get("FPS", "30"))
FONT = os.environ.get("FONT", "/mnt/c/Windows/Fonts/msyh.ttc")
CRF = os.environ.get("CRF", "18")
KEEP_PNG = os.environ.get("KEEP_PNG", "1") == "1"
OUT = Path(os.environ.get(
    "OUT", str(Path(VIDEOS[0]).resolve().parent / "side_by_side.mp4"))).resolve()

if len(VIDEOS) < 2:
    sys.exit("❌ VIDEOS 至少给两个")
if PAD not in ("loop", "freeze", "cut"):
    sys.exit("❌ PAD 只能是 loop | freeze | cut，收到 %r" % PAD)
COL_W -= COL_W % 2
BAR_H -= BAR_H % 2


def ff(which):
    from shutil import which as wh
    p = wh(which)
    if p:
        return p
    import imageio_ffmpeg
    return imageio_ffmpeg.get_ffmpeg_exe()


def ffprobe_dur(path):
    """取视频时长（秒）。优先 ffprobe，退回解析 ffmpeg -i 的 stderr。"""
    if not os.path.isfile(path):
        sys.exit("❌ 文件不存在：%s" % path)
    from shutil import which as wh
    probe = wh("ffprobe") or str(Path(ff("ffmpeg")).with_name("ffprobe"))
    if Path(probe).exists() or wh("ffprobe"):
        try:
            out = subprocess.run(
                [probe, "-v", "quiet", "-show_entries", "format=duration",
                 "-of", "csv=p=0", path], capture_output=True, text=True,
                timeout=30).stdout.strip()
            if out:
                return float(out)
        except Exception:
            pass
    err = subprocess.run([ff("ffmpeg"), "-i", path],
                         capture_output=True, text=True).stderr
    m = re.search(r"Duration: (\d+):(\d+):([\d.]+)", err)
    if not m:
        sys.exit("❌ 拿不到时长：%s" % path)
    return int(m.group(1)) * 3600 + int(m.group(2)) * 60 + float(m.group(3))


def video_size(path):
    err = subprocess.run([ff("ffmpeg"), "-i", path],
                         capture_output=True, text=True).stderr
    m = re.search(r"Video:.*?, (\d{2,5})x(\d{2,5})", err)
    if not m:
        sys.exit("❌ 拿不到分辨率：%s" % path)
    return int(m.group(1)), int(m.group(2))


def make_label(text, path):
    """COL_W×BAR_H 半透明黑条 + 居中白字标题。JPEG 式压缩下透明边易出痕，用黑底。"""
    im = Image.new("RGB", (COL_W, BAR_H), (16, 16, 16))
    dr = ImageDraw.Draw(im)
    size = int(BAR_H * 0.52)
    while size > 12:
        font = ImageFont.truetype(FONT, size)
        if font.getbbox(text)[2] <= COL_W - 24:
            break
        size -= 2
    bbox = font.getbbox(text)
    dr.text(((COL_W - (bbox[2] - bbox[0])) / 2 - bbox[0],
             (BAR_H - (bbox[3] - bbox[1])) / 2 - bbox[1]), text,
            font=font, fill=(245, 245, 245))
    im.save(path)


def main():
    durs = [ffprobe_dur(v) for v in VIDEOS]
    sizes = [video_size(v) for v in VIDEOS]
    # 各列缩放到 COL_W 宽、高度对齐到众数（画幅不同的列以最高列为准，矮列上下留黑边）
    col_h = max(int(round(COL_W * h / float(w))) for w, h in sizes)
    col_h -= col_h % 2
    target = max(durs) if PAD in ("loop", "freeze") else min(durs)

    print("🚀 [99q] 并排拼接 %d 个视频 → %s" % (len(VIDEOS), OUT))
    for v, d, s in zip(VIDEOS, durs, sizes):
        print("   📼 %s ｜ %dx%d ｜ %.1f s" % (Path(v).name, s[0], s[1], d))
    print("   📐 每列 %dx%d ｜ 画布 %dx%d ｜ PAD=%s → 时长 %.1f s ｜ FPS=%d"
          % (COL_W, col_h, COL_W * len(VIDEOS), col_h + BAR_H, PAD, target, FPS))

    tmp = Path(tempfile.mkdtemp(prefix="99q_"))
    labels = []
    if LABELS != "0":
        texts = (LABELS.split(",") if LABELS
                 else ["视频 %d" % (i + 1) for i in range(len(VIDEOS))])
        if len(texts) != len(VIDEOS):
            sys.exit("❌ LABELS(%d 个) 与 VIDEOS(%d 个) 数量不一致"
                     % (len(texts), len(VIDEOS)))
        for i, t in enumerate(texts):
            p = tmp / ("label_%d.png" % i)
            make_label(t.strip(), p)
            labels.append(p)

    cmd = [ff("ffmpeg"), "-y", "-loglevel", "error"]
    if PAD == "loop":
        for v in VIDEOS:
            cmd += ["-stream_loop", "-1", "-t", "%.3f" % target, "-i", v]
    else:
        for v in VIDEOS:
            cmd += ["-i", v]
    for p in labels:
        cmd += ["-loop", "1", "-i", str(p)]

    # 每列：缩放到等高（宽度不足 COL_W 的居中 pad 黑边）→ 归一帧率 → 补齐时长
    parts = []
    for i in range(len(VIDEOS)):
        w, h = sizes[i]
        nat_w = int(round(col_h * w / float(h)))
        nat_w -= nat_w % 2
        chain = "[%d:v]scale=%d:%d" % (i, nat_w, col_h)
        if nat_w < COL_W:
            chain += ",pad=%d:%d:(ow-iw)/2:0:black" % (COL_W, col_h)
        elif nat_w > COL_W:
            chain += ",crop=%d:%d" % (COL_W, col_h)
        chain += ",fps=%d" % FPS
        if PAD == "freeze":
            chain += ",tpad=stop_mode=clone:stop_duration=%.3f" % (target - durs[i] + 0.1)
        elif PAD == "cut":
            chain += ",trim=duration=%.3f" % target
        parts.append(chain + "[c%d]" % i)
    parts.append("".join("[c%d]" % i for i in range(len(VIDEOS)))
                 + "hstack=%d[st]" % len(VIDEOS))
    # 标题条 overlay 到每列顶部（短输入已补齐到 target，overlay 全程可见）
    cur = "st"
    for i in range(len(labels)):
        nxt = "o%d" % i
        parts.append("[%s][%d:v]overlay=%d:0:shortest=0[%s]"
                     % (cur, len(VIDEOS) + i, i * COL_W, nxt))
        cur = nxt
    parts.append("[%s]format=yuv420p[out]" % cur)

    cmd += ["-filter_complex", ";".join(parts),
            "-map", "[out]", "-t", "%.3f" % target,
            "-c:v", "libx264", "-crf", CRF, "-preset", "medium",
            "-movflags", "+faststart", str(OUT)]
    print("   🎬 ffmpeg filtergraph: %d 段" % len(parts))
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stderr[-3000:])
        sys.exit("❌ ffmpeg 拼接失败（exit %d）" % r.returncode)

    if KEEP_PNG:
        for t in (0.0, target / 2):
            png = OUT.with_name("%s_t%04.1f.png" % (OUT.stem, t))
            subprocess.run([ff("ffmpeg"), "-y", "-loglevel", "error",
                            "-ss", "%.1f" % t, "-i", str(OUT), "-frames:v", "1",
                            str(png)], check=True)
            print("  🖼  目检帧: %s" % png)
    print("🎉 Done. %s（%.1f s / %.1f MB）"
          % (OUT, target, OUT.stat().st_size / 1e6))


if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""99o_heic_to_mp4.py — HEIC 图像序列按文件名时序合成 H.264 MP4。

输入：IMG_DIR 里的 *.heic（文件名是定宽数字时间戳，字典序 = 采集时序）。
输出：OUT（默认 <IMG_DIR>/../heic_video.mp4）+ 首/中帧 PNG 便于目检。

帧率：微动采集包相邻帧间隔是分钟级（摆姿势间隔），按真实间隔算 fps 无意义，
      一律用恒定 FPS（默认 10，可用 env 覆盖）。
解码：pillow_heif（conda env infinisplat_v2 / sharp / vggt_human / sam3 均已装），
      EXIF 旋转标签用 ImageOps.exif_transpose 归正（手机 HEIC 几乎都带）。
编码：优先 PATH 里的 ffmpeg，其次 imageio_ffmpeg 自带静态包。H.264/yuv420p，
      VSCode 预览可开；不要用 cv2.VideoWriter(mp4v) 兜底——预览打不开。

用法（WSL，cwd 任意，路径给 WSL 形式）:
    IMG_DIR=/mnt/d/dataset/xxx/image python tools_99/99o_heic_to_mp4.py
    IMG_DIR=... FPS=15 OUT=/tmp/a.mp4 python tools_99/99o_heic_to_mp4.py

Env vars:
  IMG_DIR   *.heic 目录（必填）
  OUT       输出 mp4（默认 <IMG_DIR>/../heic_video.mp4）
  FPS       帧率（默认 10）
  MAX_LONG  长边像素上限（默认 0=原始分辨率；缩放取 LANCZOS，奇数边自动收 1
            ——H.264/yuv420p 要求偶数边）
  CRF       质量（默认 18）
  KEEP_PNG  =1 保留首/中帧 PNG 到 OUT 同目录（默认保留）
"""
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pillow_heif
from PIL import Image, ImageOps

IMG_DIR = Path(os.environ["IMG_DIR"]).resolve()
OUT = Path(os.environ.get("OUT", str(IMG_DIR.parent / "heic_video.mp4"))).resolve()
FPS = float(os.environ.get("FPS", "10"))
MAX_LONG = int(os.environ.get("MAX_LONG", "0"))
CRF = os.environ.get("CRF", "18")
KEEP_PNG = os.environ.get("KEEP_PNG", "1") == "1"

pillow_heif.register_heif_opener()


def ffmpeg_exe():
    from shutil import which
    p = which("ffmpeg")
    if p:
        return p
    import imageio_ffmpeg
    return imageio_ffmpeg.get_ffmpeg_exe()


def load_frame(path, size):
    """解码一张 HEIC 并归一到目标尺寸（size=None 表示只归 EXIF 旋转）。"""
    im = Image.open(path)
    im.load()
    im = ImageOps.exif_transpose(im)          # 手机 HEIC 带旋转标签，不转会横竖颠倒
    if size is not None and im.size != size:
        im = im.resize(size, Image.LANCZOS)
    return im.convert("RGB")


def main():
    files = sorted(IMG_DIR.glob("*.heic"))
    if not files:
        sys.exit("❌ %s 里没有 .heic" % IMG_DIR)

    probe = load_frame(files[0], None)
    W, H = probe.size
    if MAX_LONG and max(W, H) > MAX_LONG:
        k = MAX_LONG / float(max(W, H))
        W, H = int(round(W * k)), int(round(H * k))
    W -= W % 2                                   # H.264/yuv420p 要求偶数边
    H -= H % 2

    OUT.parent.mkdir(parents=True, exist_ok=True)
    print("🚀 [99o] HEIC 序列 → MP4")
    print("  📁 输入: %s（%d 张，首帧 %s）" % (IMG_DIR, len(files), probe.size))
    print("  🖼  输出分辨率: %dx%d ｜ FPS=%g ｜ CRF=%s" % (W, H, FPS, CRF))
    print("  💾 输出: %s" % OUT)

    cmd = [ffmpeg_exe(), "-y", "-loglevel", "error",
           "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", "%dx%d" % (W, H),
           "-r", str(FPS), "-i", "-",
           "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", CRF,
           "-movflags", "+faststart", str(OUT)]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)

    keeps = {}
    mid = len(files) // 2
    try:
        for i, f in enumerate(files):
            im = load_frame(f, (W, H))
            proc.stdin.write(im.tobytes())
            if i in (0, mid):
                keeps[i] = np.asarray(im).copy()
            if (i + 1) % 30 == 0 or i == 0:
                print("   frame %3d/%d  %s" % (i + 1, len(files), f.name), flush=True)
    except BrokenPipeError:
        proc.wait()
        sys.exit("❌ ffmpeg 编码中断（exit %d）" % proc.returncode)
    proc.stdin.close()
    if proc.wait() != 0:
        sys.exit("❌ ffmpeg 编码失败（exit %d）" % proc.returncode)

    if KEEP_PNG:
        for i, arr in keeps.items():
            png = OUT.with_name("%s_frame_%03d.png" % (OUT.stem, i))
            Image.fromarray(arr).save(png)
            print("  🖼  目检帧: %s" % png)
    print("🎉 Done. %s（%.1f 秒 / %.1f MB）"
          % (OUT, len(files) / FPS, OUT.stat().st_size / 1e6))


if __name__ == "__main__":
    main()

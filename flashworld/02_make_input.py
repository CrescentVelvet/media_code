#!/usr/bin/env python3
"""单图 → FlashWorld 输入 json（含相机轨迹）。

官方 cli.py 只吃「一个目录下的若干 json」，UI 里相机轨迹是前端 three.js 生成的
（index.html），仓库里没有 Python 版。本脚本把这套约定搬过来：

  json 字段：image_prompt / text_prompt / image_index / resolution[N,H,W] / cameras[]

相机约定（从 index.html + utils.py:create_rays 反推，逐条对过）：
  - 位姿是 **c2w**（相机→世界），位置单位任意（normalize_cameras 会归一化）
  - 四元数顺序 **[w, x, y, z]**（three.js 顺序，不是 scipy 的 xyzw）
  - 相机看向 **-Z**，上方向 +Y（OpenGL 约定）
  - fx = fy = 0.5 / tan(fov/2) * H，cx = W/2，cy = H/2（index.html:SaveAllToJson 原文）
  - resolution = [帧数, 高, 宽]，官方 UI 只给 24x480x704 / 24x480x704(竖)

轨迹模式（TRAJ）：
  orbit          绕注视点水平环绕，覆盖 YAW_SPAN 度（默认 40°）—— 本仓默认，
                 比官方 demo 的 ±15° 覆盖更宽，更适合「抽 3D 资产」
  official_orbit 复刻官方 demo 的 ±15° 小轨道（对照复现官方结果用）
  forward        复刻官方 "Move Forward" 模板（沿 -Z 推 DOLLY，姿态不变）

⚠️ 单图没有真实内参：FOV 只能假设。官方 UI 默认 60°。输入图与假设 FOV 不符
   会让重建出的场景透视偏胖/偏瘦，但没有真值可对齐，只能按画面观感调。

Env vars:
  IMAGE        单张图或图片目录（必填）
  OUTPUT       json 输出目录
  TEXT         文本提示词（默认空；官方会在前面拼 "[Static] "）
  N_FRAME      帧数（默认 24，官方训练/推理固定 24）
  HEIGHT/WIDTH 分辨率（默认 480/704）
  FOV          竖直 FOV 度（默认 60）
  TRAJ         orbit | official_orbit | forward（默认 orbit）
  YAW_SPAN     环绕角度（度，默认 40）
  PITCH        环绕仰角（度，默认 0，限 ±80）
  RADIUS       环绕半径（默认 1.0）
  DOLLY        forward 模式推进距离（默认 0.5）
  IMAGE_INDEX  输入图落在哪一帧（默认 0）
  LIMIT        目录模式最多处理多少张（默认全部）
"""
import json
import math
import os
import sys
from pathlib import Path

IMAGE = os.environ.get("IMAGE", "")
OUTPUT = os.environ.get("OUTPUT", "")
TEXT = os.environ.get("TEXT", "")
N_FRAME = int(os.environ.get("N_FRAME", "24"))
HEIGHT = int(os.environ.get("HEIGHT", "480"))
WIDTH = int(os.environ.get("WIDTH", "704"))
FOV = float(os.environ.get("FOV", "60"))
TRAJ = os.environ.get("TRAJ", "orbit")
YAW_SPAN = float(os.environ.get("YAW_SPAN", "40"))
PITCH = float(os.environ.get("PITCH", "0"))
RADIUS = float(os.environ.get("RADIUS", "1.0"))
DOLLY = float(os.environ.get("DOLLY", "0.5"))
IMAGE_INDEX = int(os.environ.get("IMAGE_INDEX", "0"))
LIMIT = int(os.environ.get("LIMIT", "0"))

EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".heic", ".heif", ".tif", ".tiff"}


# ── 四元数与朝向 ────────────────────────────────────────────────────────────
def mat_to_quat_wxyz(R):
    """3x3 旋转矩阵 → 四元数 [w, x, y, z]（Shepperd 法，取最大分量分支避免除零）。"""
    m00, m01, m02 = R[0]
    m10, m11, m12 = R[1]
    m20, m21, m22 = R[2]
    tr = m00 + m11 + m22
    if tr > 0.0:
        s = math.sqrt(tr + 1.0) * 2.0
        w = 0.25 * s
        x = (m21 - m12) / s
        y = (m02 - m20) / s
        z = (m10 - m01) / s
    elif m00 > m11 and m00 > m22:
        s = math.sqrt(1.0 + m00 - m11 - m22) * 2.0
        w = (m21 - m12) / s
        x = 0.25 * s
        y = (m01 + m10) / s
        z = (m02 + m20) / s
    elif m11 > m22:
        s = math.sqrt(1.0 + m11 - m00 - m22) * 2.0
        w = (m02 - m20) / s
        x = (m01 + m10) / s
        y = 0.25 * s
        z = (m12 + m21) / s
    else:
        s = math.sqrt(1.0 + m22 - m00 - m11) * 2.0
        w = (m10 - m01) / s
        x = (m02 + m20) / s
        y = (m12 + m21) / s
        z = 0.25 * s
    n = math.sqrt(w * w + x * x + y * y + z * z)
    q = [w / n, x / n, y / n, z / n]
    # 归一化后 w<0 与 w>0 是同一个旋转；统一取 w>=0，便于和官方 json 对齐比较
    if q[0] < 0:
        q = [-c for c in q]
    return q


def look_at_quat(eye, target, up=(0.0, 1.0, 0.0)):
    """相机在 eye 看向 target → 四元数 [w,x,y,z]。相机空间 +Z 指离目标（看向 -Z）。"""
    z = [eye[i] - target[i] for i in range(3)]
    zn = math.sqrt(sum(c * c for c in z))
    if zn < 1e-9:
        raise ValueError("相机位置与注视点重合，无法定向")
    z = [c / zn for c in z]
    x = [up[1] * z[2] - up[2] * z[1],
         up[2] * z[0] - up[0] * z[2],
         up[0] * z[1] - up[1] * z[0]]
    xn = math.sqrt(sum(c * c for c in x))
    if xn < 1e-9:  # 正上方/正下方俯视时 up 与视轴共线，退化
        raise ValueError("视线与 up 共线（PITCH 太接近 ±90°）")
    x = [c / xn for c in x]
    y = [z[1] * x[2] - z[2] * x[1],
         z[2] * x[0] - z[0] * x[2],
         z[0] * x[1] - z[1] * x[0]]
    # 列向量 = 基向量：R = [x y z]
    R = [[x[0], y[0], z[0]],
         [x[1], y[1], z[1]],
         [x[2], y[2], z[2]]]
    return mat_to_quat_wxyz(R)


# ── 轨迹 ────────────────────────────────────────────────────────────────────
def trajectory(n, mode, yaw_span, pitch, radius, dolly):
    """返回 n 个 (position, quaternion) —— c2w。"""
    cams = []
    if mode in ("orbit", "official_orbit"):
        span = 30.0 if mode == "official_orbit" else yaw_span
        el = math.radians(max(-80.0, min(80.0, pitch)))
        target = (0.0, 0.0, 0.0)
        for i in range(n):
            # 单帧时落在轨道中点，避免除零
            frac = 0.5 if n == 1 else i / (n - 1)
            az = math.radians(-span / 2.0 + span * frac)
            pos = (radius * math.cos(el) * math.sin(az),
                   radius * math.sin(el),
                   radius * math.cos(el) * math.cos(az))
            cams.append((pos, look_at_quat(pos, target)))
    elif mode == "forward":
        for i in range(n):
            frac = 0.0 if n == 1 else i / (n - 1)
            pos = (0.0, 0.0, -dolly * frac)  # 官方 Move Forward：z -= stepSize，姿态不变
            cams.append((pos, [1.0, 0.0, 0.0, 0.0]))
    else:
        raise ValueError(f"未知 TRAJ: {mode}")
    return cams


# ── 输入图处理 ──────────────────────────────────────────────────────────────
def ensure_readable_image(path, out_dir):
    """HEIC/HEIF 转 PNG（PIL 默认读不了）；其它格式原样返回。"""
    ext = path.suffix.lower()
    if ext not in (".heic", ".heif"):
        return path
    try:
        import pillow_heif  # noqa: F401
        from PIL import Image
    except ImportError:
        sys.exit("❌ HEIC 需要 pillow-heif：pip install pillow-heif pillow\n"
                 "   （或先把图转成 png/jpg 再喂进来）")
    pillow_heif.register_heif_opener()
    dst = out_dir / "images"
    dst.mkdir(parents=True, exist_ok=True)
    out = dst / (path.stem + ".png")
    if not out.exists():
        Image.open(path).convert("RGB").save(out)
    print(f"  🖼️  HEIC → PNG: {out}")
    return out


def check_aspect(path):
    """输入图长宽比和目标分辨率不一致时给个提醒（cli.py 会 center-crop）。"""
    try:
        from PIL import Image
        with Image.open(path) as im:
            w, h = im.size
    except Exception:
        return
    r_img, r_dst = w / h, WIDTH / HEIGHT
    if abs(r_img - r_dst) / r_dst > 0.02:
        print(f"  ⚠️  输入图 {w}x{h}（{r_img:.3f}）与目标 {WIDTH}x{HEIGHT}（{r_dst:.3f}）"
              f"长宽比不一致 → cli.py 会 center-crop 后缩放")


def build_one(img_path, out_dir):
    fx = 0.5 / math.tan(0.5 * math.radians(FOV)) * HEIGHT
    fy = fx
    cx, cy = WIDTH / 2.0, HEIGHT / 2.0

    cams = trajectory(N_FRAME, TRAJ, YAW_SPAN, PITCH, RADIUS, DOLLY)
    payload = {
        "image_prompt": str(img_path),
        "text_prompt": TEXT,
        "image_index": IMAGE_INDEX,
        "resolution": [N_FRAME, HEIGHT, WIDTH],
        "cameras": [
            {"position": [round(c, 6) for c in pos],
             "quaternion": [round(c, 8) for c in quat],
             "fx": round(fx, 4), "fy": round(fy, 4),
             "cx": round(cx, 2), "cy": round(cy, 2)}
            for pos, quat in cams
        ],
    }
    return payload


def main():
    if not IMAGE:
        sys.exit("❌ IMAGE 未设置（单张图片或图片目录）")
    src = Path(IMAGE).expanduser()
    if not src.exists():
        sys.exit(f"❌ IMAGE 不存在: {src}")
    out_dir = Path(OUTPUT).expanduser() if OUTPUT else Path.cwd() / "flashworld_inputs"
    out_dir.mkdir(parents=True, exist_ok=True)

    if src.is_dir():
        images = sorted(p for p in src.rglob("*") if p.suffix.lower() in EXTS)
        if LIMIT > 0:
            images = images[:LIMIT]
    else:
        images = [src]
    if not images:
        sys.exit(f"❌ 目录里没有图片: {src}")

    print(f"🚀 [02] 生成 FlashWorld 输入 json")
    print(f"  🖼️  输入:   {src}（{len(images)} 张）")
    print(f"  💾 输出:   {out_dir}")
    print(f"  📐 分辨率: {N_FRAME}×{HEIGHT}×{WIDTH} (N×H×W)  fov={FOV}°  "
          f"fx=fy={0.5 / math.tan(0.5 * math.radians(FOV)) * HEIGHT:.1f}")
    print(f"  🎥 轨迹:   {TRAJ}  yaw_span={YAW_SPAN}°  pitch={PITCH}°  r={RADIUS}")
    print("")

    for p in images:
        p = p.expanduser()
        if not p.is_absolute():
            p = p.resolve()
        readable = ensure_readable_image(p, out_dir)
        check_aspect(readable)
        payload = build_one(readable, out_dir)
        dst = out_dir / f"{p.stem}.json"
        # 官方 1.json 是单行紧凑格式；统一 LF（Windows 上 open("w") 会翻 CRLF）
        with dst.open("w", encoding="utf-8", newline="\n") as f:
            json.dump(payload, f, ensure_ascii=False)
        print(f"  ✅ {dst.name}  ({len(payload['cameras'])} cams)")

    print("")
    print(f"🎉 Done. {len(images)} 个 json 在 {out_dir}")
    print(f"   下一步：INPUT_DIR={out_dir} bash flashworld/02_run_inference.sh")


if __name__ == "__main__":
    main()

#!/usr/bin/env bash
# 05_pose_colmap.sh — 用 COLMAP 从图像估位姿，导出 InfiniSplat 需要的 cameras.npz。
#
# 为什么要这个：官方发布包只接受"图像 + 位姿"或"图像 + DA3 预测位姿"。实测弱纹理/透明
# 物体上 DA3 位姿分支会退化（三视角两两夹角被估成 <1°），此时推理仍"成功"但结果是废的。
# 换成 COLMAP 的 SfM 位姿可以救回来。
#
# 用法：
#   IMAGES_DIR=/path/to/imgs OUT_NPZ=/path/out/cameras.npz bash 05_pose_colmap.sh
#
# 输出 npz 约定（与官方 examples 一致）：
#   c2w        [N,4,4] float32  OpenCV 约定（x 右, y 下, z 朝前）
#   intrinsics [N,3,3] float32  像素内参，对应 --long-side 处理后的图像
#   names      [N]     str      文件名，顺序与 c2w 对应
#
# Env:
#   IMAGES_DIR=        输入图像目录（JPG/PNG）
#   OUT_NPZ=           输出 npz 路径（默认 <图像目录>/cameras_colmap.npz）
#   WORK_DIR=          COLMAP 工作目录（默认 $RESULTS_DIR/_colmap/<目录名>）
#   LONG_SIDE=518      与 02 保持一致；COLMAP 内部用 COLMAP_LONG_SIDE 缩放加速
#   COLMAP_LONG_SIDE=1024  特征提取时的缩放长边（越大越准越慢）
#   COLMAP_ENV=colmap  含 colmap 二进制的 conda env
set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/_env.sh"

: "${IMAGES_DIR:?ERROR: 需要 IMAGES_DIR}"
: "${LONG_SIDE:=518}"
: "${COLMAP_LONG_SIDE:=1024}"
: "${COLMAP_ENV:=colmap}"
NAME=$(basename "$IMAGES_DIR")
: "${WORK_DIR:=$RESULTS_DIR/_colmap/$NAME}"
: "${OUT_NPZ:=$WORK_DIR/cameras_colmap.npz}"

COLMAP_BIN="$HOME/miniconda3/envs/$COLMAP_ENV/bin/colmap"
if [ ! -x "$COLMAP_BIN" ]; then
    echo "❌ ERROR: 找不到 colmap: $COLMAP_BIN（先建 $COLMAP_ENV env）" >&2
    exit 1
fi
# conda-forge 的 colmap 4.x 需要显式 LD_LIBRARY_PATH 找 libopenimageio 等
export LD_LIBRARY_PATH="$HOME/miniconda3/envs/$COLMAP_ENV/lib:${LD_LIBRARY_PATH:-}"

echo "🚀 [05] COLMAP 位姿估计 — $NAME"
echo "  🖼️  输入: $IMAGES_DIR"
echo "  💾 工作目录: $WORK_DIR"
echo "  📐 COLMAP 长边: $COLMAP_LONG_SIDE | 推理长边: $LONG_SIDE"

rm -rf "$WORK_DIR"
mkdir -p "$WORK_DIR/images" "$WORK_DIR/sparse"

# 缩放到 COLMAP_LONG_SIDE 再提特征：22MP 原图 SIFT 很慢，且对 3 视图无收益
python - "$IMAGES_DIR" "$WORK_DIR/images" "$COLMAP_LONG_SIDE" <<'PY'
import sys
from pathlib import Path
from PIL import Image, ImageOps
src, dst, long_side = Path(sys.argv[1]), Path(sys.argv[2]), int(sys.argv[3])
files = sorted(p for p in src.iterdir() if p.is_file() and p.suffix.lower() in {'.jpg', '.jpeg', '.png', '.webp'})
for p in files:
    with Image.open(p) as im:
        im = ImageOps.exif_transpose(im).convert('RGB')
        f = long_side / max(im.size)
        if f < 1.0:
            im = im.resize((max(2, round(im.width * f)), max(2, round(im.height * f))), Image.LANCZOS)
        im.save(dst / (p.stem + '.jpg'), quality=95)
print(f"  ✂️  {len(files)} 张缩放至长边 {long_side}")
PY

DB="$WORK_DIR/database.db"
# COLMAP 4.x 把 SiftExtraction.use_gpu 改名为 FeatureExtraction.use_gpu（匹配器同理），
# 服务器上的 3.x 还是旧名，探一下再决定。
if "$COLMAP_BIN" feature_extractor -h 2>&1 | grep -q -- "--FeatureExtraction.use_gpu"; then
    FE_GPU="--FeatureExtraction.use_gpu 1"; FM_GPU="--FeatureMatching.use_gpu 1"
else
    FE_GPU="--SiftExtraction.use_gpu 1"; FM_GPU="--SiftMatching.use_gpu 1"
fi
"$COLMAP_BIN" feature_extractor --database_path "$DB" --image_path "$WORK_DIR/images" \
    --ImageReader.single_camera 1 $FE_GPU > "$WORK_DIR/feat.log" 2>&1 || {
    echo "❌ feature_extractor 失败（$WORK_DIR/feat.log）" >&2; tail -5 "$WORK_DIR/feat.log" >&2; exit 1; }

"$COLMAP_BIN" exhaustive_matcher --database_path "$DB" $FM_GPU \
    > "$WORK_DIR/match.log" 2>&1 || {
    echo "❌ exhaustive_matcher 失败（$WORK_DIR/match.log）" >&2; tail -5 "$WORK_DIR/match.log" >&2; exit 1; }

"$COLMAP_BIN" mapper --database_path "$DB" --image_path "$WORK_DIR/images" \
    --output_path "$WORK_DIR/sparse" > "$WORK_DIR/mapper.log" 2>&1 || {
    echo "❌ mapper 失败（$WORK_DIR/mapper.log）" >&2; tail -5 "$WORK_DIR/mapper.log" >&2; exit 1; }

if [ ! -f "$WORK_DIR/sparse/0/images.txt" ]; then
    echo "❌ COLMAP 未能重建（sparse/0 为空）。视图太少或纹理不足。" >&2
    echo "   看 $WORK_DIR/mapper.log 里的 registered images 数" >&2
    exit 1
fi

REG=$(grep -c "^[0-9].*\.jpg$" "$WORK_DIR/sparse/0/images.txt" 2>/dev/null || echo "?")
echo "  🔍 注册图像: $REG"

# COLMAP world→camera 转 OpenCV c2w。内参必须换算到**原图**像素坐标：
# prepare_images() 会拿原图尺寸再缩放/裁剪，传半分辨率的内参会让视场角错一倍。
python - "$WORK_DIR/sparse/0" "$WORK_DIR/images" "$IMAGES_DIR" "$OUT_NPZ" <<'PY'
import sys
from pathlib import Path
import numpy as np
from PIL import Image

sparse, det_dir, src_dir, out_npz = (Path(sys.argv[1]), Path(sys.argv[2]),
                                     Path(sys.argv[3]), Path(sys.argv[4]))
src = {p.stem: p for p in src_dir.iterdir()
       if p.is_file() and p.suffix.lower() in {'.jpg', '.jpeg', '.png', '.webp'}}

def read_cameras(p):
    cams = {}
    lines = [l for l in p.read_text().splitlines() if l.strip() and not l.startswith('#')]
    i = 0
    while i < len(lines):
        cid, model, w, h = lines[i].split()[:4]
        params = np.array([float(x) for x in lines[i + 1].split()])
        cams[int(cid)] = (model, int(w), int(h), params)
        i += 2
    return cams

def read_images(p):
    out, lines = [], [l for l in p.read_text().splitlines() if l.strip() and not l.startswith('#')]
    i = 0
    while i < len(lines):
        parts = lines[i].split()
        qw, qx, qy, qz = map(float, parts[1:5])
        tx, ty, tz = map(float, parts[5:8])
        cid, name = int(parts[8]), parts[9]
        out.append(((qw, qx, qy, qz), (tx, ty, tz), cid, name))
        i += 2
    return out

def qvec2rot(q):
    w, x, y, z = q
    return np.array([[1-2*(y*y+z*z), 2*(x*y-w*z),   2*(x*z+w*y)],
                     [2*(x*y+w*z),   1-2*(x*x+z*z), 2*(y*z-w*x)],
                     [2*(x*z-w*y),   2*(y*z+w*x),   1-2*(x*x+y*y)]])

cams = read_cameras(sparse / 'cameras.txt')
imgs = read_images(sparse / 'images.txt')
# 顺序必须与 02 的 sorted(iterdir, key=name) 一致，否则位姿会配错图
imgs.sort(key=lambda r: src[Path(r[3]).stem].name)

c2ws, Ks, names = [], [], []
for q, t, cid, name in imgs:
    R = qvec2rot(q)                      # world→camera 旋转
    w2c = np.eye(4); w2c[:3, :3] = R; w2c[:3, 3] = np.array(t)
    c2ws.append(np.linalg.inv(w2c))
    model, cw, ch, prm = cams[cid]
    orig = src[Path(name).stem]
    with Image.open(orig) as im:
        ow, oh = im.size
    K = np.eye(3)
    if model in ('SIMPLE_PINHOLE', 'SIMPLE_RADIAL'):
        f, cx, cy = prm[0], prm[1], prm[2]
        K[0, 0] = K[1, 1] = f
    else:                                 # PINHOLE / OPENCV / RADIAL
        fx, fy, cx, cy = prm[:4]
        K[0, 0], K[1, 1] = fx, fy
    K[0, 2], K[1, 2] = cx, cy
    K[0] *= ow / cw; K[1] *= oh / ch      # COLMAP 记录尺寸 → 原图尺寸
    Ks.append(K)
    names.append(orig.name)

np.savez(out_npz, c2w=np.stack(c2ws).astype(np.float32),
         intrinsics=np.stack(Ks).astype(np.float32), names=np.array(names))
print(f"  ✅ {out_npz}  N={len(names)}  {names}")
PY

echo "🎉 Done: $OUT_NPZ"
echo "   下一步：CAMERAS=$OUT_NPZ IMAGES_DIR=$IMAGES_DIR POSE_MODE=provided bash 02_run_inference.sh"

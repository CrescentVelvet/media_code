#!/usr/bin/env bash
# 一次性：把 32v 渲染裁图 + 3v 环绕视频从 WSL 产物区拷到 D 盘素材目录（供 PPT 用）
set -o pipefail
A=/mnt/d/output/results/compare_4methods_assets
rm -rf "$A/_peek"
mkdir -p "$A/render_crops" "$A/video"
for m in GT DA3-GS QuerySplat InfiniSplat_v2 VGGT-Ω_3DGS DA3_3DGS; do
  for v in v002 v021; do
    cp "/home/velvet/output/batch_4methods/render_crops/$m/$v.png" "$A/render_crops/${m}_${v}.png"
  done
done
cp /home/velvet/output/batch_4methods/orbit_video_3view/five_way_arc_3view.mp4 "$A/video/"
cp /home/velvet/output/batch_4methods/orbit_video_3view/five_way_arc_3view_frame_000.png "$A/video/poster.png"
echo "crops: $(ls "$A/render_crops" | wc -l) 张"
ffprobe -v error -select_streams v:0 -show_entries stream=width,height,nb_frames -of csv=p=0 "$A/video/five_way_arc_3view.mp4" 2>/dev/null \
  || python3 -c "
import cv2
c = cv2.VideoCapture('$A/video/five_way_arc_3view.mp4')
print('video:', int(c.get(3)), 'x', int(c.get(4)), 'frames', int(c.get(7)))" 2>/dev/null \
  || echo "(无 ffprobe/cv2，尺寸在 Windows 侧读)"

#!/usr/bin/env bash
# 21_batch_portrait.sh — 批量：Remy 采集包（heic+transforms.json）→ DA3 高斯 → orbit 视频
#
# 用法:
#   bash 21_batch_portrait.sh                          # 全量
#   bash 21_batch_portrait.sh <数据根> <输出根>        # 自定义路径
#   bash 21_batch_portrait.sh <数据根> <输出根> <包名> # 只跑一包（验证用）
#
# 每包三级产物（已存在即跳过，可断点续跑）：
#   <输出根>/<包>/images/*.jpg          20_convert_heic.py
#   <输出根>/<包>/gs_max/0000.ply       08_bench_official_gs.py（32 视角/504p/first/全SH）
#   <输出根>/<包>/videos/orbit.mp4      19_orbit_video.py
# 日志: <输出根>/_logs/<包>.log；汇总: <输出根>/_summary.tsv
set +u
SRC="${1:-/mnt/d/dataset/人像验收数据汇总-含佳佳}"
OUT="${2:-$HOME/output/da3_portrait_batch}"
ONLY="${3:-}"
LOGD="$OUT/_logs"
mkdir -p "$LOGD"

M=/mnt/c/code/media_code/da3_human
source "$HOME/miniconda3/etc/profile.d/conda.sh"

echo "🚀 [21] 批量 DA3 高斯 + orbit 视频"
echo "  📁 数据: $SRC"
echo "  💾 输出: $OUT"

printf "pkg\tn_imgs\tgs_points\tfwd_s\tply_MB\tvideo_MB\tstatus\n" > "$OUT/_summary.tsv"

for d in "$SRC"/*/; do
    name="$(basename "$d")"
    if [ -n "$ONLY" ] && [ "$name" != "$ONLY" ]; then continue; fi
    log="$LOGD/$name.log"
    t0=$(date +%s)
    echo "━━ $name ━━"

    # 1) heic → jpg
    if [ ! -f "$OUT/$name/images/.done" ]; then
        conda activate da3_human
        SRC_DIR="$d" OUT_DIR="$OUT/$name" python "$M/20_convert_heic.py" > "$log" 2>&1
        if [ $? -ne 0 ]; then echo "❌ convert 失败（见 $log）"; continue; fi
        touch "$OUT/$name/images/.done"
    fi
    n_imgs=$(ls "$OUT/$name/images"/*.jpg | wc -l)

    # 2) DA3 推理（32 视角 / 504p / ref=first / 不裁剪+全SH / 快速导出）
    if [ ! -f "$OUT/$name/gs_max/0000.ply" ]; then
        conda activate da3_human
        DA3_MODEL_DIR=/mnt/d/wheel/da3_human_ms \
        IMAGES_DIR="$OUT/$name/images" \
        OUT_DIR="$OUT/$name" \
        PROCESS_RES=504 N_VIEWS=32 REF_STRATEGY=first \
        EXPORT=max FAST_EXPORT=1 \
        python "$M/08_bench_official_gs.py" >> "$log" 2>&1
        if [ ! -f "$OUT/$name/gs_max/0000.ply" ]; then
            echo "❌ DA3 推理失败（见 $log）"
            printf "%s\t%s\t-\t-\t-\t-\tda3_fail\n" "$name" "$n_imgs" >> "$OUT/_summary.tsv"
            continue
        fi
    fi

    # 3) orbit 视频（手持采集是开弧，--arc 只在观测弧内往返；mp4 <100KB 视为坏文件重渲）
    mp4="$OUT/$name/videos/orbit.mp4"
    if [ -f "$mp4" ] && [ "$(stat -c%s "$mp4")" -lt 100000 ]; then rm -f "$mp4"; fi
    if [ ! -f "$mp4" ]; then
        conda activate vggt_human
        python "$M/19_orbit_video.py" \
            --result "$OUT/$name" --out "$OUT/$name/videos" \
            --arc \
            >> "$log" 2>&1
        if [ ! -f "$mp4" ]; then
            echo "❌ 视频渲染失败（见 $log）"
            printf "%s\t%s\t-\t-\t-\t-\tvideo_fail\n" "$name" "$n_imgs" >> "$OUT/_summary.tsv"
            continue
        fi
    fi

    n_pts=$(grep -oP 'PLY: \K[0-9,]+' "$log" | tail -1)
    ply_mb=$(du -m "$OUT/$name/gs_max/0000.ply" | cut -f1)
    vid_mb=$(du -m "$OUT/$name/videos/orbit.mp4" | cut -f1)
    dt=$(( $(date +%s) - t0 ))
    printf "%s\t%s\t%s\t-\t%s\t%s\tok\n" "$name" "$n_imgs" "$n_pts" "$ply_mb" "$vid_mb" >> "$OUT/_summary.tsv"
    echo "✅ $name | ${n_imgs} 图 | ${n_pts} 高斯 | PLY ${ply_mb}M | MP4 ${vid_mb}M | ${dt}s"
done

echo "🎉 Done. 汇总: $OUT/_summary.tsv"

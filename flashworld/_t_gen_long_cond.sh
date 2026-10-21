#!/bin/bash
# 生成长轨迹多条件实验的 5 组输入 json（在 Windows Git Bash 下跑）
PY=/c/Users/wangyufeng/.workbuddy/binaries/python/envs/default/Scripts/python.exe
S=/c/code/media_code/flashworld/05_make_multi_input.py
PNG_DIR="D:/output/flashworld_input/jiabao_png"
F41="0,2,4,6,8,10,12,14,16,18,20,22,24,26,28,30,32,34,36,38,40,43,45,47,49,51,53,55,57,59,61,63,65,67,69,71,73,75,77,79,81"
B41="D:/output/flashworld_input/json_traj41/01e954de1b8a4cde9315c976f4dcb964-佳佳抱枕_frame000.json"
B82="D:/output/flashworld_input/json_traj82/01e954de1b8a4cde9315c976f4dcb964-佳佳抱枕_frame000.json"
OUTDIR="D:/output/flashworld_input/json_lc"
mkdir -p "$OUTDIR"

ALL41="0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40"
ALL82="0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,66,67,68,69,70,71,72,73,74,75,76,77,78,79,80,81"
C41_T41="0,4,7,11,15,18,22,25,29,33,36,40"
C24_T82="0,4,7,11,14,18,21,25,28,32,35,39,42,46,49,53,56,60,63,67,70,74,77,81"
C41_T82="0,2,4,6,8,10,12,14,16,18,20,22,24,26,28,30,32,34,36,38,40,43,45,47,49,51,53,55,57,59,61,63,65,67,69,71,73,75,77,79,81"

BASE="$B41" PNG_DIR="$PNG_DIR" FRAME_IDXS="$F41" COND="$ALL41" OUT="$OUTDIR/traj41_cond41.json" "$PY" "$S" || exit 1
BASE="$B41" PNG_DIR="$PNG_DIR" FRAME_IDXS="$F41" COND="$C41_T41" OUT="$OUTDIR/traj41_cond12.json" "$PY" "$S" || exit 1
BASE="$B82" PNG_DIR="$PNG_DIR" COND="$C24_T82" OUT="$OUTDIR/traj82_cond24.json" "$PY" "$S" || exit 1
BASE="$B82" PNG_DIR="$PNG_DIR" COND="$C41_T82" OUT="$OUTDIR/traj82_cond41.json" "$PY" "$S" || exit 1
BASE="$B82" PNG_DIR="$PNG_DIR" COND="$ALL82"  OUT="$OUTDIR/traj82_cond82.json" "$PY" "$S" || exit 1
echo ALL_JSON_DONE

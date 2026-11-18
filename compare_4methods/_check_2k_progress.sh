#!/usr/bin/env bash
# 2K 批量的进度体检：进程 CPU/内存 + 各步骤日志尾 + GPU
echo "== python 进程（CPU% MEM% 启动时长 命令）=="
ps -eo pcpu,pmem,etime,comm | grep -i python | head -5
echo ""
echo "== da3 日志尾 =="
tail -3 ~/output/batch_4methods/2k_da3_佳佳抱枕_2k.log 2>/dev/null
echo ""
echo "== GPU =="
nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader
echo ""
echo "== 已产出物 =="
ls ~/output/da3_results/佳佳抱枕_2k/ 2>/dev/null
ls ~/output/da3_gs_results/ 2>/dev/null | grep 2k

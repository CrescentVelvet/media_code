#!/usr/bin/env bash
# audit_32（三图五方法视频）的环境包装：querysplat env（cv2/PIL/gsplat 齐全）。
# 教训（2026-11-10）：audit_31 直接 conda activate da3_human 缺 skimage；
# 且与别的重度 GPU 任务共存会 rasterization 活锁——audit_32 自带显存预检，这里再挡一层。
set -o pipefail
LOG=/mnt/c/code/media_code/compare_4methods/_run_32_3view_wrapped.log
: > "$LOG"
{
    source ~/miniconda3/etc/profile.d/conda.sh
    conda activate querysplat
    cd /mnt/c/code/media_code
    free_gib=$(python -c "import torch;print(torch.cuda.mem_get_info()[0]/2**30)")
    echo "GPU free = ${free_gib} GiB"
    if python -c "import torch,sys;sys.exit(0 if torch.cuda.mem_get_info()[0]>=4*2**30 else 1)"; then
        echo ">>> start audit_32 $(date)"
        python compare_4methods/audit_32_five_way_video_3view.py
        rc=$?
        echo "<<< audit_32 exit=$rc $(date)"
        exit $rc
    else
        echo "REFUSE: GPU busy (<4 GiB free)"
        exit 3
    fi
} >> "$LOG" 2>&1

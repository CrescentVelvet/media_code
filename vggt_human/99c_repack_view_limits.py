#!/usr/bin/env python3
"""99c_repack_view_limits.py — 收窄视角重封装：基于 99b 中间产物重打 MP4。

手机端观看视角过大时能看到重建边缘的残缺/伪影。本脚本不动 PLY 编码和视频，
只收窄 view_limits.json 的可视角范围（必要时夹紧初始相机），重跑封装两步：

    [1/2] 99b 的 bin + camera.json + 新 view_limits/init_camera → GLB
    [2/2] 99b 的视频 output.mp4 + 新 GLB → mp4_crop/<task>.mp4

前置条件（缺一报错提示先跑 99b，均由 99b 产出并默认保留在 mp4_work/）:
    mp4_work/<task>/GSCompressed_B*.bin     压缩码流
    mp4_work/<task>/output.mp4              帧视频
    三件套 json: task 目录下（UWA 型）或 mp4_work/<task>/uwa_json/（COLMAP 型）

收窄规则（对 view_limits.json 原值内缩，保持中心不动）:
    Phi  (水平)   [minPhi,   maxPhi]   → [minPhi+LEFT,   maxPhi-RIGHT]
    Theta(垂直)   [minTheta, maxTheta] → [minTheta+BOTTOM, maxTheta-TOP]
    Radius        [minRadius,maxRadius]→ [minR+MARGIN,   maxR-MARGIN]（默认 0 不动）
    初始相机若落到收窄区外 → 自动夹到边界并打印警告
    单侧收窄量 ≥ 半跨度 → 该 task 报错跳过（收没了）

用法:
    python vggt_human/99c_repack_view_limits.py                  # 默认 10° 对称收窄
    PHI_MARGIN=15 THETA_MARGIN=5 python vggt_human/99c_repack_view_limits.py
    PHI_LEFT=5 PHI_RIGHT=25 python vggt_human/99c_repack_view_limits.py  # 不对称
    DRY_RUN=1  python ...   # 只打印收窄前后范围，不执行
    FORCE=1 / ONLY=task_a,task_b   # 同 99b

Env vars:
    TOOL_DIR      UWA 工具链根目录（含 muxer.py / build/gltf_packer）
    SRC_ROOT      批次根目录（同 99b，用于定位 task 目录与三件套 json）
    RESULTS_ROOT  统一结果根（默认 ../../output/recon_human_results）
    OUT_DIR       输出目录，默认 <RESULTS_ROOT>/<批次名>/mp4_crop（与 99b 的
                  mp4/ 平级共存，不覆盖 99b 成品）
    WORK_ROOT     99b 中间产物根，默认 <RESULTS_ROOT>/<批次名>/mp4_work
    PHI_MARGIN    Phi 两侧对称收窄角度（度，默认 10；LEFT/RIGHT 未设时用它）
    THETA_MARGIN  Theta 两侧对称收窄角度（度，默认 10；TOP/BOTTOM 未设时用它）
    PHI_LEFT / PHI_RIGHT / THETA_TOP / THETA_BOTTOM
                  单侧收窄角度（度，未设则取对应 *_MARGIN）
    RADIUS_MARGIN Radius 两侧内缩（米，默认 0 不动）
    ONLY / FORCE / DRY_RUN / PYTHON_BIN / ASTC_BLOCK   含义同 99b
"""
import json
import math
import os
import shutil
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

# UWA 三件套文件名（与 99b 一致；camera.json 原样透传，另两件是收窄对象）
UWA_JSONS = {
    "init_camera": "init_camera.json",
    "camera": "camera.json",
    "view_params": "view_limits.json",
}
# 收窄版落盘名（放 mp4_work/<task>/crop_json/，不污染 99b 的 uwa_json/）
CROP_JSONS = {
    "init_camera": "init_camera_crop.json",
    "view_params": "view_limits_crop.json",
}


# --------------------------------------------------------------------------
# 基础工具（与 99b 相同）
# --------------------------------------------------------------------------
def _tail(text: str, n: int = 15) -> str:
    lines = [l for l in text.strip().splitlines() if l.strip()]
    return "\n".join(lines[-n:])


def run_cmd(cmd, cwd=None, env=None, log_tail: int = 15) -> int:
    """执行命令，失败时打印输出尾部（不吞异常，返回码交给调用方判断）。"""
    proc = subprocess.run(
        [str(c) for c in cmd],
        cwd=str(cwd) if cwd else None,
        env=env,
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        print(f"  ❌ 命令失败 (rc={proc.returncode}): {' '.join(str(c) for c in cmd)}")
        if proc.stdout:
            print("  ---- stdout ----")
            print(_tail(proc.stdout, log_tail))
        if proc.stderr:
            print("  ---- stderr ----")
            print(_tail(proc.stderr, log_tail))
    return proc.returncode

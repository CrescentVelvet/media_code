#!/usr/bin/env python3
"""Probe RetouchFormer's attention_list + test attention-mask de-color (方案 C).

One-shot diagnostic: loads the model, runs ONE face, and prints/visualizes:

  1. attention_list format — len, per-tensor [B,C,H,W]/dtype/min/max/mean
     (is it sigmoid'd [0,1]? C=? spatial size? — needed to build a pixel mask)
  2. reddish root-cause — beauty vs src per-channel mean (ΔR-B>0 = 偏红)
  3. beauty_high per-channel DC — is the red in the high-freq global DC?
     (if yes, high_freq_dc already fixed it; if ~0, red is local → needs mask)
  4. 方案 C trial — channel-mean of the finest attention scale → resize to 512 →
     soft/hard pixel fusion: decolor_C = mask·beauty + (1-mask)·src
     (normal-skin region = orig pixel = ZERO redness; blemish region = beauty repair)
  5. per-channel mean of each method (which kills redness best?) + compare PNG
     [src | beauty | mask | C_soft | C_hard | high_freq_dc]

Env (same as 03e): RETOUCH_DIR, WEIGHT_PATH (default $MODEL_DIR/release_model/gen_best.pth),
  MODEL_NAME, INPUT_DIR, OUTPUT_DIR, RESIZE_MODE, SIZE, DEVICE
Run:
  GPU=0 INPUT_DIR=../HYPIR/input/test_faces_hq OUTPUT_DIR=../../output/hypir_test_results/probe \
    python hypir/decolor_probe.py
"""
import os
import sys
import importlib
from pathlib import Path

# 直接 python 运行(不走 .sh/_env.sh)时 GPU->CUDA_VISIBLE_DEVICES 映射不存在；
# torch 只认 CUDA_VISIBLE_DEVICES，必须在 import torch / 任何 cuda 调用前设，
# 否则 CUDA context 已初始化、改了无效(进程仍跑默认卡 → OOM)
_gpu = os.environ.get("GPU")
if _gpu:
    os.environ["CUDA_VISIBLE_DEVICES"] = _gpu

import torch
from PIL import Image
from torchvision.utils import save_image

# Reuse 03e's pure helpers (wavelet_*, build_transform, decolor_image, IMG_EXTS).
# `import decolor_beauty_dataset` runs its module-level env reads + sys.path.insert
# (benign); we re-read the few vars we need (WEIGHT_PATH default etc.) below.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import decolor_beauty_dataset as D  # noqa: E402

MODEL_NAME = os.environ.get("MODEL_NAME", "RetouchFormer")
MODEL_DIR = os.environ.get("MODEL_DIR", "../../model/RetouchFormer")
WEIGHT_PATH = os.environ.get("WEIGHT_PATH") or f"{MODEL_DIR}/release_model/gen_best.pth"
INPUT_DIR = os.environ.get("INPUT_DIR")
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "./probe_out")
DEVICE = os.environ.get("DEVICE", "cuda")


def main():
    if not INPUT_DIR:
        sys.exit("ERROR: set INPUT_DIR (folder of face images; takes the first one)")
    if not os.path.isfile(WEIGHT_PATH):
        sys.exit(f"ERROR: WEIGHT_PATH not found: {WEIGHT_PATH} (set WEIGHT_PATH or MODEL_DIR)")

    if DEVICE == "cuda" and torch.cuda.is_available():
        torch.cuda.set_device(0)
    device = torch.device(DEVICE if (DEVICE == "cuda" and torch.cuda.is_available()) else "cpu")
    print(f"[*] device={device}")

    print(f"[*] building model '{MODEL_NAME}'...")
    net = importlib.import_module("model." + MODEL_NAME)
    model = net.InpaintGenerator().to(device)
    print(f"[*] loading checkpoint: {WEIGHT_PATH}")
    state = torch.load(WEIGHT_PATH, map_location=device, weights_only=False)
    model.load_state_dict(state)
    model.eval()

    tfm = D.build_transform()
    input_dir = Path(INPUT_DIR)
    imgs = []
    for r, _, fs in os.walk(input_dir):
        for f in fs:
            if os.path.splitext(f)[1].lower() in D.IMG_EXTS:
                imgs.append(Path(r) / f)
    imgs.sort()
    if not imgs:
        sys.exit(f"ERROR: no images in {input_dir}")
    fp = imgs[0]
    print(f"[*] probe image: {fp.name}  (first of {len(imgs)})")

    src = tfm(Image.open(fp).convert("RGB")).unsqueeze(0).to(device)  # [1,3,512,512] in [-1,1]
    print(f"[*] src shape={tuple(src.shape)} range=[{src.min():.3f},{src.max():.3f}]")

    with torch.no_grad():
        beauty, attention_list = model(src)   # 接住 attention_list (之前 `beauty, _` 丢了)

    # ── 1. attention_list 格式 ──
    print()
    print("=" * 64)
    print("=== 1. attention_list 格式 ===")
    print("=" * 64)
    print(f"len(attention_list) = {len(attention_list)}")
    # 双引号串里撇号免转义；f-string 表达式部分({}内)不能含反斜杠(Python<3.12)，故提前定义而非内联三元字符串
    TAG_OK = "  (sigmoid'd [0,1])"
    TAG_BAD = "  (NOT [0,1])"
    for i, a in enumerate(attention_list):
        is01 = (a.min() >= -1e-3 and a.max() <= 1 + 1e-3)
        print(f"  [{i}] shape={tuple(a.shape)} dtype={a.dtype} "
              f"min={a.min():.4f} max={a.max():.4f} mean={a.mean():.4f}"
              f"{TAG_OK if is01 else TAG_BAD}")

    # ── 2. 红润根源: beauty vs src per-channel mean ──
    print()
    print("=" * 64)
    print("=== 2. 红润根源: per-channel mean ([-1,1] -> [0,1] 读) ===")
    print("=" * 64)
    print("  ΔR-B > 0 = 偏红暖; 越大越红")
    for name, t in [("src(orig)  ", src), ("beauty     ", beauty)]:
        m = (t.mean(dim=[0, 2, 3]) + 1) / 2   # [-1,1] -> [0,1]
        print(f"  {name} R={m[0]:.4f} G={m[1]:.4f} B={m[2]:.4f}  ΔR-B={m[0]-m[2]:+.4f}")

    # ── 3. beauty_high per-channel DC (红在高频全局 DC?) ──
    print()
    print("=" * 64)
    print("=== 3. beauty_high per-channel DC (红在高频全局 DC?) ===")
    print("=" * 64)
    print("  理论 ~0; R>0 且 B<0 = 高频带红 DC (high_freq_dc 能去); ~0 = 红是局部, 需 mask")
    beauty_high, _ = D.wavelet_decomposition(beauty)
    src_high, _ = D.wavelet_decomposition(src)
    for name, t in [("beauty_high", beauty_high), ("src_high   ", src_high)]:
        m = t.mean(dim=[0, 2, 3])
        print(f"  {name} R={m[0]:+.5f} G={m[1]:+.5f} B={m[2]:+.5f}")

    # ── 4. 方案 C: attention mask 像素融合 ──
    print()
    print("=" * 64)
    print("=== 4. 方案 C: attention mask 像素融合 ===")
    print("=" * 64)
    # attention_list: 6 个 [B,C,H,W]; 取空间分辨率最大的那个(最精细), C 可能很大
    # (如 512) → channel-mean 降到 [B,1,H,W] 当瑕疵概率图
    best = max(attention_list, key=lambda a: a.shape[-1])
    print(f"  finest scale: shape={tuple(best.shape)}")
    # C 可能很大(如 512): 降维到 [B,1,H,W]. max = 每位置取最像瑕疵的通道(更对,
    # 对应 forward 里 per-channel >0.5 的判断); mean 会稀释(打印作对比). 用 max 融合.
    mask_mean = best.mean(dim=1, keepdim=True)
    mask = best.max(dim=1, keepdim=True)[0]
    print(f"  channel-mean: range=[{mask_mean.min():.4f},{mask_mean.max():.4f}] mean={mask_mean.mean():.4f}")
    print(f"  channel-max : range=[{mask.min():.4f},{mask.max():.4f}] mean={mask.mean():.4f}  <- 用这个")
    blemish = (mask > 0.5).float().mean()
    print(f"  >0.5 blemish area (max): {blemish:.4f}  (小=瑕疵少/正常皮肤多; 大≈1=mask 方向反了, 用 1-mask)")
    # resize 到 512 (attention 在 64x64 潜空间, 上采样做像素 mask)
    if mask.shape[-1] != src.shape[-1]:
        mask = torch.nn.functional.interpolate(
            mask, size=(src.shape[-1], src.shape[-1]), mode='bilinear', align_corners=False)
        print(f"  resized mask -> {tuple(mask.shape)} (上采样到像素空间)")
    # soft / hard fusion: blemish region (>0.5) 用 beauty(修复), 正常区 用 src(原图零红润)
    decolor_C_soft = mask * beauty + (1 - mask) * src
    mask_hard = (mask > 0.5).float()
    decolor_C_hard = mask_hard * beauty + (1 - mask_hard) * src
    # high_freq_dc 对比 (已实现)
    decolor_dc = D.decolor_image(beauty, src, "high_freq_dc")

    # ── 5. 各方案 per-channel mean + 对比图 ──
    print()
    print("=" * 64)
    print("=== 5. 各方案结果 per-channel mean (ΔR-B 越接近 src 越不红) ===")
    print("=" * 64)
    for name, t in [("src(orig)     ", src),
                    ("beauty(红)    ", beauty),
                    ("C_soft       ", decolor_C_soft),
                    ("C_hard       ", decolor_C_hard),
                    ("high_freq_dc ", decolor_dc)]:
        m = (t.mean(dim=[0, 2, 3]) + 1) / 2
        print(f"  {name} R={m[0]:.4f} G={m[1]:.4f} B={m[2]:.4f}  ΔR-B={m[0]-m[2]:+.4f}")

    out = Path(OUTPUT_DIR)
    out.mkdir(parents=True, exist_ok=True)
    mask_vis = mask * 2 - 1   # [0,1] -> [-1,1] for save_image value_range
    panels = [src, beauty, mask_vis, decolor_C_soft, decolor_C_hard, decolor_dc]
    labels = ["src(orig)", "beauty(red)", "mask(C-max)", "C_soft", "C_hard", "high_freq_dc"]
    cmp = torch.cat(panels, dim=3)
    cmp_path = out / f"probe_{fp.stem}.png"
    save_image(cmp, str(cmp_path), normalize=True, value_range=(-1, 1))
    print(f"\n[*] saved compare: {cmp_path}")
    print(f"    panels L->R: {' | '.join(labels)}")
    print(f"[*] 看图: beauty 列红, mask 列亮=瑕疵区; C_soft/C_hard 列正常皮肤应回到 src 色, "
          f"瑕疵区保留美颜修复; high_freq_dc 列若仍红=红是局部(非全局DC), 方案 C 应更对症")


if __name__ == "__main__":
    main()

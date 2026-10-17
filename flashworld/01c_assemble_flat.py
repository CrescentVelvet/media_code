#!/usr/bin/env python3
"""把「迅雷批量下载的一个扁平目录」整理成 HF 仓库目录结构，并补下 KB 级小文件。

为什么需要：迅雷批量任务按 URL 末段命名、全部落在同一个目录，而 diffusers 加载
Wan2.2 时要的是 `Wan2.2-TI2V-5B-Diffusers/{vae,transformer,text_encoder,tokenizer,scheduler}`
的子目录结构。另外 `config.json` 在三个组件里重名，迅雷无法区分（会变成 config(1).json），
所以这类文件不放进迅雷清单，由本脚本按完整相对路径补下。

用法：
    python flashworld/01c_assemble_flat.py --src "D:/wheel/flashworld_ms/thunder"
    python flashworld/01c_assemble_flat.py --src /mnt/d/wheel/flashworld_ms/thunder

整理完再跑 `bash flashworld/01b_link_local_weights.sh` 挂进 HF cache。

Env vars:
  FLASHWORLD_MODEL_DIR  目标根目录（默认 /mnt/d/wheel/flashworld_ms）
  MS_ENDPOINT           小文件下载源（默认 modelscope；WSL 下也可换 hf-mirror）
"""
import argparse
import io
import os
import re
import shutil
import sys
import urllib.request

# 相对路径 -> 字节数。2026-10-17 取自 ModelScope
# /api/v1/models/Wan-AI/Wan2.2-TI2V-5B-Diffusers/repo/files?Recursive=true
# 与 HF API 的 size 字段逐个核对一致（合计 34.20 GB）。
MANIFEST = {
    'configuration.json': 74,
    'model_index.json': 499,
    'scheduler/scheduler_config.json': 820,
    'text_encoder/config.json': 855,
    'text_encoder/model-00001-of-00003.safetensors': 4935812536,
    'text_encoder/model-00002-of-00003.safetensors': 4983103192,
    'text_encoder/model-00003-of-00003.safetensors': 1442935480,
    'text_encoder/model.safetensors.index.json': 22476,
    'tokenizer/special_tokens_map.json': 7079,
    'tokenizer/spiece.model': 4548313,
    'tokenizer/tokenizer.json': 16837459,
    'tokenizer/tokenizer_config.json': 61758,
    'transformer/config.json': 495,
    'transformer/diffusion_pytorch_model-00001-of-00005.safetensors': 4978254344,
    'transformer/diffusion_pytorch_model-00002-of-00005.safetensors': 4846784976,
    'transformer/diffusion_pytorch_model-00003-of-00005.safetensors': 4972658392,
    'transformer/diffusion_pytorch_model-00004-of-00005.safetensors': 4846785080,
    'transformer/diffusion_pytorch_model-00005-of-00005.safetensors': 354751840,
    'transformer/diffusion_pytorch_model.safetensors.index.json': 73297,
    'vae/config.json': 1701,
    'vae/diffusion_pytorch_model.safetensors': 2818777808,
}

MS_BASE = "https://www.modelscope.cn/models/Wan-AI/Wan2.2-TI2V-5B-Diffusers/resolve/master/"
WAN_SUBDIR = "Wan2.2-TI2V-5B-Diffusers"

# 小于这个体积的文件不指望迅雷，直接 HTTP 拉（总共不到 100 KB）
SMALL = 100 * 1024
# 移动大文件前如果两边不在同一个文件系统，copytree 会很慢；这里只用 shutil.move，跨 fs 时自动退化


def norm_src(p):
    """Windows 路径 -> WSL 路径（在 Linux 下才有意义）。"""
    if os.name == "nt":
        return p
    m = re.match(r"^([A-Za-z]):[\\/](.*)$", p)
    if m:
        return "/mnt/%s/%s" % (m.group(1).lower(), m.group(2).replace("\\", "/"))
    return p


def human(n):
    return "%.2f GB" % (n / 1e9) if n >= 1e9 else "%.1f MB" % (n / 1e6)


def scan_src(src):
    """递归收集 src 下所有文件的 basename -> 路径列表（迅雷可能新建子目录）。"""
    idx = {}
    for dp, dns, fns in os.walk(src):
        dns[:] = [d for d in dns if d not in (".td", "$RECYCLE.BIN")]
        for fn in fns:
            idx.setdefault(fn, []).append(os.path.join(dp, fn))
    return idx


def pick(idx, base):
    """按 basename 找文件；容忍迅雷的「name (1).ext」重名后缀。"""
    cands = list(idx.get(base) or [])
    stem, ext = os.path.splitext(base)
    pat = re.compile(r"^%s \(\d+\)%s$" % (re.escape(stem), re.escape(ext)))
    for k, vs in idx.items():
        if pat.match(k):
            cands.extend(vs)
    return cands


def fetch(url, dst):
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    print("  ⬇️  %s" % os.path.relpath(dst, os.path.dirname(os.path.dirname(dst))))
    with urllib.request.urlopen(url, timeout=60) as r, open(dst, "wb") as f:
        shutil.copyfileobj(r, f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="迅雷批量下载的目录（扁平结构即可）")
    ap.add_argument("--dst", default=None, help="Wan 仓库目录，默认 $FLASHWORLD_MODEL_DIR/Wan2.2-TI2V-5B-Diffusers")
    ap.add_argument("--keep", action="store_true", help="只复制不移动（默认移动，省 D 盘空间要多跑一次拷贝）")
    args = ap.parse_args()

    src = norm_src(args.src)
    root = os.environ.get("FLASHWORLD_MODEL_DIR", "/mnt/d/wheel/flashworld_ms")
    dst = args.dst or os.path.join(root, WAN_SUBDIR)
    ms_endpoint = os.environ.get("MS_ENDPOINT", MS_BASE).rstrip("/") + "/"

    if not os.path.isdir(src):
        sys.exit("❌ 源目录不存在: %s" % src)

    print("🚀 [01c] 整理扁平下载 -> HF 仓库结构")
    print("  📥 源:  %s" % src)
    print("  📦 目标: %s" % dst)
    print("  🔧 方式: %s" % ("复制" if args.keep else "移动"))
    print("")

    idx = scan_src(src)
    print("  🔍 源目录内文件数: %d" % sum(len(v) for v in idx.values()))

    # basename 在清单里出现多次的（三个 config.json）绝不按名字取，避免张冠李戴
    seen = {}
    for rel in MANIFEST:
        seen.setdefault(os.path.basename(rel), []).append(rel)
    ambiguous = set(b for b, rs in seen.items() if len(rs) > 1)

    moved, fetched, skipped, missing, wrong = [], [], [], [], []
    for rel, size in sorted(MANIFEST.items()):
        d = os.path.join(dst, rel)
        base = os.path.basename(rel)

        if os.path.exists(d) and os.path.getsize(d) == size:
            skipped.append(rel)
            continue

        hit = None
        if base not in ambiguous:
            for c in pick(idx, base):
                if os.path.getsize(c) == size:
                    hit = c
                    break
                if hit is None:
                    wrong.append((rel, os.path.getsize(c), size))

        if hit is not None:
            os.makedirs(os.path.dirname(d), exist_ok=True)
            if os.path.exists(d):
                os.remove(d)
            if args.keep:
                shutil.copy2(hit, d)
            else:
                shutil.move(hit, d)
            moved.append(rel)
        elif size <= SMALL or base in ambiguous:
            try:
                fetch(ms_endpoint + rel, d)
                if os.path.getsize(d) != size:
                    wrong.append((rel, os.path.getsize(d), size))
                else:
                    fetched.append(rel)
            except Exception as e:
                missing.append((rel, "下载失败: %s" % e))
        else:
            missing.append((rel, "源目录里没找到同尺寸文件"))

    print("")
    print("  ✅ 移动 %d / 下载 %d / 已就绪 %d" % (len(moved), len(fetched), len(skipped)))
    if wrong:
        print("  ⚠️ 尺寸不符 %d 个（可能还没下完）:" % len(wrong))
        for rel, got, want in wrong[:5]:
            print("     %-60s %s != %s" % (rel, human(got), human(want)))
    if missing:
        print("  ❌ 缺失 %d 个:" % len(missing))
        for rel, why in missing[:10]:
            print("     %-60s %s" % (rel, why))
        print("")
        sys.exit("❌ 未整理完整，补齐后重跑本脚本即可（已就位的会跳过）")

    total = sum(os.path.getsize(os.path.join(dst, r)) for r in MANIFEST)
    print("  📊 目标目录合计 %s" % human(total))
    print("")
    print("🎉 Done. 下一步: bash flashworld/01b_link_local_weights.sh")


if __name__ == "__main__":
    main()

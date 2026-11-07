#!/usr/bin/env python3
"""AUDIT-3：参数空间扫掠 —— 判「opacity / scale 是否被误读」的判官

## 为什么扫这两个量

PLY 里 opacity 存 logit、scale 存 log，这是 3DGS 的约定；只要有一处口径错
（少做/多做一次 sigmoid、漏乘一次 depth 或 scale_factor、把 σ 当 3σ 用），
表现就是**全局系数偏了**，而不是某个视角单独坏。此时扫一个全局增益 k，
PSNR 会在 k≠1 处出现明显峰值：

  PSNR 峰在 k≈1     → 参数空间读对了，剩下的糊/重影不是口径问题
  PSNR 峰在 k≫1     → alpha 偏薄（或散点偏小），是系统性口径 bug，可修

同时记录 alpha 中位数，作为"薄"的独立证据（k 放大到什么程度才不透光）。

用法（经 audit_2_render_decomp.sh 的同一套 CUDA 环境）：
  PLY=... NPZ=... OUT=... OP_GAINS=1,2,4,8,16 SC_GAINS=1,1.5,2,3
"""
import json
import os
import re
from pathlib import Path

import numpy as np
import torch
from gsplat import rasterization

PLY = Path(os.path.expanduser(os.environ["PLY"]))
NPZ = Path(os.path.expanduser(os.environ["NPZ"]))
OUT = Path(os.path.expanduser(os.environ["OUT"]))
VIEWS = [int(x) for x in os.environ.get("VIEWS", "0,8,16").split(",")]
OP_GAINS = [float(x) for x in os.environ.get("OP_GAINS", "1,2,4,8,16").split(",")]
SC_GAINS = [float(x) for x in os.environ.get("SC_GAINS", "1,1.5,2,3").split(",")]
SIZE = os.environ.get("RENDER_SIZE", "378x504")
REF_DIR = Path(os.path.expanduser(os.environ.get(
    "REF", "~/output/da3_results/_captures/佳佳抱枕/images")))


def log(s):
    print(s, flush=True)


def load_ply(path):
    with open(path, "rb") as f:
        raw = f.read(65536)
        end = raw.index(b"end_header") + len(b"end_header") + 1
        head = raw[:end].decode("ascii", "replace")
        n = int(re.search(r"element vertex (\d+)", head).group(1))
        props = [l.split()[2] for l in head.splitlines() if l.startswith("property")]
        f.seek(end)
        arr = np.fromfile(f, dtype=np.float32).reshape(n, len(props))
    nm = {k: i for i, k in enumerate(props)}
    means = arr[:, [nm["x"], nm["y"], nm["z"]]]
    sh = arr[:, [nm["f_dc_0"], nm["f_dc_1"], nm["f_dc_2"]]]
    ri = sorted((int(k.split("_")[-1]), i) for k, i in nm.items() if k.startswith("f_rest_"))
    if ri:
        rest = arr[:, [i for _, i in ri]]
        n_rest = rest.shape[1] // 3
        rest = rest.reshape(n, 3, n_rest).transpose(0, 2, 1)
        sh = np.concatenate([sh[:, None, :], rest], axis=1)
    else:
        sh = sh[:, None, :]
    return (means, sh, arr[:, nm["opacity"]],
            arr[:, [nm["scale_0"], nm["scale_1"], nm["scale_2"]]],
            arr[:, [nm["rot_0"], nm["rot_1"], nm["rot_2"], nm["rot_3"]]])


def psnr(a, b):
    mse = float(np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2))
    return 99.0 if mse == 0 else 10 * np.log10(255.0 ** 2 / mse)


def main():
    import cv2
    dev = "cuda"
    means_np, sh_np, op_np, sc_np, q_np = load_ply(PLY)
    log(f"📦 {PLY.name}: {len(means_np):,} 高斯")
    sh_degree = int(round(sh_np.shape[1] ** 0.5)) - 1
    means = torch.from_numpy(means_np).to(dev)
    sh = torch.from_numpy(sh_np).to(dev)
    op_raw = torch.from_numpy(op_np).to(dev)
    sc_raw = torch.from_numpy(sc_np).to(dev)
    quats = torch.from_numpy(q_np).to(dev)
    quats = quats / quats.norm(dim=-1, keepdim=True).clamp_min(1e-9)

    d = np.load(NPZ)
    w2c4 = np.concatenate([d["extrinsics"], np.tile(np.array([[[0, 0, 0, 1]]], np.float32),
                                                    (d["extrinsics"].shape[0], 1, 1))], axis=1)
    H0, W0 = d["depth"].shape[-2:]
    W, H = (int(v) for v in SIZE.lower().replace("*", "x").split("x"))
    Ks = d["intrinsics"].astype(np.float32).copy()
    Ks[:, 0, :] *= W / W0
    Ks[:, 1, :] *= H / H0
    refs = sorted([p for p in REF_DIR.iterdir()
                   if p.suffix.lower() in (".png", ".jpg", ".jpeg")])
    log(f"📷 {SIZE}  SH degree={sh_degree}  视角 {VIEWS}")

    def run(ogain, sgain):
        opacity = torch.sigmoid(op_raw + float(np.log(max(ogain, 1e-9))))
        scales = torch.exp(sc_raw) * sgain
        ps, al = [], []
        for v in VIEWS:
            ref = cv2.cvtColor(cv2.imread(str(refs[v])), cv2.COLOR_BGR2RGB)
            ref = cv2.resize(ref, (W, H), interpolation=cv2.INTER_AREA)
            vm = torch.from_numpy(w2c4[v]).float().to(dev)[None]
            K = torch.from_numpy(Ks[v]).float().to(dev)[None]
            with torch.no_grad():
                img, a, _ = rasterization(means, quats, scales, opacity, sh, vm, K, W, H,
                                          sh_degree=sh_degree)
            im = img[0].clamp(0, 1).mul(255).byte().cpu().numpy()
            ps.append(psnr(im, ref))
            al.append(float(np.squeeze(a).mean()))
        return float(np.mean(ps)), float(np.mean(al))

    rows = []
    log("\n[opacity 扫掠]  gain  PSNR   mean_alpha")
    for g in OP_GAINS:
        p, a = run(g, 1.0)
        rows.append(dict(kind="opacity", gain=g, psnr=p, alpha=a))
        log(f"                {g:5.2f}  {p:6.2f}  {a:.3f}")
    log("\n[scale 扫掠]    gain  PSNR   mean_alpha")
    for g in SC_GAINS:
        p, a = run(1.0, g)
        rows.append(dict(kind="scale", gain=g, psnr=p, alpha=a))
        log(f"                {g:5.2f}  {p:6.2f}  {a:.3f}")

    # 判据
    op_rows = [r for r in rows if r["kind"] == "opacity"]
    sc_rows = [r for r in rows if r["kind"] == "scale"]
    best_op = max(op_rows, key=lambda r: r["psnr"])
    best_sc = max(sc_rows, key=lambda r: r["psnr"])
    base_op = op_rows[0]["psnr"]
    base_sc = sc_rows[0]["psnr"]
    log("\n" + "=" * 60)
    log(f"opacity 最优增益 {best_op['gain']:.2f}（{best_op['psnr']:.2f} dB，"
        f"相对 k=1 的 {base_op - base_op:+.2f} → {best_op['psnr'] - base_op:+.2f} dB）")
    log(f"scale   最优增益 {best_sc['gain']:.2f}（{best_sc['psnr']:.2f} dB，"
        f"相对 k=1 {best_sc['psnr'] - base_sc:+.2f} dB）")
    log(f"⇒ opacity 口径：{'✅ 正确（峰在 1）' if best_op['gain'] <= 2.01 else '❌ 偏薄，疑似口径 bug'}")
    log(f"⇒ scale   口径：{'✅ 正确（峰在 1）' if best_sc['gain'] <= 1.51 else '❌ 偏小，疑似口径 bug'}")

    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / "audit_param_sweep.json", "w", encoding="utf-8", newline="\n") as f:
        json.dump(dict(ply=str(PLY), size=SIZE, views=VIEWS, rows=rows), f,
                  ensure_ascii=False, indent=1)
    log(f"💾 {OUT / 'audit_param_sweep.json'}")


if __name__ == "__main__":
    main()

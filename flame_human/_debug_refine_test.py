"""_debug_refine_test.py — 05 对齐"欠训练"验证：加长迭代 + 放缓 LR 衰减。

假设（来自本日诊断）：05 的 run_stage 用 gamma=0.95 逐迭代衰减，300 步后
LR 只剩 2e-7（后 200 步空转）；而单帧自拟合能到 ~9px、全局却是 ~20px。
本脚本以 05 结果为初值，用 gamma=0.9995 + 3000 步重跑 4.3 式联合优化，
看 RMS 能降到多少 → 判定「欠训练」假设是否成立。
"""
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from colmap_io import read_cameras  # noqa: E402
from align_3dmm import FlameLM, run_stage, forward, reproj_rms  # noqa: E402


def main():
    results_dir = Path(os.environ["RESULTS_DIR"])
    source_dir = os.environ["SOURCE_DIR"]
    pid = os.environ.get("PID", "0")
    n_iter = int(os.environ.get("REFINE_ITERS", "3000"))
    gamma = float(os.environ.get("REFINE_GAMMA", "0.9995"))
    lr = float(os.environ.get("REFINE_LR", "5e-3"))
    dev = torch.device("cuda")

    recon = json.loads((results_dir / "04_recon/face_recon.json").read_text())
    al = json.loads(
        (results_dir / "05_align/head_align.json").read_text())["persons"][pid]
    flame = FlameLM(os.environ["FLAME_MODEL"],
                    os.environ["FLAME_LM468_EMBEDDING"], dev)
    views = read_cameras(source_dir)
    view_by = {v["stem"]: v for v in views}
    stems = al["frames"]

    Rc = torch.tensor(np.stack([view_by[s]["R"] for s in stems]),
                      dtype=torch.float32, device=dev)
    tc = torch.tensor(np.stack([view_by[s]["T"] for s in stems]),
                      dtype=torch.float32, device=dev)
    K = torch.tensor(np.stack([
        [[view_by[s]["fx"], 0, view_by[s]["cx"]],
         [0, view_by[s]["fy"], view_by[s]["cy"]],
         [0, 0, 1.0]] for s in stems]), dtype=torch.float32, device=dev)
    uv_gt = torch.tensor(np.stack([
        np.asarray(recon["frames"][s]["persons"][pid]["lm468"], dtype=np.float32)
        for s in stems]), dtype=torch.float32, device=dev)

    pack = {
        "dev": dev, "Rc": Rc, "tc": tc, "K": K, "uv_gt": uv_gt,
        "log_s": torch.tensor(np.log(al["scale"]), dtype=torch.float32, device=dev),
        "gq": torch.tensor(al["global_q"], dtype=torch.float32, device=dev),
        "gt": torch.tensor(al["global_t"], dtype=torch.float32, device=dev),
        "lq": torch.tensor(np.array(al["local_q"]), dtype=torch.float32, device=dev),
        "lt": torch.tensor(np.array(al["local_t"]), dtype=torch.float32, device=dev),
        "id": torch.tensor(al["id_coeff"], dtype=torch.float32, device=dev),
        "exp": torch.tensor(np.array(al["exp_coeff"]), dtype=torch.float32, device=dev),
    }
    with torch.no_grad():
        uv = forward(flame.lm3d(pack["id"], pack["exp"]), pack["log_s"],
                     pack["gq"], pack["gt"], pack["lq"], pack["lt"], Rc, tc, K)
        f0 = reproj_rms(uv, uv_gt).cpu().numpy()
    print(f"初始: rms 中值={np.median(f0):.2f}px 均值={f0.mean():.2f}px")

    r = run_stage(flame, pack, n_iter, lr, 0.1,
                  {"global": True, "local": True, "id": True, "exp": True},
                  lam_id=0.0001, lam_exp=0.001, anchor_w=0.005,
                  verbose_tag="refine", gamma=gamma)
    pack.update({k: r[k] for k in ("log_s", "gq", "gt", "lq", "lt", "id", "exp")})
    with torch.no_grad():
        uv = forward(flame.lm3d(pack["id"], pack["exp"]), pack["log_s"],
                     pack["gq"], pack["gt"], pack["lq"], pack["lt"], Rc, tc, K)
        f1 = reproj_rms(uv, uv_gt).cpu().numpy()
    print(f"精修后: rms 中值={np.median(f1):.2f}px 均值={f1.mean():.2f}px  "
          f"max={f1.max():.1f}px")


if __name__ == "__main__":
    main()

"""_debug_selffit.py — 单帧自拟合上限探测（判定 20px 误差的性质）。

对指定帧，只优化该帧的 exp/pose（+可选 id），把 FLAME 拟合到该帧的 DECA
landmarks，测「单帧可达 RMS」：
  - 若自拟合 RMS 很低(~5px) → 模型表达力够；全局 20px 来自**跨帧一致性**
    （共享 id/exp 无法同时满足各帧的 landmark）
  - 若自拟合 RMS 仍 ~15-20px → **landmark↔FLAME 嵌入/标签本身**是瓶颈
"""
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from colmap_io import read_cameras  # noqa: E402
from align_3dmm import FlameLM, forward  # noqa: E402


def main():
    results_dir = Path(os.environ["RESULTS_DIR"])
    source_dir = os.environ["SOURCE_DIR"]
    pid = os.environ.get("PID", "0")
    dev = torch.device("cuda")
    recon = json.loads((results_dir / "04_recon/face_recon.json").read_text())
    al = json.loads(
        (results_dir / "05_align/head_align.json").read_text())["persons"][pid]
    flame = FlameLM(os.environ["FLAME_MODEL"],
                    os.environ["FLAME_LM468_EMBEDDING"], dev)
    views = read_cameras(source_dir)
    view_by = {v["stem"]: v for v in views}
    stems = al["frames"]

    gq = torch.tensor(al["global_q"], dtype=torch.float32, device=dev)
    gt = torch.tensor(al["global_t"], dtype=torch.float32, device=dev)
    log_s = torch.tensor(np.log(al["scale"]), dtype=torch.float32, device=dev)
    id0 = torch.tensor(al["id_coeff"], dtype=torch.float32, device=dev)

    def K_of(s):
        v = view_by[s]
        return torch.tensor([[v["fx"], 0, v["cx"]], [0, v["fy"], v["cy"]],
                             [0, 0, 1.0]], dtype=torch.float32, device=dev)

    for i in (0, 15, 29):
        s = stems[i]
        v = view_by[s]
        uv_gt = torch.tensor(
            np.asarray(recon["frames"][s]["persons"][pid]["lm468"],
                       dtype=np.float32), device=dev)
        Rc = torch.tensor(v["R"], dtype=torch.float32, device=dev)[None]
        tc = torch.tensor(v["T"], dtype=torch.float32, device=dev)[None]
        K = K_of(s)[None]
        for id_free in (False, True):
            exp = torch.zeros(1, 100, device=dev, requires_grad=True)
            lq = torch.tensor(np.array(al["local_q"])[i:i + 1], dtype=torch.float32,
                              device=dev).requires_grad_(True)
            lt = torch.tensor(np.array(al["local_t"])[i:i + 1], dtype=torch.float32,
                              device=dev).requires_grad_(True)
            idp = id0.clone().requires_grad_(True) if id_free else id0
            ps = [{"params": [exp, lq, lt], "lr": 0.01}]
            if id_free:
                ps.append({"params": [idp], "lr": 0.003})
            opt = torch.optim.Adam(ps)
            sch = torch.optim.lr_scheduler.ExponentialLR(opt, gamma=0.997)
            for it in range(1500):
                opt.zero_grad()
                uv = forward(flame.lm3d(idp, exp), log_s, gq, gt, lq, lt,
                             Rc, tc, K)
                loss = ((uv[0] - uv_gt) ** 2).sum(-1).mean()
                loss.backward()
                opt.step()
                sch.step()
            with torch.no_grad():
                uv = forward(flame.lm3d(idp, exp), log_s, gq, gt, lq, lt,
                             Rc, tc, K)
                rms = ((uv[0] - uv_gt) ** 2).sum(-1).sqrt().mean().item()
            print(f"  frame {i:2d}  id_{'free' if id_free else 'fix'}  "
                  f"自拟合 RMS = {rms:6.2f} px")


if __name__ == "__main__":
    main()

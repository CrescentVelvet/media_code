#!/usr/bin/env python3
"""align_3dmm.py — 阶段四：FLAME 三阶段对齐（global / local / coeff）。

变换链（设计文档 §7，顺序保证 SRT 与 exp 在表示层面独立）：
    p_can   = mean + B_id @ id + B_exp @ exp_f     ← 表情（中性空间）
    p_local = R_local_f @ p_can + t_local_f        ← 局部刚体
    p_world = global_s * (R_global @ p_local + t_global)   ← 全局相似
    uv      = π(K [R_cam|t_cam] p_world)           ← COLMAP 相机

三阶段（关键修正都在这里）：
  4.1 optimize_global：local_SRT 固定为 PnP/DECA 初值不参与优化。
      原设计「4.1 只用一个全局 SRT」会让残差由「偏离平均朝向多少」主导，
      median×reject_extent 就会系统性砍掉大角度转头帧。固定 local 后残差才
      真正反映 landmark 质量；reject 因此推迟到 4.2 之后。
  4.2 optimize_local：放开 local_SRT（global 参数 lr×0.1），anchor 正则 0.005。
      → check_local：局部旋转偏离中位 >55° 的帧**重置为 PnP 初值**（不是归零；
        归零等于把帧扔回 4.1 的平均朝向，4.3 从零学 >55° 很难收敛）
      → reject：per-frame RMS > median×reject_extent 的帧剔除，重跑 4.2
  4.3 optimize_coeff：联合 global + local + id(300) + exp(100)，L1 正则 + anchor 0.01

尺度初始化：COLMAP 的 t_cam 是 SfM 任意单位，PnP 的 t_pnp 是米，两者不能直接混。
用「头在 SfM 世界的位置 + SfM↔米 的尺度 s」做一次线性最小二乘解出来
（每帧 3 个方程，F 帧解 4 个未知数），避免让 Adam 从 1 开始爬好几个数量级。

Env: RECON_JSON / SOURCE_DIR / OUT_JSON / FLAME_MODEL / FLAME_LM468_EMBEDDING
     GLOBAL_ITERS / LOCAL_ITERS / COEFF_ITERS / LR_GLOBAL / LR_LOCAL / LR_COEFF
     REJECT_EXTENT / RESET_DEG / ANCHOR_W_LOCAL / ANCHOR_W_COEFF
     LAM_ID / LAM_EXP / DROP_OUTBOUND / MIN_FRAMES
"""
import os
import sys
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
from colmap_io import read_cameras, view_index  # noqa: E402

N_LM = 468
N_SHAPE = 300
N_EXPR = 100


def log(m):
    print(m, flush=True)


# ── 四元数 / 旋转 ────────────────────────────────────────────────────────────
def quat_to_mat(q):
    """q (...,4) [w,x,y,z] 已归一化 → R (...,3,3)，可微。"""
    q = F.normalize(q, dim=-1)
    w, x, y, z = q.unbind(-1)
    return torch.stack([
        1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y),
        2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
        2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y),
    ], dim=-1).reshape(q.shape[:-1] + (3, 3))


def mat_to_quat_np(R):
    from scipy.spatial.transform import Rotation
    q = Rotation.from_matrix(R).as_quat()  # (x,y,z,w)
    return np.concatenate([q[3:4], q[:3]]).astype(np.float64)


def quat_to_mat_np(q):
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ], dtype=np.float64)


def median_quat(qs):
    """输入 (N,4) [w,x,y,z] → 返回 [w,x,y,z]。

    用 scipy 的四元数平均，比逐维中位数稳（逐维中位数可能拼出非单位四元数）。
    """
    from scipy.spatial.transform import Rotation
    q = Rotation.from_quat(np.c_[qs[:, 1:4], qs[:, 0]]).mean().as_quat()  # (x,y,z,w)
    return np.concatenate([q[3:4], q[:3]])


# ── FLAME ────────────────────────────────────────────────────────────────────
class FlameLM:
    """给定 id / exp 求 468 个 landmark 的 3D 位置（可微）。"""

    def __init__(self, model_path, emb_path, device):
        import smplx
        self.flame = smplx.create(
            model_path=model_path, model_type="flame",
            num_expression_coeffs=N_EXPR, use_face_contour=False).to(device)
        for p in self.flame.parameters():
            p.requires_grad_(False)
        z = np.load(emb_path)
        self.lm_verts = torch.as_tensor(z["lm_verts"], device=device)
        self.lm_bary = torch.as_tensor(z["lm_bary"], dtype=torch.float32,
                                       device=device)
        self.verts_mean = torch.as_tensor(z["verts_mean"], dtype=torch.float32,
                                          device=device)
        # anchor：正则化 local_SRT 幅度用的固定点。
        # 优先用 FLAME 的 neck joint（头部旋转的枢轴），取不到就退回顶点质心。
        anchor = None
        try:
            J = self.flame.J_regressor
            if J is not None:
                Jt = torch.as_tensor(np.asarray(J), dtype=torch.float32,
                                     device=device)
                anchor = (Jt @ self.verts_mean)[0]
        except Exception:
            anchor = None
        if anchor is None:
            anchor = self.verts_mean.mean(0)
        env_a = os.environ.get("ANCHOR_XYZ", "")
        if env_a:
            anchor = torch.tensor([float(v) for v in env_a.split(",")],
                                  device=device)
        self.anchor = anchor
        self.device = device

    def lm3d(self, id_coeff, local_exp):
        """id (300,), exp (F,100) → (F,468,3)。"""
        f = local_exp.shape[0]
        betas = id_coeff.unsqueeze(0).expand(f, -1)
        out = self.flame(betas=betas, expression=local_exp)
        v = out.vertices
        return (v[:, self.lm_verts] * self.lm_bary[:, :, None]).sum(2)


# ── 正向模型 ─────────────────────────────────────────────────────────────────
def forward(lm3d, log_s, gq, gt, lq, lt, Rc, tc, K):
    """→ uv (F,468,2)。K (F,3,3)。"""
    Rl = quat_to_mat(lq)                       # (F,3,3)
    p_local = torch.bmm(Rl, lm3d.transpose(1, 2)).transpose(1, 2) + lt[:, None, :]
    Rg = quat_to_mat(gq)                       # (3,3)
    p_w = torch.matmul(p_local, Rg.T) + gt[None, None, :]
    p_w = p_w * torch.exp(log_s)
    p_c = torch.bmm(Rc, p_w.transpose(1, 2)).transpose(1, 2) + tc[:, None, :]
    uv = torch.bmm(K, p_c.transpose(1, 2)).transpose(1, 2)
    z = uv[:, :, 2:3]
    return uv[:, :, :2] / z.clamp(min=1e-9)


def reproj_rms(uv_pred, uv_gt):
    return torch.sqrt(((uv_pred - uv_gt) ** 2).sum(-1).mean(-1))  # (F,)


# ── 初始化 ───────────────────────────────────────────────────────────────────
def init_from_pnp(items, views, device):
    """PnP 位姿 → 全局/局部初值 + SfM↔米 尺度 s（线性最小二乘）。

    对每帧：R_cam·p_head + t_cam = s·t_pnp
    未知数 [p_head(3), s(1)]，F 帧共 3F 个方程，lstsq 解。
    """
    stems = [it["stem"] for it in items]
    A, b = [], []
    for s in stems:
        v = views[s]
        Rs.append(v["R"])
        ts_sfm.append(v["T"])
    Rs = np.stack(Rs)
    ts_sfm = np.stack(ts_sfm)

    for i, s in enumerate(stems):
        v = views[s]
        t_pnp = np.asarray(items[i]["pnp_t"], dtype=np.float64)
        A.append(np.hstack([v["R"], -t_pnp.reshape(3, 1)]))
        b.append(-v["T"])
    A = np.vstack(A)
    b = np.concatenate(b)
    sol, *_ = np.linalg.lstsq(A, b, rcond=None)
    p_head, s_scale = sol[:3], float(sol[3])
    if not np.isfinite(s_scale) or abs(s_scale) < 1e-12:
        s_scale = 1.0

    # 每帧头部在世界（SfM 单位）的位置与朝向
    R_f, t_f = [], []
    for i, s in enumerate(stems):
        v = views[s]
        R_pnp = np.asarray(items[i]["pnp_R"], dtype=np.float64).reshape(3, 3)
        t_pnp = np.asarray(items[i]["pnp_t"], dtype=np.float64)
        R_f.append(v["R"].T @ R_pnp)
        t_f.append(v["R"].T @ (s_scale * t_pnp - v["T"]))

    # 全局 = 中位（旋转用四元数平均，位置用逐维中位）
    q_f = np.stack([mat_to_quat_np(R) for R in R_f])
    q_g = median_quat(q_f)
    R_g = quat_to_mat_np(q_g)
    t_g = np.median(np.stack(t_f), axis=0)

    # 局部 = 全局的逆作用到每帧
    lq, lt = [], []
    for i in range(len(stems)):
        Rl = R_g.T @ R_f[i]
        tl = R_g.T @ (t_f[i] - t_g)
        lq.append(mat_to_quat_np(Rl))
        lt.append(tl)
    init = {
        "log_s": float(np.log(abs(s_scale))),
        "gq": q_g, "gt": t_g,
        "lq": np.stack(lq), "lt": np.stack(lt),
        "pnp_lq": np.stack(lq).copy(), "pnp_lt": np.stack(lt).copy(),
    }
    return init


# ── 三阶段 ───────────────────────────────────────────────────────────────────
def run_stage(flame, pack, n_iter, lr, lr_global_scale, optimize, lam_id=0.0,
              lam_exp=0.0, anchor_w=0.0, verbose_tag=""):
    """optimize: dict 指定哪些参数进优化器。"""
    dev = pack["dev"]
    log_s = pack["log_s"]
    gq, gt = pack["gq"], pack["gt"]
    lq, lt = pack["lq"], pack["lt"]
    idc, exp = pack["id"], pack["exp"]

    params = []
    if optimize.get("local"):
        lq = lq.detach().clone().requires_grad_(True)
        lt = lt.detach().clone().requires_grad_(True)
        params += [lq, lt]
    if optimize.get("global"):
        log_s = log_s.detach().clone().requires_grad_(True)
        gq = gq.detach().clone().requires_grad_(True)
        gt = gt.detach().clone().requires_grad_(True)
        params += [{"params": [log_s, gq, gt], "lr": lr * lr_global_scale}]
    if optimize.get("id"):
        idc = idc.detach().clone().requires_grad_(True)
        params.append(idc)
    if optimize.get("exp"):
        exp = exp.detach().clone().requires_grad_(True)
        params.append(exp)

    opt = torch.optim.Adam(params, lr=lr)
    sch = torch.optim.lr_scheduler.ExponentialLR(opt, gamma=0.95)
    hist = []
    for it in range(n_iter):
        opt.zero_grad()
        lm3d = flame.lm3d(idc, exp)
        uv = forward(lm3d, log_s, gq, gt, lq, lt,
                     pack["Rc"], pack["tc"], pack["K"])
        loss = ((uv - pack["uv_gt"]) ** 2).sum(-1).mean()
        if lam_id:
            loss = loss + lam_id * idc.abs().mean()
        if lam_exp:
            loss = loss + lam_exp * exp.abs().mean()
        if anchor_w and optimize.get("local"):
            Rl = quat_to_mat(lq)
            a_loc = torch.bmm(Rl, flame.anchor.expand(Rl.shape[0], 3, 1)) \
                .squeeze(-1) + lt
            loss = loss + anchor_w * ((a_loc - flame.anchor) ** 2).sum(-1).mean()
        loss.backward()
        opt.step()
        sch.step()
        hist.append(float(loss.item()))
        if (it + 1) % max(n_iter // 5, 1) == 0 or it == 0:
            with torch.no_grad():
                rms = reproj_rms(uv, pack["uv_gt"]).mean().item()
            log(f"      {verbose_tag} iter {it+1}/{n_iter} "
                f"loss={hist[-1]:.4f} rms={rms:.2f}px")

    return {
        "log_s": log_s.detach(), "gq": gq.detach(), "gt": gt.detach(),
        "lq": lq.detach(), "lt": lt.detach(),
        "id": idc.detach(), "exp": exp.detach(), "hist": hist,
    }


def check_local(lq, lt, pnp_lq, pnp_lt, thr_deg=55.0):
    """局部旋转偏离中位 > thr_deg 的帧 → 重置为 PnP 初值（不是归零）。"""
    q_np = lq.detach().cpu().numpy()
    q_med = median_quat(q_np)
    R_med = quat_to_mat_np(q_med)
    mask = np.zeros(len(q_np), dtype=bool)
    for i, q in enumerate(q_np):
        rel = quat_to_mat_np(q) @ R_med.T
        ang = np.degrees(np.arccos(np.clip((np.trace(rel) - 1) * 0.5, -1, 1)))
        if ang > thr_deg:
            mask[i] = True
    if mask.any():
        lq = lq.clone()
        lt = lt.clone()
        lq[mask] = torch.as_tensor(pnp_lq[mask], device=lq.device,
                                   dtype=lq.dtype)
        lt[mask] = torch.as_tensor(pnp_lt[mask], device=lt.device,
                                   dtype=lt.dtype)
    return lq, lt, mask


def main():
    recon_json = Path(os.environ.get(
        "RECON_JSON", f"{os.environ.get('RESULTS_DIR','')}/04_recon/face_recon.json"))
    source_dir = os.environ.get("SOURCE_DIR", "")
    out_json = Path(os.environ.get(
        "OUT_JSON", f"{os.environ.get('RESULTS_DIR','')}/05_align/head_align.json"))
    flame_model = os.environ.get("FLAME_MODEL", "")
    emb = os.environ.get("FLAME_LM468_EMBEDDING", "")
    g_iters = int(os.environ.get("GLOBAL_ITERS", "300"))
    l_iters = int(os.environ.get("LOCAL_ITERS", "300"))
    c_iters = int(os.environ.get("COEFF_ITERS", "300"))
    lr_g = float(os.environ.get("LR_GLOBAL", "1e-2"))
    lr_l = float(os.environ.get("LR_LOCAL", "1e-2"))
    lr_c = float(os.environ.get("LR_COEFF", "5e-3"))
    reject_extent = float(os.environ.get("REJECT_EXTENT", "5.0"))
    reset_deg = float(os.environ.get("RESET_DEG", "55"))
    aw_local = float(os.environ.get("ANCHOR_W_LOCAL", "0.005"))
    aw_coeff = float(os.environ.get("ANCHOR_W_COEFF", "0.01"))
    lam_id = float(os.environ.get("LAM_ID", "1e-4"))
    lam_exp = float(os.environ.get("LAM_EXP", "1e-3"))
    drop_outbound = int(os.environ.get("DROP_OUTBOUND", "1"))
    min_frames = int(os.environ.get("MIN_FRAMES", "8"))

    for p, tag in ((recon_json, "阶段三输出"), (flame_model, "FLAME"),
                   (emb, "lm468 嵌入")):
        if not p or not Path(p).exists():
            sys.exit(f"❌ 缺少 {tag}: {p}")

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(f"🗿 [阶段四] FLAME 三阶段对齐  device={dev}")
    flame = FlameLM(flame_model, emb, dev)
    views = view_index(read_cameras(source_dir))
    recon = json.loads(recon_json.read_text())

    # ── 按人聚合 ─────────────────────────────────────────────────────────
    persons = {}
    for stem, rec in recon.get("frames", {}).items():
        if stem not in views:
            continue
        for oid, r in rec.get("persons", {}).items():
            if not r.get("ok") or not r.get("lm468"):
                continue
            if not r.get("pnp"):
                continue
            if drop_outbound and r.get("crop_lm", {}).get("flag_outbound"):
                continue
            persons.setdefault(oid, []).append({
                "stem": stem,
                "lm": np.asarray(r["lm468"], dtype=np.float64),
                "pnp_R": r["pnp"]["R"], "pnp_t": r["pnp"]["t"],
                "deca": r.get("deca"),
            })
    log(f"  👥 有观测的人: { {k: len(v) for k, v in sorted(persons.items())} }")

    out = {"meta": {
        "recon_json": str(recon_json), "flame": flame_model,
        "g_iters": g_iters, "l_iters": l_iters, "c_iters": c_iters,
        "reject_extent": reject_extent, "reset_deg": reset_deg,
        "lam_id": lam_id, "lam_exp": lam_exp, "drop_outbound": bool(drop_outbound),
    }, "persons": {}}

    for oid in sorted(persons):
        items = sorted(persons[oid], key=lambda x: x["stem"])
        if len(items) < min_frames:
            log(f"  ⚠️ p{oid}: 帧数 {len(items)} < {min_frames}，跳过")
            continue
        log(f"  🔧 拟合 p{oid} ({len(items)} 帧)…")
        t0 = time.time()

        stems = [it["stem"] for it in items]
        Rc = torch.tensor(np.stack([views[s]["R"] for s in stems]),
                          dtype=torch.float32, device=dev)
        tc = torch.tensor(np.stack([views[s]["T"] for s in stems]),
                          dtype=torch.float32, device=dev)
        K = torch.tensor(np.stack([
            [[views[s]["fx"], 0, views[s]["cx"]],
             [0, views[s]["fy"], views[s]["cy"]],
             [0, 0, 1.0]] for s in stems]), dtype=torch.float32, device=dev)
        uv_gt = torch.tensor(np.stack([it["lm"] for it in items]),
                             dtype=torch.float32, device=dev)

        ini = init_from_pnp(items, views, dev)
        f = len(stems)
        # id / exp 初值：DECA 给了就用，否则零（exp=0 即中性脸）
        d0 = next((it["deca"] for it in items if it.get("deca")), None)
        id0 = np.asarray(d0["shape"], dtype=np.float32) if d0 else np.zeros(N_SHAPE, np.float32)
        if len(id0) < N_SHAPE:
            id0 = np.pad(id0, (0, N_SHAPE - len(id0)))
        exp0 = np.zeros((f, N_EXPR), dtype=np.float32)
        if d0 and d0.get("exp"):
            # 同一初值铺到所有帧；逐帧差异由 4.3 学
            e = np.asarray(d0["exp"], dtype=np.float32)[:N_EXPR]
            exp0[:, :len(e)] = e

        pack = {
            "dev": dev, "Rc": Rc, "tc": tc, "K": K, "uv_gt": uv_gt,
            "log_s": torch.tensor(ini["log_s"], dtype=torch.float32, device=dev),
            "gq": torch.tensor(ini["gq"], dtype=torch.float32, device=dev),
            "gt": torch.tensor(ini["gt"], dtype=torch.float32, device=dev),
            "lq": torch.tensor(ini["lq"], dtype=torch.float32, device=dev),
            "lt": torch.tensor(ini["lt"], dtype=torch.float32, device=dev),
            "id": torch.tensor(id0, dtype=torch.float32, device=dev),
            "exp": torch.tensor(exp0, dtype=torch.float32, device=dev),
        }
        pnp_lq = torch.tensor(ini["pnp_lq"], dtype=torch.float32, device=dev)
        pnp_lt = torch.tensor(ini["pnp_lt"], dtype=torch.float32, device=dev)

        with torch.no_grad():
            uv0 = forward(flame.lm3d(pack["id"], pack["exp"]), pack["log_s"],
                          pack["gq"], pack["gt"], pack["lq"], pack["lt"],
                          Rc, tc, K)
            log(f"     初值 RMS: {reproj_rms(uv0, uv_gt).mean().item():.2f} px "
                f"(s0={np.exp(ini['log_s']):.4g})")

        # 4.1 全局（local / id / exp 全部冻结）
        r = run_stage(flame, pack, g_iters, lr_g, 1.0,
                      {"global": True}, verbose_tag="4.1")
        pack.update({k: r[k] for k in ("log_s", "gq", "gt", "lq", "lt", "id", "exp")})

        # 4.2 局部（global 走 0.1× lr）
        r = run_stage(flame, pack, l_iters, lr_l, 0.1,
                      {"global": True, "local": True}, anchor_w=aw_local,
                      verbose_tag="4.2")
        pack.update({k: r[k] for k in ("log_s", "gq", "gt", "lq", "lt", "id", "exp")})

        # 4.2 校验：>55° 重置为 PnP 初值后重跑
        lq, lt, mask = check_local(pack["lq"], pack["lt"],
                                   pnp_lq.cpu().numpy(), pnp_lt.cpu().numpy(),
                                   reset_deg)
        if mask.any():
            log(f"     🔄 {int(mask.sum())} 帧局部旋转 >{reset_deg}°，"
                f"重置为 PnP 初值并重跑 4.2")
            pack["lq"], pack["lt"] = lq, lt
            r = run_stage(flame, pack, l_iters, lr_l, 0.1,
                          {"global": True, "local": True}, anchor_w=aw_local,
                          verbose_tag="4.2b")
            pack.update({k: r[k] for k in ("log_s", "gq", "gt", "lq", "lt", "id", "exp")})

        # reject：此处每帧已有独立 local_SRT，残差才干净
        with torch.no_grad():
            uv = forward(flame.lm3d(pack["id"], pack["exp"]), pack["log_s"],
                         pack["gq"], pack["gt"], pack["lq"], pack["lt"],
                         Rc, tc, K)
            frms = reproj_rms(uv, uv_gt).cpu().numpy()
        thr = max(3.0, float(np.median(frms)) * reject_extent)
        keep = frms <= thr
        if (~keep).sum() and keep.sum() >= min_frames:
            log(f"     ✂️  剔除 {int((~keep).sum())} 帧 "
                f"(rms>{thr:.1f}px, 最差 {frms.max():.0f}px)，重跑 4.2")
            idx = torch.tensor(np.where(keep)[0], device=dev)
            for kk in ("Rc", "tc", "K", "uv_gt"):
                pack[kk] = pack[kk][idx]
            pack["lq"], pack["lt"] = pack["lq"][idx], pack["lt"][idx]
            pack["exp"] = pack["exp"][idx]
            pnp_lq, pnp_lt = pnp_lq[idx], pnp_lt[idx]
            stems = [stems[i] for i in np.where(keep)[0]]
            r = run_stage(flame, pack, l_iters, lr_l, 0.1,
                          {"global": True, "local": True}, anchor_w=aw_local,
                          verbose_tag="4.2c")
            pack.update({k: r[k] for k in ("log_s", "gq", "gt", "lq", "lt", "id", "exp")})

        # 4.3 系数联合优化
        r = run_stage(flame, pack, c_iters, lr_c, 0.1,
                      {"global": True, "local": True, "id": True, "exp": True},
                      lam_id=lam_id, lam_exp=lam_exp, anchor_w=aw_coeff,
                      verbose_tag="4.3")
        pack.update({k: r[k] for k in ("log_s", "gq", "gt", "lq", "lt", "id", "exp")})

        with torch.no_grad():
            uv = forward(flame.lm3d(pack["id"], pack["exp"]), pack["log_s"],
                         pack["gq"], pack["gt"], pack["lq"], pack["lt"],
                         pack["Rc"], pack["tc"], pack["K"])
            frms = reproj_rms(uv, pack["uv_gt"]).cpu().numpy()
        log(f"     ✅ rms 中值 {np.median(frms):.2f}px / 均值 {frms.mean():.2f}px "
            f"({time.time()-t0:.0f}s, {len(stems)} 帧)")

        out["persons"][oid] = {
            "n_frames": len(stems),
            "frames": stems,
            "scale": float(torch.exp(pack["log_s"]).item()),
            "global_q": pack["gq"].cpu().tolist(),
            "global_t": pack["gt"].cpu().tolist(),
            "local_q": pack["lq"].cpu().tolist(),
            "local_t": pack["lt"].cpu().tolist(),
            "id_coeff": pack["id"].cpu().tolist(),
            "exp_coeff": pack["exp"].cpu().tolist(),
            "rms_px": frms.tolist(),
            "rms_median": float(np.median(frms)),
            "rms_mean": float(frms.mean()),
        }

    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(out))
    log(f"💾 {out_json}")
    log(f"🎉 阶段四完成，成功 {len(out['persons'])} 人")


if __name__ == "__main__":
    main()

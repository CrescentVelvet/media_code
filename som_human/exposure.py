#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""exposure.py — 方案 A：逐相机可学习曝光仿射补偿（ExposureBank）

来源：BACKLOG「方案 A」（服务器 Reconstruction exposure.py 的 ExposureModule）。
动机：手持采集 auto-exposure 逐帧波动（亮度/色温漂移），3DGS 的 loss 假设
所有帧光度一致，颜色漂移系统性干扰几何学习。double_wave 是户外场景，
01b 诊断里静态区色差偏高疑似含曝光成分。

模型：每相机一个 4×3 仿射 A，对渲染图做 RGB 齐次校正：
    img_adj = A^T @ [img_render; 1]      （A 初始化 = [I₃|0]，恒等）
训练口径（train_som.py 集成）：
  - iteration < EXPOSURE_FROM_ITER（默认 = warm_up）前 A 冻结为恒等（门控）；
  - 门控开启瞬间做**闭式初始化**：对全部训练帧渲染一次，
    X=[render;1]ᵀ(N×4)，torch.linalg.lstsq(X, GT) 解出 A；
    再做一轮抗外点（剔除残差 > 2×中位数的像素后重解）；
  - 之后 A 以独立小 lr（默认 1e-3）联合训练，DSSIM 也用校正后的图（口径一致）；
  - 正则 reg = REG_W * Σ|A - A_eye|（拉回恒等，防止与 SH 颜色互相吸收震荡）；
  - test 帧没训过 A：按 fid 与相邻 train 帧的 A 线性插值，评测同样 apply
    （train/test 指标与基线可比的前提）。

用法（全部走 env，与 SOM_* 同惯例；02_train.sh 有透传）：
  SOM_EXPOSURE=1                总开关（默认 0）
  SOM_EXPOSURE_FROM_ITER=-1     门控起点（-1 = warm_up）
  SOM_EXPOSURE_LR=1e-3          A 的独立 lr
  SOM_EXPOSURE_REG=0.01         拉回恒等正则权重
"""
import bisect
import os

import torch


class ExposureBank:
    """逐训练相机一个 4×3 齐次仿射；test 相机按 fid 线性插值。"""

    def __init__(self, names, fids, lr=1e-3, reg_weight=0.01):
        # names/fids：训练相机的 image_name / fid（float, 0~1），按 fid 升序
        order = sorted(range(len(names)), key=lambda i: fids[i])
        self.names = [names[i] for i in order]
        self.fids = [float(fids[i]) for i in order]
        self.name2idx = {n: i for i, n in enumerate(self.names)}
        self.reg_weight = reg_weight
        self.initialized = False  # 闭式初始化是否已做（门控用）

        A_eye = torch.zeros(4, 3)
        A_eye[:3, :] = torch.eye(3)
        self.A_eye = A_eye.cuda()
        self.A = torch.nn.Parameter(
            self.A_eye.unsqueeze(0).repeat(len(self.names), 1, 1))
        self.optimizer = torch.optim.Adam([self.A], lr=lr)

    # ── 应用 ─────────────────────────────────────────────────────────────
    @staticmethod
    def _apply_A(A, image):
        # image (3,H,W) → x (4,H,W) → out_c = Σ_j A[j,c]·x_j
        x = torch.cat([image, torch.ones_like(image[:1])], dim=0)
        return torch.einsum("jc,jhw->chw", A, x)

    def apply(self, name, image):
        """训练路径（可导）：训练相机查自己的 A。"""
        return self._apply_A(self.A[self.name2idx[name]], image)

    def interp_A(self, fid):
        """test 相机：按 fid 与相邻训练帧的 A 线性插值（A 近恒等，lerp 安全）。"""
        i = bisect.bisect_left(self.fids, float(fid))
        if i <= 0:
            return self.A[0]
        if i >= len(self.fids):
            return self.A[-1]
        t = (float(fid) - self.fids[i - 1]) / max(
            self.fids[i] - self.fids[i - 1], 1e-12)
        return self.A[i - 1] * (1.0 - t) + self.A[i] * t

    def apply_eval(self, name, fid, image):
        """评测路径（detach）：训练相机查表，test 相机插值。"""
        if name in self.name2idx:
            A = self.A[self.name2idx[name]]
        else:
            A = self.interp_A(fid)
        return self._apply_A(A.detach(), image)

    # ── 正则 ─────────────────────────────────────────────────────────────
    def reg_loss(self):
        """Σ|A - A_eye|（BACKLOG 口径：sum 而非 mean，权重默认 0.01）。"""
        return (self.A - self.A_eye).abs().sum()

    def max_dev(self):
        """诊断：当前 A 与恒等的最大绝对偏差（打印用）。"""
        return float((self.A.detach() - self.A_eye).abs().max())

    # ── 闭式初始化 ────────────────────────────────────────────────────────
    @torch.no_grad()
    def closed_form_init(self, renders, outlier_reject=True):
        """renders: {name: (render(3,H,W), gt(3,H,W), alpha|None)}，CPU 张量。
        每相机：lstsq 解 A；可选抗外点（剔残差>2×中位像素后重解）。"""
        A_all = self.A_eye.unsqueeze(0).repeat(len(self.names), 1, 1).cpu()
        n_pix_used = []
        for name, (r, g, alpha) in renders.items():
            X = torch.cat([r, torch.ones_like(r[:1])], 0).reshape(4, -1).T.double()
            Y = g.reshape(3, -1).T.double()
            if alpha is not None:
                m = alpha.reshape(-1) > 0.5
                X, Y = X[m], Y[m]
            sol = torch.linalg.lstsq(X, Y).solution  # (4,3)
            if outlier_reject:
                res = (X @ sol - Y).abs().mean(dim=1)
                keep = res <= 2.0 * res.median()
                if int(keep.sum()) > 100:
                    sol = torch.linalg.lstsq(X[keep], Y[keep]).solution
                    n_pix_used.append(int(keep.sum()))
            A_all[self.name2idx[name]] = sol.float()
        self.A.data = A_all.cuda()
        self.initialized = True
        dev = (A_all - self.A_eye.cpu()).abs()
        print("[EXPOSURE] closed-form init over {} cams: max|A-I|={:.4f} "
              "mean|A-I|={:.4f}".format(len(renders), float(dev.max()),
                                        float(dev.mean())), flush=True)

    # ── 存取 ─────────────────────────────────────────────────────────────
    def save_weights(self, model_path, iteration):
        out = os.path.join(model_path, "exposure",
                           "iteration_{}".format(iteration))
        os.makedirs(out, exist_ok=True)
        torch.save({"A": self.A.detach().cpu(), "names": self.names,
                    "fids": self.fids},
                   os.path.join(out, "exposure.pth"))

    @classmethod
    def load(cls, model_path, iteration=None):
        """从 model_path/exposure/iteration_N/exposure.pth 恢复（评测/渲染用）。
        iteration=None 取最大迭代号；文件不存在返回 None（未开曝光的模型）。"""
        exp_dir = os.path.join(model_path, "exposure")
        if not os.path.isdir(exp_dir):
            return None
        if iteration is None:
            its = [int(d.split("_")[1]) for d in os.listdir(exp_dir)
                   if d.startswith("iteration_")]
            if not its:
                return None
            iteration = max(its)
        p = os.path.join(exp_dir, "iteration_{}".format(iteration),
                         "exposure.pth")
        if not os.path.isfile(p):
            return None
        ckpt = torch.load(p, map_location="cpu")
        obj = cls.__new__(cls)
        obj.names = ckpt["names"]
        obj.fids = ckpt["fids"]
        obj.name2idx = {n: i for i, n in enumerate(obj.names)}
        A_eye = torch.zeros(4, 3)
        A_eye[:3, :] = torch.eye(3)
        obj.A_eye = A_eye.cuda()
        obj.A = ckpt["A"].cuda()  # 普通张量（评测不需要梯度）
        obj.optimizer = None
        obj.reg_weight = 0.0
        obj.initialized = True
        return obj

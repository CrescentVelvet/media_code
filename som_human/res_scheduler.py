# -*- coding: utf-8 -*-
"""res_scheduler.py — 分阶段分辨率调度（coarse-to-fine），供 train_som.py 使用。

动机（峰哥 2026-10-13 提供的加速方案，源自服务器 Reconstruction
optim_schduler.py 的 ResolutionScheduler；本文件是按其文字描述的重实现）：
  前段 1/4 分辨率 → 中段 1/2 → 末段全分辨率。光栅化成本 ∝ 像素数，
  前段渲染便宜 ~16 倍；D-3DGS 官方只有固定 -r，这里按 iteration 门控。

可行性关键点（已核实，见 README「分阶段分辨率调度」节）：
  - D-3DGS 的 Camera.projection_matrix 只依赖 FoV/znear/zfar，与分辨率无关；
    rasterizer 每次调用时读 viewpoint_camera.image_height/width（gaussian_renderer/
    __init__.py:54-55）→ 就地改属性即可改渲染分辨率，无需克隆相机。
  - cameras.json 在 Scene.__init__ 时落盘（scene/__init__.py:69-78），
    训练中的属性改动不会污染导出。
  - 评测（training_report）始终用 original_image 全分辨率 → PSNR 与基线可比。
  - SoM 运动基只消费 fid/xyz/rotation，不接触图像，与分辨率完全正交。

稠密化阈值补偿（默认开）：
  screenspace 梯度是像素单位，分辨率降为 1/s 时同一场景的梯度模长 ≈ 1/s。
  为保持「该稠密化多少」的判定不变，有效阈值 = densify_grad_threshold / s。
  ⚠️ 这是按量纲推导的启发式，未做消融；关掉置 SOM_RES_COMPENSATE_DENSIFY=0。

已知取舍（写入 README）：
  1. 细节结构（手指等）在 1/4 分辨率下梯度信号弱，稠密化集中在低频结构——
     我们本来就 --densify_until_iter 2000 早冻结（防爆显存），影响有限；
     若验收发现手部细节退化，第一个怀疑这里。
  2. SSIM 窗固定 11×11，低分辨率下等效感受野变大（loss 略偏平滑），可接受。
"""
import torch
import torch.nn.functional as F


def parse_schedule(spec):
    """解析 "0.3:4,0.7:2" → [(end_frac=0.3, scale=4), (0.7, 2), (1.0, 1)]。
    空串/None → 恒 1（等价关闭）。末段恒为全分辨率，无需显式写。"""
    if not spec:
        return [(1.0, 1)]
    phases = []
    for tok in spec.split(","):
        frac_s, scale_s = tok.split(":")
        frac, scale = float(frac_s), int(scale_s)
        if not (0.0 < frac < 1.0):
            raise ValueError("res_schedule 的分段点必须在 (0,1): %r" % tok)
        if scale < 1:
            raise ValueError("res_schedule 的 scale 必须 >=1: %r" % tok)
        phases.append((frac, scale))
    phases.sort()
    if any(phases[i][0] >= phases[i + 1][0] for i in range(len(phases) - 1)):
        raise ValueError("res_schedule 分段点必须严格递增: %r" % spec)
    phases.append((1.0, 1))
    return phases


class ResolutionScheduler:
    """按 iteration 返回当前降采样倍数，并管理训练相机的低分辨率 GT 缓存。

    用法（train_som.py 主循环）：
        sched = ResolutionScheduler("0.3:4,0.7:2", opt.iterations)
        sched.prepare(scene.getTrainCameras())
        ...
        scale = sched.scale(iteration)
        sched.apply(viewpoint_cam, scale)      # 就地改 image_height/width
        render_pkg = render(viewpoint_cam, ...) # rasterizer 此时读到低分辨率
        sched.restore(viewpoint_cam)            # 立即还原（eval/保存安全）
        gt = sched.gt_image(viewpoint_cam, scale)
    """

    def __init__(self, spec, iterations, min_side=64):
        self.phases = parse_schedule(spec)
        self.iterations = iterations
        self.min_side = min_side
        self._full = {}       # uid -> (H, W) 原始分辨率
        self._gt_cache = {}   # (uid, scale) -> Tensor[C, H/s, W/s]（与原图同 device）
        self._last_scale = None

    @property
    def enabled(self):
        return len(self.phases) > 1

    def scale(self, iteration):
        frac = iteration / float(self.iterations)
        for end_frac, s in self.phases:
            if frac <= end_frac:
                return s
        return 1

    def prepare(self, train_cameras):
        for cam in train_cameras:
            self._full[cam.uid] = (cam.image_height, cam.image_width)
        if not self.enabled:
            return
        scales = sorted({s for _, s in self.phases if s > 1})
        for cam in train_cameras:
            H, W = self._full[cam.uid]
            for s in scales:
                h2, w2 = self._scaled_hw(H, W, s)
                img = cam.original_image
                if img.device.type != "cuda":
                    img = img.cuda()
                with torch.no_grad():
                    ds = F.interpolate(img.unsqueeze(0), size=(h2, w2),
                                       mode="bilinear", antialias=True, align_corners=False)
                self._gt_cache[(cam.uid, s)] = ds.squeeze(0).clamp(0.0, 1.0)
        desc = ", ".join("1/%d (%dx%d)" % (s, w2, h2)
                         for s in scales
                         for h2, w2 in [self._scaled_hw(H, W, s)])
        print("[RES] 分阶段分辨率: %s；末段全分辨率 (%dx%d)" % (desc, W, H))

    def _scaled_hw(self, H, W, s):
        return max(self.min_side, int(round(H / s))), max(self.min_side, int(round(W / s)))

    def apply(self, cam, scale):
        """就地改相机的渲染分辨率；返回实际生效的 scale（1 时不动）。"""
        if scale <= 1:
            return 1
        H, W = self._full[cam.uid]
        h2, w2 = self._scaled_hw(H, W, s=scale)
        cam.image_height, cam.image_width = h2, w2
        return scale

    def restore(self, cam):
        if cam.uid in self._full:
            cam.image_height, cam.image_width = self._full[cam.uid]

    def gt_image(self, cam, scale):
        """取当前 scale 下的 GT（已在 cuda 上，clamp 过）。"""
        if scale <= 1:
            return cam.original_image
        return self._gt_cache[(cam.uid, scale)]

    def densify_threshold(self, base_threshold, scale):
        """低分辨率下梯度模长 ≈1/s，阈值同步除以 s 保持稠密化判定不变。"""
        return base_threshold / max(1, scale)

    def log_transition(self, iteration, scale, H, W):
        """阶段切换时打一行日志（幂等，只在变化时打）。"""
        if scale != self._last_scale:
            h2, w2 = (H, W) if scale <= 1 else self._scaled_hw(H, W, scale)
            print("\n[RES] iter %d/%d → 分辨率 1/%d (%dx%d)"
                  % (iteration, self.iterations, scale, w2, h2))
            self._last_scale = scale

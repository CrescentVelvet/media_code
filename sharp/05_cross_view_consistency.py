#!/usr/bin/env python3
"""SHARP 跨视角自洽 + 两层错位的跨视角放大（S1a）。

承接 03/04（S0.5）：两层不是"表面+背面"，而是**同一表面的两次亚像素采样**
（source 视角下横向错位中位 0.746 px）。但那一切只在 extrinsic=I 的 source
视角成立。本脚本回答 EXPERIMENT_PLAN.md 里 S1a 的问题：

    两层那 0.75 px 错位，在别的视角下会不会摊开成可见重影？

配套要同时回答的三条前提：

* Q1 尺度自洽：三张单图各自独立前馈，米制尺度是否一致？
* Q2 深度自洽：A 的点从 T 视角**渲染**出的深度图 vs T 自己的深度图，残差多大？
  （首版用「点投影 + 整像素深度缓冲」比深度，被遮挡严重污染：92° 基线时 A 的点
  落到 T 的另一张表面，dz 大头来自遮挡关系而非几何误差，<10mm 只占 2.5%，该数
  不可用。现改为两边都交给 gsplat 做 z 排序 + alpha 合成，遮挡一视同仁。）
* Q3 错位放大：两层的 3D 位移 delta 在 B 视角投影成多少像素？（核心）
* Q4 跨视角收益：用 A 的点从 B 渲染，第二层是帮忙还是捣乱？

## 为什么这次的数比 S0.5 可信

S0.5 的佳佳 HEIC 无 EXIF，disparity_factor 走 30mm 兜底，绝对尺度不可信。
本脚本输入 jiajia_hold_3v，同目录 prior_camera.json 给了真实 K
(fx=1380.82, fy=1384.18, cx=717.65, cy=952.85 @ 1439x1919)，且三相机 K 完全相同
（同一设备定焦）。焦距直接取 K，主点用真实 cx/cy（不是 w/2）。

但仍然不是 GT 标定：prior_camera.json 是 SplatAnything 从这三图估计的。
初版这里写的是「R0 实测三中心解 Sim(3) 残差 2.0~2.7 度」，那个数与本脚本相关：
实测两两相机中心距离 1.1619 / 2.1636 / 2.7946 m，光轴夹角 47.6 / 86.1 / 124.8 度，
是**大基线**（不是"附近视角"）。

## ⚠️ Q4 的绝对值不能直接与自渲比（v2 修正，务必先读）

v1 报「自渲 26~27 dB vs 跨视角 6.3~7.8 dB，差 19 dB」并据此判定"几何完全没对上"。
**这个对照不公平，已撤回。** Q4 的 `psnr` 是全图：渲染底色是纯黑，GT 是真实房间，
两者背景本来就不同，6~8 dB 里绝大部分来自"两张不同的照片本来就不一样"，
不是几何误差。`psnr_covered`（alpha>0.5 内）已经排除了背景，但仍不能说明对齐好坏。

v2 的判据是**三方同掩码对拍**（identity / 单平面单应 / SHARP 渲染，都只在 alpha>0.5
内算 PSNR）：

    identity  = 完全不做 warp，直接逐像素比（"两张照片的天然差异"这个下界）
    planeH    = 单个前向平行平面的 2D 单应 Kb(R + t n^T/d) Ka^-1，d 扫描取最优
    SHARP     = A 的点经 R,t 变换后从 B 渲染

下表是 **06_cross_view_fair_compare.py 的实测输出**（v3，2026-11-17 修双重变换后重跑；
`S1a_crossview_13v_*/fair_compare.json` 为唯一权威来源；`d` 为每对扫描取最优值）：

| 组 | 视角对 | ang | mask | identity | planeH@d | SHARP | S−I | S−P |
|---|---|---|---|---|---|---|---|---|
| 13v_small22 | 000→002 | 21.7° | 91% | 11.64 | 13.21@1.6 | 15.44 | +3.81 | +2.23 |
| 13v_small22 | 000→015 | 11.5° | 96% | 12.40 | 14.72@1.6 | 16.36 | +3.96 | +1.63 |
| 13v_small22 | 002→000 | 21.7° | 90% | 11.70 | 13.53@1.4 | 15.37 | +3.67 | +1.83 |
| 13v_small22 | 002→015 | 10.4° | 97% | 12.11 | 14.78@1.4 | 16.85 | +4.74 | +2.07 |
| 13v_small22 | 015→000 | 11.5° | 99% | 12.44 | 13.99@1.6 | 16.34 | +3.90 | +2.35 |
| 13v_small22 | 015→002 | 10.4° | 100% | 12.07 | 14.05@1.6 | 17.08 | +5.01 | +3.02 |
| 13v_small26 | 000→003 | 35.9° | 80% | 11.20 | 12.42@1.6 | 14.78 | +3.58 | +2.36 |
| 13v_small26 | 002→003 | 14.2° | 96% | 12.41 | 13.39@1.4 | 15.54 | +3.13 | +2.15 |
| 13v_small26 | 003→000 | 35.9° | 82% | 11.08 | 12.41@1.4 | 14.88 | +3.79 | +2.47 |
| 13v_small26 | 003→002 | 14.2° | 99% | 12.08 | 13.25@1.6 | 16.49 | +4.41 | +3.24 |
| 13v_large125 | 000→004 | 47.6° | 69% | 10.90 | 12.05@1.6 | 14.12 | +3.22 | +2.07 |
| 13v_large125 | 000→022 | 86.1° | 39% | 10.47 | 12.99@4.0 | 11.98 | +1.51 | **−1.01** |
| 13v_large125 | 004→000 | 47.6° | 71% | 10.45 | 11.87@1.6 | 13.44 | +3.00 | +1.57 |
| 13v_large125 | 004→022 | 124.8° | 32% | 10.85 | 12.59@0.8 | 12.01 | +1.16 | **−0.59** |
| 13v_large125 | 022→000 | 86.1° | 45% | 10.18 | 9.45@2.0 | 13.59 | +3.41 | +4.14 |
| 13v_large125 | 022→004 | 124.8° | 41% | 9.28 | 11.89@0.6 | 12.49 | +3.21 | +0.60 |

（`13v_small26` 的 000→002 / 002→000 与 `13v_small22` 同名对是同一对视角，数字相同，
去重后合计 16 对。）

**汇总（16 对）：`S−I` 中位 +3.73 dB（16/16 全为正，范围 +1.16 ~ +5.01）；
`S−P` 中位 +2.11 dB（14/16 为正，范围 −1.01 ~ +3.24）。**

## 🔴 v2 的结论整个是错的（2026-11-17 修 bug 后撤回）

v2 曾报「`S−I` 中位 +0.15、`S−P` 中位 −1.65，17/18 对输给单平面单应」，据此判定
「354 万点打不过一个 2D homography」。**那个结论建立在双重变换 bug 上，已全部撤回。**
修正后 SHARP **16/16 对赢 identity、14/16 对赢 planeH**，量级 +2~+4 dB。

正确的读法：

* **净贡献是真实的**：+3.73 dB 说明跨视角几何确实在起作用，不是"再准一点就能用"
  也不是"系统性错"。v2 那句「跨视角位置/朝向是系统性错的」**是错的**。
* **小基线更好但不构成平台期**：10~22° 组 S−P 中位 +2.19、25~60° 组 +2.21，
  到 86°/125° 才转负（−1.01 / −0.59）。两条转负的对都是 `→022`，
  而 022 是全组最远的视角（基线 2.79 m）且覆盖率只有 32~41%。
  **这是覆盖率问题不是几何问题**——在只剩三成像素上算 PSNR，本身方差就大。
* **但 11~17 dB 绝对水平仍远低于"可用"**。赢 identity 不等于能当融合基座：
  同掩码 17 dB 意味着纳伪融合时是**破坏性**的（对照 SA 自渲 26~27 dB）。
  结论从「几何信息≈0 甚至为负」改成「**几何方向对、精度不够**」。

所以本脚本报的**相对关系（层间 delta、跨视角放大倍数）比绝对 PSNR 可信**。

## 坐标约定

* SHARP 输出在**各自相机系**（OpenCV: x 右 y 下 z 前），无外参。
* prior_camera.json 的 extrinsics 是 **OpenCV c2w**（见 SplatAnything
  src/utils/camera_io.py:49 "Read OpenCV c2w poses"）。
* 把 A 相机系的点搬到 B 相机系：p_B = inv(c2w_B) @ c2w_A @ p_A。
* 刚体变换必须同时处理均值与协方差，用官方 apply_transform
  （sharp.utils.gaussians:101，内部走 Sigma' = R Sigma R^T 再分解）。
  注意它接受 3x4（取 [...,:3,:3] 与 [...,:3,3]），传 4x4 也能跑但语义要自己保证。

## 口径边界

* 不训练、不微调（官方未开源训练代码），只前向 + 渲染。
* 渲染底色默认 black（贴合官方 cli/render.py）；单层对底色敏感（S0.5 实测差 6.4 dB），
  故跨视角单层对照一律同底色，跨底色比较禁止。
* 跨视角渲染的 PSNR **不能与 source 自洽 PSNR 横比**（前者含相机误差与被摄对象
  未共视的部分；而且底色不同就不可比）。能做的对照只有三个：
  ①同一 target 视角下 full vs layer0/layer1；
  ②同一 target 视角、同一掩码下的 **identity**（不 warp 直接贴）与 **planeH**
    （单平面单应）——这两个都是"不含 SHARP 几何"的基线，用来判断 SHARP 有没有
    贡献，见上「Q4 的绝对值不能直接与自渲比」；
  ③跨基线长度的对比（须同掩码、同视角对类型）。
  禁止单独拿 Q4 的 `psnr`（全图）去和自渲数比大小。
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch


# ------------------------------------------------------------------ 基础工具


def load_camera_json(path: Path) -> tuple[dict, dict]:
    """读 prior_camera.json -> (c2w dict, K dict)，key 为 image stem。"""
    data = json.loads(path.read_text(encoding="utf-8"))
    c2w = {e["camera_id"]: np.asarray(e["matrix"], dtype=np.float64) for e in data["extrinsics"]}
    kmat = {e["camera_id"]: np.asarray(e["matrix"], dtype=np.float64) for e in data["intrinsics"]}
    return c2w, kmat


def psnr(pred: np.ndarray, target: np.ndarray, mask: np.ndarray | None = None) -> float:
    if mask is not None and mask.any():
        pred, target = pred[mask], target[mask]
    mse = float(np.mean((pred.astype(np.float64) - target.astype(np.float64)) ** 2))
    return 10.0 * math.log10(1.0 / max(mse, 1e-12))


def load_rgb(path: Path) -> np.ndarray:
    """读原图 sRGB **uint8 0..255**，与官方 `io.load_rgb` 同口径。

    ⚠️ 必须返回 0..255：`predict()` 内部会再 `/255`（复刻官方
    `cli/predict.py:168` 的 `image.copy().float().permute(2,0,1) / 255.0`）。
    本脚本首版此处返回 0..1，实际输入被压到 1/255，网络吃到的图几乎全黑，
    层间几何随之崩坏：dxy 2.33mm → 19.3mm、dz +0.19mm → −4.14mm、
    dpx 1.59px → 21.5px，z 中位 2.43m → 1.56m。派生的 Q1~Q4 全部作废重跑。
    GT 请用 `load_rgb_gt()`。
    """
    from sharp.utils import io

    image_u8, _, _ = io.load_rgb(path)
    return np.asarray(image_u8, dtype=np.uint8)


def load_rgb_gt(path: Path) -> np.ndarray:
    """GT 口径：sRGB float32 0..1。"""
    return load_rgb(path).astype(np.float32) / 255.0


def save_png(arr: np.ndarray, path: Path) -> None:
    from PIL import Image

    Image.fromarray(np.clip(arr * 255.0, 0, 255).astype(np.uint8)).save(path)


def save_gray(arr: np.ndarray, path: Path, vmin=None, vmax=None) -> None:
    from PIL import Image

    a = np.asarray(arr, dtype=np.float32)
    lo = float(np.nanmin(a)) if vmin is None else float(vmin)
    hi = float(np.nanmax(a)) if vmax is None else float(vmax)
    if hi <= lo:
        hi = lo + 1e-6
    norm = np.clip((a - lo) / (hi - lo), 0.0, 1.0)
    Image.fromarray((norm * 255).astype(np.uint8)).save(path)


def subset_gaussians(g, start: int, end: int):
    from sharp.utils.gaussians import Gaussians3D

    return Gaussians3D(
        mean_vectors=g.mean_vectors[:, start:end].contiguous(),
        singular_values=g.singular_values[:, start:end].contiguous(),
        quaternions=g.quaternions[:, start:end].contiguous(),
        colors=g.colors[:, start:end].contiguous(),
        opacities=g.opacities[:, start:end].contiguous(),
    )


def concat_g(parts: list):
    from sharp.utils.gaussians import Gaussians3D

    return Gaussians3D(
        mean_vectors=torch.cat([p.mean_vectors for p in parts], dim=1),
        singular_values=torch.cat([p.singular_values for p in parts], dim=1),
        quaternions=torch.cat([p.quaternions for p in parts], dim=1),
        colors=torch.cat([p.colors for p in parts], dim=1),
        opacities=torch.cat([p.opacities for p in parts], dim=1),
    )


def render(renderer, g, extrinsics: torch.Tensor, intrinsics: torch.Tensor, h: int, w: int):
    """渲染一张。extrinsics 是 **w2c**（gsplat 约定，OpenCV）。返回 numpy。

    ⚠️⚠️ **必须与高斯所处的坐标系配对，这是本脚本踩过的最大坑。**

    两种合法用法，二选一，**绝不能叠加**：

      (a) 高斯在**相机系**（已用 `apply_w2c_to_gaussians` 搬过去）
          → `extrinsics` 必须传 **``torch.eye(4)``**（点已经在目标相机系了）。
      (b) 高斯在**世界系**（原始前馈输出，未变换）
          → `extrinsics` 传 **w2c**。

    2026-11-17 实测的 bug：Q4 用了 `apply_transform(g, w2c)` **再**传 `extrinsics=w2c`，
    等于 `p → R(R p + t) + t`，**变换施加两次**。后果是画面整体错位、alpha 覆盖率虚低
    （73% vs 正确的 91%）、跨视角 PSNR 被压低 **+3.45~+3.97 dB**（同一批像素，
    intersection 掩码下比较，见 `isolate_extrinsic.txt`）。这个 bug 让 SHARP 看起来
    「打不过单平面单应」，而修正后它其实赢——**结论被反转**。
    """
    with torch.no_grad():
        out = renderer(
            g,
            extrinsics=extrinsics[None],
            intrinsics=intrinsics[None],
            image_width=w,
            image_height=h,
        )
    color = np.clip(out.color[0].permute(1, 2, 0).float().cpu().numpy(), 0.0, 1.0)
    alpha = out.alpha[0, 0].float().cpu().numpy()
    depth = out.depth[0, 0].float().cpu().numpy()
    return color, alpha, depth


def check_render_equivalence(
    renderer,
    g_world,
    w2c_np: np.ndarray,
    Kt4_t: torch.Tensor,
    rh: int,
    rw: int,
    device,
    tag: str = "",
):
    """**回归检查（守门人）**：路径 (a) 与路径 (b) 必须等价。

    render() 的 docstring 写了两种合法用法。它们数学上等价：

        (a) 先把点搬到 T 相机系 `p' = R p + t`，再用 `extrinsics = eye` 渲染
        (b) 点留在世界系，用 `extrinsics = w2c` 渲染

    两条路径只应差浮点误差。**若不等价，说明 R,t 被施加了两次或零次**——
    2026-11-17 那个把结论反转的 bug 正是这样漏过两轮评审的。

    ── 阈值怎么定的（v5，2026-11-18）────────────────────────────────
    首版拍脑袋用「max|Δalpha|<1e-3 且 max|Δcolor|<1e-2」，在正常批次上报 1/6 等价，
    差点把整批结果白扔。原因是两个：

    1. **用了 max 统计**。gsplat 是 alpha 合成，像素级 max 由单个边界像素主导；
       两条路径的矩阵乘顺序不同（`inv(ndc@K@E)` vs `inv(E) 后 apply_transform`），
       边界像素的 alpha 分界必然漂 1~2 个量化阶。几何没错位时 coverage 一位不差。
    2. **阈值没有对照**。不知道"真故障"长什么样，就无法判断 1e-3 是噪声还是事故。

    所以本版改为：**先注入一次真故障**（把 w2c 施加两遍，正是 11-17 那个 bug 的形态），
    量出它的量级，再把判据定在"比噪声大、比故障小"的区间上，并检查分离度
    `fault_median / equiv_median >= 10`。分离度不够（比如 w2c 接近单位阵）时
    直接报 `inconclusive`，不假装通过。

    返回 dict（可直接进 json 台账）。
    """
    g_cam = apply_w2c_to_gaussians(g_world, w2c_np)
    eye = torch.eye(4, device=device)
    w2c_t = torch.from_numpy(w2c_np.astype(np.float32)).to(device)

    ca, aa, _ = render(renderer, g_cam, eye, Kt4_t, rh, rw)     # (a) 相机系 + eye
    cb, ab, _ = render(renderer, g_world, w2c_t, Kt4_t, rh, rw)  # (b) 世界系 + w2c

    # 故障注入：再乘一次 w2c，复现「变换被施加两次」的事故形态。
    g_fault = apply_w2c_to_gaussians(g_cam, w2c_np)
    cf, af, _ = render(renderer, g_fault, eye, Kt4_t, rh, rw)   # (c) 双重变换

    def _med(a, b):
        return float(np.median(np.abs(a - b)))

    d_alpha_max = float(np.max(np.abs(aa - ab)))
    d_alpha_med = _med(aa, ab)
    d_color_med = _med(ca, cb)
    cov_a, cov_b = float((aa > 0.5).mean()), float((ab > 0.5).mean())
    d_cov = abs(cov_a - cov_b)

    f_alpha_max = float(np.max(np.abs(aa - af)))
    f_alpha_med = _med(aa, af)
    f_color_med = _med(ca, cf)
    cov_f = float((af > 0.5).mean())
    f_cov = abs(cov_a - cov_f)

    # 判据：等价必须"贴地"（对 coverage 与中位差异都应几乎为零），
    # 且必须与故障量级拉开 ≥10x，否则这次检查本身无效。
    sep_alpha = f_alpha_med / max(d_alpha_med, 1e-12)
    sep_cov = f_cov / max(d_cov, 1e-12)
    near = (d_alpha_med < 1e-3) and (d_cov < 1e-3)
    separated = (sep_alpha >= 10.0) or (f_cov >= 1e-2)
    ok = bool(near and separated)
    inconclusive = bool(near and not separated)

    r = {
        "pair": tag,
        "max_abs_diff_alpha": d_alpha_max,
        "median_abs_diff_alpha": d_alpha_med,
        "median_abs_diff_color": d_color_med,
        "coverage_a_camera_space": cov_a,
        "coverage_b_world_space": cov_b,
        "abs_diff_coverage": d_cov,
        "fault_injection": {
            "max_abs_diff_alpha": f_alpha_max,
            "median_abs_diff_alpha": f_alpha_med,
            "median_abs_diff_color": f_color_med,
            "coverage_double_transform": cov_f,
            "abs_diff_coverage": f_cov,
            "separation_alpha": sep_alpha,
            "separation_coverage": sep_cov,
        },
        "equivalent": ok,
        "inconclusive": inconclusive,
        "note": (
            "判据：(a)(b) 中位 alpha 差 <1e-3 且 coverage 差 <1e-3（贴地）；"
            "且与双重变换故障的分离度 ≥10x（否则 inconclusive）。"
        ),
    }
    if inconclusive:
        flag, verdict = "⚠️", "检查无效（与故障未拉开），不作废也不声称通过"
    elif ok:
        flag, verdict = "✅", "等价"
    else:
        flag, verdict = "❌", "不等价，整批作废"
    print(
        f"   {flag} 回归检查 {tag}: Δalpha(med/max)={d_alpha_med:.2e}/{d_alpha_max:.2e} "
        f"Δcov={d_cov:.2e} | 故障注入 med={f_alpha_med:.2e} Δcov={f_cov:.2e} "
        f"分离={sep_alpha:.0f}x → {verdict}"
    )
    return r


# ------------------------------------------------------------------ 推理


def predict(
    predictor,
    image_u8: np.ndarray,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    device,
):
    """单图前馈 -> 相机坐标高斯。

    与 03/04 的差别：主点用真实 cx/cy（不用 w/2、h/2），且 **fx / fy 分开传**。
    官方 `cli/predict.py:166-203` 只接受一个 `f_px`（因为它拿不到标定），
    这里 `prior_camera.json` 给了 fx=1380.82 / fy=1384.18，丢掉其中一个没有道理。
    v4 复核（2026-11-18）把「05 把 fy 当成 fx」列为评测协议缺陷，此处修正。

    ⚠️ 但 `disparity_factor` **仍用 fx**：网络只有一档焦距输入槽
    （`disparity_factor = f_px / width`），训练分布就是单焦距，
    喂 fy 会改变网络行为而不只是几何。所以：
      * 网络输入口径 = fx（与官方一致，保证可比）
      * 反投影 K 的口径 = (fx, fy)（几何正确）
    两者的差只有 0.24%，实测影响记在 `_q0_protocol_audit["fy_vs_fx"]`
    （同一份 NDC 高斯只换反投影 K 的 fy 再反投影一次，量出点位差）。
    """
    import torch.nn.functional as F

    from sharp.utils.gaussians import unproject_gaussians

    internal_shape = (1536, 1536)
    # 防复发：输入必须是 0..255。若有人传 0..1，下面 /255 后整幅图变成 1/255 亮度，
    # 网络输出仍"看起来正常"（无 NaN），但几何全错——首版就是这样静默错了 8 倍。
    _mx = float(np.asarray(image_u8).max())
    if _mx <= 1.5:
        raise ValueError(
            f"predict() 期望 0..255 的 uint8 图，收到最大值 {_mx:.3f}；"
            "疑似传了 0..1 的 float。请用 load_rgb() 取原图、load_rgb_gt() 取 GT。"
        )
    image_pt = torch.from_numpy(image_u8.copy()).float().to(device).permute(2, 0, 1) / 255.0
    _, height, width = image_pt.shape
    disparity_factor = torch.tensor([fx / width]).float().to(device)

    image_resized_pt = F.interpolate(
        image_pt[None],
        size=(internal_shape[1], internal_shape[0]),
        mode="bilinear",
        align_corners=True,
    )

    # 4x4：unproject_gaussians 内部做 ndc @ K @ E 的 4x4 连乘
    intrinsics = torch.tensor(
        [
            [fx, 0, cx, 0],
            [0, fy, cy, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ],
        device=device,
        dtype=torch.float32,
    )
    intrinsics_resized = intrinsics.clone()
    intrinsics_resized[0] *= internal_shape[0] / width
    intrinsics_resized[1] *= internal_shape[1] / height

    with torch.no_grad():
        gaussians_ndc = predictor(image_resized_pt, disparity_factor)
        gaussians = unproject_gaussians(
            gaussians_ndc, torch.eye(4, device=device), intrinsics_resized, internal_shape
        )
        # fy 分离的实测影响：同一份 NDC 高斯，只把反投影 K 的 fy 换成旧口径值，看点位差多少。
        # 纯审计用途（v4 §4 要求量化「05 把 fy 当 fx」的代价），不参与任何结论。
        #
        # ⚠️ 必须按 resize 后的同尺度替换。`intrinsics_resized` 的 [0,0] 与 [1,1] 已分别乘过
        # 1536/w 与 1536/h（本例 1.067 vs 0.800，相差 33%），直接写 [1,1]=[0,0] 等于把焦距
        # 放大 33%，量出 0.153 m —— 比真实影响大两个数量级，首版就是这么错的。
        # 旧口径是 `f_px = fx` 走完 resize，即 K[1,1] = fx * 1536/h。
        K_fy_eq_fx = intrinsics_resized.clone()
        K_fy_eq_fx[1, 1] = fx * (internal_shape[1] / height)
        g_old = unproject_gaussians(
            gaussians_ndc, torch.eye(4, device=device), K_fy_eq_fx, internal_shape
        )
    mu_c = gaussians.mean_vectors[0]
    mu_o = g_old.mean_vectors[0]
    d_xyz = (mu_c - mu_o).abs()
    d_norm = torch.linalg.norm(mu_c - mu_o, dim=-1)
    rel = d_norm / torch.linalg.norm(mu_c, dim=-1).clamp_min(1e-9)
    fy_audit = {
        "fy_over_fx": float(fy / max(fx, 1e-9)),
        "median_abs_dz_m": float(d_xyz[:, 2].median()),
        "median_abs_dx_m": float(d_xyz[:, 0].median()),
        "median_abs_dy_m": float(d_xyz[:, 1].median()),
        "median_abs_dxyz_norm_m": float(d_norm.median()),
        "p90_abs_dxyz_norm_m": float(d_norm.quantile(0.90)),
        "median_rel_change": float(rel.median()),
        "note": "正数=用真 fy 相对旧口径（fy=fx）的位移；两者 dz 恒为 0（K 的 fy 只影响 y 与 z 的耦合）",
    }
    return {
        "gaussians": gaussians,
        "intrinsics": intrinsics,
        "size": (height, width),
        "fy_audit": fy_audit,
    }


# ------------------------------------------------------------------ 几何


def project_points(mu: np.ndarray, w2c: np.ndarray, K: np.ndarray) -> tuple:
    """把相机系点投影到像素。mu [N,3] -> (uv [N,2], z [N])。"""
    pc = mu @ w2c[:3, :3].T + w2c[:3, 3]
    z = pc[:, 2]
    valid = z > 1e-6
    uv = np.full((mu.shape[0], 2), np.nan, dtype=np.float64)
    uv[valid, 0] = K[0, 0] * pc[valid, 0] / z[valid] + K[0, 2]
    uv[valid, 1] = K[1, 1] * pc[valid, 1] / z[valid] + K[1, 2]
    return uv, z, valid


def stat(name: str, a: np.ndarray, unit: str = "", pct=(10, 50, 90)) -> dict:
    a = np.asarray(a, dtype=np.float64)
    a = a[np.isfinite(a)]
    if a.size == 0:
        return {}
    d = {
        "name": name,
        "n": int(a.size),
        "mean": float(a.mean()),
        "median": float(np.median(a)),
    }
    for p in pct:
        d[f"p{p}"] = float(np.percentile(a, p))
    print(
        f"   {name:34s} n={a.size:>9,}  med={d['median']:+10.5f}  "
        f"p10={d['p10']:+10.5f}  p90={d['p90']:+10.5f}{unit}"
    )
    return d


def layer_pair(g, n_layers: int = 2):
    """按 layer-major 切层（S0.5 用 hook 逐元素对拍证明过）。"""
    n_total = g.mean_vectors.shape[1]
    hw = n_total // n_layers
    side = int(round(math.sqrt(hw)))
    if hw * n_layers != n_total or side * side != hw:
        raise RuntimeError(f"点数 {n_total} 与 {n_layers} x {side}^2 不符")
    return [subset_gaussians(g, l * hw, (l + 1) * hw) for l in range(n_layers)], hw, side


def apply_w2c_to_gaussians(g, w2c_np: np.ndarray):
    """刚体变换（相机系 A -> 相机系 B）。用官方 apply_transform。

    ⚠️ `transform` 必须传 **3x4**。官方实现里是
    `mean_vectors @ transform_linear.T`（gaussians.py:116），`.T` 对 >2 维张量
    是"整体倒序"而非转置——传 `[1,3,4]` 会得到 `[3,10,3]` 这类错维，`len(mv)` 变成 3，
    下游 gsplat 按 `len(mean_vectors)` 定 batch 就会 `IndexError`。
    实测：3x4 → `(1,N,3)` 正确；1x3x4 → `(3,N,3)` 错误。
    """
    from sharp.utils.gaussians import apply_transform

    t = torch.from_numpy(w2c_np[:3, :].astype(np.float32)).to(g.mean_vectors.device)
    out = apply_transform(g, t)
    assert out.mean_vectors.shape[0] == g.mean_vectors.shape[0], (
        f"apply_transform 批维异常: {tuple(out.mean_vectors.shape)} vs "
        f"{tuple(g.mean_vectors.shape)}；检查 transform 是否被误加 batch 维"
    )
    return out


# ------------------------------------------------------------------ 主流程


def main() -> int:
    ap = argparse.ArgumentParser(description="SHARP 跨视角自洽（S1a）")
    ap.add_argument("--scene", required=True, type=Path,
                    help="场景目录，需含 images/ 与 prior_camera.json")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--ckpt", type=Path, default=None)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--render-scale", type=float, default=0.5,
                    help="渲染下采样；跨视角对照只需同口径，0.5 省 4x")
    ap.add_argument("--bg", default="black", help="底色。跨视角单层对照必须与 source 一致")
    ap.add_argument("--max-views", type=int, default=0, help="限制视角数（调试用，0=全部）")
    ap.add_argument("--views", default="",
                    help="只跑指定视角，逗号分隔，支持唯一前缀（如 000,002,015）。"
                         "用于从多视角包（13v/8v）里挑小基线三元组，不复制图片。")
    args = ap.parse_args()

    device = torch.device(args.device)
    args.out.mkdir(parents=True, exist_ok=True)

    from sharp.models import PredictorParams, create_predictor
    from sharp.utils.gsplat import GSplatRenderer

    scene = args.scene
    c2w_all, k_all = load_camera_json(scene / "prior_camera.json")
    stems = sorted(c2w_all.keys())
    if args.views:
        want = [v.strip() for v in args.views.split(",") if v.strip()]
        picked = []
        for w_ in want:
            hit = [s for s in stems if s == w_]
            if not hit:
                hit = [s for s in stems if s.startswith(w_)]
            if len(hit) != 1:
                raise SystemExit(
                    f"--views 的 '{w_}' 匹配到 {len(hit)} 个视角 {hit}；请给更长的前缀或全名"
                )
            picked.append(hit[0])
        stems = picked
        print(f"   子集 {len(stems)} 个（来自 {args.views}）")
    if args.max_views:
        stems = stems[: args.max_views]
    print(f"📁 场景 {scene}")
    print(f"   视角 {len(stems)} 个: {', '.join(stems)}")

    # K 必须是同一台相机（同一设备定焦）；不一致就要单独处理
    k_uniq = {tuple(np.round(k_all[s].ravel(), 4)) for s in stems}
    print(f"   K 唯一值个数 = {len(k_uniq)} " + ("（同一相机）" if len(k_uniq) == 1 else "⚠️ 相机不一致"))
    K0 = k_all[stems[0]]

    t0 = time.time()
    predictor = create_predictor(PredictorParams())
    if args.ckpt:
        state = torch.load(args.ckpt, weights_only=True)
    else:
        import sharp.cli.predict as _p

        print(f"⬇️  加载权重 {_p.DEFAULT_MODEL_URL}")
        state = torch.hub.load_state_dict_from_url(_p.DEFAULT_MODEL_URL, progress=True)
    predictor.load_state_dict(state)
    predictor.eval().to(device)
    print(f"🏋️  权重 {time.time() - t0:.1f}s")

    # ---------------- 逐视角前馈 ----------------
    preds = {}
    for s in stems:
        img_path = scene / "images" / f"{s}.png"
        img = load_rgb(img_path)          # uint8 0..255，喂网络
        h, w = img.shape[:2]
        Kh = k_all[s]
        t0 = time.time()
        pred = predict(
            predictor,
            img,
            float(Kh[0, 0]), float(Kh[1, 1]),
            float(Kh[0, 2]), float(Kh[1, 2]),
            device,
        )
        g = pred["gaussians"]
        if s == stems[0]:
            print(f"🔮 尺寸 {w}x{h}  前向 {time.time() - t0:.1f}s  点数 {g.mean_vectors.shape[1]:,}")
        layers, hw, side = layer_pair(g)
        mu = g.mean_vectors[0].cpu().numpy()
        preds[s] = {
            "g": g,
            "layers": layers,
            "hw": hw,
            "side": side,
            "mu": mu,
            "img": load_rgb_gt(img_path),   # GT 口径 0..1，只用于 Q4 算 PSNR
            "size": (h, w),
            "c2w": c2w_all[s],
            "K": Kh,
            "fy_audit": pred["fy_audit"],
        }

    # 渲染尺寸（半分辨率，与 04 同口径）
    h0, w0 = preds[stems[0]]["size"]
    rh = int(round(h0 * args.render_scale)) if args.render_scale != 1.0 else h0
    rw = int(round(w0 * args.render_scale)) if args.render_scale != 1.0 else w0
    ratio = rw / w0
    print(f"🖥️  渲染 {rw}x{rh} (scale={args.render_scale})")

    renderer = GSplatRenderer(color_space="linearRGB", background_color=args.bg)
    results: dict = {"scene": str(scene), "views": stems, "render_size": [rw, rh], "bg": args.bg}
    audit: dict = {}

    # ---------------- Q0：评测协议审计（v4 复核要求，先自证口径再谈结论） ----------------
    # 这一段不产出新结论，只把「这次评测的内参/掩码/变换链到底用了什么」落成台账，
    # 免得下次又出现「旧表不能沿用，但没人知道为什么」。
    print("\n=== Q0 评测协议审计 ===")
    ks = []
    for s in stems:
        Kh = preds[s]["K"]
        ks.append(
            {
                "view": s,
                "fx": float(Kh[0, 0]), "fy": float(Kh[1, 1]),
                "cx": float(Kh[0, 2]), "cy": float(Kh[1, 2]),
                "aspect_fy_over_fx": float(Kh[1, 1] / max(Kh[0, 0], 1e-9)),
            }
        )
    fy_over_fx = [k["aspect_fy_over_fx"] for k in ks]
    audit["intrinsics"] = {
        "per_view": ks,
        "fy_over_fx_min": float(min(fy_over_fx)),
        "fy_over_fx_max": float(max(fy_over_fx)),
        "source": "prior_camera.json（SplatAnything 从这三图估计，不是标定 GT）",
        "disparity_factor_uses": "fx（网络只有单焦距输入槽，与官方一致；反投影 K 用 fx,fy）",
    }
    print(
        f"   内参 fx={ks[0]['fx']:.2f} fy={ks[0]['fy']:.2f} cx={ks[0]['cx']:.2f} cy={ks[0]['cy']:.2f}"
        f"  fy/fx={ks[0]['aspect_fy_over_fx']:.5f}"
    )
    print(f"   来源：prior_camera.json（估计值，非标定 GT）")

    # fy 分离的实测代价：反面投影一次，量点位差。0.24% 的焦距差到底等不等于"可忽略"，
    # 用数字说话，不靠感觉。
    fy_audit = {s: preds[s]["fy_audit"] for s in stems}
    audit["fy_vs_fx"] = fy_audit
    s0 = stems[0]
    a0 = fy_audit[s0]
    print(
        f"   fy/fx 分离代价（{s0}）：|Δy|中位={a0['median_abs_dy_m']:.5f} m  "
        f"|Δxyz|中位={a0['median_abs_dxyz_norm_m']:.5f} m "
        f"p90={a0['p90_abs_dxyz_norm_m']:.5f} m  相对={a0['median_rel_change'] * 100:.4f}%"
    )
    results["_q0_protocol_audit"] = audit

    # ---------------- Q1：尺度自洽 ----------------
    # 三张单图各自独立前馈，米制量级是否一致？不一致则任何跨视角残差都无意义。
    print("\n=== Q1 尺度自洽（各视角 z 分布）===")
    q1 = {}
    for s in stems:
        mu = preds[s]["mu"]
        z = mu[:, 2]
        r = {
            "z_median_m": float(np.median(z)),
            "z_p10_m": float(np.percentile(z, 10)),
            "z_p90_m": float(np.percentile(z, 90)),
        }
        q1[s] = r
        print(f"   {s}: med={r['z_median_m']:.4f} p10={r['z_p10_m']:.4f} p90={r['z_p90_m']:.4f} m")
    zmeds = [q1[s]["z_median_m"] for s in stems]
    results["_q1_scale"] = {
        "per_view": q1,
        "z_median_spread": float(max(zmeds) / max(min(zmeds), 1e-9)),
    }
    print(f"   → z 中位极差比 = {max(zmeds)/max(min(zmeds),1e-9):.3f}（越接近 1 越自洽）")

    # ---------------- Q3：层间 3D 位移的跨视角放大（核心） ----------------
    # 这是 S1a 的主问题：source 视角下两层错位 0.75px，换到 target 视角会放大成多少？
    print("\n=== Q3 层间错位的跨视角放大（核心）===")
    q3 = {}
    for s in stems:
        d = preds[s]
        l0, l1 = d["layers"]
        m0 = l0.mean_vectors[0].cpu().numpy()
        m1 = l1.mean_vectors[0].cpu().numpy()
        Kh = d["K"]
        # source 视角（相机系自身，w2c = I）
        uv0, z0, v0 = project_points(m0, np.eye(4), Kh)
        uv1, z1, v1 = project_points(m1, np.eye(4), Kh)
        dpx_src = np.linalg.norm(uv1 - uv0, axis=1) * ratio
        ok = v0 & v1 & np.isfinite(dpx_src)
        src_med = float(np.median(dpx_src[ok]))

        per_target = {}
        for t in stems:
            if t == s:
                continue
            # A 相机系 -> T 相机系
            w2c = np.linalg.inv(preds[t]["c2w"]) @ d["c2w"]
            uvt0, zt0, vt0 = project_points(m0, w2c, preds[t]["K"])
            uvt1, zt1, vt1 = project_points(m1, w2c, preds[t]["K"])
            dpx_t = np.linalg.norm(uvt1 - uvt0, axis=1) * ratio
            okt = vt0 & vt1 & np.isfinite(dpx_t)
            med = float(np.median(dpx_t[okt])) if okt.any() else float("nan")
            per_target[t] = {
                "dpx_median_render": med,
                "dpx_p90_render": float(np.percentile(dpx_t[okt], 90)) if okt.any() else float("nan"),
                "amplification_vs_source": float(med / max(src_med, 1e-9)),
                "n_projected": int(okt.sum()),
                "frac_projected": float(okt.mean()),
            }
        q3[s] = {
            "dpx_median_source_render": src_med,
            "per_target": per_target,
        }
        print(f"   {s}  source 错位 {src_med:.3f} px")
        for t, r in per_target.items():
            print(
                f"      -> {t}: {r['dpx_median_render']:.3f} px "
                f"(x{r['amplification_vs_source']:.2f})  p90={r['dpx_p90_render']:.2f} px  "
                f"可见 {r['frac_projected']*100:.1f}%"
            )
    results["_q3_layer_offset_cross_view"] = q3

    # ---------------- Q2：深度自洽（跨视角深度图残差，渲染口径） ----------------
    # ⚠️ 首版用「点投影 + 整像素深度缓冲」比深度，被遮挡严重污染：92° 基线时
    # A 的点投到 T，落点常常是 T 的另一张表面（前臂遮住躯干等），dz 差异大部分
    # 来自遮挡关系不同，而不是几何不准。于是 <10mm 只占 2.5%，这个数不可用。
    # 改用渲染口径：两边都交给 gsplat 做完整的 z 排序 + alpha 合成，
    # 得到的是各自「T 视角下的最前表面深度」，遮挡对两边一视同仁。
    print("\n=== Q2 深度自洽（A 渲到 T 视角的深度图 vs T 自己的深度图）===")
    print("   （渲染口径：两边都是最前表面，z 排序由 gsplat 负责，避免遮挡污染）")
    q2 = {}
    q2_depth_maps: dict = {}
    for s in stems:
        d = preds[s]
        for t in stems:
            if t == s:
                continue
            w2c_np = np.linalg.inv(preds[t]["c2w"]) @ d["c2w"]
            w2c = torch.from_numpy(w2c_np.astype(np.float32)).to(device)
            Kt4 = np.eye(4, dtype=np.float64)
            Kt4[:3, :3] = preds[t]["K"]
            if ratio != 1.0:
                Kt4[0] *= ratio
                Kt4[1] *= ratio
            Kt4_t = torch.from_numpy(Kt4.astype(np.float32)).to(device)

            # A 的点从 T 视角渲染。点已被 apply_w2c_to_gaussians 搬到 T 相机系，
            # 所以这里必须用单位外参；T 自己那份本来就在 T 相机系，同样用单位外参
            # ——两边口径一致，dz 才有可比性（2026-11-17 修正）。
            gA = apply_w2c_to_gaussians(d["g"], w2c_np)
            _, aA, dpA = render(renderer, gA, torch.eye(4, device=device), Kt4_t, rh, rw)
            # T 自己的点从 T 视角渲染（恒等外参）
            _, aT, dpT = render(
                renderer, preds[t]["g"],
                torch.eye(4, device=device), Kt4_t, rh, rw,
            )
            # 共可见：两边都有实体覆盖。alpha 阈值 0.5 与 Q4 一致。
            both = (aA > 0.5) & (aT > 0.5) & (dpA > 0) & (dpT > 0)
            dz = (dpA - dpT)[both]
            q2[f"{s}->{t}"] = {
                "dz_median_m": float(np.median(dz)) if dz.size else None,
                "dz_p10_m": float(np.percentile(dz, 10)) if dz.size else None,
                "dz_p90_m": float(np.percentile(dz, 90)) if dz.size else None,
                "abs_dz_median_m": float(np.median(np.abs(dz))) if dz.size else None,
                "frac_abs_lt_10mm": float(np.mean(np.abs(dz) < 0.010)) if dz.size else None,
                "frac_abs_lt_50mm": float(np.mean(np.abs(dz) < 0.050)) if dz.size else None,
                "frac_cov_both": float(both.mean()),
                "n_overlap_px": int(both.sum()),
            }
            q2_depth_maps[f"{s}->{t}"] = (dpA * both, dpT * both)
            r = q2[f"{s}->{t}"]
            if r["dz_median_m"] is not None:
                print(
                    f"   {s} -> {t}: dz med={r['dz_median_m']:+.4f} m  "
                    f"|dz| med={r['abs_dz_median_m']:.4f} m  "
                    f"<10mm {r['frac_abs_lt_10mm']*100:5.1f}%  "
                    f"<50mm {r['frac_abs_lt_50mm']*100:5.1f}%  "
                    f"共可见 {r['frac_cov_both']*100:.1f}% ({r['n_overlap_px']:,} px)"
                )
                save_gray(dpA * both, args.out / f"q2_{s}_to_{t}_depthA.png", 0.0, 4.0)
                save_gray(dpT * both, args.out / f"q2_{s}_to_{t}_depthT.png", 0.0, 4.0)
                dzmap = np.zeros_like(dpA)
                dzmap[both] = np.clip(np.abs(dpA - dpT)[both] * 1000.0, 0.0, 500.0)
                save_gray(dzmap, args.out / f"q2_{s}_to_{t}_absdz_mm.png", 0.0, 500.0)
    results["_q2_depth_consistency"] = q2

    # ---------------- Q4：跨视角渲染（A 的点从 T 视角渲染） ----------------
    # 这是「会不会变重影」的直接证据：不只看位移像素数，而是真的渲出来比 PSNR/alpha。
    print("\n=== Q4 跨视角渲染：A 的点从 T 视角渲染 vs T 的 GT ===")
    print("   （只用 A 的单视角点云，不融合；看第二层在别的视角是帮忙还是捣乱）")
    q4 = {}
    for s in stems:
        d = preds[s]
        g_full = d["g"]
        l0, l1 = d["layers"]
        for t in stems:
            if t == s:
                continue
            w2c = torch.from_numpy(
                (np.linalg.inv(preds[t]["c2w"]) @ d["c2w"]).astype(np.float32)
            ).to(device)
            Kt = preds[t]["K"].copy()
            Kt4 = np.eye(4, dtype=np.float64)
            Kt4[:3, :3] = Kt
            if ratio != 1.0:
                Kt4[0] *= ratio
                Kt4[1] *= ratio
            Kt4_t = torch.from_numpy(Kt4.astype(np.float32)).to(device)

            gt_t = preds[t]["img"]
            if ratio != 1.0:
                from PIL import Image

                gt_t = np.asarray(
                    Image.fromarray((gt_t * 255).astype(np.uint8)).resize((rw, rh), Image.LANCZOS),
                    dtype=np.float32,
                ) / 255.0

            row = {}
            for nm, gg in (("full", g_full), ("layer0", l0), ("layer1", l1)):
                # 点已由 apply_w2c_to_gaussians 搬进 T 相机系 → 渲染必须用单位外参。
                # 传 w2c 会二次施加 R,t（2026-11-17 修正，压低 3.45~3.97 dB）。
                gg_t = apply_w2c_to_gaussians(gg, (np.linalg.inv(preds[t]["c2w"]) @ d["c2w"]))
                c, a, dp = render(renderer, gg_t, torch.eye(4, device=device), Kt4_t, rh, rw)
                pv = psnr(c, gt_t)
                # 只看 alpha 覆盖到的区域，避免背景拉高 PSNR
                m = a > 0.5
                pv_cov = psnr(c, gt_t, m) if m.any() else float("nan")
                row[nm] = {
                    "psnr": pv,
                    "psnr_covered": pv_cov,
                    "alpha_mean": float(a.mean()),
                    "coverage_frac": float(m.mean()),
                    "depth_median": float(np.median(dp[m])) if m.any() else None,
                }
                if nm == "full":
                    save_png(c, args.out / f"q4_{s}_to_{t}_full.png")
                    save_gray(a, args.out / f"q4_{s}_to_{t}_alpha.png", 0.0, 1.0)
                    # 存无损 float32：06 要在"公共掩码"上重算三方的 PSNR，
                    # 若只能读 8bit PNG，重算值会与这里差零点几 dB（量化 + 反量化）。
                    np.save(args.out / f"q4_{s}_to_{t}_alpha.npy", a.astype(np.float32))
                    np.save(args.out / f"q4_{s}_to_{t}_full.npy", c.astype(np.float32))
            row["gain_layer1_over_layer0_db"] = row["full"]["psnr"] - row["layer0"]["psnr"]
            q4[f"{s}->{t}"] = row
            print(
                f"   {s} -> {t}:  full={row['full']['psnr']:6.2f}  "
                f"l0={row['layer0']['psnr']:6.2f}  l1={row['layer1']['psnr']:6.2f}  "
                f"full-l0={row['gain_layer1_over_layer0_db']:+5.2f} dB  "
                f"alpha={row['full']['alpha_mean']:.3f}"
            )

            # 守门人：每对都验一次 (a) 相机系+eye 与 (b) 世界系+w2c 是否等价。
            # 不等价就别往下看了，说明变换又被重复施加了。
            w2c_np = np.linalg.inv(preds[t]["c2w"]) @ d["c2w"]
            chk = check_render_equivalence(
                renderer, g_full, w2c_np, Kt4_t, rh, rw, device, tag=f"{s}->{t}"
            )
            audit.setdefault("render_equivalence", []).append(chk)
    results["_q4_cross_view_render"] = q4

    # 回归检查汇总：任何一对不等价 ⇒ 整批作废，不许往下写结论。
    # `inconclusive`（与故障未拉开）单独计，不算通过也不算失败——但要在汇总里点名，
    # 免得把「检查本身无效」静默当成绿。
    eqs = audit.get("render_equivalence", [])
    if eqs:
        bad = [e for e in eqs if not e["equivalent"] and not e.get("inconclusive")]
        unc = [e for e in eqs if e.get("inconclusive")]
        audit["render_equivalence_all_pass"] = not bad
        audit["render_equivalence_inconclusive"] = [e["pair"] for e in unc]
        results["_q0_protocol_audit"] = audit
        print(
            f"\n=== 回归检查（渲染等价性）===\n   "
            f"{len(eqs) - len(bad) - len(unc)}/{len(eqs)} 对等价"
            + (f"  ⚠️ {len(unc)} 对检查无效：{[e['pair'] for e in unc]}" if unc else "")
            + ("" if not bad else f"  ❌ 不等价：{[e['pair'] for e in bad]}")
        )
        if bad:
            print("   ❌ 变换链又被重复施加了，本批 PSNR 全部不可用。")
            return 2

    print(f"\n✅ 主分析完成，产物目录 {args.out}")
    (args.out / "cross_view.json").write_text(
        json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8", newline="\n"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

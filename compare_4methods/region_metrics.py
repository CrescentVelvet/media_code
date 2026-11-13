#!/usr/bin/env python3
"""region_metrics.py — 不看 PSNR 也能说清的画质指标（模糊 / 重影 / 纹理 / 错位）。

为什么需要：PSNR 对「整体偏糊」「局部重影」「纹理被抹平」几乎不敏感——把 GT 高斯
模糊 σ=0.6 后 PSNR 仍有三四十 dB，但人眼已经明显觉得糊；反过来一个几何错位 1~2 px
的模型 PSNR 掉 10 dB 以上，观感却可能只是"有点虚"。所以区域指标要分开测：

  指标              回答什么                       糊了会怎样
  ----------------  -----------------------------  ----------------------------
  sharp_ratio       细节还在不在（Laplacian 方差比） 明显 < 1
  hf_ratio          高频占自身能量的比例            下降（纹理被抹平）
  contrast_ratio    局部对比度是否被压              下降
  edge_f1@tol       结构边缘对得上吗                糊 → recall 掉；位移 → 也掉
  ghost_rate        有没有 GT 里不存在的多余边缘    重影/噪点 → 升高
  flow_med_px       区域内整体位移多少像素          错位 → 升高（糊也可能升）
  flow_div          局部尺度膨胀/收缩                "鼻子变宽" → 正值

所有指标都在 mask 内统计（mask 先腐蚀，避开边界混色），且 mask 由 GT 图分割得到，
渲染图用同一 mask——渲染与 GT 已按 GT 相机对齐，这是可比的。

依赖只有 numpy + cv2（skimage 可选，用于 ssim）。**不用 LPIPS/DISTS**：要下载权重，
且它们对"重影"这种结构性伪影的响应不如 edge/flow 直接。
"""
import numpy as np
import cv2

try:
    from skimage.metrics import structural_similarity as _ssim
except Exception:                                     # pragma: no cover
    _ssim = None


# ── 工具 ────────────────────────────────────────────────────────────────────
def _gray(rgb):
    if rgb.ndim == 2:
        return rgb.astype(np.uint8)
    return cv2.cvtColor(rgb.astype(np.uint8), cv2.COLOR_RGB2GRAY)


def _erode(mask, px=2):
    if px <= 0:
        return mask
    k = np.ones((2 * px + 1, 2 * px + 1), np.uint8)
    return cv2.erode(mask.astype(np.uint8), k)


def prep_mask(mask, shape=None, erode_px=2, min_px=64):
    """二值化 + （可选）缩放到 shape + 腐蚀。返回 uint8 0/1；太小返回 None。"""
    if mask is None:
        return None
    m = np.asarray(mask)
    if m.ndim == 3:
        m = m[..., 0]
    if shape is not None and m.shape[:2] != tuple(shape[:2]):
        m = cv2.resize(m.astype(np.uint8), (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST)
    m = (m > 127).astype(np.uint8) if m.max() > 1 else (m > 0).astype(np.uint8)
    m = _erode(m, erode_px)
    return m if int(m.sum()) >= min_px else None


def _lap_var(gray, m):
    lap = cv2.Laplacian(gray.astype(np.float32), cv2.CV_32F)
    return float(lap[m > 0].var())


def _edges(gray, lo=60, hi=160):
    return cv2.Canny(gray, lo, hi, L2gradient=True)


def _dist_to(edge01):
    """到最近边缘像素的欧氏距离图（边缘本身为 0）。"""
    if edge01.max() == 0:
        return np.full(edge01.shape, 1e9, np.float32)
    return cv2.distanceTransform(1 - edge01, cv2.DIST_L2, 5)


# ── 单项指标 ────────────────────────────────────────────────────────────────
def psnr(gt, rd, m=None):
    g = gt.astype(np.float64)
    r = rd.astype(np.float64)
    if m is not None:
        g, r = g[m > 0], r[m > 0]
    mse = float(np.mean((g - r) ** 2))
    return 99.0 if mse == 0 else 10.0 * np.log10(255.0 ** 2 / mse)


def ssim(gt, rd, m=None):
    if _ssim is None:
        return float("nan")
    full = _ssim(gt, rd, channel_axis=2, data_range=255, full=True)[1]
    return float(full.mean()) if m is None else float(full[m > 0].mean())


def sharp_ratio(gt, rd, m):
    """渲染 / GT 的 Laplacian 方差比。<1 糊，>1 过锐/噪。"""
    a = _lap_var(_gray(gt), m)
    b = _lap_var(_gray(rd), m)
    return float(b / a) if a > 1e-9 else float("nan")


def hf_ratio(gt, rd, m):
    """高频能量占自身总能量的比例（渲染 / GT）。纹理被抹平则下降。"""
    def hf(im):
        g = _gray(im).astype(np.float32)
        lap = cv2.Laplacian(g, cv2.CV_32F)
        tot = float(np.sqrt(np.mean((g[m > 0] - g[m > 0].mean()) ** 2))) + 1e-9
        return float(np.sqrt(np.mean(lap[m > 0] ** 2)) / tot)
    a = hf(gt)
    return float(hf(rd) / a) if a > 1e-9 else float("nan")


def contrast_ratio(gt, rd, m):
    a = float(_gray(gt)[m > 0].std()) + 1e-9
    return float(_gray(rd)[m > 0].std() / a)


def edge_f1(gt, rd, m, tol=2.0, lo=60, hi=160):
    """遮罩内结构边缘的 chamfer F1（tol px 容差内算命中）。

    precision = 渲染边缘里离 GT 边缘 ≤ tol 的比例（凭空多出来的边缘扣分）
    recall    = GT 边缘里离渲染边缘 ≤ tol 的比例（丢了细节扣分）
    ⚠️ 2026-11-13 修：初版把 precision 写成「到渲染边缘自身」的距离，恒为 1。
    """
    eg = _edges(_gray(gt), lo, hi) * (m > 0).astype(np.uint8)
    er = _edges(_gray(rd), lo, hi) * (m > 0).astype(np.uint8)
    if eg.sum() < 20 or er.sum() < 20:
        return float("nan"), float("nan"), float("nan")
    dg = _dist_to((eg > 0).astype(np.uint8))     # 每个像素到最近 GT 边缘
    dr = _dist_to((er > 0).astype(np.uint8))     # 每个像素到最近渲染边缘
    prec = float((dg[er > 0] <= tol).mean())     # 渲染边缘 → GT 边缘
    rec = float((dr[eg > 0] <= tol).mean())      # GT 边缘 → 渲染边缘
    f1 = 0.0 if prec + rec == 0 else 2 * prec * rec / (prec + rec)
    return f1, prec, rec


def ghost_rate(gt, rd, m, tol=4.0, lo=60, hi=160):
    """渲染边缘中距 GT 边缘 > tol px 的比例 —— 重影/凭空长出的纹理。"""
    eg = _edges(_gray(gt), lo, hi) * (m > 0).astype(np.uint8)
    er = _edges(_gray(rd), lo, hi) * (m > 0).astype(np.uint8)
    if er.sum() < 20:
        return float("nan")
    if eg.sum() < 20:
        return 1.0
    dg = _dist_to((eg > 0).astype(np.uint8))
    return float((dg[er > 0] > tol).mean())


def flow_stats(gt, rd, m, max_side=512):
    """Farneback 光流（GT→渲染）：区域内中位位移与散度。

    flow_med_px：整体对齐残差（px），错位/重影都会抬高。
    flow_div   ：散度均值，>0 表示渲染在局部被"撑大"（如鼻子变宽），<0 收缩。
    ⚠️ 辅助判据：纹理极弱时 Farneback 不可靠，读数要配 edge_f1 一起看。
    """
    g = _gray(gt)
    r = _gray(rd)
    h, w = g.shape
    s = min(1.0, max_side / float(max(h, w)))
    if s < 1.0:
        g = cv2.resize(g, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
        r = cv2.resize(r, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
        m = cv2.resize(m, (g.shape[1], g.shape[0]), interpolation=cv2.INTER_NEAREST)
    fl = cv2.calcOpticalFlowFarneback(g, r, None, 0.5, 4, 21, 3, 5, 1.2, 0)
    mag = np.linalg.norm(fl, axis=2) / max(s, 1e-6)     # 换回原像素
    fx, fy = fl[..., 0], fl[..., 1]
    div = (np.gradient(fx, axis=1) + np.gradient(fy, axis=0)) / max(s, 1e-6)
    sel = m > 0
    return float(np.median(mag[sel])), float(div[sel].mean())


# ── 汇总 ────────────────────────────────────────────────────────────────────
def all_metrics(gt, rd, mask, erode_px=2, tol_edge=2.0, tol_ghost=4.0,
                with_flow=True, with_ssim=True):
    """对一块区域算全套指标。mask 为 None 或过小时返回 None。"""
    m = prep_mask(mask, shape=gt.shape, erode_px=erode_px)
    if m is None:
        return None
    f1, prec, rec = edge_f1(gt, rd, m, tol=tol_edge)
    out = dict(
        px=int(m.sum()),
        psnr=psnr(gt, rd, m),
        sharp_ratio=sharp_ratio(gt, rd, m),
        hf_ratio=hf_ratio(gt, rd, m),
        contrast_ratio=contrast_ratio(gt, rd, m),
        edge_f1=f1, edge_prec=prec, edge_rec=rec,
        ghost_rate=ghost_rate(gt, rd, m, tol=tol_ghost),
    )
    if with_ssim:
        out["ssim"] = ssim(gt, rd, m)
    if with_flow:
        out["flow_med_px"], out["flow_div"] = flow_stats(gt, rd, m)
    return out


def summarize(rows):
    """若干帧的指标 dict 列表 → 各指标均值（跳过 nan）。"""
    if not rows:
        return None
    keys = set().union(*[set(r) for r in rows])
    out = {}
    for k in keys:
        vals = [r[k] for r in rows if k in r and r[k] is not None
                and not (isinstance(r[k], float) and np.isnan(r[k]))]
        out[k] = float(np.mean(vals)) if vals else float("nan")
    out["n_frames"] = len(rows)
    return out

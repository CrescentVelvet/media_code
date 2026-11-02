#!/usr/bin/env python3
"""糊 / 重影的诊断指标（CPU，只依赖 numpy + PIL）。

与 `render_check.py` 的分工：那个出 PSNR/SSIM（质量分），这个回答三个工程问题：
  1. **糊了多少**——渲染等价于把 GT 模糊几像素（`equivalent_blur_sigma`）；
  2. **糊在哪**——分块高频保留率（`tile_retention`）；
  3. **边缘有没有分裂**——强边缘处渲染梯度剖面的峰数与过渡宽度（`edge_stats`）。
     「双峰」是重影的形态学证据：真鬼影会在同一处出现两条平行梯度峰；
     单纯发散只会让一条峰变宽。

被 09_diag_ghost.py（诊断已有渲染）和 10_sweep_merge.py（扫合并参数）共用，
所以指标口径只能在这里改一次。
"""
import numpy as np
from PIL import Image, ImageOps


def load_gt_processed(path, w, h):
    """复刻 pipeline.prepare_images 的 bicubic 中心裁剪缩放（与 render_check 同源）。

    必须与模型预处理逐位一致，否则 GT 与渲染之间差一个裁剪偏移，指标会被系统性拉低。
    改动时同步 render_check.py:model_preprocess。
    """
    with Image.open(path) as im:
        im = ImageOps.exif_transpose(im).convert("RGB")
        scale = max(w / im.width, h / im.height)
        left = max(0.0, (im.width - w / scale) / 2)
        top = max(0.0, (im.height - h / scale) / 2)
        box = (left, top, min(float(im.width), left + w / scale),
               min(float(im.height), top + h / scale))
        resized = im.resize((w, h), Image.Resampling.BICUBIC, box=box)
    return np.asarray(resized, dtype=np.float32) / 255.0


def to_gray(a):
    return a @ np.array([0.299, 0.587, 0.114], np.float32)


def gauss_blur(img, sigma):
    """可分离高斯（reflect 边界）。sigma<=0 原样返回。"""
    if sigma <= 0:
        return img
    r = int(np.ceil(3 * sigma))
    x = np.arange(-r, r + 1, dtype=np.float32)
    k = np.exp(-(x ** 2) / (2 * sigma ** 2))
    k /= k.sum()
    out = img.astype(np.float32)
    for axis in (0, 1):
        pad = [(0, 0)] * out.ndim
        pad[axis] = (r, r)
        p = np.pad(out, pad, mode="reflect")
        acc = np.zeros_like(out)
        for i, ki in enumerate(k):
            sl = [slice(None)] * out.ndim
            sl[axis] = slice(i, i + out.shape[axis])
            acc += ki * p[tuple(sl)]
        out = acc
    return out


def psnr(a, b):
    mse = float(np.mean((a.astype(np.float32) - b.astype(np.float32)) ** 2))
    return float("inf") if mse <= 1e-12 else 10.0 * np.log10(1.0 / mse)


def equivalent_blur_sigma(gt, rd, sigmas=(0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 6.0)):
    """找使「GT 模糊 σ 后」与渲染最像的 σ —— 把糊量化成一个像素数。

    返回 (sigma_eff, psnr_at_best, curve)。curve 为 [(sigma, psnr), ...]，
    用来判断峰值是否落在网格端点（说明真实模糊超出搜索范围）。
    """
    curve = []
    for s in sigmas:
        curve.append((float(s), psnr(gauss_blur(gt, s), rd)))
    best = max(curve, key=lambda t: t[1])
    return best[0], best[1], curve


def _lap(a):
    """3x3 拉普拉斯绝对值（reflect 边界），作为高频能量的代理。"""
    p = np.pad(a, 1, mode="reflect")
    lap = (4 * p[1:-1, 1:-1] - p[:-2, 1:-1] - p[2:, 1:-1] - p[1:-1, :-2] - p[1:-1, 2:])
    return np.abs(lap)


def tile_retention(gt, rd, grid=4):
    """分块高频保留率 = mean|lap(render)| / mean|lap(GT)|。

    <1 说明该块被过度平滑；>1 说明引入了 GT 没有的高频（噪点/振铃）。
    背景块也参与，所以整幅均值会被大平面拉高，判断时要看逐块矩阵。
    """
    hf_g, hf_r = _lap(to_gray(gt)), _lap(to_gray(rd))
    H, W = hf_g.shape
    rows = np.linspace(0, H, grid + 1).astype(int)
    cols = np.linspace(0, W, grid + 1).astype(int)
    mat = np.zeros((grid, grid), np.float32)
    for i in range(grid):
        for j in range(grid):
            g = hf_g[rows[i]:rows[i + 1], cols[j]:cols[j + 1]].mean()
            r = hf_r[rows[i]:rows[i + 1], cols[j]:cols[j + 1]].mean()
            mat[i, j] = r / g if g > 1e-6 else np.nan
    return mat


def edge_stats(gt, rd, pct=95, half=4, min_gap=2, prom_frac=0.45, valley_frac=0.75,
               clean_frac=0.4):
    """强水平边缘处的梯度剖面统计（子像素，避免 FWHM 取整导致的伪饱和）。

    只在 GT 本身是**单峰干净边缘**的位置取样（窗口内距主峰 ≥2px 处必须 < clean_frac×主峰），
    然后读渲染同位置剖面：
      - spread_ratio     : 渲染梯度分布标准差 / GT 的（>1 是发散 —— 糊）
      - double_peak_ratio: 剖面出现「双峰 + 中间有谷」的比例（>0 是重影）

    两条口径都是为了压假阳性，被第一版实测逼出来的：
      * 只数局部极大 → 噪声抖动全算峰，真数据误报 73%；
      * 不筛 GT 干净度 → 窗口里相邻第二条边缘自成"双峰"，底噪 57%。
    现在 GT 与自身比的双峰率应 ≈0，可用作指标的零假设检验（见 09 --selftest）。

    返回 (spread_ratio_median, double_peak_ratio, n_edges, examples)。
    """
    def prof_stats(p):
        x = np.arange(len(p), dtype=np.float32)
        s = p.sum()
        if s <= 1e-9:
            return None, 0.0
        c = float((x * p).sum() / s)
        spread = float(np.sqrt(max((p * (x - c) ** 2).sum() / s, 0.0)))
        return spread, c

    gx_g = np.abs(np.diff(to_gray(gt), axis=1))
    gx_r = np.abs(np.diff(to_gray(rd), axis=1))
    thr = np.percentile(gx_g, pct)
    H, W = gx_g.shape
    loc = gx_g > thr
    loc[:, 1:-1] &= (gx_g[:, 1:-1] >= gx_g[:, :-2]) & (gx_g[:, 1:-1] >= gx_g[:, 2:])
    ys, xs = np.nonzero(loc)
    ratios, doubles, examples = [], 0, []
    for y, x in zip(ys, xs):
        lo, hi = max(0, x - half), min(W, x + half + 1)
        pg, pr = gx_g[y, lo:hi], gx_r[y, lo:hi]
        if len(pg) < 5 or pg.max() <= 1e-6 or pr.max() <= 1e-6:
            continue
        # GT 必须是单峰干净边缘，否则这个位置测不出"渲染额外造出的第二条"
        i1g = int(np.argmax(pg))
        far = [pg[j] for j in range(len(pg)) if abs(j - i1g) >= 2]
        if far and max(far) > clean_frac * pg.max():
            continue
        sg, _ = prof_stats(pg)
        sr, _ = prof_stats(pr)
        if sg and sg > 1e-3 and sr is not None:
            ratios.append(sr / sg)
        idx = [i for i in range(1, len(pr) - 1)
               if pr[i] >= pr[i - 1] and pr[i] >= pr[i + 1]]
        if len(idx) >= 2:
            idx.sort(key=lambda i: -pr[i])
            i1 = idx[0]
            i2 = next((i for i in idx[1:] if abs(i - i1) >= min_gap), None)
            if i2 is not None:
                a, b = min(i1, i2), max(i1, i2)
                valley = float(pr[a:b + 1].min())
                if pr[i2] >= prom_frac * pr[i1] and valley <= valley_frac * pr[i2]:
                    doubles += 1
                    if len(examples) < 5:
                        examples.append(dict(y=int(y), x=int(x),
                                             peaks=[int(lo + a), int(lo + b)],
                                             gap=int(b - a), valley_ratio=round(valley / pr[i2], 3),
                                             rd=pr.round(3).tolist()))
    n = max(1, len(ratios))
    return (float(np.median(ratios)) if ratios else float("nan"),
            doubles / n, len(ratios), examples)


def synthetic_controls(gt, sigma=1.5, shift=3, alpha=0.35):
    """给指标做灵敏度自检用：两个已知答案的对照。

    blur : 模拟纯发散（应只有 spread↑，双峰≈0）
    ghost: 模拟真鬼影（0.65×GT + 0.35×右移 3px 的 GT，应出现双峰）
    """
    g = gauss_blur(gt, sigma)
    sh = np.roll(gt, shift, axis=1)
    sh[:, :shift] = gt[:, :shift]
    ghost = (1 - alpha) * gt + alpha * sh
    return {"blur": g, "ghost": ghost}



def retention_map_png(gt, rd, mat, path, grid=4):
    """GT | 渲染 | 分块保留率热图（红=过度平滑，绿=正常，蓝=引入伪高频）。"""
    H, W, _ = gt.shape
    heat = np.zeros_like(gt)
    rows = np.linspace(0, H, grid + 1).astype(int)
    cols = np.linspace(0, W, grid + 1).astype(int)
    for i in range(grid):
        for j in range(grid):
            v = np.nan_to_num(mat[i, j], nan=1.0)
            if v < 1.0:                      # 平滑：红
                c = np.array([1.0, 0.15, 0.15], np.float32) * (1 - v)
            else:                            # 伪高频：蓝
                c = np.array([0.15, 0.3, 1.0], np.float32) * min(v - 1.0, 1.0)
            blk = heat[rows[i]:rows[i + 1], cols[j]:cols[j + 1]]
            blk[:] = 0.45 * c + 0.55 * blk
    out = np.concatenate([gt, rd, np.clip(gt * 0.55 + heat, 0, 1)], axis=1)
    Image.fromarray((out * 255).round().astype(np.uint8)).save(path)

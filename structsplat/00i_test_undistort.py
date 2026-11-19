#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
structsplat/00i_test_undistort.py — 去畸变链路判据（不需要权重 / GPU，只需 transformers.json）

背景（2026-11-19 实测）：华为 Remy 包 camera_model=OPENCV，带
    k1=+0.0909  k2=-0.2374  p1=-0.00021  p2=+0.00088  k3=+0.1914
而 load_gt_cameras 早期只读 fx/fy/cx/cy，等于把畸变当 0。实测误差：
    原图 1440×1920 尺度  中位 7.06px  P95 10.68px  最大 18.29px
    缩到 518 宽后        中位 2.54px  P95  3.84px  最大  6.58px
这个量级足以让「渲染输入视角」的 PSNR 自检把畸变残差误读成相机估计不准。

判据（五条，缺一不可）：
  A. 真实 OPENCV 数据必须识别出畸变，且 newK 与 cv2 一致
  B. 全零畸变 / 无畸变字段（PINHOLE 包）必须返回 None —— 不能误触发
  C. 自洽性：渲染侧针孔投影 == 内容在去畸变图中的真实落点
       cv2.undistort 对输出像素的反向映射是 src = K0·distort(newK⁻¹·p_d)
       ⟹ 3D 点 Xc 落点：p_d = undistortPoints(K0·Xc_norm, P=newK)
       而 infer.py 渲染是纯针孔 p = newK·Xc_norm，两者必须相等
  D. 误差量级必须与上面记录的一致（证明修的是真问题，不是空转）
  E. 实图去畸变后尺寸不变、无黑边（roi 未裁掉的像素不能超过边框）

用法：
  python 00i_test_undistort.py [transforms.json]
  # 不给参数则用默认的佳佳抱枕包
"""

import importlib.util
import json
import os
import sys
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.environ.get("STRUCTSPLAT_REPO") or os.path.expanduser("~/repos/StructSplat")
sys.path.insert(0, REPO)

DEFAULT_TP = ("/mnt/d/dataset/手部微动样例/"
              "01e954de1b8a4cde9315c976f4dcb964-佳佳抱枕/transforms.json")

# 记录在案的误差量级（D 判据用；改数据后需同步更新）
EXPECT_IGNORE_MEDIAN_PX = 7.06
EXPECT_IGNORE_MAX_PX = 18.29

_ok = _fail = 0


def check(name, cond, detail=""):
    global _ok, _fail
    if cond:
        _ok += 1
        print(f"  ✅ {name}" + (f"  {detail}" if detail else ""))
    else:
        _fail += 1
        print(f"  ❌ {name}" + (f"  {detail}" if detail else ""))
    return cond


def load_infer():
    spec = importlib.util.spec_from_file_location("inf_ud", os.path.join(HERE, "infer.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def main():
    tp = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_TP
    if not os.path.exists(tp):
        print(f"❌ transforms.json 不存在: {tp}")
        print("   把样例包放到默认位置，或显式传路径: python 00i_test_undistort.py <path>")
        return 2

    import cv2
    inf = load_infer()
    meta = json.load(open(tp, encoding="utf-8"))
    frames = meta.get("frames", [])
    if not frames:
        print(f"❌ {tp} 里没有 frames")
        return 2
    fr0 = frames[0]
    W0, H0 = int(fr0["w"]), int(fr0["h"])
    K0 = np.array([[fr0["fl_x"], 0.0, fr0["cx"]],
                   [0.0, fr0["fl_y"], fr0["cy"]],
                   [0.0, 0.0, 1.0]], dtype=np.float64)

    # ---------- A. 真实数据必须识别 ----------
    print(f"[A] 真实 OPENCV 包必须识别出去畸变   {os.path.basename(os.path.dirname(tp))}")
    und = inf.probe_distortion(tp)
    if not check("probe_distortion 返回非 None", und is not None):
        print("\n❌ A 失败：这个包可能没有畸变字段，后续判据无法进行")
        return 1
    K_a, dist, newK = und
    check("dist 按 cv2 顺序 k1,k2,p1,p2,k3",
          dist.shape == (5,),
          f"= {np.round(dist, 5).tolist()}")
    check("K 来自 JSON 原始内参", np.allclose(K_a, K0),
          f"fx={K_a[0,0]:.1f}")
    # newK 必须与 cv2 自己算的一致（防手写公式漂移）
    newK_ref, _roi = cv2.getOptimalNewCameraMatrix(K0, dist, (W0, H0), 0.0, (W0, H0))
    check("newK 与 cv2.getOptimalNewCameraMatrix 一致",
          np.allclose(newK, newK_ref, atol=1e-6),
          f"fx {newK[0,0]:.3f} (cv2 {newK_ref[0,0]:.3f}), cx {newK[0,2]:.3f}")
    # 内参可以变，但焦距变化不能太离谱（去畸变只是轻微缩放）
    ratio = newK[0, 0] / K0[0, 0]
    check("焦距缩放比在 [0.9, 1.1]（否则是畸变系数或顺序错）",
          0.9 < ratio < 1.1, f"{ratio:.5f}x")

    # ---------- B. 不该触发的情况 ----------
    print("\n[B] 无畸变数据必须返回 None（不能误触发）")
    tmps = []
    m_zero = json.loads(json.dumps(meta))
    for fr in m_zero["frames"]:
        for k in ("k1", "k2", "k3", "p1", "p2"):
            if k in fr:
                fr[k] = 0.0
    m_none = json.loads(json.dumps(meta))
    for fr in m_none["frames"]:
        for k in ("k1", "k2", "k3", "p1", "p2"):
            fr.pop(k, None)
    for tag, obj in (("全零畸变", m_zero), ("无畸变字段（PINHOLE 包）", m_none)):
        f = tempfile.NamedTemporaryFile("w", suffix=".json", delete=False,
                                        encoding="utf-8")
        json.dump(obj, f)
        f.close()
        tmps.append(f.name)
        got = inf.probe_distortion(f.name)
        check(f"{tag} → None", got is None, "" if got is None else f"返回了 {type(got)}")

    # ---------- C. 自洽性 ----------
    print("\n[C] 自洽性：渲染针孔投影 == 内容在去畸变图中的真实落点")
    rng = np.random.default_rng(0)
    N = 2000
    zc = rng.uniform(0.4, 6.0, N)
    xn = rng.uniform(-1.0, 1.0, N) * (W0 / 2) / K0[0, 0]
    yn = rng.uniform(-1.0, 1.0, N) * (H0 / 2) / K0[1, 1]
    Xc = np.stack([xn * zc, yn * zc, zc], 1)
    # 渲染侧（infer.py）：纯针孔 + newK
    p_render = np.stack([Xc[:, 0] / Xc[:, 2] * newK[0, 0] + newK[0, 2],
                         Xc[:, 1] / Xc[:, 2] * newK[1, 1] + newK[1, 2]], 1)
    # 去畸变图侧：真实带畸变落点 → 映射到新网格
    uv_true = cv2.projectPoints(Xc, np.zeros(3), np.zeros(3), K0, dist)[0][:, 0, :]
    p_und = cv2.undistortPoints(uv_true.reshape(-1, 1, 2), K0, dist, P=newK)[:, 0, :]
    err = np.linalg.norm(p_render - p_und, axis=1)
    check("|渲染 - 去畸变真实落点| < 1e-3 px", err.max() < 1e-3,
          f"中位 {np.median(err):.2e}  最大 {err.max():.2e}px")

    # ---------- D. 误差量级 ----------
    print("\n[D] 修之前的误差量级（忽略畸变）必须与记录一致")
    p_naive = np.stack([Xc[:, 0] / Xc[:, 2] * K0[0, 0] + K0[0, 2],
                        Xc[:, 1] / Xc[:, 2] * K0[1, 1] + K0[1, 2]], 1)
    e = np.linalg.norm(p_naive - uv_true, axis=1)
    med, mx = float(np.median(e)), float(e.max())
    print(f"     原图 {W0}×{H0}: 中位 {med:.2f}px  P95 {np.percentile(e,95):.2f}px  最大 {mx:.2f}px")
    print(f"     缩到 518 宽:  中位 {med*518/W0:.2f}px  P95 {np.percentile(e,95)*518/W0:.2f}px  "
          f"最大 {mx*518/W0:.2f}px")
    # 允许 ±15% 波动（随机点采样不同）
    check(f"中位误差 ≈ {EXPECT_IGNORE_MEDIAN_PX}px (±15%)",
          abs(med - EXPECT_IGNORE_MEDIAN_PX) / EXPECT_IGNORE_MEDIAN_PX < 0.15,
          f"实测 {med:.2f}px")
    check(f"最大误差 ≈ {EXPECT_IGNORE_MAX_PX}px (±15%)",
          abs(mx - EXPECT_IGNORE_MAX_PX) / EXPECT_IGNORE_MAX_PX < 0.15,
          f"实测 {mx:.2f}px")
    check("忽略畸变在 518 宽下仍 > 2px（确属必须修的缺陷）",
          med * 518 / W0 > 2.0, f"{med*518/W0:.2f}px")

    # ---------- E. 实图检查 ----------
    print("\n[E] 实图去畸变：尺寸不变、无黑边")
    img_dir = os.path.join(os.path.dirname(tp), "image")
    p0 = None
    if os.path.isdir(img_dir):
        cand = sorted(os.listdir(img_dir))
        cand = [c for c in cand if os.path.splitext(c)[1].lower() in inf.IMAGE_EXTS]
        if cand:
            p0 = os.path.join(img_dir, cand[0])
    if p0 is None:
        check("找到实图（跳过）", True, "无 image/ 目录，E 段跳过")
    else:
        from PIL import Image
        ext = os.path.splitext(p0)[1].lower()
        if ext in (".heic", ".heif"):
            import pillow_heif
            pillow_heif.register_heif_opener()
        im = np.asarray(Image.open(p0).convert("RGB"))
        check("实图尺寸与 JSON 一致", im.shape[:2] == (H0, W0),
              f"{im.shape[1]}×{im.shape[0]}")
        uim = cv2.undistort(im, K0, dist, None, newK)
        check("去畸变后尺寸不变", uim.shape == im.shape, str(uim.shape))
        b = 0
        for sl in (uim[:4], uim[-4:], uim[:, :4], uim[:, -4:]):
            b += int((sl.sum(axis=2) == 0).sum())
        check("边框黑像素 < 100（roi 覆盖良好）", b < 100, f"{b} 个")
        mad = float(np.abs(uim.astype(np.int16) - im.astype(np.int16)).mean())
        check("与原图有重采样差异（>0，证明真的变形了）", mad > 0.5,
              f"MAD {mad:.2f}/255")

    for t in tmps:
        os.unlink(t)

    print("\n" + "=" * 60)
    if _fail == 0:
        print(f"🎉 全部通过（{_ok} 项）")
        print("   结论：OPENCV 畸变已按 newK 正确接入，错位从 ~7px 降到 ~1e-5px")
        return 0
    print(f"❌ 失败 {_fail} 项 / 通过 {_ok} 项")
    return 1


if __name__ == "__main__":
    sys.exit(main())

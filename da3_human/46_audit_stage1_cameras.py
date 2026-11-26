#!/usr/bin/env python3
"""旧海思 stage1 相机的图像几何审计，不改输入或已有重建。

仅用图像自身 SIFT + USAC 筛选对应，再比较登记的 OpenGL c2w 解释。
输出各视角对的中位 Sampson 距离、双正深度比例和有效匹配数；
该检验约束相对几何，不证明绝对尺度、动态人脸几何或相机是真值。

Env vars:
  ROOT            海思数据根目录
  CASE_FILTER     可选单个案例目录名
  RESULTS_DIR     新结果目录（必须不存在）
  SIFT_FEATURES   每图最大 SIFT 特征数，默认 6000
  MIN_MATCHES     一对图进入统计的最少内点，默认 20
"""
import json
import os
from pathlib import Path

import cv2
import numpy as np

FLIP_CV = np.diag([1., -1., -1., 1.])


def dump_json(path, data):
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.write("\n")


def sampson(F, p, q):
    x = np.column_stack((p, np.ones(len(p))))
    y = np.column_stack((q, np.ones(len(q))))
    fx, fy = x @ F.T, y @ F
    return np.abs(np.sum(y * fx, axis=1)) / np.sqrt(
        np.sum(fx[:, :2] ** 2 + fy[:, :2] ** 2, axis=1).clip(1e-20))


def fundamental(ext, K, i, j):
    rel = ext[j] @ np.linalg.inv(ext[i])
    x, y, z = rel[:3, 3]
    skew = np.array([[0., -z, y], [z, 0., -x], [-y, x, 0.]])
    return np.linalg.inv(K[j]).T @ skew @ rel[:3, :3] @ np.linalg.inv(K[i])


def stats(values):
    a = np.asarray(values)
    if not len(a):
        return None
    return {"n": len(a), "median": float(np.median(a)),
            "p90": float(np.quantile(a, .9)), "p99": float(np.quantile(a, .99))}


def load_case(case):
    manifest_path = case / "splatanything/extraction.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    data = json.loads((case / manifest["camera_file"]).read_text(encoding="utf-8"))
    frames, views = manifest["frames"], data["views"]
    if not (len(frames) == len(views) == data["viewCount"] == 10):
        raise ValueError(f"{case.name}: 需要十个配对视角")
    if [r["sourceIndex"] for r in frames] != [v["sourceIndex"] for v in views]:
        raise ValueError(f"{case.name}: sourceIndex 不一致")
    c2w, intrinsics, files = [], [], []
    for frame, view in zip(frames, views):
        file = case / "splatanything/images" / Path(frame["image"]).name
        if not file.is_file():
            raise FileNotFoundError(file)
        files.append(file)
        transform = np.eye(4)
        transform[:3, :3] = np.asarray(view["rotation3x3"]).reshape(3, 3)
        transform[:3, 3] = view["position"]
        if (np.abs(transform[:3, :3] @ transform[:3, :3].T - np.eye(3)).max() > 1e-4
                or abs(np.linalg.det(transform[:3, :3]) - 1) > 1e-4):
            raise ValueError(f"{case.name}: 非 SO(3) 旋转")
        c2w.append(transform)
        intrinsics.append([[view["fx"], 0., view["cx"]],
                           [0., view["fy"], view["cy"]], [0., 0., 1.]])
    return files, np.asarray(c2w), np.asarray(intrinsics)


def main():
    root = Path(os.environ.get("ROOT", "/mnt/d/dataset/测试数据/海思端侧重建拍摄数据"))
    selected = os.environ.get("CASE_FILTER", "")
    cases = ([root / selected] if selected else sorted(root.glob("case*_nopose_recentered"),
             key=lambda p: int(p.name.split("_")[0].removeprefix("case"))))
    if not cases:
        raise FileNotFoundError(f"未发现旧 stage1 案例: {root}")
    out = Path(os.environ["RESULTS_DIR"])
    if out.exists() or out.is_symlink():
        raise FileExistsError(f"结果目录须不存在且非符号链接: {out}")
    out.mkdir(parents=True)
    cv2.setRNGSeed(42)
    sift = cv2.SIFT_create(nfeatures=int(os.environ.get("SIFT_FEATURES", "6000")))
    matcher = cv2.BFMatcher()
    min_matches = int(os.environ.get("MIN_MATCHES", "20"))
    report = {"root": str(root), "image_filter": "SIFT ratio=.7 + 图像独立 USAC_MAGSAC 1px",
              "selection_policy": "先全部案例、全部视角对；不按结果挑视角",
              "limits": "图像极线约束相对几何；真人非同步/动态区域可能违反静态场景假设，须结合背景复核。",
              "cases": []}
    for case in cases:
        files, c2w, K = load_case(case)
        negative = c2w.copy()
        negative[:, :3, 3] *= -1
        variants = {"stage1_cv_flip": np.linalg.inv(c2w @ FLIP_CV),
                    "no_axis_flip": np.linalg.inv(c2w),
                    "no_flip_negative_centers": np.linalg.inv(negative)}
        errors = {name: [] for name in (*variants, "image_fit_F")}
        front = {name: [0, 0] for name in variants}
        pairs = []
        features = [sift.detectAndCompute(cv2.imread(str(file), 0), None) for file in files]
        for i in range(len(files)):
            for j in range(i + 1, len(files)):
                ki, di = features[i]
                kj, dj = features[j]
                # 无纹理视角可能没有SIFT描述子；不能让空数组触发knnMatch的dtype断言。
                if di is None or dj is None or len(di) < 2 or len(dj) < 2:
                    pairs.append({"views": [i, j], "matches": 0,
                                  "status": "insufficient_descriptors"})
                    continue
                di, dj = np.asarray(di, dtype=np.float32), np.asarray(dj, dtype=np.float32)
                good = [a for a, b in matcher.knnMatch(di, dj, k=2) if a.distance < .7 * b.distance]
                if len(good) < min_matches:
                    pairs.append({"views": [i, j], "matches": 0, "status": "insufficient_raw_matches"})
                    continue
                p = np.asarray([ki[m.queryIdx].pt for m in good])
                q = np.asarray([kj[m.trainIdx].pt for m in good])
                F, mask = cv2.findFundamentalMat(p, q, cv2.USAC_MAGSAC, 1., .999, 10000)
                if F is None or mask is None or mask.sum() < min_matches:
                    pairs.append({"views": [i, j], "matches": 0, "status": "insufficient_image_F_inliers"})
                    continue
                p, q = p[mask.ravel() != 0], q[mask.ravel() != 0]
                row = {"views": [i, j], "matches": len(p), "status": "evaluated"}
                matrices = {name: fundamental(E, K, i, j) for name, E in variants.items()}
                matrices["image_fit_F"] = F
                for name, matrix in matrices.items():
                    error = sampson(matrix, p, q)
                    errors[name].extend(error.tolist())
                    row[name] = stats(error)
                    if name in variants:
                        E = variants[name]
                        xyz = cv2.triangulatePoints(K[i] @ E[i, :3], K[j] @ E[j, :3], p.T, q.T)
                        xyz = xyz[:3] / xyz[3:]
                        positive = ((E[i, 2, :3] @ xyz + E[i, 2, 3] > 0)
                                    & (E[j, 2, :3] @ xyz + E[j, 2, 3] > 0))
                        front[name][0] += int(positive.sum())
                        front[name][1] += len(p)
                pairs.append(row)
        info = {"case": case.name, "n_images": len(files),
                "epipolar_pixel_error": {name: stats(value) for name, value in errors.items()},
                "positive_depth_fraction": {name: a / b if b else None for name, (a, b) in front.items()},
                "n_evaluated_pairs": sum(p["status"] == "evaluated" for p in pairs), "pairs": pairs}
        report["cases"].append(info)
        dump_json(out / f"{case.name}.json", info)
        dump_json(out / "summary.json", report)
        print(json.dumps({"case": case.name, "pairs": info["n_evaluated_pairs"],
                          "errors": info["epipolar_pixel_error"],
                          "front": info["positive_depth_fraction"]}, ensure_ascii=False), flush=True)
    dump_json(out / "summary.json", report)
    print(f"🎉 完成 {len(cases)} 例；结果：{out}", flush=True)


if __name__ == "__main__":
    main()

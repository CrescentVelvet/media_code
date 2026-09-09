#!/usr/bin/env python3
"""从 DECA 的 landmark_embedding.npy 转出 smplx 需要的 flame_static_embedding.pkl。

smplx FLAME 构造函数读取的 pkl 需要 keys：
  - lmk_face_idx : (68,) int64 —— 68 个 landmark 所在三角面 id
  - lmk_b_coords : (68, 3) —— 重心坐标
DECA 的 landmark_embedding.npy 里 full_lmk_* 就是这份数据（来源同为 FLAME 官网
landmark embeddings 包），直接转换即可。
"""
import argparse
import pickle

import numpy as np
import torch  # DECA 的 npy 里存了 torch 张量


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="DECA landmark_embedding.npy")
    ap.add_argument("--dst", required=True, help="输出 flame_static_embedding.pkl")
    args = ap.parse_args()

    d = np.load(args.src, allow_pickle=True, encoding="latin1")
    d = d.item() if d.shape == () else d

    faces_idx = d["full_lmk_faces_idx"]
    b_coords = d["full_lmk_bary_coords"]
    # 统一转 numpy，去掉 batch 维 (1,68)->(68,)
    if hasattr(faces_idx, "numpy"):
        faces_idx = faces_idx.numpy()
    if hasattr(b_coords, "numpy"):
        b_coords = b_coords.numpy()
    faces_idx = np.asarray(faces_idx).reshape(-1).astype(np.int64)
    b_coords = np.asarray(b_coords).reshape(-1, 3).astype(np.float64)
    assert faces_idx.shape == (68,), f"lmk_face_idx 形状异常: {faces_idx.shape}"
    assert b_coords.shape == (68, 3), f"lmk_b_coords 形状异常: {b_coords.shape}"

    out = {
        "lmk_face_idx": faces_idx,
        "lmk_b_coords": b_coords,
    }
    with open(args.dst, "wb") as f:
        pickle.dump(out, f)
    print(f"✅ 写出 {args.dst}")
    print(f"   lmk_face_idx: {faces_idx.shape} (范围 {faces_idx.min()}-{faces_idx.max()})")
    print(f"   lmk_b_coords: {b_coords.shape} (每行和≈1: {b_coords.sum(1).mean():.4f})")


if __name__ == "__main__":
    main()

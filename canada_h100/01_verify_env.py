#!/usr/bin/env python
"""01_verify_env.py — 在计算节点上验证环境是否真能跑 3DGS 训练。

只验证「能不能跑」，不测性能。三关：
  1) torch 能认到 GPU（sm_90 / H100）
  2) gsplat JIT 编译 CUDA kernel 成功（这是最容易失败的一关）
  3) 真的跑一次光栅化 + 反传，数值有限

用法（必须写在 sbatch 里提交，登录节点无 GPU）：
    python 01_verify_env.py
"""
import os
import sys
import time

SEP = "=" * 60


def step(n, title):
    print(f"\n{SEP}\n[{n}] {title}\n{SEP}")


def main():
    import torch
    print("torch      :", torch.__version__)
    print("cuda build :", torch.version.cuda)
    print("cuda avail :", torch.cuda.is_available())

    step(1, "GPU 检查")
    if not torch.cuda.is_available():
        print("❌ 没有 GPU —— 是不是在登录节点跑的？必须 sbatch 提交。")
        return 1
    n = torch.cuda.device_count()
    print(f"GPU 数量: {n}")
    for i in range(n):
        p = torch.cuda.get_device_properties(i)
        print(f"  [{i}] {p.name} | {p.total_memory / 2**30:.1f} GiB | sm_{p.major}{p.minor} | {p.multi_processor_count} SM")
    cap = torch.cuda.get_device_capability(0)
    if cap != (9, 0):
        print(f"⚠️  预期 sm_90 (H100)，实际 {cap}")
    print("✅ GPU OK")

    step(2, "gsplat JIT 编译（首次约 30-90 秒）")
    t0 = time.time()
    try:
        from gsplat import rasterization
        print("import gsplat.rasterization OK")
    except Exception as e:
        print("❌ import gsplat 失败:", e)
        return 1

    # 构造最小但能真正出图的高斯场景。
    # ⚠️ 踩过的坑：scale=0.1 + 相机 z=4 + fx=80 时，投影半径只有 2 像素，
    #    落不到像素中心 → alpha 全 0、像素全 0、grad 全 0，看起来"通过"其实是空渲染。
    #    所以这里用 scale=0.5、相机拉近到 z=3，并且强制检查 alpha 真的有覆盖。
    dev = "cuda"
    N = 8
    means = torch.tensor([[0.0, 0.0, 0.0]] * N, device=dev, requires_grad=True)
    quats = torch.tensor([[1.0, 0.0, 0.0, 0.0]] * N, device=dev, requires_grad=True)
    scales = torch.full((N, 3), 0.5, device=dev, requires_grad=True)
    opac = torch.full((N,), 0.9, device=dev, requires_grad=True)
    colors = torch.rand(N, 3, device=dev, requires_grad=True)

    H = W = 64
    # gsplat 用 OpenCV 相机约定：相机朝 +z 看，可见要求 z_cam > 0。
    # ⚠️ 踩过的坑：viewmat[2,3] 必须写 +3（世界原点在相机前方 3 米），
    #    写成 -3 会让高斯落到相机背后被裁掉 → alpha 全 0，看着"没报错"其实没渲染。
    viewmat = torch.eye(4, device=dev)
    viewmat[2, 3] = 3.0
    K = torch.tensor([[80.0, 0, 32.0], [0, 80.0, 32.0], [0, 0, 1]], device=dev)

    try:
        out, alpha, meta = rasterization(
            means, quats, scales, opac, colors,
            viewmat[None], K[None], W, H, packed=True,
        )
        torch.cuda.synchronize()
        dt = time.time() - t0
        print(f"✅ 光栅化成功（含 JIT 编译）耗时 {dt:.1f}s")
        print(f"   渲染输出 {tuple(out.shape)}, alpha {tuple(alpha.shape)}")

        # 关键断言：真的画出了东西，而不是空渲染
        a_max = alpha.max().item()
        a_sum = alpha.sum().item()
        c_max = out.max().item()
        print(f"   alpha max={a_max:.4f} sum={a_sum:.1f} | 像素值 max={c_max:.4f}")
        if a_max <= 0 or a_sum <= 0:
            print("❌ alpha 全 0 —— 高斯没覆盖任何像素，渲染是空的（验证无效）")
            return 1
        if c_max <= 0:
            print("❌ 像素全 0 —— 渲染是空的（验证无效）")
            return 1
        print(f"✅ 渲染非空，覆盖 {int((alpha > 0.01).sum().item())} 个像素")
    except Exception as e:
        print("❌ 光栅化失败:", type(e).__name__, e)
        import traceback
        traceback.print_exc()
        return 1

    step(3, "反传（训练必需）")
    try:
        loss = (out ** 2).mean()
        loss.backward()
        torch.cuda.synchronize()
        g = means.grad
        if g is None:
            print("❌ means.grad 为 None —— 计算图没接上")
            return 1
        gn = g.norm().item()
        if not torch.isfinite(g).all():
            print("❌ means.grad 含 inf/nan")
            return 1
        if gn <= 0:
            print("❌ means.grad 全 0 —— 梯度没传回来，训练会不收敛")
            return 1
        print(f"✅ 反传成功, loss={loss.item():.6f}, |grad|={gn:.6f}")
    except Exception as e:
        print("❌ 反传失败:", type(e).__name__, e)
        import traceback
        traceback.print_exc()
        return 1

    step(4, "显存与扩展性样本")
    for bs in [1, 8, 64]:
        try:
            m = torch.randn(bs, 1024, 1024, device=dev, dtype=torch.float16)
            torch.cuda.synchronize()
            print(f"  fp16 张量 {bs}×1024×1024 分配 OK（{bs * 1024 * 1024 * 2 / 2**20:.0f} MiB）")
            del m
        except torch.cuda.OutOfMemoryError:
            print(f"  {bs}×1024×1024 OOM")
            break
    free, total = torch.cuda.mem_get_info()
    print(f"  显存: 已用 {(total - free) / 2**30:.1f} / {total / 2**30:.1f} GiB")

    print(f"\n{SEP}\n🎉 环境可用：可以直接跑 3DGS 训练\n{SEP}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

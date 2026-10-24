#!/usr/bin/env python3
"""一次性补丁：splannequin-gs 官方发布代码的发布级 bug（幂等，可重复跑）。

补丁 1 — gaussian_model.py add_densification_stats 调用/定义不一致：
train.py:317 以 2 参调用，而定义要求 3 参（color_point_tensor）。
4DGaussians 的 color 梯度来自其定制 rasterizer，本仓 renderer 只建
screenspace_points，color 梯度 tensor 根本不存在；且 densify() 的
color_grad_iter 恒为 0（train.py:327 未传），color_gradient_accum
从不参与 clone/split 判据 —— 故改为可选参数，行为与发布配置一致。

补丁 2 — render.py render_fixed_time 帧数越界：
官方硬编码 frames=300（面向合成数据 300 测试视角），真实场景测试
视角数 < 300 时 views[idx] IndexError —— 钳制到 len(views)。
"""
import io
import sys

PATCHES = [
    (
        "/home/velvet/repos/splannequin-gs/scene/gaussian_model.py",
        (
            "    def add_densification_stats(self, viewspace_point_tensor, update_filter, color_point_tensor):\n"
            "        self.xyz_gradient_accum[update_filter] += torch.norm(viewspace_point_tensor[update_filter,:2], dim=-1, keepdim=True)\n"
            "        self.color_gradient_accum[update_filter] += torch.norm(color_point_tensor[update_filter,:, :], dim=(1,2)).unsqueeze(1)\n"
            "        self.denom[update_filter] += 1"
        ),
        (
            "    def add_densification_stats(self, viewspace_point_tensor, update_filter, color_point_tensor=None):\n"
            "        # 官方 train.py 以 2 参调用本函数（renderer 不产出 color 梯度 tensor），\n"
            "        # 且 densify() 的 color_grad_iter 恒为 0 → color_gradient_accum 从不参与\n"
            "        # clone/split 判据。改为可选参数：None 时跳过累积，行为与发布配置一致。\n"
            "        self.xyz_gradient_accum[update_filter] += torch.norm(viewspace_point_tensor[update_filter,:2], dim=-1, keepdim=True)\n"
            "        if color_point_tensor is not None:\n"
            "            self.color_gradient_accum[update_filter] += torch.norm(color_point_tensor[update_filter,:, :], dim=(1,2)).unsqueeze(1)\n"
            "        self.denom[update_filter] += 1"
        ),
    ),
    (
        "/home/velvet/repos/splannequin-gs/render.py",
        (
            '    pbar = tqdm(range(frames), desc=f"Fixed-time rendering (t={time_index})")\n'
            "    for idx in pbar:\n"
            "        view = views[idx]\n"
        ),
        (
            "    # patched: 官方硬编码 frames=300（合成数据），真实场景测试视角数不足时越界 → 钳制。\n"
            "    n_frames = min(frames, len(views))\n"
            '    pbar = tqdm(range(n_frames), desc=f"Fixed-time rendering (t={time_index})")\n'
            "    for idx in pbar:\n"
            "        view = views[idx]\n"
        ),
    ),
]


def main():
    for path, old, new in PATCHES:
        src = io.open(path, "r", encoding="utf-8", newline="").read()
        if new in src:
            print("already patched, skip:", path)
            continue
        if src.count(old) != 1:
            sys.exit(f"❌ pattern not found or not unique: {path} — 检查官方代码是否更新")
        io.open(path, "w", encoding="utf-8", newline="").write(src.replace(old, new))
        print("🎉 patched:", path)


if __name__ == "__main__":
    main()

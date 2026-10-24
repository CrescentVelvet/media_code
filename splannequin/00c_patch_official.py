#!/usr/bin/env python3
"""一次性补丁：splannequin-gs 官方发布代码的调用/定义不一致。

train.py:317 以 2 参调用 add_densification_stats，而 gaussian_model.py
定义要求 3 参（color_point_tensor）。4DGaussians 的 color 梯度来自其
定制 rasterizer，本仓 renderer（gaussian_renderer/__init__.py:28）只建
screenspace_points，color 梯度 tensor 根本不存在；且 densify() 的
color_grad_iter 恒为 0（train.py:327 未传），color_gradient_accum
从不参与 clone/split 判据 —— 故改为可选参数，行为与发布配置一致。
"""
import io
import sys

PATH = "/home/velvet/repos/splannequin-gs/scene/gaussian_model.py"

OLD = (
    "    def add_densification_stats(self, viewspace_point_tensor, update_filter, color_point_tensor):\n"
    "        self.xyz_gradient_accum[update_filter] += torch.norm(viewspace_point_tensor[update_filter,:2], dim=-1, keepdim=True)\n"
    "        self.color_gradient_accum[update_filter] += torch.norm(color_point_tensor[update_filter,:, :], dim=(1,2)).unsqueeze(1)\n"
    "        self.denom[update_filter] += 1"
)
NEW = (
    "    def add_densification_stats(self, viewspace_point_tensor, update_filter, color_point_tensor=None):\n"
    "        # 官方 train.py 以 2 参调用本函数（renderer 不产出 color 梯度 tensor），\n"
    "        # 且 densify() 的 color_grad_iter 恒为 0 → color_gradient_accum 从不参与\n"
    "        # clone/split 判据。改为可选参数：None 时跳过累积，行为与发布配置一致。\n"
    "        self.xyz_gradient_accum[update_filter] += torch.norm(viewspace_point_tensor[update_filter,:2], dim=-1, keepdim=True)\n"
    "        if color_point_tensor is not None:\n"
    "            self.color_gradient_accum[update_filter] += torch.norm(color_point_tensor[update_filter,:, :], dim=(1,2)).unsqueeze(1)\n"
    "        self.denom[update_filter] += 1"
)


def main():
    src = io.open(PATH, "r", encoding="utf-8", newline="").read()
    if NEW in src:
        print("already patched, skip")
        return
    if src.count(OLD) != 1:
        sys.exit("❌ pattern not found or not unique — 检查官方代码是否更新")
    io.open(PATH, "w", encoding="utf-8", newline="").write(src.replace(OLD, NEW))
    print("🎉 patched:", PATH)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""给官方 scenesplatter.py 打补丁：main() return 前导出 scene.ply。

背景：官方 main() 只 return gaussians 不落盘任何 3D 资产（只有过程视频
flash3d_video.mp4 / render_video_{i}.mp4 / confidence_maps）。横评需要可渲染
PLY（对齐 FlashWorld 的 gaussians.ply / Gen3R 的 pcds.ply 口径），必须补这行。

幂等：以标记注释判断，已打过则跳过。
"""
import os
import sys

SS_DIR = os.environ.get("SS_DIR", os.path.expanduser("~/repos/Scene-Splatter"))
TARGET = os.path.join(SS_DIR, "scenesplatter.py")
MARKER = "# media_code: save scene.ply patch"

def main():
    if not os.path.exists(TARGET):
        sys.exit(f"❌ {TARGET} 不存在（先跑 00a_setup_env.sh clone 官方仓）")
    with open(TARGET, "r", encoding="utf-8") as f:
        src = f.read()
    if MARKER in src:
        print(f"⏭️  补丁已存在: {TARGET}")
        return
    # main() 末尾的 "    return gaussians" 是全文唯一裸 return gaussians 行
    # （optimize_gaussian 返回的是三元组，不会误匹配）
    needle = "\n    return gaussians\n"
    if src.count(needle) != 1:
        sys.exit(f"❌ 锚点不唯一或缺失（count={src.count(needle)}），人工检查 {TARGET}")
    patched = src.replace(
        needle,
        "\n    # 保存最终 3D 高斯（media_code 补丁，横评要 PLY）\n"
        "    gaussians.save_ply(os.path.join(save_dir, \"scene.ply\"))  " + MARKER + "\n"
        "    return gaussians\n")
    with open(TARGET, "w", encoding="utf-8", newline="\n") as f:
        f.write(patched)
    print(f"✅ 已打补丁: {TARGET}（main() 末尾新增 scene.ply 导出）")

if __name__ == "__main__":
    main()

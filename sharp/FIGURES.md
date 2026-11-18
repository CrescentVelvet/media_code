# SHARP 图像与视频台账

## 2026-11-17 原生三图工程试跑（2026-11-18归档）

- 总览：`D:/ppt/figures/sharp_portrait/figures/sharp_portrait_probe_contact.png`；
  1920×1080，三行对应000/004/022，四列为source与各自原生轨迹frame 0/30/59。
- 源与视频/PLY：`D:/output/results/sharp_portrait_probe_20261117/`，`source/`、`02_gaussians/`；
  本机原输出`~/output/sharp_results/portrait_probe_20261117/02_gaussians/`保留未移动。
- 每张图独立建场，没有三图配准或融合。三张PNG均无EXIF，官方30mm兜底；
  **不是相同target上的SA/SHARP对拍，不据此宣称几何更准或融合可用**。
- 渲染调用`sharp/02_run_inference.sh`，官方`aed6527`；完整输入/权重/PLY哈希与帧数在
  输出根的`sharp_probe_manifest.json`。原生输出检查见EXPERIMENTS的SH0补记。
- 一次性生成器：`D:/ppt/figures/sharp_portrait/_build/inspect_probe_20261117.py`，不入库；
  WSL运行：`~/miniconda3/envs/sharp/bin/python /mnt/d/ppt/figures/sharp_portrait/_build/inspect_probe_20261117.py`。
- 已目视查看contact sheet；附近视角保持可辨人像，未据小缩略图裁决鼻部精度。

最新03～06数值与结论以[EXPERIMENTS.md](EXPERIMENTS.md)顶部11月18日v4为准，
上述原生视频不替代跨视角评测，旧v1/v2/v3分析图也不得标成当前路线结论。

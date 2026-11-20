# Pi3X — WSL 复现指南

与 README.md 并列（服务器版略）。WSL 本机实测路径：官方仓 `~/repos/Pi3`，权重 `/mnt/d/wheel/pi3x_ms`（软链到 `~/model/pi3x`），输出 `~/output/pi3x_results`（跑完 08 搬到 `/mnt/d/output/results/pi3x_results`）。

```bash
# 0) 环境（幂等；torch 优先用 /mnt/d/wheel 本地 cu124 wheel，缺了走 download.pytorch.org）
INSTALL_DEPS=1 bash pi3x/00a_setup_env.sh

# 1) 权重（~4GB，HF 走 proxy.env 代理；断点续传）
bash pi3x/01_download_models.sh

# 2) 一键跑佳佳抱枕（82 帧 1440x1920 HEIC → 去畸变 → 纯图像 + 条件注入两组推理 + GT 评测）
GPU=0 SCENE=jiajia_baozhen \
  RAW_DIR="/mnt/d/dataset/手部微动样例/01e954de1b8a4cde9315c976f4dcb964-佳佳抱枕" \
  bash pi3x/run_all.sh

# 3) 搬结果到 Windows 盘
bash pi3x/08_move_output.sh
```

显存：82 帧 @ PIXEL_LIMIT=255000 在 3090 24G 上可跑；条件注入（multimodal 分支）显存略高，OOM 就降 PIXEL_LIMIT。

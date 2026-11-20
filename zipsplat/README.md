# ZipSplat — 无位姿多图前馈 3DGS + TokenBA

官方仓：<https://github.com/cvg/ZipSplat>；论文：<https://arxiv.org/abs/2606.05102>。
本目录只放编排/评测脚本，不复制官方代码。`zipsplat`（504px）用于真实图像；论文表格用 `zipsplat-252`，二者不可混报。

## 常用命令

```bash
# 在 code-dir/media_code 下；路径按服务器实际位置显式给出
CONDA_ENV=zipsplat ZIPSPLAT_REPO=../ZipSplat INSTALL_DEPS=1 bash zipsplat/00_setup_env.sh
WEIGHT_PATH=../../model/zipsplat/zipsplat-da3g-504.tar bash zipsplat/01_download_models.sh
IMAGES_DIR=../data/jiajia/images SCENE_NAME=jiajia_16v MAX_VIEWS=16 \
  WEIGHT_PATH=../../model/zipsplat/zipsplat-da3g-504.tar \
  RESULTS_DIR=../zipsplat_results BA=1 RENDER_VIDEO=1 bash zipsplat/02_run_inference.sh
```

输出：`gaussians_feedforward.ply`、TokenBA 后 `gaussians.ply`、`predicted_cameras.*`、`renders/`、`metrics.json`，可选 `orbit_wiggle.mp4`。输入图自动中心裁到官方支持的最近长宽比；504 权重的竖图为 378×504。

## 配置

| 变量 | 默认 | 说明 |
|---|---|---|
| `CONDA_ENV` | `zipsplat` | 独立环境，不修改其它算法 |
| `ZIPSPLAT_REPO` | `~/repos/ZipSplat` | 官方代码（服务器 `00_setup_env.sh` 默认 sibling） |
| `WEIGHT_PATH` | `~/model/zipsplat/zipsplat-da3g-504.tar` | 本地官方权重 |
| `RESULTS_DIR` | `~/output/zipsplat_results` | Linux fs 输出 |
| `MAX_VIEWS` | `16` | 从图片目录等间隔采样；上限 64 |
| `BA` | `1` | 1=官方 TokenBA；0=仅前馈 |
| `RENDER_VIDEO` | `0` | 1=官方 wiggle 环绕视频，不是留出评测 |

佳佳抱枕按 [`README_wsl.md`](README_wsl.md) 执行。实验记录见 [`EXPERIMENTS.md`](EXPERIMENTS.md)。

## 许可与边界

- 代码 Apache-2.0；**预训练权重 CC BY-NC 4.0，仅限非商业用途**。
- 本目录验证的是官方发布权重在本机数据上的推理，不是从随机初始化重训论文模型。
- `source` 自洽指标与 `target` 留出指标分开；后者由输入相机 Sim(3) 对齐，轨迹近平面时须检查退化标志。
- 环绕视频用于目检，不是新视角定量证明。

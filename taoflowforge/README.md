# TaoFlowForge — 单图 → 原生三角网格（三阶段级联流匹配）

阿里 Taobao3D（arXiv 2609.37139，Apache-2.0）：单张图直接产出带朝向的三角网格
（≤2 万顶点 / ≤10 万面，512³ 量化），Stage0 64³ 占据 → Stage1 顶点生成 → Stage2 拓扑
解码 + 补洞/法向校正。官方只发布了推理代码 + 权重，无训练代码。
权重在 ModelScope `Taobao3D/TaoFlowForge`（stage0/1/2.pt，非 gated）。

## 常用命令
> 假设已在本机 WSL（Ubuntu2404）。首次跑前先做下方「首次准备」。
> **铁律：每条命令都必须显式写出模型地址、输入路径、输出路径，不能全靠脚本里的默认值。**

```bash
# 一键（建环境 + 下权重 + 推理）
GPU=0 INPUT=/mnt/d/data/my_object.png \
  bash taoflowforge/run_all.sh

# 分步
# 1) 下权重（ModelScope；已下好会自动跳过）
bash taoflowforge/01_download_models.sh

# 2) 单图推理（输出到 ~/output/taoflowforge_results/<图片名>/）
GPU=0 INPUT=/mnt/d/data/my_object.png \
  NAME=my_object \
  bash taoflowforge/02_run_inference.sh

# 断点续跑（stage1 完成后中断，从中间产物恢复）
cd ~/repos/Taobao3D/TaoFlowForge && \
  python -m taoflowforge.cli run \
    --image /mnt/d/data/my_object.png \
    --stage0-checkpoint /mnt/d/wheel/taoflowforge_ms/stage0.pt \
    --stage1-checkpoint /mnt/d/wheel/taoflowforge_ms/stage1.pt \
    --stage2-checkpoint /mnt/d/wheel/taoflowforge_ms/stage2.pt \
    --output-dir ~/output/taoflowforge_results/my_object \
    --resume-from ~/output/taoflowforge_results/my_object/stage1.npz \
    --seed 42
```

## 首次准备
```bash
cd ~/code  # 或本仓所在目录
git clone <media_code 仓> && cd media_code && cp proxy.env.example proxy.env
# ⚠️ 确认 proxy.env 中 http_proxy / https_proxy 按需配置（本机 GitHub 直连可用）

# 建 env + 装依赖 + clone 官方仓（WSL）
INSTALL_DEPS=1 bash taoflowforge/00a_setup_env.sh

# 下权重（ModelScope，国内直连）
bash taoflowforge/01_download_models.sh
```

权重目录布局：
```
/mnt/d/wheel/taoflowforge_ms/     # proxy.env: TAOFLOWFORGE_MODEL_DIR
  stage0.pt                        # 64³ 粗占据
  stage1.pt                        # Level-MoE 顶点生成（512³）
  stage2.pt                        # 拓扑解码
```

---

以下为详细参考。

## Config (env vars)
| var | default (WSL) | note |
|---|---|---|
| `INPUT` | —（必填） | 输入图像路径（RGB/RGBA，白底最佳） |
| `NAME` | 输入文件名去扩展名 | 输出子目录名 |
| `SEED` | `42` | 官方固定种子，复现用 |
| `OFFLOAD` | `auto` | 上游 choices：`auto` / `on` / `off`（**不是**早期文档写的 `stage`） |
| `EXTRA_ARGS` | 空 | 透传给 CLI，如 `--no-fill-holes` |
| `TFF_REMBG_DIR` | 空 | RMBG-2.0 目录（HF gated，可选）；不设则不做背景移除 |
| `TAOFLOWFORGE_DIR` | `~/repos/Taobao3D` | 官方仓（proxy.env 覆盖） |
| `TAOFLOWFORGE_MODEL_DIR` | `/mnt/d/wheel/taoflowforge_ms` | 权重根（proxy.env 覆盖） |
| `TAOFLOWFORGE_RESULTS_DIR` | `~/output/taoflowforge_results` | 输出根（proxy.env 覆盖） |

## 输出布局
```
~/output/taoflowforge_results/<name>/
├── input.png        # 预处理后的实际输入（抠背景/白底合成后）
├── stage0.npz       # 中间产物（含 RNG 状态，可 --resume-from）
├── stage1.npz
├── stage2.npz
├── mesh_raw.obj     # Stage 2 原始输出
├── mesh.obj         # 补洞 + 法向校正后（最终交付用）
├── mesh.glb         # 归一到 [-0.5,0.5]³ 的可视化版
└── metadata.json
```

## 目录布局
```
<code-dir>/
├── media_code/taoflowforge/      # 本目录（编排脚本）
├── ~/repos/Taobao3D/             # 官方仓（WSL Linux fs）
│   └── TaoFlowForge/             # 实际包目录（python -m taoflowforge.cli 以此为 cwd）
├── /mnt/d/wheel/taoflowforge_ms/ # 权重（D 盘）
└── ~/output/taoflowforge_results/# 输出（跑完按需 08 搬到 /mnt/d）
```

## Notes
- **论文/代码 stage 编号错一位**：论文 Stage 1/2/3 ↔ 代码 stage0/1/2。读源码时注意。
- **论文写 Stage 3 是 32 层 DiT，发布权重实为 28 层**（strict load 会校验架构元数据）。
- **ModelScope SDK 有总时限**：26GB 权重 1 小时下不完会报 `[E1001]`。大文件推荐
  Windows 侧迅雷/curl 直下（`https://www.modelscope.cn/models/Taobao3D/TaoFlowForge/resolve/master/stageN.pt`），
  放进 `/mnt/d/wheel/taoflowforge_ms/` 后 01 自动跳过。预期尺寸：stage0 5.47G / stage1 14.5G / stage2 6.03G。
- **上游 requirements bug**：`Pillow==12.0.0` 与 gradio 5.50.0（pillow<12.0）冲突，
  00a 运行时 sed 降级 Pillow==11.3.0 解决；open3d 需 `conda install -c conda-forge libgomp`。
- 批量推理：`GPU=0 bash taoflowforge/03_run_batch.sh`（默认每物体首视角 8 张；
  `VIEW_SEL=all` 跑 3 视角 24 张；输入数据集 `D:\dataset\物体三图数据`）。
- 可选 DINOv3：若 stage0.pt 未内嵌 DINO 权重，需 `--dino-checkpoint <HF目录>`；
  HF gated 下不了时用 ModelScope 镜像 `facebook/dinov3-vitl16-pretrain-lvd1689m`（非 gated）。
- RMBG-2.0 是 HF gated + CC BY-NC 4.0（非商用），默认不装；输入用白底图即可。
- 适用边界：道具/场景/商品类物体；**不适合高精度人脸**（顶点上限 2 万）。

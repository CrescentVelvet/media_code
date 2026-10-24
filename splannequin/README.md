# splannequin — 双检测时间锚定正则，冻结 MC 视频（WACV 2026）

复现 [Splannequin](https://github.com/chien90190/splannequin-gs)（arXiv 2512.05113，基于 4DGaussians/HexPlane）。
动机：验证其「hidden/defective Gaussian 检测 + 时间锚定」正则可能缓解 som_human 的手部糊/幽灵问题
（两者同属「弱监督时间位置上的 Gaussian 自由漂移」）。调研笔记见 media_paper `3D-20251204-Splannequin.html`。

## 常用命令
> WSL 本机；首次跑前先做下方「首次准备」。

```bash
# 自备数据（<SCENE_DIR>/images/*.png）
SCENE_DIR=~/my_data/hand01 bash splannequin/01_colmap.sh
SCENE_DIR=~/my_data/hand01 bash splannequin/02_train.sh          # splannequin（锚定开）
BASELINE=1 SCENE_DIR=~/my_data/hand01 bash splannequin/02_train.sh  # 4DGS 基线（消融对照）
SCENE_DIR=~/my_data/hand01 bash splannequin/03_render.sh          # fixed-time-view 渲染
```

## 首次准备

```bash
bash splannequin/00a_setup_env.sh   # 建 env + torch1.13.1+cu117 + nvcc11.7 + gcc11 + 编译扩展
```

官方仓放 `~/repos/splannequin-gs`（Linux fs）。GitHub 直连不稳时，00a 回退用
`/mnt/d/wheel/splannequin_src/` 下的 tarball（codeload 打包，子模块为锁定 commit）。

### 需手动下载的文件（迅雷/代理，WSL 直连不通）
| 文件 | URL | 放到 |
|---|---|---|
| 主仓 tarball | https://codeload.github.com/chien90190/splannequin-gs/tar.gz/refs/heads/main | `/mnt/d/wheel/splannequin_src/splannequin-gs-main.tar.gz` |
| depth-diff-gaussian-rasterization（锁定 e495066） | https://codeload.github.com/ingra14m/depth-diff-gaussian-rasterization/tar.gz/e49506654e8e11ed8a62d22bcb693e943fdecacf | 同目录 |
| simple-knn（锁定 44f7642） | https://gitlab.inria.fr/bkerbl/simple-knn/-/archive/44f764299fa305faf6ec5ebd99939e0508331503/simple-knn-44f764299fa305faf6ec5ebd99939e0508331503.tar.gz | 同目录 |
| glm（rasterizer 的子子模块，锁定 5c46b9c） | https://codeload.github.com/g-truc/glm/tar.gz/5c46b9c07008ae65cb81ab79cd677ecc1934b903 | 同目录 |
| EvaluationNet.pth.tar（仅 aesthetic 评测需要） | https://drive.google.com/uc?id=1IXBtvwn8fMCmDRczQExIR3ZSDhinoqbi | `~/model/splannequin/`（软链或复制到仓内 `./models/`） |
| dataset.csv（仅官方 YouTube benchmark 需要） | https://huggingface.co/datasets/chien90190/Splannequin/resolve/main/dataset.csv | 仓内 `./data/dataset.csv` |

注：eval 的 musiq/topiq_nr/hyperiqa/clipiqa 权重由 pyiqa 首次运行时自动下载（源在 GitHub，
WSL 不通时需代理或预下载到 `~/.cache/pyiqa`）。

## Config (env vars)
| var | default | note |
|---|---|---|
| `SCENE_DIR` | （必填） | 含 images/ 的场景目录 |
| `SCENE_NAME` | basename(SCENE_DIR) | 实验名 |
| `BASELINE` | 0 | 1 = 4DGaussians 基线（关锚定） |
| `RESOLUTION` | 1 | 训练分辨率除数 |
| `EXP_DIR` | `~/output/splannequin_results/exp` | 训练输出根 |
| `DATA_DIR` | `~/output/splannequin_results/data` | 数据根（预留） |
| `SPLANNEQUIN_DIR` | `~/repos/splannequin-gs` | 官方仓 |
| `SRC_CACHE` | `/mnt/d/wheel/splannequin_src` | tarball 缓存 |
| `COLMAP_ENV` | `colmap` | 01 用的 colmap conda env |
| `TIME_INDEX` | 空（每 8 帧全渲） | 03 的时刻选择，如 `0,1,2` |
| `RENDER_MODES` | `normal fixed-time-view` | 03 的渲染模式 |
| `EXTRA_TRAIN_ARGS` | 空 | 02 追加 train.py 参数 |

## 目录布局
```
~/repos/splannequin-gs/          # 官方仓（tarball 落盘，无 .git）
~/output/splannequin_results/    # data/ + exp/（训练输出）
~/model/splannequin/             # EvaluationNet 等权重
/mnt/d/wheel/splannequin_src/    # 源码 tarball 缓存
```

## 关键超参（arguments/splannequin.py）
- iterations 30k / batch_size 2 / densify 到 15k
- `occlusion_*`（defective→未来锚定）与 `hidden_*`（hidden→过去锚定）两组：
  steepness=3、from_iter=10k、l1 从 20k 起、权重 L1=10（occlusion 为 1e1）
- 与 som_human E-07 的对照点：锚定是「先验给」约束（从其它时刻的已监督状态借信号），
  不是「梯度学」运动——恰好绕开弱纹理手部梯度不足的问题。

## 已知坑
- mmcv 只用于 `mmcv.Config.fromfile` 读配置，但必须装全版（openmmlab cu117/torch1.13 预编译轮）。
- 官方 README 写 gcc-10；WSL Ubuntu 24.04 无 gcc-10，用 conda gcc 11（nvcc 11.7 host 上限）。
- `torch.cuda.amp` 旧 API：py3.8+torch1.13 环境内正常，别升 torch。
- 官方 train.sh 每次自动选最空 GPU 并写死 `./exp`；本仓 02 改为显式 `GPU=` + 外置 `$EXP_DIR`。

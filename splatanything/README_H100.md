# SplatAnything — 蓝区 H100 集群运行说明

本文只适用于**蓝区 H100 集群**（ca-uae / 迪拜，账号 `00950754`）。**黄区普通服务器**使用 [README.md](README.md)，本机 WSL 使用 [README_wsl.md](README_wsl.md)。三者路径、环境和验证状态不能互相套用。

> 集群登录、隐藏分区、GPU 节点、作业提交和监控等公共规则见 [canada_h100/README.md](../canada_h100/README.md)。本文只记录 SA 在蓝区的专用落点与已实测的 LGTM 小试。

## 环境与路径

| 用途 | 蓝区路径或设置 |
|---|---|
| 代码仓 | `/data/00950754/code/media_code` |
| SA 官方仓 | `/data/00950754/repos/SplatAnything` |
| SA 公开权重 | `/data/00950754/models/splatanything` |
| 原 SA Python 环境 | `/data/00950754/envs/splatanything` |
| 原 SA 环境激活脚本 | `/data/00950754/envs/activate_splatanything.sh` |
| LGTM 独立环境 | `/data/00950754/envs/splatanything_lgtm` |
| LGTM 官方只读仓 | `/data/00950754/repos/ml-lgtm_official` |
| LGTM 实验脚本与产物 | `/data/00950754/runs/splatanything_lgtm` |

- 登录节点没有 GPU；渲染、编译和训练都应在 Slurm 分配的 GPU 节点执行。
- LGTM 使用 Python 3.11 + torch 2.7.1+cu126，patched gsplat 来自 LGTM 仓的 `extern/gsplat`。**不要覆盖原 SA 环境的 gsplat，也不要加载未采用的 `gsplat_rayfix` 副本。**
- 本目录的通用 `00–11` 脚本可以按显式路径在蓝区使用，但蓝区的 Slurm 调度与集群约束不能从黄区 README 推断。基础前馈目前只有黄区通用命令模板，不能写成黄区或蓝区均已完整验收。
- 下面的 Slurm 模板是依据已实测环境和集群公共规则整理的复跑模板，尚未以独立作业实跑；已验证的实验产物见末节。

## 原 SA 八卡训练

这是 `canada_h100` 的自研训练链，不是 SA 官方发布的训练代码，也不是 35 号 LGTM 小试。

```bash
ssh h100
cd /data/00950754/code/media_code/canada_h100
chmod +x *.sh *.py
./12_sa_train.sh start                  # 8 卡 / 168h / 全参 / 25 万步
./12_sa_train.sh start 2 1h gshead     # 2 卡解码头短试
./12_sa_train.sh stat                   # 队列与显卡
./12_sa_train.sh tail                   # 训练日志
./12_sa_train.sh metrics                # 逐 rank 指标
./12_sa_train.sh stop                   # 停止后必须再 ls 复查
./12_sa_train.sh ls
```

完整参数、数据轮转、恢复训练及注意事项见 [canada_h100/README.md](../canada_h100/README.md)。不要因为复跑 LGTM 而停止或修改原有 SA 作业。

## LGTM 式局部纹理小试（35）

`35_lgtm_texture.py` 已实测：三张 source=[0,10,21]、一张 held-out=[22]、252²，冻结几何和透明度，对比 K=4 投影纹理与普通 RGB 残差，各拟合 40 步。**它不是 LGTM 官方跨场景训练或 4K 复现；投影和训练链通过，但画质未通过，不进入人物主线。** 读数及实图结论见 [EXPERIMENTS.md](EXPERIMENTS.md)。

### 新作业复跑

以下是蓝区的独立 Slurm 作业模板，**不依赖历史作业号**。在登录节点将内容写入仓外，例如 `/data/00950754/runs/splatanything_lgtm/run_lgtm.sbatch`，再执行 `sbatch`。模板不会自动安装依赖或覆盖已有环境。

```bash
#!/bin/bash
#SBATCH -J sa_lgtm_probe
#SBATCH -p h100a011
#SBATCH --gres=gpu:1
#SBATCH -c 16
#SBATCH --mem=120G
#SBATCH --time=04:00:00
#SBATCH -o /data/00950754/logs/%x_%j.out

set -eo pipefail
RUN=/data/00950754/runs/splatanything_lgtm
export SPLATANYTHING_DIR=/data/00950754/repos/SplatAnything
export SPLATANYTHING_MODEL_DIR=/data/00950754/models/splatanything
export LGTM_DIR=/data/00950754/repos/ml-lgtm_official

/data/00950754/envs/splatanything_lgtm/bin/python \
  "$RUN/35_lgtm_texture.py" \
  --scene "$SPLATANYTHING_DIR/examples/object" \
  --out "$RUN/lgtm40_repeat_${SLURM_JOB_ID}" \
  --res 252 --n-source 3 --steps 40 --pose-mode \
  --texture-size 4 --log-every 10
```

复跑前确认实验脚本与待复现报告的 `code_sha256` 一致。集群作业查看：

```bash
squeue -a -u "$(whoami)"
tail -f /data/00950754/logs/sa_lgtm_probe_<JOBID>.out
```

仅在需要临时进入自己**已拥有且仍在运行**的作业节点时，才使用 `srun --overlap --jobid=<自己的JOBID> ...`。历史作业 `201373222` 是本次实验时的原 SA 作业记录，**不是新实验的固定入口**；不要进入、停止或修改不属于自己的作业。

### 已完成的正式实验

- 产物：`/data/00950754/runs/splatanything_lgtm/lgtm40_verified_v4/`。
- 报告：`report.json`、`smoke.json`，含输入、代码 SHA、投影硬门、参数更新、训练曲线及逐视角指标。
- 图像：双臂 `*_source/`、`*_held_out/` 中的 GT / before / after / diff / alpha。
- 结论：LGTM held-out PSNR 14.264 → 15.256 dB，普通对照 13.951 → 15.430 dB；纹理臂留出增益少 0.486 dB，且仍有网格、局部过曝和摩尔纹，**视觉验收未通过**。
- 运行期间原 SA 作业未被停止或修改。

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

## 人像 UE53 短训：手动上传、审核、训练与多 checkpoint 评测（38–44）

**状态（2026-11-27）：本机 RTX 3090 上合成及真实 UE 的零更新 GPU 门已通过；蓝区 H100 尚未运行 38–44，也没有训练 loss 或重影改善证据。** 训练前必须人工完成下面第 2 步；不能由 gate-only 代替审核，也不能停止或修改已有通用作业。

### 1. 本机上传数据与审核模板

数据必须落在 **SSD 数据集目录**，不要放家目录（只有 51 G，装不下 121.6 G）。用 `canada_h100/13_rsync_dataset.sh` 在 WSL 里跑，它带进度条、断点续传和断线重试：

```bash
# 本机 WSL，先看清单
wsl -d Ubuntu2404 -- bash /mnt/c/code/media_code/canada_h100/13_rsync_dataset.sh --dry-run
# 去掉 --dry-run 正式传；Ctrl-C 或断网后重跑同一条命令即续传
wsl -d Ubuntu2404 -- bash /mnt/c/code/media_code/canada_h100/13_rsync_dataset.sh
# 传完对拍，应报「无待传文件」
wsl -d Ubuntu2404 -- bash /mnt/c/code/media_code/canada_h100/13_rsync_dataset.sh --verify
```

默认落点 `/data/00950754/datasets/ue53_portrait/`。审核模板是小文件，单独传：

```bash
ssh h100 "mkdir -p /data/00950754/runs/splatanything_portrait"
rsync -az -e ssh \
  "/mnt/d/output/results/splatanything_results/portrait_ue53/short_v3_review.json" \
  "h100:/data/00950754/runs/splatanything_portrait/review_template.json"
```

H100 上生成与机器路径匹配的 manifest；`short_v1` 必须不存在：

```bash
ssh h100
source /data/00950754/envs/activate_splatanything.sh
export SPLATANYTHING_DIR=/data/00950754/repos/SplatAnything
export SPLATANYTHING_MODEL_DIR=/data/00950754/models/splatanything
cd /data/00950754/code/media_code
git pull --ff-only
mkdir -p /data/00950754/runs/splatanything_portrait
python splatanything/38_prepare_portrait_ue53.py \
  --dataset-root /data/00950754/datasets/ue53_portrait \
  --audit-views 2 \
  --out /data/00950754/runs/splatanything_portrait/manifest_v1 \
  --write-review-template /data/00950754/runs/splatanything_portrait/review.json
```

### 2. 逐视角人工审核（训练硬门）

`38` 固定 `s03/s06/s09` 三源、`e06/e19_p00` 训练 target、四个近输入评测及四个背面试图。每包另有 2 个仅结构 QA 视角。将 `review.json` 中 `views` 的每项 `status` 按原 RGB 改成 `accepted` 或 `rejected`，并核对 `exposure_flag`。由于这是三源固定协议，source 不可静默换图；rejected 序列暂不进入本轮，需要新协议时另建 manifest，不能用开关放行。13 个 c6-fail 身份只能作 hard-example 诊断，不进入优化。

### 3. 在已分配的 GPU 上先跑零更新门

不要在登录节点跑 CUDA。用独立单卡作业或 `srun -p h100a011 --gres=gpu:1 --pty bash` 获取 GPU；分区与作业纪律见 `canada_h100/README.md`。下面 `RUN` 换成唯一新目录；已存在或失败目录不覆盖。

```bash
RUN=/data/00950754/runs/splatanything_portrait/gate_ada_v1
python splatanything/43_smoke_portrait.py \
  --official-repo "$SPLATANYTHING_DIR" --res 112 --out "$RUN/synthetic"
python splatanything/40_train_portrait.py \
  --manifest /data/00950754/runs/splatanything_portrait/manifest_v1/manifest.json \
  --official-repo "$SPLATANYTHING_DIR" \
  --sequences ada_env17 --gate-only --res 504 --out "$RUN/real504"
```

成功标记分别是 `SMOKE_PASS` 和 `GATE_ONLY_PASS`。检查 `gate.json` 的 `optimizer_steps=0`、非空 person coverage、ED 最大差 ≤1e-5、有限非零学生梯度与全部 `parameter_max_delta=0`，并目检 RGB/alpha。**本机 `ada_env17` 504² 实测：n_merged=425019、人物 α>0.5 覆盖 0.813587、有效深度覆盖 0.828069、ED 最大差 0、峰值分配 12.32 GiB；B0 仍有拖影和大块黑色遮挡，不是质量通过。** 本机证据位于 `D:/output/results/splatanything_results/portrait_ue53/gate504_v1/`，合成证据在同目录树 `synthetic_v21/`。

### 4. 100 step 工程冒烟、按实测步速续训

只有人工审核与 H100 两项门均通过才执行。`RUN` 用新的训练目录，先保存 B0；只开放 `gs_head` 和 `gs_renderer.decoder.gs_head`，冻结外部教师和 source encoder。

```bash
TRAIN=/data/00950754/runs/splatanything_portrait/ue53_v1
python splatanything/40_train_portrait.py \
  --manifest /data/00950754/runs/splatanything_portrait/manifest_v1/manifest.json \
  --review /data/00950754/runs/splatanything_portrait/review.json \
  --official-repo "$SPLATANYTHING_DIR" \
  --out "$TRAIN" --res 504 --steps 100 --lr 1e-4 --depth-weight 0.1 \
  --max-train-sequences 16 --checkpoint-at 1,25,50,100
python splatanything/41_plot_portrait.py --run "$TRAIN"
```

先读 `run.json` 的实测 `measured_mean_step_s` / `eta_2000_only_steps_hours`、真实 `metrics.jsonl` 和 loss 图，再决定是否续训。目前不能给可信 H100 2000-step ETA；不能拿本机 4.85 秒的单次门检乘步数。

```bash
python splatanything/40_train_portrait.py \
  --manifest /data/00950754/runs/splatanything_portrait/manifest_v1/manifest.json \
  --review /data/00950754/runs/splatanything_portrait/review.json \
  --official-repo "$SPLATANYTHING_DIR" \
  --out "$TRAIN" --res 504 --steps 2000 --lr 1e-4 --depth-weight 0.1 \
  --resume "$TRAIN/step_0100.pt" --checkpoint-at 250,500,1000,2000
python splatanything/41_plot_portrait.py --run "$TRAIN"
```

### 5. 独立身份、多 checkpoint 与真人开发集回放

先用 val 身份选 checkpoint，再对 test 只评一次；不要用 target 拟合 Sim(3) 或选参数。现有两测试身份只支持探索性正/负信号。H100 上尚未实测 `42`；若蓝区原环境的 gsplat 与本机 D-pass 接口不同，先停在 gate，不切换或覆盖现有环境。

```bash
python splatanything/42_eval_portrait.py \
  --manifest /data/00950754/runs/splatanything_portrait/manifest_v1/manifest.json \
  --rundir "$TRAIN" --split test --res 504 \
  --official-repo "$SPLATANYTHING_DIR" \
  --out /data/00950754/runs/splatanything_portrait/eval_test_v1 --save-ply
# 先上传冻结真人 protocol.json / real_eval_v18/n1_real_eval.json 及其引用的原图后，
# 另用 --real-protocol ... --real-report ... --real-res 756；不能省掉真实输入目录。
```

输出 `per_view_metrics.csv`、`checkpoint_eval.json`、`*_B0_vs_checkpoints.png` 和 alpha/ED 图。逐项看人物 bbox 与整图重影、低通外观、人物覆盖和有效 ED 覆盖；PSNR/loss 上升或把人物抹掉都不是通过。当前 UE 包没有鼻区地标，**不能输出虚构的鼻区毫米误差**，也不能把静态 UE 结果外推成同步真人泛化。

本机 WSL 回传（不覆盖本地同名旧实验）：

```bash
rsync -az --info=progress2 -e ssh \
  "h100:/data/00950754/runs/splatanything_portrait/" \
  "/mnt/d/output/results/splatanything_results/portrait_ue53/h100_v1/"
```

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

# flame_human NOTES — 原理 / 机制 / 排错手册

README 只讲怎么跑；实验数据见 [EXPERIMENTS.md](EXPERIMENTS.md)。本文件讲**为什么这么做**与**踩过的坑**。

## 1. 三分支架构

```
head_gs  : 绑定 FLAME 网格（重心插值）+ 自由高斯，逐帧由 exp/local_q/local_t 形变
body_gs  : 上游 nn ply 经多视角投票提取的静态高斯（人物 region）
scene_gs : 同一份 ply 的剩余部分（背景）
```

**合成方式**：三个分支的已激活属性 `cat` 成一个 `MergedGS`，**单次渲染**。
跨分支遮挡由光栅器的深度排序正确处理 —— 不做逐分支渲染再叠加（那样遮挡是错的）。

**head 的形变机制**（`train_avatar.py:181 deform`）：

```python
exp_f   = self.exp[f]                     # 表情逐帧
V       = flame(betas=id_coeff, expression=exp_f).vertices
p_bound = (V[self.tri] * bary).sum(1)     # 绑定点：逐帧网格插值
p_can[is_free] += self._free_can          # 自由点：+ 全局固定偏移
p_local = Rl(local_q[f]) @ p_can + local_t[f]   # 头旋转/平移逐帧
```

**逐帧**：表情、局部旋转/平移、网格形变。**全局共享**：外观（SH/opacity/scale）、`free_can` 偏移、
身份 `id_coeff`。所以"canonical 空间重建 + 逐帧反变换"这套机制**本来就在做**；
它要生效的前提是**拟合足够准**（当前 lm-RMS 7.45px 已经是这条路的边界）。

## 2. 最重要的一条教训：监督区必须 = 分支职责边界

本链路在这个坑上**踩了四次**，每次症状不同但根因相同：

| 次数 | 位置 | 错误 | 症状 |
|---|---|---|---|
| 1 | 08b body finetune | 用 person mask（**含头**）监督 body | body 高斯长进头盒 → 头部重影 |
| 2 | 08 头训练 | 用 person mask（**含身体**）监督 head | head 分支 2/3 梯度在拟合夹克/胳膊（浪费）|
| 3 | 08d scene finetune | 罚「person 区渲染**亮度**」 | scene 学会"**涂黑**"而非变透明（inside=0.022）|
| 4 | 08e 剪枝 | scene 用 body 的阈值（DROP_RATIO=0.6） | 误删"被人物长期遮挡的合法背景"→ 轮廓空洞 |

**推论**：任何"区域监督"改动，先问三个问题 ——
① 该区域是否正好是本分支的职责？② 惩罚项能否被"作弊式"满足？③ 阈值是否按**分支语义**设定
（head=细节物体可取小尺度门控；scene=大平面不能）？

## 3. 05 对齐：为什么 4.3 要 3000 步 / gamma 0.9995

`run_stage` 原本对所有阶段用 `ExponentialLR(gamma=0.95)` **逐迭代**衰减 —— 300 步后
LR 只剩 `0.95^300 ≈ 2e-7`，**后 200 步等于空转**。而 4.3 要联合优化
59 帧 × (pose 7 + exp 100) + 300 维 id。

判定实验（`_debug_selffit.py`）：单帧自拟合可达 ~9px，全局却 26.5px
→ **不是模型表达力不足，是优化不足**。修复后 7.45px，且各 landmark 区域误差
从"嘴 29px / 左眼 19px 极不均"变为"全区域 3-6px 均匀"（真实对齐改善，非过拟合）。

## 4. 排错手册

| 症状 | 根因 | 修法 |
|---|---|---|
| 训练 loss **单调爬升**（先降后升） | densify 把 `_xyz` 换成新 `nn.Parameter`，**Adam 仍持旧引用** → 活参数零更新 | densify 后 `opt = Adam(build_params())` 重挂（镜像 `finetune_body.py:396`）|
| `_RasterizeGaussiansBackward ... got [120000,16,3] expected [70000,16,3]` | 恢复 06 产物时 **ply 与 `avatar_bind_p0.npz` 不同步** | 两者必须成对：重跑 `06_init_avatar_gs.sh` 重新生成 |
| 脚本 **3 秒静默退出**、无 echo | `set -u` 与 `proxy.env` 的 `PYTHONPATH="...:$PYTHONPATH"`（未绑定变量）冲突 | 用 `set -eo pipefail`（**不要 `-u`**）；08b 用 `set -o pipefail` 无此问题 |
| `❌ 缺 AvatarGaussian 初始 PLY` | 06 产出 `avatar_p0.ply`（**一位**），08 默认 `PID=00` 找 `avatar_p00.ply` | `PID=0`；08 脚本默认值已改 |
| `ModuleNotFoundError: accelerate/diffusers` | HYPIR 依赖只在 **vggt_human** 环境 | 脚本内 `conda activate vggt_human`（`ENHANCE_ENV`）|
| WSL 里 git 报 `Author identity unknown` | git 身份只配在 **Windows 侧** | commit 走 Windows 侧 `cd /c/code/media_code && git ...` |
| 闸门阈值"设了但没用" | 清晰度算在**加了 padding 的大裁剪**上，背景纹理拉高方差 | 度量必须用**未加 padding 的脸框** |
| `PLC` 找不到产物文件 | `source ./_env.sh` 前未设 `SCRIPT_DIR=$PWD` → `REPO_DIR` 算错、proxy.env 不加载 | 先 `SCRIPT_DIR=$PWD source ./_env.sh` |

## 5. 调试方法论（本链路验证有效）

1. **白底单分支渲染**：分别渲染 scene/body/head，**背景必须用白色** —— 黑底时分不清
   "黑=没有高斯（背景色）"和"黑=有暗色高斯"。工具：`_debug_branch_split.py`。
2. **指标裁到分支职责区**：评 head 质量用 **head 框内 PSNR**。全图 PSNR 被大面积
   scene 稀释（head 真实提升 +3.38dB，全图只值 +0.37dB）。
3. **单帧自拟合上限探测**（`_debug_selffit.py`）：判断"优化不足"还是"表达力不足"。
4. **消融要单分支单独跑评测**：08e 的归因就是靠"仅 body / 仅 scene / 仅 head"三组
   分别跑，才发现收益 100% 来自 head。
5. **改配置要公平 A/B**：两个产物用**同一套后处理**（如同套 08e 剪枝）再比，
   否则差异可能来自后处理而非被测变量。
6. **度量口径要固定**：时序抖动的首测 max ratio=1.847 是**裁框尺寸跳变**造成的假象，
   固定口径后最差仅 1.02-1.04。

# ZipSplat 自训手册：要多少数据 · 数据长什么样 · 参数怎么配

> 配套：[官方训练数据与训练配置明细](OFFICIAL_TRAINING_DETAILS.md) ｜ [训练指南](TRAINING_GUIDE_3DGS.md) ｜ [数据集总清单](TRAINING_DATASETS.md)
>
> 核对日期：2026-11-28，逐条对齐本机官方仓 `~/repos/ZipSplat` 的 `splatfactory/`（HEAD `73ab63f`）。凡标“代码”的结论都来自源码，不来自论文描述。**本文未执行下载与训练**，体积为发布规模近似值。

本文分两半：**§1–§7 是“自己造数据去训”**（数据量 → 数据形态 → 结构 → 代码坑 → 参数），**§8–§13 是“下载官方数据集复现”**。只想跑通就只看 §1、§6、§6.1。

## 0. 先分清三种视角

一次训练样本（sample）= **一个场景 + 一组抽出来的视角**，视角被分成两组：

| 术语 | 在本仓/论文里的叫法 | 是否进网络编码器 | 作用 |
|---|---|---|---|
| **context**（上下文视角） | 也叫 input / source / 输入视角 | **是** | 网络只看这些图，预测深度、相机和 3D 高斯；它们定义坐标系与基线 |
| **target**（目标视角） | 监督视角 / novel view | **否** | 只拿 GT 图和新视角相机算渲染损失，不参与建表示 |
| **reference view** | 第一个 context | 是 | 固定为世界坐标原点（GSN 归一化的基准），因此 `random_reference_view` 会改变尺度与朝向后处理 |

几个容易误解的点：

1. **context 不是“历史帧”**。ZipSplat 是前馈模型，没有时序状态；每个 step 从同一场景随机抽 N 个视角当 context，N 在 2–24 之间随机（504 配置从 step 0 就是全范围）。
2. **target 也不是“下一帧”**。它是**留出的其它视角**，只用来算 `L1 + LPIPS`。RE10K/DL3DV 的 target 通过帧距约束选（`bounded`），ScanNet++ 等通过共视度选（`rank`），物体类随机选（`random`）。
3. **context 与 target 索引在同一样本内不重复**，且都来自同一场景。所以一个 scene 至少要 `N + 4` 张图，否则该 scene 被跳过。
4. **一个 batch 只含一个数据集（child）**，跨数据集会截断当前 batch —— 这就是“多数据集混合”实现上不等于把十份数据混成一锅的原因。

```text
sample（一个 scene）
├── context: N 张 (N ~ U[2,24])  ──► encoder ──► depth/camera ──► Gaussians
│                                                    │
└── target: 4 张 (GT RGB + 相机) ────────────────────┴──► render ──► L1 + LPIPS
```

## 1. 要准备多少数据

### 1.1 先算每场景的下限（硬门槛，代码）

`_scene_ok()` 与各采样器的 `min_required_images()` 会**同时**过滤，任一不满足就跳过该 scene：

| 约束 | 数值 | 出处 |
|---|---|---|
| 场景视角数 ≥ context + target | **≥ V + 4**，V 最大 24 → **≥ 28** | `multi_view_dataset.py:454` |
| `bounded` 采样器自身要求 | `max(2·min_dist_to_ctx+1, min_dist_between_ctx+1)`；RE10K 参数下 = **36**，DL3DV = **9** | `view_sampler/bounded.py:162` |
| `rank` 采样器 | ≥ 3 | `view_sampler/rank.py:35` |
| `random` 采样器 | ≥ 1 | `view_sampler/random.py:20` |

**结论：按每场景 ≥ 36 张准备**（用 RE10K 那套 `bounded` 参数时）。只满足 28 张仍然会在抽到 V=24 且 `min_dist` 判定不过时被丢掉。跳到 V=2 时 6 张就够，但 V 是随机抽的，不能赌。

被跳过的 scene **不报错**：`_process_scene` 的异常被 `except Exception` 吞掉（`multi_view_dataset.py:200`），只在 `logger.debug` 里留一行。所以“损失不降但也不崩”通常就是全被跳了。

### 1.2 要多少个 scene

| 目标 | scene 数 | 每场景图 | 能验证什么 | 直观耗时 |
|---|---|---|---|---|
| **冒烟** | 20–50 | 36–80 | 转换、补深度、shard 读取、训练循环、loss 下降、无 NaN | 单卡数小时 |
| **方向性验证** | 200–500 | 36–120 | 改动的方向对不对（不是复现指标） | 单卡 1–3 天 |
| **小规模重训** | 2,000–5,000 | 36–200 | 能否在新域（例如人像）上稳定收敛 | 多卡数天 |
| **官方复现** | RE10K ≈80K clip + DL3DV 10,510 scene | 数十到数百 | 论文数字 | 16×GH200 |

500 scene 以下不要期待泛化。官方 252 配方是 ~9 万个 scene 抽 45 万步、全局 batch 384 场景/步，等于同一批场景被反复访问上千次——小数据的“看过很多步”不等于“见过很多场景”。

### 1.3 换算成磁盘

```text
磁盘 ≈ scenes × 平均视角数 × (单张 JPEG + 单张 depth WebP) × 2~3
```

后一个 ×2~3 是 shard 冗余 + 补深度时的临时重写（`extract_depth` 走临时文件再原子重命名，峰值接近 2 倍）。参考量级：RE10K + DL3DV 960P ≈ 2.8 TB 原始，转换加补深度后按 TB 计。

自己造数据时可以省：**先把长边缩到训练分辨率再打包**（`resize` 脚本），别让 4K 原图进 shard 再靠 `preprocessing.resize` 在线缩——那是白烧 IO 和 CPU。

### 1.4 开跑前必须具备的环境

| 类别 | 具体内容 | 说明 |
|---|---|---|
| 代码 | 官方仓 `cvg/ZipSplat` + `splatfactory` 环境（Python 3.12、torch、gsplat、nvcc） | 与本仓轻量推理 env 分开 |
| 权重 | **DA3 `depth-anything/DA3NESTED-GIANT-LARGE-1.1`** 转成 `weights/da3/da3-giant-1.1.pth` | 训练初始化必需；补深度用另一个 `DA3NESTED-GIANT-LARGE` |
| 包 | `depth_anything_3`（pip） | 只补深度时需要 |
| 显卡 | 补深度、precompute rankings 都要 GPU | 单卡可跑，`torchrun` 可多卡 |
| 授权 | DL3DV / ScanNet++ / Matterport 类需申请或签协议 | 脚本只从官方 host 取，不绕过条款 |

## 2. 什么样的数据

### 2.1 五条硬要求

| 项 | 要求 | 不满足会怎样 |
|---|---|---|
| **相机模型** | pinhole，**无畸变**。有畸变必须先 undistort（官方 convert 会做） | 边缘几何错位，网络学到系统性偏移 |
| **位姿** | **OpenCV c2w**，12 个数（R 展平 + t） | 存成 w2c 或 4×4 会在归一化后得到镜像场景，训练不报错但结果错 |
| **深度** | 默认配方**必须有**（见 2.2） | 该 scene 静默跳过，训练跑空 |
| **基线** | 相邻视角要有平移，不能纯旋转 | 三角化退化，几何监督失效 |
| **分辨率** | 不低于训练裁剪尺寸，**不能靠 upscale** | 官方按“源图 ≥ 裁剪尺寸”过滤宽高比预设 |

### 2.2 深度为什么是“必须”，而不是“可选”

`compute_scene_scale()` 在 `normalize_baseline=False`（默认）时走的是“深度点到原点距离的中位数”，**没有有效深度就直接 raise**：

```python
if not depth_mask.any():
    raise ValueError("No valid depth for scale normalization")
```

这个异常被 `_process_scene` 外层吞掉 → 该 scene 被跳过。**所以“只给 RGB 文件夹”等于全量静默跳过**，屏幕上看不出任何错误。

只有三种情况能不要深度：

| 写法 | 结果 |
|---|---|
| `normalize_baseline: start_end` | 用首尾 context 相机的距离定尺度，只需位姿 |
| `normalize_baseline: max_pairwise_d` | 用相机间最大距离 |
| `scene_scale_constant: <float>` | 直接给固定逆尺度 |

但注意：即使这样绕开尺度，**GeoCE（0.05 权重）仍然读 `depth_mask`** 来判定 patch 有效性。无深度时 `min_valid_frac=0.5` 判不过，GeoCE 对整场景退化为 0，你训练的就是“去掉几何约束项的 ZipSplat”。`depth_weight` 默认 0，所以渲染深度 L1 不是问题，问题是尺度与 GeoCE。

### 2.3 人像/UE 渲染特有的三条

1. **脸在画面里太小等于没数据**。504 长边、9:16 裁剪后一件 1080p 人像的脸可能只有 40–60 px；网络把脸压进 token 的 patch 后只剩几个 token。相机要推近，或裁到上半身。
2. **背景占主导**。人像场景里大面积的墙/地面会主导渲染损失与 GeoCE。要么背景干净，要么接受“模型主要在学背景”。
3. **多机位要有真实视差**。绕人 360° 但半径固定、只转向 = 纯旋转，对几何无信息。机位要同时有角度和距离变化。

## 3. 数据长什么样：shard 结构

官方不用图片文件夹，用 **tar shard 流式读取**（`datasets/README.md`、`utils/io.py`）。

### 3.1 目录树

```text
data/<dataset>/<HxW>/
├── train-scenes/
│   ├── shard-000000.tar        # 默认 glob 是 shard-*.tar
│   ├── shard-000001.tar
│   └── index.json              # {scene_key: "shard-000000.tar"}
└── test-scenes/                # 同结构；验证集从这里读
    ├── shard-000000.tar
    └── index.json
```

- `<HxW>` 是**源分辨率**（如 `360x640`、`540x960`），不是训练分辨率；目录名写错不会有自动纠正。
- `index.json` 只用于**数 scene**（`composed.py:_count_scenes`）。缺了会退化成扫所有 tar，慢但能跑（有 warning）。
- `shard_glob` 默认 `shard-*.tar`；只有 `mipnerf360` 那种“一场景一 tar”的写法用 `*.tar`。

### 3.2 一个 tar 内部的成员（精确命名）

由 `io.write_scene_to_tar()` 决定，`{key}` 是场景 ID（也当文件名前缀）：

| 成员 | 形状 / 格式 | 必需 |
|---|---|---|
| `{key}.meta.json` | `{"num_views": N, "has_depth": bool, ...}` | 是 |
| `{key}.cameras.npy` | `(N, D) float32`，`D ∈ {6, 8, 10}` | 是 |
| `{key}.poses.npy` | `(N, 12) float32`，**c2w** | 是 |
| `{key}.img{i}.{jpg\|png\|webp}` | 原始字节透传，扩展名按 magic bytes 判 | 是 |
| `{key}.dep{i}.webp` | log 量化 uint8，`0` = 无效 | `has_depth=true` 时 |
| `{key}.depth_ranges.json` | `[[d_min, d_max], ...]` 每视角一对 | `has_depth=true` 时 |
| `{key}.rankings.npy` / `.overlaps.npy` | `(N, k) int16` / `(N, k) float32`，每视角 k 个最共视视角（自己排第一） | `rank` 采样器需要 |
| `{key}.seen_by.npy` / `.sees.npy` | 共视反向索引（`RANKING_MEMBERS` 另两项） | `rank` 采样器需要 |
| `{key}.cov{i}.webp` / `.seg{i}.webp` / `.dyn{i}.webp` | 共视掩码 / 实例图 / 动态掩码，都是二进制 WebP | 可选 |
| `{key}.times.npy` / `.time_ids.npy` / `.cam_ids.npy` | `float32 (N,)` / `int32 (N,)` / `int32 (N,)` | 可选 |
| `{key}.tracks_xyz.npy` / `.tracks_vis.npy` / `.tracks_seg.npy` | 轨迹坐标 / 可见性 / 语义（`TRACK_MEMBERS`） | 可选 |

**`{key}` 有三条命名约束**（`io.py` 的读取端决定，写错了不报错、只错位）：

1. **不能含 `.`**。`_parse_tar_entry` 的调用方用 `member.name.split(".")[0]` 反推 scene key（`io.py` 的 `read_scenes_from_tar` 与 `iter_scenes_from_tar` 都是），所以 `scene.001` 会被截成 `scene`——同一 shard 里所有带点号的场景会**合并成一个**，`num_views` 对不上后被 `_scene_ok` 静默丢弃。用 `scene001` / `scene-001` / `scene_001` 都行。
2. **不能含 `.img` / `.dep` / `.cov` / `.seg` / `.dyn` 这些子串**。解析按 `elif` 链做子串判定（`".img" in name` → `".dep" in name` → …），取 index 时写死 `name.split(".img")[1].split(".")[0]` 再 `int()`。key 一旦带这些子串，**别的成员**会先命中错的分支：key=`a.img` 时 `a.img.dep0.webp` 命中 `.img`，`split(".img")[1]` 得到 `.dep0.webp`，再去 `split(".")[0]` 得空串，`int('')` 直接抛 `ValueError: invalid literal for int() with base 10: ''`（已实测）。**用 `scene_0001` 这种纯字母数字加下划线的 key 最稳。**
3. **同一 scene 的成员必须连续写入**。`iter_scenes_from_tar` 的注释写明 "Assumes scenes are stored contiguously (guaranteed by write_scene_to_tar)"——它是遇到 key 变化就 `yield`。所以追加写 shard 时不能多进程各自往同一 tar 加不同 scene，要串行 or 一进程一 tar。

### 3.3 相机与位姿的具体编码

**Camera**（`geometry/cameras.py`）——`data_` 最后一位可以是 6/8/10：

```python
# 6 维：无畸变
[w, h, fx, fy, cx, cy]
# 8 / 10 维：后接畸变参数
[w, h, fx, fy, cx, cy, k1, k2, ...]
```

主点按 COLMAP 口径（左上像素中心在 `(0.5, 0.5)`）。缩放/裁剪时官方用 3×3 图像变换矩阵同步改 `w,h,fx,fy,cx,cy`，不建议自己手算。

**Pose**（`geometry/poses.py`）——固定 12 维，**c2w**：

```python
# 4x4 w2c 或 c2w 转成 12 维
R, t = T[:3, :3], T[:3, 3]
pose_12 = np.concatenate([R.flatten(), t])   # (12,)
```

### 3.4 深度编码

`io.encode_depth()` 做的是 **log 空间量化到 uint8，再 WebP lossless**：

1. 取有效像素的 p0.1 / p99.9 作为 `[d_min, d_max]`；
2. 范围外像素写成 `0` = 无效；
3. 范围内按对数映射到 `1..255`；
4. 返回 `(webp_bytes, d_min, d_max)`，后者存进 `depth_ranges.json`。

这意味着：**深度是相对量化的，必须和 `depth_ranges` 一起存**。只拷 `.dep{i}.webp` 不给 ranges，解码会得到全错的比例。

**精度实测**（本机跑 `encode_depth`→`decode_depth` 往返，40×56 梯度深度 `0.8→12.0` 米）：

| 项 | 结果 |
|---|---|
| 形状 | `(40,56)` → `(40,56)`，**不转置** |
| 范围内相对误差 | 均值 **0.27%**，p99 0.53% |
| 无效像素（输入 `-1`） | 解码后仍是负值，正确保留 |
| 有效像素被误标无效 | **6 / 2240**（0.27%），都在 0.8 和 12.0 两个端点上 |

最后一行是唯一需要留意的：`d_min/d_max` 取自 p0.1/p99.9，**落在这两个百分位之外的像素会被写成 0（无效）**，即使它原本是有效深度。所以极少数端点像素会丢，但 `min_valid_frac=0.5` 和 `depth_mask` 的统计不会被影响。

两个退化分支也实测过：

| 输入 | `d_min/d_max` | 解码结果 |
|---|---|---|
| 整图常数 `3.0` | `3.0 / 3.0`（`d_max-d_min < _DEPTH_EPS`） | 全图 `3.0`，精度无损 |
| 常数 `2.5` + 对角 `-1` | `2.5 / 2.5` | `-1` 保住，其余 `2.5` |
| **整图全无效**（`-1`） | **`0.0 / 0.0`** | 全图 `-1` |

也就是说“深度图大面积常数值”不会崩，只是丢掉真实起伏；而**全无效的深度会写 `d_min=d_max=0.0`**，此时 `compute_scene_scale` 拿到的 mask 全 False → 直接 raise → 场景被静默跳过。这正是 §2.2 那条结论的底层机制。

## 4. 从自己的图造 shard

仓库没有“给一个 COLMAP 文件夹就出 shard”的通用入口（每个数据集一个 convert 脚本），所以要自己拼装。最小可用流程：

```python
# 在官方仓根目录，conda activate splatfactory 后执行
import json, tarfile
from pathlib import Path
import numpy as np

from splatfactory.datasets.utils.io import encode_depth, write_scene_to_tar

def build_shard(scenes, out_tar: Path):
    """scenes: list[dict]，每项对应一个 scene 的全部视角。同一 tar 必须一次写完。"""
    out_tar = Path(out_tar)
    out_tar.parent.mkdir(parents=True, exist_ok=True)
    index = {}
    with tarfile.open(out_tar, "w") as tar:      # 不压缩，成员本来就是 JPEG/WebP
        for scene in scenes:
            write_scene_to_tar(tar, scene)
            index[scene["key"]] = out_tar.name
    return index

def make_scene(key, images, cameras_12, poses_12, depths=None):
    """
    key        : str       scene ID，不能含 '.'、也不能含 .img/.dep/.cov/.seg/.dyn
    images     : list[bytes]  每个视角一张 JPEG/PNG 的原始字节（透传，不解码不重编码）
    cameras_12 : (N, 6)  [w, h, fx, fy, cx, cy]
    poses_12   : (N, 12) c2w，R 展平 + t
    depths     : list[np.ndarray] | None  每视角 (H, W) float32，米
    """
    assert "." not in key, f"scene key 不能含点号: {key}"
    scene = {
        "key": key,
        "cameras": np.asarray(cameras_12, dtype=np.float32),
        "poses": np.asarray(poses_12, dtype=np.float32),
        "images": images,
        "num_views": len(images),
        "has_depth": depths is not None,
        "depths": None,
        "depth_ranges": None,
    }
    if depths is not None:
        enc = [encode_depth(d) for d in depths]     # -> (bytes, d_min, d_max)
        scene["depths"] = [e[0] for e in enc]
        scene["depth_ranges"] = [(e[1], e[2]) for e in enc]
    return scene
```

用法：`train-scenes` / `test-scenes` 各写一次，`index.json` 放在 shard 同级。下面这段是最小完整可跑版，只需把 `load_views()` 换成你自己的读图逻辑：

```python
import cv2
from pathlib import Path

def load_views(scene_dir: Path):
    """读一个场景目录，返回 (images, cameras, poses, depths)。
    这里给的是最小骨架——换成你自己的 COLMAP / UE 导出读法。"""
    files = sorted(scene_dir.glob("*.jpg"))                 # 视角按名字排序
    images, cams, poses, depths = [], [], [], []
    for f in files:
        images.append(f.read_bytes())                       # 原始字节，不重编码
        meta = np.load(f.with_suffix(".npz"))               # 你自己的相机/位姿/深度
        cams.append(meta["K"])                              # [w,h,fx,fy,cx,cy]
        poses.append(meta["c2w"][:3, :4].reshape(12))       # 4x4 先截 3x4 再展平！
        depths.append(meta["depth"].astype(np.float32))     # (H,W) 米
    return images, np.stack(cams), np.stack(poses), (depths or None)

def build_tree(src_root: Path, dst_root: Path, names, per_shard=200):
    dst_root.mkdir(parents=True, exist_ok=True)
    index, pending, shard_id = {}, [], 0

    def flush():
        """把 pending 里的场景写成当前 shard，然后清空。必须一次写完。"""
        nonlocal shard_id
        if not pending:
            return
        name = f"shard-{shard_id:06d}.tar"
        with tarfile.open(dst_root / name, "w") as tar:
            for s in pending:
                write_scene_to_tar(tar, s)
                index[s["key"]] = name          # key -> 该 key 所在的 tar 名
        pending.clear()
        shard_id += 1

    for name in names:
        pending.append(make_scene(f"scene_{name}", *load_views(src_root / name)))
        if len(pending) >= per_shard:
            flush()
    flush()                                      # 别漏掉最后不足 per_shard 的那批
    (dst_root / "index.json").write_text(json.dumps(index), encoding="utf-8")

build_tree(Path("raw/mydata/train"), Path("data/mydata/252/train-scenes"), ["s001", "s002"])
build_tree(Path("raw/mydata/test"),  Path("data/mydata/252/test-scenes"),  ["s001"])
```

收尾三件事：

1. **写 `index.json`**：`{scene_key: out_tar.name}`，否则数 scene 会退化成扫 tar。
2. **train / test 必须两棵树**。`trainer.py:1322` 无条件 `DatasetClass(data_conf, split="val")`，它读的是 `test_shard_dir`；只建 train 会在启动时 `FileNotFoundError`。
3. **shard 大小**：默认按 512 MB 轮转。已有 shard 想重切用 `python -m splatfactory.datasets.scripts.repack <dir> --target-size-mb 512`（原地、按场景原子提交，峰值接近 1× 而不是 2×）。

另外两点容易在“分批生成”时踩到：

- **一个 tar 从头写到尾，别中途关掉再以 `"a"` 追加**。`iter_scenes_from_tar` 靠“key 变化就 yield”切场景，追加时若上一个 scene 的成员被切成两段，第二段会被当成一个新场景（`num_views` 只剩后半），然后被静默丢弃。数据量大就多开几个 `shard-00000i.tar`，别往同一个 tar 里补。
- **`{key}` 不要用带点号的原始文件名**（见 §3.2 的命名约束），UE/Blender 导出的 `scene.0001` 这类要先替换成 `scene_0001`。

**这段代码的实测结果**（3 场景 × 5 视角、`per_shard=2`，跑通写入 + `iter_scenes_from_tar` 读回）：

```text
shard-000000.tar -> 2 scenes ['scene_s001', 'scene_s002']
shard-000001.tar -> 1 scene  ['scene_s003']      # flush() 收尾的那批
index.json       -> {'scene_s001': 'shard-000000.tar', 'scene_s002': 'shard-000000.tar',
                     'scene_s003': 'shard-000001.tar'}
```

两点校准过的细节：

- **`index.json` 只被用来数数量**：`_count_scenes`（`composed.py:109`）就是 `len(json.load(...))`，值（tar 文件名）其实没被读。所以 map 写成 `{key: tar_name}` 只是可读性需要；真正找场景是扫 tar 内容。文件缺失会 warning 并退化成扫 tar，慢但能跑。
- **`c2w` 存 4×4 时不能直接 `.reshape(12)`**，会 `cannot reshape array of size 16 into shape (12,)`。一定先 `[:3, :4]` 再展平——这是 `Pose.data_` 只认 12 维（`R.flatten() + t`）决定的。


### 4.1 没有真实深度怎么办

三种路子，按可靠性排序：

| 做法 | 命令 | 代价与风险 |
|---|---|---|
| **DA3 补深度**（官方做法） | `python -m splatfactory.datasets.scripts.extract_depth data/<ds>/<HxW>/train-scenes` | 需要 GPU + `depth_anything_3`；**原地重写 shard**；长序列用 `--max-views 150` 轮转切块保尺度一致 |
| **UE/渲染器直接出 depth** | 打包时按 3.4 的 log 量化写 | 最准，但要确认是**轴向 Z** 不是射线距离（差一个 `cos`） |
| 绕开尺度归一化 | `normalize_baseline: start_end` 或 `scene_scale_constant` | 能跑，但失去 GeoCE 的深度支撑，等于弱化配方 |

`extract_depth` 只补“没有深度”的 scene，已有深度默认跳过；要重算加 `--overwrite`。

## 5. 训练代码必须注意的 6 件事

这几条都是读代码才能发现、看论文看不出来的。

### 5.1 `data.batch_size` 是“全局场景数”，会被除以卡数

```python
# trainer.py:1293 scale_by_device_count
if "batch_size" in data_conf and not batch_size_per_gpu:
    data_conf.batch_size = int(data_conf.batch_size / num_gpus)
logger.info(f"Batch size: global={data_conf.batch_size * num_gpus}, "
            f"per-device={data_conf.batch_size}")
```

252 配置的 384 是 **16 卡 × 24 场景/卡**。单卡直接跑就是 384 个场景/步——按 V=24 算约 1 万张图，必然 OOM。

单卡该填多少有官方参照：`zipsplat-4090.yaml` 写的是 `batch_size: 96`，配 4 张 24 GB 卡，也就是 **24 场景/卡**。所以 3090/4090 单卡从 `24` 起步，显存不够再往下走 §7.6 的退让表。

**注意有个开关能改变语义**：`data.batch_size_per_gpu: true` 时不做除法，`batch_size` 直接当每卡值。自己写配置时如果按“每卡 N 场景”填了数又没设这个开关，多卡跑就会变成 N/卡数，每卡只剩零头。

### 5.2 每 step 的图像数不是常数，会反向波动

采样器每步先随机抽一个视角数 `vc ∈ image_num_range`，再用“图数预算”反推这个 batch 装几个场景：

```python
# multi_view_dataset.py:_sample_batch_params
vc = rng.randint(min_views, max_views)          # 例如 [2, 24]
bs = max(1, self.max_img_per_gpu // vc)         # 图数预算 / 视角数
```

`max_img_per_gpu` 取的是 `conf.get(split + "_batch_size", conf.batch_size)`（`multi_view_dataset.py:153`）——训练时读 `train_batch_size`，没有才退回 `batch_size`；验证时读 `val_batch_size`。按 16 卡 × 24 场景算，每卡每步的真实图像数是 `bs × (vc + 4)`：

| 抽到的 `vc` | `bs`（场景/卡） | 每步图像数 |
|---|---|---|
| 2 | 12 | 72 |
| 4 | 6 | 48 |
| 8 | 3 | 36 |
| 12 | 2 | 32 |
| 24 | 1 | 28 |

**峰值在 `vc` 最小处**（2 个 context 时反推的场景数最多，4 个 target 的附加开销被放大）。所以：

- 估显存按**最坏情况**（约 `bs_max × 6`）算，别按平均值；
- 单卡把 `batch_size` 调小后，`bs` 会更快被 `max(1, ...)` 卡住，此时“多视角”那档实际变成 1 场景 28 图——不会 OOM，但吞吐被拉低；
- 验证集固定 `val_batch_size: 1`，不参与这套退让逻辑，别指望它省显存。

### 5.3 `cam_enc` 恒冻结，backbone 可训但只给 0.1 倍学习率

`models/encoders/dav3_encoder.py:123` 里 `self.cam_enc.requires_grad_(False)` 是**无条件**的，`trainable: True` 也解不开；紧跟着连 `backbone.camera_token` 也一起冻了。源码给的注释很直白：它只在 prior 注入的步上跑，放开会让 DDP 报 unused params。所以：

- 相机 tokenizer 不动，别去找它的 lr 配置项；
- backbone 权重会更新，但 `lr_scaling` 里 `backbone: 0.1` 让它只拿 1/10 学习率；
- `cam_dec.fc` 被单独抬回 `1.0`，这是有意为之——解码头要重新适配你的相机分布。

如果你的数据相机分布和官方差很远（比如 UE 出的理想 pinhole），**优先动 `cam_dec.fc` 的学习率，而不是放开 `cam_enc`**。

### 5.4 `view_schedule` 会偷偷改你的视角上限

`_step_view_schedule()`（`trainer.py:533`）在前 `num_steps × view_schedule` 步里，把训练集的最大视角数从 `image_num_range[0]` 线性抬到 `image_num_range[1]`：

```python
T = int(self.conf.num_steps * schedule_frac)
current_max = min_views + (self._view_target - min_views) * (min(self.tot_it, T) / T)
self._train_dataset.update_image_num_range(new_max=round(current_max))
```

252 配置是 `0.5`（前 22.5 万步视角数才升满），504 微调配置是 `0`。**它只影响训练集**，验证集不动。含义：

- 训练早期实际只看 2–4 个 context，别以为一上来就在练 24 视角；
- 你的数据如果每场景刚好 36 张（下限），`vc` 想抽到 24 是有可能的，但 `bounded` 采样器还要凑够距离约束，实际常抽到中低段；
- 想让模型早点见到多视角，把 `view_schedule` 调小（如 `0.1`），或直接设 `0`。

### 5.5 `--restore` 会用旧 `config.yaml` 覆盖你的新参数

```python
# train.py:140
if args.restore:
    restore_conf = OmegaConf.load(output_dir / "config.yaml")
    conf = OmegaConf.merge(restore_conf, conf)
```

`--restore` 会先把上次实验目录里的 `config.yaml` 读回来，再把命令行 dotlist **合并**上去。所以：

- 命令行显式给的值优先，能改；
- 命令行没给的字段，**沿用上次的值**，哪怕你改的是父级 yaml；
- 换了数据集还想接着训，一定要显式传 `data.dataset_dir=...`，否则继续吃旧路径。

另外 `create_training_dir` 对已存在目录默认 `FileExistsError`，四个开关语义要分清（`train.py:93`）：

| 开关 | 行为 |
|---|---|
| `--restore` | 允许多次写同一目录 + 读回旧 `config.yaml` 合并，续训用 |
| `--overwrite` | 只做 `exist_ok=True`，**不删旧文件**；旧 ckpt 还留着，`keep_last_checkpoints` 会照旧清理 |
| `--clean` | `shutil.rmtree(output_dir)` **真删整个实验目录**再重建，谨慎 |
| `--ablate` | 在目录下自动建递增子目录 `0/ 1/ 2/`，做消融时避免互相覆盖 |

### 5.6 验证集的图像来自 `test` 树，不是从 `train` 里切

`launch_training`（`trainer.py:1315`）无条件构造两份数据集：

```python
train_dataset = DatasetClass(data_conf, split="train")   # 读 train_shard_dir
val_dataset   = DatasetClass(data_conf, split="val")     # 读 test_shard_dir
```

**只建 train 目录会在启动时直接 `FileNotFoundError`**，不会“先训着再说”。所以你至少要准备两棵树，验证集几十个场景即可（`val_samples: 512` 是按样本抽，不是按场景）。

## 6. 开工前检查清单

自建数据踩的坑 90% 集中在“能启动但静默跳过”。逐条自查：

| # | 检查项 | 怎么验 | 不对会怎样 |
|---|---|---|---|
| 1 | `dataset_dir` 下两棵树都在 | `ls data/mydata/252/` 应有 `train-scenes/`、`test-scenes/`，各自含 `shard-*.tar` | `resolve_shard_paths` 拿不到 shard，直接 `FileNotFoundError` |
| 2 | 每场景视角数 ≥ 采样器下限 | 见 §1.1（RE10K 参数 = 36） | 场景被 `_scene_ok` 静默丢弃 |
| 3 | 深度存在且有效 | 抽查 `decode_depth`，正数占比 > 50% | `compute_scene_scale` raise → 被吞 → 全量跳过 |
| 4 | 位姿是 c2w 12 维 | 打印一个场景的 `poses.npy` | 场景镜像，loss 能降但结果错乱 |
| 5 | 内参主点用 COLMAP 口径 | `cx,cy` 为 `w/2, h/2` 附近 | 相机偏移，重影 |
| 6 | `rank` 采样器需要 rankings | tar 内有 `*.rankings.npy`、`*.overlaps.npy` | 用 `rank` 时全部被判“成员缺失” |
| 7 | 目录名与实际内容分辨率一致 | 读一个 scene 的图 shape，确认与目录名（如 `540x960`）相符 | aspect 预设全被过滤 |
| 8 | `preprocessing.resize` 与数据准备阶段对齐 | 252 与 504 不混用同一个实验 | 尺寸不匹配，报错或视野被裁 |
| 9 | scene key 不含 `.` / `.img` 等子串，成员连续 | 读一个 tar，用 `m.name.split(".")[0]` 反推 key 数，应等于你写入的场景数 | 多个场景被并成一个：`meta`/`cameras`/`poses` 互相覆盖、`num_views` 与数组长度对不上 → 抛异常被吞，静默丢场景 |


**最容易忽略的是第 3 条**：日志只有 `debug` 级别，默认看不到。跑之前先把 `logging` 调到 `DEBUG` 一分钟，确认没有成片 `Failed to process scene`。

### 6.1 三十秒连通性自检

不用真训，先确认数据集能被读出来。注意类要和 yaml 的 `name` 对上——`composed` 对应 `ComposedDataset`，它要求 `childs`（见 §3.1 的目录约定）：

```bash
python - <<'PY'
from omegaconf import OmegaConf
from splatfactory.datasets import get_dataset
import glob, tarfile

# 先查 shard 本身有没有崩（第 9 条，不用构造数据集）
for tar_path in sorted(glob.glob("data/mydata/252/train-scenes/shard-*.tar"))[:1]:
    keys = set()
    with tarfile.open(tar_path) as tf:
        for m in tf:
            keys.add(m.name.split(".")[0])
    print(f"{tar_path}: {len(keys)} scene(s), 前三个 = {sorted(keys)[:3]}")
    # 这里如果只有 1 个 key 而你写了多个场景 → key 里带点号了

# 单 child 的最小配置，不必准备十份数据
conf = OmegaConf.create({
    "name": "composed",
    "batch_size": 4,
    "num_workers": 0,
    "image_num_range": [2, 8],
    "num_target_views": 4,
    "childs": {
        "mine": {
            "dataset_dir": "data/mydata/252",     # 下面应有 train-scenes / test-scenes
            "dataset_name": "mydata",
            "weight": 1.0,
            "preprocessing": {"resize": 252, "edge_divisible_by": 14},
            "view_sampler": {"name": "bounded",
                             "min_distance_between_context_views": 8,
                             "gap_multiplier": 11,
                             "max_distance_between_consecutive_views": 22},
        }
    },
})

DatasetClass = get_dataset(conf.name)
ds = DatasetClass(conf, split="train")
print("shards:", len(ds.shard_paths))
b = next(iter(ds.get_loader(num_workers=0)))
print("scene:", b["scene"])
print("context views:", b["context"]["image"].shape)
print("target  views:", b["target"]["image"].shape)
print("depth valid frac:", b["context"]["depth_mask"].float().mean().item())
PY
```

看四个数就够：

- 第一行打印的 **scene 数**对不对（第 9 条）：写进去 10 个场景却只打印 1 个，就是 key 里带了点号或成员不连续；
- `shards: 0` 或直接 `FileNotFoundError` → 路径/命名不对（第 1 条）；
- 构造函数成功但第一次取 batch 就抛异常 → 大概率是场景全被过滤（第 2、3 条），此时把 `logging` 调到 `DEBUG` 会看到成片 `Failed to process scene`；
- `depth valid frac` 接近 0 → 深度没写进去或全被裁掉，`compute_scene_scale` 会 raise；
- `context`/`target` 的 batch 维是 `num_context_views` 和 `num_target_views`，`H×W` 应等于 `resize` 后被 `edge_divisible_by` 对齐的结果。

注意 `get_loader` 用的是 `DataLoader(self, batch_size=None, ...)`——**不做 collate**，`next(iter(...))` 拿到的就是一个 sample 的原始 dict。所以 `b["context"]["image"].shape` 是 `[V, 3, H, W]`（`V` = 这个 sample 抽到的 context 数），而不是带 batch 维的 `[B, V, 3, H, W]`。看到没有 batch 维别以为配置错了；真正的拼 batch 在 `trainer` 侧按场景列表做。


单个 sample 的结构是 `{name, scene, scene_index, context, target, overlap}`，`context`/`target` 各自含 `image`、`camera`、`pose`、`depth`、`depth_mask`、`index` 等（见 `_sample_scene` 的返回处）。所以上面不要去找 `num_views` 这类扁平字段。

`get_dataset()` 走 `splatfactory.datasets.__init__.get_dataset`，按 `name` 动态找 `BaseDataset` 子类，和 `trainer.py:1320` 用的是同一个入口——这个自检通过，训练时的数据构造就不会再出问题。

## 7. 参数配置一览

字段来源三个 yaml 层层合并：`configs/zipsplat.yaml` → `configs/data/*.yaml` → `configs/model/zipsplat.yaml`。改哪一层都行，命令行 dotlist 优先级最高。

### 7.1 数据集相关（`data:`）

| 字段 | 252 默认 | 说明 |
|---|---|---|
| `dataset_dir` | — | 数据根目录，下面直接是 `<HxW>/` |
| `train_shard_dir` / `test_shard_dir` | `${.dataset_dir}/train-shards` / `test-shards` | **base 默认是 `-shards`，官方 yaml 全改成 `-scenes`**；自己建目录时按 yaml 写，别照 base 默认 |
| `shard_glob` | `shard-*.tar` | 文件命名模式，`mipnerf360` 例外为 `*.tar` |
| `batch_size` | 384 | **全局场景数**，会被除以卡数（§5.1） |
| `train_batch_size` / `val_batch_size` / `test_batch_size` | 未设 / 1 / 1 | 设了就优先于 `batch_size`（§5.2 的 `max_img_per_gpu` 读这个） |
| `num_workers` | 8 | 会被"不超过 shard 数"自动压；单机小数据改成 2 |
| `prefetch_factor` | 2 | `num_workers=0` 时自动失效 |
| `image_num_range` | `[2, 24]` | 每步随机抽的视角数区间 |
| `aspect_ratios` | `[1.0]` | 每 batch 抽一次；252 只有 1:1，504 有七档 |
| `zoom_range` | `None` | 逐场景中心缩放裁剪，**只在训练生效**；504 为 `[0.8, 1.0]` |
| `view_schedule` | 0.5 | 视角上限爬坡所占训练比例（§5.4） |
| `num_target_views` | 4 | 每场景额外渲染 4 个 target 做监督 |
| `reference_frame` | `first_cam` | 五选一：`first_cam` / `mean_cam` / `mean_cam_down` / `mean_point` / `mean_point_const` |
| `use_relative_poses` | true | 相对位姿归一化开关 |
| `random_reference_view` | true | 训练时打乱 context 顺序 |
| `normalize_baseline` | false | 尺度归一化方式（§2.2），true 等价 `start_end` |
| `scene_scale_constant` | `None` | 直接给固定逆尺度，跳过 `compute_scene_scale` |
| `near` / `far` | 0.1 / 100.0 | base 默认 `0.01/1000.0`，官方数据配置收紧到 `0.1/100.0`；**在归一化后的坐标系里，不除以场景 scale** |
| `max_pose_jump_ratio` | 10.0 | 相邻帧位移 > 中位数的多少倍就丢该场景；`0` 关闭（504 全局关、视频类单独开） |
| `augmentations` | `{"name": "identity"}` | 默认无增强 |
| `samples_per_epoch` / `val_samples` | 未设 | 见 §7.3 的 epoch 长度说明 |
| `max_samples` | `None` | 兜底上限，`samples_per_epoch` 未设时它决定 `cap` |
| `preprocessing.resize` | 252 | 与目录名必须一致 |
| `preprocessing.edge_divisible_by` | 14 | 边长整除约束（patch 对齐） |

### 7.2 模型相关（`model:`）

| 字段 | 252 默认 | 说明 |
|---|---|---|
| `backbone` | `da3g` | 档位见 §7.4 |
| `gaussians_per_token` | 32 | 每 token 出多少高斯，504 微调降到 16 |
| `query_sample_ratio` | `[0.1, 1.0]` | 查询采样比例区间 |
| `geoce_loss.weight` | 0.05 | GeoCE 权重，**关掉深度会退化为 0** |
| `cone_loss.weight` | 0.01 | 锥形约束 |
| `pose_loss.weight` | 0.05 | 位姿正则 |
| `gaussian_head.mse_weight` | 1.0 | `use_l1_loss: true` 时作 L1 的乘子 |
| `gaussian_head.lpips_weight` | 0.1 | LPIPS 权重 |
| `gaussian_head.depth_weight` | 0.0 | 渲染深度 L1，**默认全关** |
| `gaussian_head.scale_penalty_weight` | 1.0 | 高斯尺度惩罚（防过曝大面片） |
| `gaussian_head.opacity_penalty_weight` | 1.0 | 不透明度惩罚 |

### 7.3 训练相关（`train:`）

| 字段 | 252 默认 | 504 微调 | 说明 |
|---|---|---|---|
| `num_steps` | 450_000 | 100_000 | 总步数 |
| `lr` | 3e-4 | 1e-4 | 微调降一档 |
| `warmup_steps` | 5% | 同 | `LinearLR` 预热 |
| `lr_schedule` | `SequentialLR`(Linear→Cosine) | 同 | 见 §7.5 |
| `clip_grad` | 1.0 | 同 | 梯度裁剪 |
| `optimizer_options.weight_decay` | 0.05 | 同 | |
| `betas` | `[0.9, 0.95]` | 同 | 比常规更快的二阶矩 |
| `lr_scaling` | `{"cam_dec.fc":1.0, backbone:0.1}` | 同 | 首个匹配生效（§5.3） |
| `mixed_precision` | bfloat16 | 同 | |
| `compile` | `default` | 同 | torch.compile |
| `eval_every_epoch` | 40 | 2 | 按 epoch 计，不是按步 |
| `save_every_iter` | 5000 | 同 | |
| `keep_last_checkpoints` | 1 | 同 | 只留最新，旧 ckpt 被删 |
| `best_key` | `loss/total` | 同 | 选最优 ckpt 的指标 |

**`eval_every_epoch` 是按 epoch 算的**，所以要先知道一个 epoch 有多长。`ComposedDataset.__iter__`（`composed.py:168`）的逻辑是：

```python
total = conf.samples_per_epoch if split == "train" else conf.val_samples
cap   = ceil(total / (world_size * num_workers)) if total else conf.max_samples
...
if cap is not None and yielded >= cap: break
```

- **`samples_per_epoch` 有值**（504 配置 80_000）→ 每 worker 只产出 `cap` 个场景就结束，epoch 长度可控，约 `80_000 / 全局 batch_size` 步；
- **`samples_per_epoch` 为 `None`**（252 配置和你的自建配置默认就是 `None`）→ `cap` 也退化成 `max_samples`，通常是 `None`，**一个 epoch 就是把你所有 shard 全遍历一遍**。官方 252 配置注释说 ~680 步一个 epoch 就是按它那份全量数据算出来的。

自己造数据时这一点很关键：**你的数据越小，epoch 越短**，`eval_every_epoch: 40` 可能几十步就跑一次评估，日志被评估刷满、checkpoint 也被反复覆盖。自建数据建议显式设 `samples_per_epoch`（例如 `2000 × 全局 batch`），或把 `eval_every_epoch` 调大。

### 7.4 backbone 四档

配置在 `configs/model/backbone/da3{s,b,l,g}.yaml`，字段完全同构，只有数值不同：

| 档 | `backbone` 值 | `vit_name` | `embed_dim` | `out_layers` / `alt_start` | 权重文件名 |
|---|---|---|---|---|---|
| 小 | `da3s` | `vits` | 384 | `[5, 7, 11]` / 5 | `da3-small.pth` |
| 中 | `da3b` | `vitb` | 768 | `[5, 8, 11]` / 4 | `da3-base.pth` |
| 大 | `da3l` | `vitl` | 1024 | `[11, 17, 23]` / 8 | `da3-large.pth` |
| 巨 | `da3g` | `vitg` | 1536 | `[19, 29, 39]` / 13 | `da3-giant-1.1.pth` |

四档都带 `cat_token: true`、`with_camera_enc: true`，且 `alt_start == qknorm_start == rope_start`（三者成组移动，只改一个会让注意力层的 qk-norm / rope 起始 block 错位）。`num_heads` 不是手填的，由 `${eval:'embed_dim // 64'}` 自动算（1536/64=24 头）。

权重放在仓库根的 `weights/da3/` 下，默认 `da3g.yaml` 写死 `weights/da3/da3-giant-1.1.pth`。显存不够时最先降的是这一档，其次是 `gaussians_per_token`。

### 7.5 学习率调度是两段拼的

```python
SequentialLR(
    milestones=[warmup_steps],
    schedulers=[LinearLR(start_factor=1/warmup, end_factor=1.0),
                CosineAnnealingLR(T_max=num_steps - warmup, eta_min=0.0)],
)
```

含义：前 5% 步线性升温，之后余弦退到 0。`on_epoch: false` 表示按**步**更新，不是按 epoch。

自己改 `num_steps` 时，`warmup_steps` 是 `${eval:...}` 动态算的，会跟着变——但如果你手写死一个数，余弦段的 `T_max` 也要同步改，否则尾巴退不完。

### 7.6 降配参考

仓库自带 `configs/zipsplat-4090.yaml`（4×4090 24 GB，24 图/GPU），直接当模板：

```yaml
defaults:
  - zipsplat        # 继承主配置
model:
  use_checkpoint: true          # 梯度检查点，省显存换时间
  backbone:
    use_checkpoint: true
data:
  batch_size: 96                # 全局值 = 4 卡 × 24；单卡要改成 24
train:
  eval_every_epoch: 5           # 注意在 train: 下，不在 eval:
  benchmark_every_epoch: 5
```

单卡按这个顺序往下退，每次只动一档：

| 顺序 | 动作 | 影响 |
|---|---|---|
| 1 | `data.batch_size: 96 → 24` | 单卡必做，否则 96 场景/步直接 OOM |
| 2 | `model.use_checkpoint: true` + `backbone.use_checkpoint: true` | 省显存，慢约 20–30% |
| 3 | `model.backbone: da3g → da3l` | 1536→1024 维，参数量约减半 |
| 4 | `backbone: da3b` / `da3s` | 再降到 768 / 384 维 |
| 5 | `gaussians_per_token: 32 → 16` | 高斯数减半，渲染开销降 |

benchmark 是**必踩的坑**，必须先关掉再自己训。`launch_training` 启动时会对 `run_benchmarks` 里每个名字做 `eval.get_benchmark(bench_name)` 校验（`trainer.py:1342`）；更关键的是 `train_loop` 收尾处无条件跑一次 `run_all_benchmarks(..., force=True)`（`trainer.py:1287`）——**就算你一步都没到 `benchmark_every_epoch`，训练跑到末尾也会去读 `data/dl3dv/540x960` 和 RE10K benchmark 索引，没备数据就是这里崩**。

触发时机要看清：`self.epoch += 1` 在评估**之前**（`trainer.py:1256`），所以 epoch 末尾比较的是自增后的值。第一轮训练完 `epoch=1`，`1 % 40 != 0`，**不会**跑 benchmark；真正周期性触发要等到 `epoch=40`。而 `eval_init` 默认 `False`（`trainer.py:1221`），所以开头也不会强跑。

所以自己训、没准备 benchmark 数据时，必须显式清空：

```yaml
train:
  run_benchmarks: []      # 默认是空元组 ()，主配置覆盖成了两个 benchmark
```

---

下面进入第二半：**用官方数据集复现**。命令都在官方仓根目录执行（`cd ~/repos/ZipSplat`），前缀统一为 `python -m splatfactory.datasets.scripts.`，表里用 `<ds>` 简写。每个数据集都是“先 download 原始包、再 convert 成 shard”。

## 8. 三个层级，别一上来就全量

| 层级 | 数据 | 体积量级 | 能验证什么 |
|---|---|---|---|
| **T0 冒烟** | DL3DV `--sample`（50 scenes）+ Hypersim `--max-scenes 10` | 几十 GB | 转换、补深度、rankings、训练循环是否通（各自验一遍转换，训练只挂一个 child） |
| **T1 复现 252** | RE10K 全量 + DL3DV 960P（至少一个子集） | RE10K 未知 + DL3DV ≈2.8 TB（960P 全量） | 论文 252 配方（450K steps）能否复现趋势 |
| **T2 精调 504** | 10 个数据集 / **12 个 child** 全量 | **≈10 TB 级** | 发布 504 模型的第二阶段 |

T0 要自己写一份只含一个 child 的小 yaml（把 `re10k_dl3dv_252.yaml` 复制一份，删掉一个 child、`batch_size` 按单卡填 24、`num_workers` 改 2）；官方配置没有为小样准备开关，直接改 `batch_size` 跑全量数据会一直读盘。

## 9. T1（252 配方）：只要这两个

| 数据集 | 官方入口 | 下载命令 | shard 目录（配置写死） | 体积 | 备注 |
|---|---|---|---|---|---|
| **RE10K** | [项目页](https://google.github.io/realestate10k/)；脚本从 `http://schadenfreude.csail.mit.edu:8000/re10k.zip` 取 360×640 的 `.torch` 包 | `<ds>.re10k.download` → `<ds>.re10k.convert --num-workers 8` | `data/re10k/360x640` | 以站点为准（未取到 Content-Length） | 无需补深度（转换后由 DA3 补）；`--split all` 默认同时出 train/test |
| **DL3DV** | [GitHub](https://github.com/DL3DV-10K/Dataset)；HF：[960P](https://huggingface.co/datasets/DL3DV/DL3DV-ALL-960P)（另有 [480P](https://huggingface.co/datasets/DL3DV/DL3DV-ALL-480P) / 2K / 4K） | `<ds>.dl3dv.download --subset 1K --resolution 960P` → `<ds>.dl3dv.convert --resolution 960P --split all --num-workers 8` | `data/dl3dv/540x960` | 960P 全量 ≈2.8 TB | **gated**：先在 HF 页面同意条款并 `huggingface-cli login`；**必须 960P**，480P 出的是 270×480，目录名对不上 |

小样验证用：`<ds>.dl3dv.download --sample`（1K 子集前 50 个 scene）。它仍然走 960P 目录，只是数量少。

## 10. T2（504 精调）：追加的八个

| 数据集 | 官方入口 | 下载命令 | shard 目录 | 体积（近似） | 授权 / 前置 |
|---|---|---|---|---|---|
| **ScanNet++** | [官网](https://kaldir.vc.in.tum.de/scannetpp/)（需签署协议） | **无下载脚本**，用 [官方 toolkit](https://github.com/scannetpp/scannetpp) 拉到 `data/scannetpp_v2/`，再 `<ds>.scannetpp.convert --device dslr --max-side 1080 --num-workers 12`（iPhone 另跑一次 `--device iphone`） | `data/scannetpp_v2/720x1080/{dslr,iphone}` | 1,000+ 场景，体积以协议页为准 | 需注册与签署；转换要 `open3d`（自己渲染 mesh 深度） |
| **ARKitScenes** | [GitHub](https://github.com/apple/ARKitScenes)；资源域 `docs-assets.developer.apple.com/.../arkitscenes/v1` | `<ds>.arkitscenes.download --max-scenes 20`（全量去掉该参数）→ `<ds>.arkitscenes.convert --num-workers 8` | `data/arkitscenes/{480x640,640x480}`（横/竖两棵树） | raw 数 TB（5,047 videos） | 无 gating；转换按 Apple sky_direction 转正，横竖分树 |
| **Hypersim** | [GitHub](https://github.com/apple/ml-hypersim)；资源域 `.../hypersim/v1/scenes` | `<ds>.hypersim.download --max-scenes 10` → `<ds>.hypersim.convert --num-workers 8` | `data/ml-hypersim/768x1024` | 461 场景 ≈1.9 TB | 无 gating；自带深度 |
| **UnrealStereo4K** | [SMD-Nets](https://github.com/fabiotosi92/SMD-Nets)；HF [fabiotosi92/UnrealStereo4K](https://huggingface.co/datasets/fabiotosi92/UnrealStereo4K) | `<ds>.unrealstereo4k.download --sample`（8 环境/80 chunks）→ `<ds>.unrealstereo4k.convert` | `data/unrealstereo4k/540x960` | 较小（80 chunks） | 无 gating；帧间曝光会闪，权重只给 0.02 |
| **BlendedMVS** | [GitHub](https://github.com/YoYo000/BlendedMVS) / [v1.0.0](https://github.com/YoYo000/BlendedMVS/releases/tag/v1.0.0) | `<ds>.blendedmvs.download --num-workers 4` → `<ds>.blendedmvs.convert` | `data/blendedmvs/576x768` | 低分辨率 113 场景 ≈30 GB | 无 gating；唯一航拍/大尺度内容 |
| **CO3D** | [GitHub](https://github.com/facebookresearch/co3d)；`links.json` 给分片地址 | `<ds>.co3d.download --single-sequence`（≈8.9 GB）或 `--categories ... --max-chunks 2` → `<ds>.co3d.convert` | `data/co3d/720s` | 完整 ≈5.5 TB | 非商业条款；50 类物体环绕 |
| **WildRGB-D** | [GitHub](https://github.com/wildrgbd/wildrgbd)；HF [hongchi/wildrgbd](https://huggingface.co/datasets/hongchi/wildrgbd) | `<ds>.wildrgbd.download --max-categories 4` → `<ds>.wildrgbd.convert` | `data/wildrgbd/640x480` | ≈3.37 TB 压缩 | 非商业条款 |
| **G-Objaverse** | [richdreamer 数据页](https://github.com/modelscope/richdreamer/tree/main/dataset/gobjaverse)；tar 源 `virutalbuy-public.oss-cn-hangzhou.aliyuncs.com/share/aigc3d` | `<ds>.gobjaverse.download --max-per-category 10`（全量去掉该参数）→ `<ds>.gobjaverse.convert` | `data/gobjaverse/512x512` | 约 280K 对象、40 视角/对象 ≈6.5 TB | 继承底层 Objaverse 资产许可；**`Human-Shape` 是静态人物形体资产，不是真人集** |

上表 8 个数据集 + RE10K + DL3DV = 10 个数据集；但在 `multi_dataset_504.yaml` 里拆成了 **12 个 child**（ScanNet++ 拆 dslr/iphone，ARKitScenes 拆 landscape/portrait），因为同数据集的两种子集目录和 aspect 预设不同。所以配 504 时要准备的是 **12 条 shard 路径**，不是 10 条。

各 child 的 `weight` 不是数据量占比，而是**场景抽样份额**（小集合靠 shard 重复被过采样）：

| child | weight | 采样器 | 说明 |
|---|---|---|---|
| `dl3dv` | 0.32 | `bounded` | 主力；`min_dist 8 / initial 5 / gap×11 / max_consec 22` |
| `scannetpp_dslr` | 0.10 | `rank` | `min_overlap 0.2 / max_overlap 0.8` |
| `co3d` | 0.10 | `random` | 物体环绕，只能随机选 |
| `wildrgbd` | 0.08 | `random` | 同上 |
| `scannetpp_iphone` | 0.08 | `rank` | 与 dslr 同参数，不同目录 |
| `arkitscenes_landscape` | 0.07 | `rank` | 480x640 源，只许横构图 |
| `arkitscenes_portrait` | 0.06 | `rank` | 640x480 源，只许竖构图 |
| `re10k` | 0.05 | `bounded` | 从 252 的 0.5 降到 0.05——"已被 init ckpt 解决，只留个底" |
| `hypersim` | 0.05 | `rank` | 自带深度 |
| `gobjaverse` | 0.05 | `random` | 512x512 渲染，能覆盖全部 aspect |
| `unrealstereo4k` | 0.02 | `rank` | 只有 8 环境，曝光会闪 |
| `blendedmvs` | 0.02 | `random` | 唯一航拍大尺度 |

两个只有 504 才有的字段：`zoom_range: [0.8, 1.0]`（逐场景中心缩放裁剪，按源分辨率 clamp 保证不上采样）、`max_pose_jump_ratio: 0`（关掉——开着会砍掉约 27% 的 ScanNet++ 窗口，那是手持晃动不是镜头切换；视频类 child 单独开回 10.0）。

## 11. 转完之后按这个顺序处理

```bash
cd ~/repos/ZipSplat
M="python -m splatfactory.datasets.scripts"

# 1) 补深度：只有 RE10K / DL3DV 需要（其它数据集自带 depth）
$M.extract_depth data/re10k/360x640/train-scenes
torchrun --nproc_per_node=4 -m splatfactory.datasets.scripts.extract_depth \
    data/dl3dv/540x960/train-scenes --max-views 150

# 2) 共视排序：只有 rank 采样器的数据集需要（注意 ScanNet++/ARKitScenes 各有两棵树）
$M.precompute_rankings data/scannetpp_v2/720x1080/dslr/train-scenes \
                        data/scannetpp_v2/720x1080/dslr/test-scenes
$M.precompute_rankings data/scannetpp_v2/720x1080/iphone/train-scenes \
                        data/scannetpp_v2/720x1080/iphone/test-scenes
$M.precompute_rankings data/arkitscenes/480x640/train-scenes \
                        data/arkitscenes/480x640/test-scenes
$M.precompute_rankings data/arkitscenes/640x480/train-scenes \
                        data/arkitscenes/640x480/test-scenes
$M.precompute_rankings data/ml-hypersim/768x1024/train-scenes \
                        data/ml-hypersim/768x1024/test-scenes
$M.precompute_rankings data/unrealstereo4k/540x960/train-scenes \
                        data/unrealstereo4k/540x960/test-scenes

# 3) 可选：先缩一次，省训练时的 CPU/IO
$M.resize data/dl3dv/540x960/train-scenes \
    --output data/dl3dv/252x252/train-scenes --size 252 --num-workers 32
```

三点约束：

- `extract_depth` 是**原地重写 shard**（临时文件 + 原子替换），会调 DA3 GIANT-LARGE，要有 GPU 与足够时间；`--max-views` 把长序列轮转切块，保证同一场景的深度尺度一致。
- `precompute_rankings` 也写回 shard，已算过的场景会跳过，重跑是 no-op；它只读 depth/camera/pose，所以**必须放在补深度之后**。
- 若做了第 3 步预缩，记得把 yaml 里的 `dataset_dir` 改到 `252x252` 那份，否则仍是训练时在线缩放。

## 12. 六个容易踩的坑（官方数据集）

1. **DL3DV 默认 480P**。不加 `--resolution 960P` 会下 270×480，而配置要 `data/dl3dv/540x960`，目录名对不上不是“改个名”能修的。
2. **DL3DV 是 gated**。脚本会先 `verify_hf_access`，没在 HF 页面同意条款就直接退出，报错里给了页面链接。
3. **ScanNet++ 没有下载脚本**。它是签协议分发的，必须用官方 toolkit 自己拉，再 `convert`；别找第三方转存。
4. **G-Objaverse 的 `Human-Shape` 不是真人数据**。它是静态人物形体资产的多视角渲染，学不到身份与真人皮肤细节；要人像得自建身份/sequence 划分。
5. **小数据集的权重不是数据量**。0.02 的 UnrealStereo4K 靠重复 shard 被过度采样，`权重 × 总样本` 不能当“访问了多少唯一场景”。
6. **全量 504 是 TB 级投入**。先跑 T0 冒烟再决定：转换链、补深度、rankings 任一环节错了，全量重来代价很高。

## 13. 与本仓的关系

本仓 `zipsplat/` 只用发布权重做推理、TokenBA 和导出，**不需要上面任何训练数据**。这套流程只在你要自己重训或微调 `splatfactory` 时才用得上；跑通 T0 也不等于复现了官方指标。
